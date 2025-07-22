import sys
import os
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from qt_opengl_canvas import QtAdvanced3DCanvas
import threading
from data_in_image import DataInImage
from estimator import Estimator
from boundary_extractor import extract_surface_only, BoundaryExtractor
from Extract_Figures_FV import (Export_Spine_as_Text, Export_Spine_as_tiff, Open_TIFF_fig, 
                                Import_3D_segment_from_tiff_figure, Generate_3D_Segment_Library, 
                                Geneate_Estimations, Plot_matrix_scatter, Plot_3D_Matrix_Line_Test,
                                set_matplotlib_display, get_matplotlib_display_status)
import Extract_Figures_FV


class QtTIFFViewer3D(QtWidgets.QMainWindow):
    """Drop-in replacement for the old TIFFViewer3D, implemented with PySide6."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D TIFF Viewer (Qt Edition)")
        self.resize(1200, 800)

        # ------------------------------------------------------------------
        # Central widget with tabs
        # ------------------------------------------------------------------
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # --------------------------------------------------------------
        # Metadata panel (same style as Color panel) – will sit left in splitter
        # --------------------------------------------------------------
        self._metadata_widget = QtWidgets.QWidget()
        self._metadata_widget.setMinimumWidth(220)
        self._metadata_widget.setVisible(False)

        # Wrap contents in a scroll area (like right panel)
        _meta_scroll = QtWidgets.QScrollArea()
        _meta_scroll.setWidgetResizable(True)
        _meta_container = QtWidgets.QWidget()
        self._metadata_form = QtWidgets.QFormLayout(_meta_container)
        _meta_scroll.setWidget(_meta_container)

        meta_layout = QtWidgets.QVBoxLayout(self._metadata_widget)
        # Header label
        _meta_header = QtWidgets.QLabel("Main Metadata")
        _meta_header.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        _meta_header.setAlignment(QtCore.Qt.AlignCenter)
        meta_layout.addWidget(_meta_header)
        meta_layout.addWidget(_meta_scroll)

        # Keep a dict of value-labels so we can update quickly
        self._metadata_labels = {}

        # (Connection to currentChanged will be added after _tab_widget is defined)

        # Main canvas area with tabs
        self._tab_widget = QtWidgets.QTabWidget()
        self._main_canvas = QtAdvanced3DCanvas(self)
        self._tab_widget.addTab(self._main_canvas, "Full Image")
        # Now that _tab_widget exists, connect the signal
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # ------------------------------------------------------------------
        # Right-hand control panel (will be placed in a splitter)
        # ------------------------------------------------------------------
        control_widget = QtWidgets.QWidget()
        control_widget.setMinimumWidth(220)  # sensible minimum; user can resize wider

        # Use a horizontal splitter so the user can drag the divider
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # Order: metadata | main canvas | right control panel
        self._splitter.addWidget(self._metadata_widget)
        self._splitter.addWidget(self._tab_widget)
        self._splitter.addWidget(control_widget)
# set stretch so main canvas gets more space by default
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setStretchFactor(2, 1)

        # Add splitter to the main layout
        main_layout.addWidget(self._splitter)

        # Control tabs
        self._control_tabs = QtWidgets.QTabWidget()
        self._color_tab = QtWidgets.QWidget()
        self._control_tabs.addTab(self._color_tab, "Colors")

        control_layout = QtWidgets.QVBoxLayout(control_widget)
        control_layout.addWidget(self._control_tabs)

        # Status bar
        self._status = QtWidgets.QStatusBar()
        self.setStatusBar(self._status)

        # ----------------- Color Tab -----------------
        color_layout = QtWidgets.QVBoxLayout(self._color_tab)

        # File open (collapsible panel)
        self._open_toggle_btn = QtWidgets.QPushButton("Open Figure ▼")
        self._open_toggle_btn.setCheckable(True)
        self._open_toggle_btn.setChecked(False)
        color_layout.addWidget(self._open_toggle_btn)

        # Collapsible, scrollable container for figure-loading options
        self._open_panel_scroll = QtWidgets.QScrollArea()
        self._open_panel_scroll.setWidgetResizable(True)
        _open_panel_container = QtWidgets.QFrame()
        _open_panel_container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._open_panel_scroll.setWidget(_open_panel_container)
        self._open_panel_scroll.setVisible(False)
        open_panel_layout = QtWidgets.QVBoxLayout(_open_panel_container)

        # Button to choose the TIFF/figure path
        self._select_path_btn = QtWidgets.QPushButton("Select Path")
        self._select_path_btn.clicked.connect(self._select_path_dialog)
        # Place Select Path at the top of dropdown
        open_panel_layout.addWidget(self._select_path_btn)

        # Display selected path
        self._selected_path_label = QtWidgets.QLabel("No file selected")
        self._selected_path_label.setWordWrap(True)
        self._selected_path_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        open_panel_layout.addWidget(self._selected_path_label)

        # Create Open button (added to layout later, after Performance Settings)
        self._open_selected_btn = QtWidgets.QPushButton("Open")
        self._open_selected_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self._open_selected_btn.clicked.connect(self._open_selected_path)

        # The performance-settings group (perf_group) is created below; we will
        # insert it into this layout once available.

        # Connect toggle to show/hide this panel
        self._open_toggle_btn.toggled.connect(lambda checked: self._open_panel_scroll.setVisible(checked))

        # Target folder selection
        target_folder_btn = QtWidgets.QPushButton("Target Folder")
        target_folder_btn.clicked.connect(self._select_target_folder)
        color_layout.addWidget(target_folder_btn)
        
        # Target folder display
        self._target_folder_label = QtWidgets.QLabel("No target folder selected")
        self._target_folder_label.setWordWrap(True)
        self._target_folder_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        color_layout.addWidget(self._target_folder_label)

        perf_group = QtWidgets.QGroupBox("Performance Settings")
        perf_form = QtWidgets.QFormLayout(perf_group)
        self._downsample_spin = QtWidgets.QSpinBox(); self._downsample_spin.setRange(1, 10); self._downsample_spin.setValue(1)
        self._max_points_spin = QtWidgets.QSpinBox(); self._max_points_spin.setRange(10, 1000); self._max_points_spin.setValue(100)
        self._chunk_size_spin = QtWidgets.QSpinBox(); self._chunk_size_spin.setRange(50, 500); self._chunk_size_spin.setValue(100)
        self._preprocess_checkbox = QtWidgets.QCheckBox(); self._preprocess_checkbox.setChecked(True)
        self._use_legacy_edges_checkbox = QtWidgets.QCheckBox(); self._use_legacy_edges_checkbox.setChecked(False)
        self._show_matplotlib_checkbox = QtWidgets.QCheckBox(); self._show_matplotlib_checkbox.setChecked(False)
        perf_form.addRow("Downsample", self._downsample_spin)
        perf_form.addRow("Max points (k)", self._max_points_spin)
        perf_form.addRow("Chunk size", self._chunk_size_spin)
        perf_form.addRow("Background preprocessing", self._preprocess_checkbox)
        perf_form.addRow("Use legacy edge detection", self._use_legacy_edges_checkbox)
        perf_form.addRow("Show matplotlib plots", self._show_matplotlib_checkbox)
        
        # Connect performance settings to cache clearing
        self._downsample_spin.valueChanged.connect(self._clear_caches)
        self._max_points_spin.valueChanged.connect(self._clear_caches)
        self._use_legacy_edges_checkbox.stateChanged.connect(self._clear_caches)
        
        # Insert Performance settings and then the Open button so it appears LAST
        open_panel_layout.addWidget(perf_group)
        open_panel_layout.addWidget(self._open_selected_btn)
        color_layout.addWidget(self._open_panel_scroll)

        # Scroll area for colour check-boxes
        self._color_scroll = QtWidgets.QScrollArea()
        self._color_scroll.setWidgetResizable(True)
        self._color_container = QtWidgets.QWidget()
        self._color_scroll.setWidget(self._color_container)
        self._color_list_layout = QtWidgets.QVBoxLayout(self._color_container)
        color_layout.addWidget(self._color_scroll, 1)

        # Selection buttons
        btn_row = QtWidgets.QHBoxLayout()
        self._select_all_btn = QtWidgets.QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all_colors)
        self._unselect_all_btn = QtWidgets.QPushButton("Unselect All")
        self._unselect_all_btn.clicked.connect(self._unselect_all_colors)
        btn_row.addWidget(self._select_all_btn)
        btn_row.addWidget(self._unselect_all_btn)
        color_layout.addLayout(btn_row)

        # Navigation buttons for single-colour mode
        nav_row = QtWidgets.QHBoxLayout()
        self._next_btn = QtWidgets.QPushButton("Next")
        self._next_btn.clicked.connect(self._display_next_color)
        self._next_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        nav_row.addWidget(self._next_btn)
        color_layout.addLayout(nav_row)
        
        # Removed the Process All Selected button

        color_layout.addStretch()

        # ----------------- Internal State -----------------
        self.image_uploaded = None
        self.image_uploaded_edge = None  # Edge-detected version from Extract_Figures_FV
        self.rgb_colors = None
        self.data_in_image = None  # Metadata from Extract_Figures_FV
        self.current_tiff_path = None  # Store current TIFF path for processing
        self.color_index_list = []
        self.selected_p_values = []
        self.current_color_index = 0
        self._color_checkboxes = {}
        # Keep track of color tabs instead of windows
        self._color_tabs = {}
        # Target folder for saving
        self.target_folder = None
        # Store data results for saving
        self._current_data_results = {}
        # Cache for preprocessed boundary data
        self._boundary_cache = {}
        # Cache for processed point clouds
        self._point_cloud_cache = {}
        # Path selected via the dropdown but not yet opened
        self._pending_tiff_path = None

    # ------------------------------------------------------------------
    # Tab change handler – ensure metadata visible only on Full Image tab
    # ------------------------------------------------------------------
    def _on_tab_changed(self, idx: int):
        """Slot connected to QTabWidget.currentChanged.
        Show metadata dock only when the first tab (Full Image) is active."""
        if idx == 0:  # Full Image tab
            if self.data_in_image is not None:
                self._metadata_widget.setVisible(True)
        else:
            self._metadata_widget.setVisible(False)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _populate_metadata_panel(self):
        """Fill the left-hand metadata dock with the values from
        `self.data_in_image`.  Called after a new TIFF has been loaded."""
        if self.data_in_image is None:
            self._metadata_widget.setVisible(False)
            return

        # Clear previous contents
        while self._metadata_form.count():
            child = self._metadata_form.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._metadata_labels.clear()

        for tag, attr in DataInImage.tag_dict.items():
            key_lbl = QtWidgets.QLabel(tag)
            val = getattr(self.data_in_image, attr)
            val_str = "—" if val in (None, "") else str(val)
            val_lbl = QtWidgets.QLabel(val_str)
            self._metadata_form.addRow(key_lbl, val_lbl)
            self._metadata_labels[tag] = val_lbl

        # Reveal the dock now that content is available
        self._metadata_widget.setVisible(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, text: str):
        self._status.showMessage(text, 5000)  # show for 5 s

    def _clear_caches(self):
        """Clear all caches when performance settings change"""
        self._boundary_cache.clear()
        self._point_cloud_cache.clear()
        self._set_status("Caches cleared due to settings change")

    def _select_target_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Target Folder for Saved Files", str(Path.home()))
        if folder:
            self.target_folder = folder
            # Show shortened path in label
            short_path = "..." + folder[-40:] if len(folder) > 40 else folder
            self._target_folder_label.setText(f"Target: {short_path}")
            self._target_folder_label.setStyleSheet("QLabel { color: black; font-size: 10px; }")
            self._set_status(f"Target folder set: {folder}")

    def _open_file_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select 3D Indexed-colour TIFF", str(Path.home()),
            "TIFF Images (*.tiff *.tif)")
        if filename:
            self._load_tiff(filename)

    # --------------------------------------------------------------
    # New path-selection helpers (dropdown panel)
    # --------------------------------------------------------------
    def _select_path_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select 3D Indexed-colour TIFF", str(Path.home()),
            "TIFF Images (*.tiff *.tif)")
        if filename:
            self._pending_tiff_path = filename
            short_path = "..." + filename[-40:] if len(filename) > 40 else filename
            self._selected_path_label.setText(short_path)
            self._selected_path_label.setStyleSheet("QLabel { color: black; font-size: 10px; }")
            self._set_status(f"Selected file: {os.path.basename(filename)}")

    def _open_selected_path(self):
        if not self._pending_tiff_path:
            QtWidgets.QMessageBox.warning(
                self, "No File Selected",
                "Please select a TIFF file first using 'Select Path'.")
            return
        # Load the selected file
        self._load_tiff(self._pending_tiff_path)

        # Collapse the dropdown automatically
        if self._open_toggle_btn.isChecked():
            # Uncheck will trigger the toggle handler to hide the panel
            self._open_toggle_btn.setChecked(False)

    def _load_tiff(self, path: str):
        self._set_status("Loading …")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # Use Import_3D_segment_from_tiff_figure for simpler loading
            print(f"Loading TIFF using Extract_Figures_FV.Import_3D_segment_from_tiff_figure(): {path}")
            
            # Store the path for later use in single color processing
            self.current_tiff_path = path
            
            # Call the simpler import function which internally calls Open_TIFF_fig
            self.data_in_image = Import_3D_segment_from_tiff_figure(path)
            
            # Access the global variables set by Open_TIFF_fig (called internally)
            self.image_uploaded = Extract_Figures_FV.image_uploaded
            self.rgb_colors = Extract_Figures_FV.rgb_colors
            self.image_uploaded_edge = Extract_Figures_FV.image_uploaded_edge
            
            # Build color index list from unique values (excluding background)
            unique_indices = np.unique(self.image_uploaded)
            self.color_index_list = [(int(idx), self.rgb_colors[int(idx)]) for idx in unique_indices 
                                   if idx != 0 and idx < len(self.rgb_colors)]

            print(f"Loaded {len(self.color_index_list)} colors from TIFF")
            print(f"Image shape: {self.image_uploaded.shape}")
            
            # Build colour panel & initial full-image plot
            self._build_color_panel()
            self._create_voxel_plot(full_image=True)

            # Populate left-hand metadata panel
            self._populate_metadata_panel()

            # Preprocess boundaries for faster loading (optional - can be toggled)
            if self._preprocess_checkbox.isChecked():
                self._preprocess_boundaries_async()

            self._set_status(os.path.basename(path) + " loaded")
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load TIFF file:\n{str(e)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _preprocess_boundaries_async(self):
        """Preprocess boundaries for all colors in background for faster access"""
        def preprocess():
            try:
                extractor = BoundaryExtractor()
                factor = self._downsample_spin.value()
                work_image = self.image_uploaded[::factor, ::factor, ::factor] if factor > 1 else self.image_uploaded
                
                for idx, rgb in self.color_index_list:
                    cache_key = f"{idx}_{factor}"
                    if cache_key not in self._boundary_cache:
                        try:
                            # Extract surface coordinates
                            surface_coords = extractor.extract_surface_coordinates(work_image, idx)
                            if factor > 1:
                                surface_coords = surface_coords * factor
                                
                            if surface_coords.size > 0:
                                # Apply point limit
                                max_pts = self._max_points_spin.value() * 1000
                                if surface_coords.shape[0] > max_pts:
                                    sel = np.random.choice(surface_coords.shape[0], max_pts, replace=False)
                                    surface_coords = surface_coords[sel]
                                
                                # Process colors and coordinates
                                rgb_norm = np.array(rgb, dtype=np.float32) / 255.0
                                colours = np.tile(rgb_norm, (surface_coords.shape[0], 1)).astype(np.float32)
                                
                                # Center and scale
                                mins = surface_coords.min(axis=0).astype(np.float32)
                                maxs = surface_coords.max(axis=0).astype(np.float32)
                                center = (mins + maxs) / 2.0
                                dims = (maxs - mins)
                                longest = np.max(dims) if np.max(dims) > 0 else 1.0
                                
                                centred = (surface_coords.astype(np.float32) - center) / longest
                                pts = np.column_stack((centred[:, 2], centred[:, 1], centred[:, 0]))
                                
                                # Cache the result
                                self._boundary_cache[cache_key] = (surface_coords, pts, colours)
                                
                        except Exception as e:
                            print(f"Failed to preprocess color {idx}: {e}")
                            continue
                            
                def update_status():
                    self._set_status("Background preprocessing complete")
                QtCore.QMetaObject.invokeMethod(self, update_status, QtCore.Qt.QueuedConnection)
                
            except Exception as e:
                print(f"Background preprocessing error: {e}")
        
        # Start preprocessing in background
        threading.Thread(target=preprocess, daemon=True).start()

    # ------------------------------------------------------------------
    # Colour Selection / Panel helpers
    # ------------------------------------------------------------------
    def _build_color_panel(self):
        # Clear existing
        for i in reversed(range(self._color_list_layout.count())):
            item = self._color_list_layout.takeAt(i)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._color_checkboxes.clear()

        for idx, rgb in self.color_index_list:
            cb = QtWidgets.QCheckBox(f"P {idx}")
            r, g, b = rgb
            cb.setStyleSheet(f"QCheckBox{{background-color: rgb({r},{g},{b});}}")
            cb.stateChanged.connect(lambda state, p=idx: self._on_color_toggle(p, state))
            self._color_list_layout.addWidget(cb)
            self._color_checkboxes[idx] = cb

        self._color_list_layout.addStretch()

    def _on_color_toggle(self, p_value: int, state: int):
        if state == QtCore.Qt.Checked:
            if p_value not in self.selected_p_values:
                self.selected_p_values.append(p_value)
        else:
            if p_value in self.selected_p_values:
                self.selected_p_values.remove(p_value)
        # Reset iteration each time selection changes
        self.current_color_index = 0

    def _select_all_colors(self):
        for cb in self._color_checkboxes.values():
            cb.setChecked(True)

    def _unselect_all_colors(self):
        for cb in self._color_checkboxes.values():
            cb.setChecked(False)

    # ------------------------------------------------------------------
    # Single-colour navigation
    # ------------------------------------------------------------------
    def _display_next_color(self):
        # Recompute the list of currently checked colours – keep it sorted to ensure
        # deterministic iteration order (ascending palette index).  This avoids the
        # impression that the “Next” button is stuck or jumps around when the
        # underlying dict order differs between runs.
        self.selected_p_values = sorted(
            [idx for idx, cb in self._color_checkboxes.items() if cb.isChecked()]
        )

        if not self.selected_p_values:
            self._set_status("No colors selected")
            return

        # If index is out of range, wrap around and notify the user once
        if self.current_color_index >= len(self.selected_p_values):
            self._set_status("All selected colours processed – cycling back to first")
            self.current_color_index = 0

        p_val = self.selected_p_values[self.current_color_index]

        # Check if this color already has a tab open
        if p_val in self._color_tabs:
            # Switch to existing tab instead of creating new one
            self._tab_widget.setCurrentWidget(self._color_tabs[p_val])
            self._set_status(f"Switched to existing Color P{p_val} tab")
        else:
            # Create new tab
            self._open_single_color_tab(p_val)
            self._set_status(f"Opened Color P{p_val}")
        
        # Advance index for the next press – wrap-around not needed here because
        # we handle it at the top on the next call.
        self.current_color_index += 1

    def _process_all_selected_colors(self):
        """Process all selected colors using Generate_3D_Segment_Library pipeline"""
        # Get currently selected colors
        selected_colors = [idx for idx, cb in self._color_checkboxes.items() if cb.isChecked()]
        
        if not selected_colors:
            QtWidgets.QMessageBox.warning(self, "No Colors Selected", 
                                        "Please select at least one color to process.")
            return
            
        if not self.current_tiff_path:
            QtWidgets.QMessageBox.critical(self, "No TIFF Loaded", 
                                         "Please load a TIFF file first.")
            return
            
        if not self.target_folder:
            QtWidgets.QMessageBox.warning(self, "No Target Folder", 
                                        "Please select a target folder first using the 'Target Folder' button.")
            return
        
        # Confirm processing
        reply = QtWidgets.QMessageBox.question(
            self, "Process All Selected Colors", 
            f"This will process {len(selected_colors)} colors using the original Extract_Figures_FV pipeline.\n\n"
            f"Selected colors: {selected_colors}\n\n"
            f"This will:\n"
            f"• Calculate all measurements (volume, surface, length, diameter)\n"
            f"• Show 3D visualizations in the OpenGL viewer\n"
            f"• {'Display matplotlib plots (if enabled)' if self._show_matplotlib_checkbox.isChecked() else 'Suppress matplotlib plots (OpenGL only)'}\n"
            f"• Export CSV and TIFF files to: {self.target_folder}\n\n"
            f"Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        try:
            self._set_status(f"Processing {len(selected_colors)} colors using Extract_Figures_FV pipeline...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Temporarily override the get_integer_list function to return our selected colors
            original_get_list = Extract_Figures_FV.get_integer_list
            Extract_Figures_FV.get_integer_list = lambda: selected_colors
            
            # Set matplotlib display based on user preference
            set_matplotlib_display(self._show_matplotlib_checkbox.isChecked())
            
            # Use the original Generate_3D_Segment_Library function
            # This will process each selected color and generate all outputs
            # We need the output CSV/TIFF files to be written to the chosen target folder.
            # The original Generate_3D_Segment_Library() builds its output base_name from the
            # *input file path*.  Therefore, we create a *temporary copy* of the TIFF inside
            # the target folder so its basename points there – no heavy pixel data is copied
            # because shutil.copy2 is fast on the same disk and the file is usually small
            # compared to the processing time.  Alternatively we could symlink, but this is
            # cross-platform-safe.

            import shutil, tempfile, os
            tmp_dir = tempfile.mkdtemp(prefix="batch_", dir=self.target_folder)
            tmp_tiff_path = os.path.join(tmp_dir, os.path.basename(self.current_tiff_path))
            shutil.copy2(self.current_tiff_path, tmp_tiff_path)

            # Now run the original pipeline on the temp file (outputs will be in target folder)
            Generate_3D_Segment_Library(tmp_tiff_path)

            # Optionally clean up the temp copy to avoid clutter (keeping outputs)
            try:
                os.remove(tmp_tiff_path)
                os.rmdir(tmp_dir)  # remove temp dir if empty
            except Exception:
                pass
            
            # Restore the original function and reset matplotlib display
            Extract_Figures_FV.get_integer_list = original_get_list
            set_matplotlib_display(True)
            
            self._set_status(f"Successfully processed {len(selected_colors)} colors")
            QtWidgets.QMessageBox.information(
                self, "Processing Complete", 
                f"Successfully processed {len(selected_colors)} colors.\n\n"
                f"Generated files saved to: {self.target_folder}\n\n"
                f"Files include:\n"
                f"• CSV files with measurements\n"
                f"• TIFF files with 3D data\n"
                f"• 3D visualizations shown in OpenGL viewer\n"
                f"• {'Matplotlib plots were displayed' if self._show_matplotlib_checkbox.isChecked() else 'Matplotlib plots were suppressed'}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Processing Error", 
                f"Failed to process selected colors:\n{str(e)}"
            )
            print(f"Error in batch processing: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _open_single_color_tab(self, p_value: int):
        # Process single color using Extract_Figures_FV pipeline
        try:
            # Check if we have a current TIFF path
            if not self.current_tiff_path:
                QtWidgets.QMessageBox.critical(self, "No TIFF Loaded", 
                                             "Please load a TIFF file first.")
                return

            # Hide metadata panel while working on a single colour
            self._metadata_widget.setVisible(False)
            
            # Create DataInImage object for this specific color
            data_in_spine = DataInImage()
            # Copy attributes from the main image data
            if self.data_in_image:
                data_in_spine.__dict__.update(self.data_in_image.__dict__)
            
            # Generate base name for output files
            base_name = os.path.splitext(os.path.basename(self.current_tiff_path))[0]
            if self.target_folder:
                base_name = os.path.join(self.target_folder, base_name)
            
            self._set_status(f"Processing Color P{p_value} using Extract_Figures_FV pipeline...")
            
            # Use the original scientific pipeline for full analysis
            print(f"Running Geneate_Estimations for spine {p_value}")
            
            # Set matplotlib display based on user preference
            set_matplotlib_display(self._show_matplotlib_checkbox.isChecked())
            
            # We need the numeric estimations but we **do not** want to write
            # CSV / TIFF files at this stage.  Temporarily monkey-patch the
            # export functions so Geneate_Estimations can run without leaving
            # files on disk.

            import types as _t
            orig_txt = Extract_Figures_FV.Export_Spine_as_Text
            orig_tiff = Extract_Figures_FV.Export_Spine_as_tiff

            Extract_Figures_FV.Export_Spine_as_Text = lambda *a, **k: None
            Extract_Figures_FV.Export_Spine_as_tiff = lambda *a, **k: None

            try:
                Geneate_Estimations(p_value, base_name, data_in_spine)
            finally:
                # Restore original functions so Save button works normally
                Extract_Figures_FV.Export_Spine_as_Text = orig_txt
                Extract_Figures_FV.Export_Spine_as_tiff = orig_tiff
            
            # Reset to default state
            set_matplotlib_display(True)
            
            # Store the results for potential UI display
            self._current_data_results[p_value] = data_in_spine
            
            # ------------------------------------------------------------------
            # Merge interior voxels with "connection" voxels (matrix value 2)
            # ------------------------------------------------------------------
            conn_coords = None
            if hasattr(data_in_spine, "spine_matrix") and hasattr(data_in_spine, "bbox_min_ijk"):
                # Ensure we have a NumPy array so comparisons yield an array, not a single bool
                spine_mat_np = np.asarray(data_in_spine.spine_matrix)
                conn_mask = (spine_mat_np == 2)
                if conn_mask.sum() > 0:
                    k_idx, i_idx, j_idx = np.nonzero(conn_mask)
                    conn_coords = np.column_stack((k_idx, i_idx, j_idx)).astype(np.int32)
                    # Translate from local matrix space (with +1 padding) to absolute
                    i_min, j_min, k_min = data_in_spine.bbox_min_ijk
                    conn_coords[:, 0] = conn_coords[:, 0] - 1 + k_min  # depth (k)
                    conn_coords[:, 1] = conn_coords[:, 1] - 1 + i_min  # row   (i)
                    conn_coords[:, 2] = conn_coords[:, 2] - 1 + j_min  # col   (j)

            surface_coords, pts, colours = self._extract_segment_voxels(p_value, extra_coords=conn_coords)
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Processing Error", 
                                         f"Failed to process color P{p_value}:\n{str(e)}")
            return

        # Create tab widget for this color
        color_widget = QtWidgets.QWidget()
        # Use splitter so metadata side panel can be resized/retracted
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, color_widget)

        # ------------ Metadata panel (left) ------------
        meta_scroll = QtWidgets.QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_container = QtWidgets.QWidget()
        meta_form = QtWidgets.QFormLayout(meta_container)
        meta_scroll.setWidget(meta_container)

        # Wrap scroll area in a container so we can enforce a robust min width
        meta_panel = QtWidgets.QWidget()
        meta_panel.setMinimumWidth(220)
        _meta_vbox = QtWidgets.QVBoxLayout(meta_panel)
        _meta_vbox.setContentsMargins(0, 0, 0, 0)
        _meta_vbox.addWidget(meta_scroll)

        # ------------------------------------------------------------
        # 1) Populate metadata directly from the values that Geneate_Estimations
        #    has just written into `data_in_spine`.  This avoids showing '…' while
        #    waiting for the slower Estimator thread and guarantees all fields are
        #    immediately filled with correct numbers/units.
        # ------------------------------------------------------------
        label_widgets = {}
        # Store editors per colour for later retrieval in _save_color_data
        if not hasattr(self, '_metadata_editors'):
            self._metadata_editors = {}
        self._metadata_editors[p_value] = {}
        read_only_tags = ("Number_of_Layers", "Image_Height", "Image_Width")
        for tag in DataInImage.tag_dict.keys():
            attr_label = QtWidgets.QLabel(tag)
            if tag in read_only_tags:
                # Show static text for non-editable core dimensions
                display_widget = QtWidgets.QLabel()
            else:
                display_widget = QtWidgets.QLineEdit()
                display_widget.setMinimumWidth(120)

            meta_form.addRow(attr_label, display_widget)
            label_widgets[tag] = display_widget
            self._metadata_editors[p_value][tag] = display_widget

        # Fill labels with computed data
        for tag, attr in DataInImage.tag_dict.items():
            val = getattr(data_in_spine, attr)
            if val not in (None, ""):
                label_widgets[tag].setText(str(val))

        # ------------------------------------------------------------
        # 2) Background re-estimation thread has been disabled because
        #    Geneate_Estimations already computed all metrics reliably
        #    above.  Keeping a second computation path caused conflicting
        #    values or errors (e.g. missing L).  If future, enable again
        #    by calling heavy_calc() in a thread.
        # ------------------------------------------------------------

        # Add save button below metadata
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_color_data(p_value, surface_coords))
        meta_form.addRow("", save_btn)  # Empty label, just the button

        # ------------ 3-D canvas (right) ------------
        canvas = QtAdvanced3DCanvas(color_widget)
        canvas.set_points(pts, colours)

        splitter.addWidget(meta_panel)
        splitter.addWidget(canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # Place splitter in the colour tab layout
        layout = QtWidgets.QHBoxLayout(color_widget)
        layout.addWidget(splitter)

        # Add tab to main tab widget
        tab_title = f"Color P{p_value}"
        self._tab_widget.addTab(color_widget, tab_title)
        self._color_tabs[p_value] = color_widget
        
        # Switch to the new tab
        self._tab_widget.setCurrentWidget(color_widget)

    def _extract_and_process_boundary(self, p_value: int):
        """Extract and process boundary data for a specific color"""
        try:
            # Apply downsampling to the image first if requested
            factor = self._downsample_spin.value()
            
            # Choose which image to use based on legacy edge detection setting
            if self._use_legacy_edges_checkbox.isChecked() and self.image_uploaded_edge is not None:
                print(f"Using legacy edge-detected image for color {p_value}")
                work_image = self.image_uploaded_edge
            else:
                print(f"Using fast boundary extraction for color {p_value}")
                work_image = self.image_uploaded
            
            # Use the faster BoundaryExtractor class
            extractor = BoundaryExtractor()
            if factor > 1:
                downsampled_img = work_image[::factor, ::factor, ::factor]
                surface_coords = extractor.extract_surface_coordinates(downsampled_img, p_value)
                # Scale coordinates back up
                surface_coords = surface_coords * factor
            else:
                surface_coords = extractor.extract_surface_coordinates(work_image, p_value)

            if surface_coords.size == 0:
                raise ValueError(f"Colour {p_value} contains no surface voxels.")

            # Apply point limit
            max_pts = self._max_points_spin.value() * 1000
            if surface_coords.shape[0] > max_pts:
                sel = np.random.choice(surface_coords.shape[0], max_pts, replace=False)
                surface_coords = surface_coords[sel]

            # Set color for all surface points
            rgb_norm = np.array(self.rgb_colors[p_value], dtype=np.float32) / 255.0
            colours = np.tile(rgb_norm, (surface_coords.shape[0], 1)).astype(np.float32)

            # Centre and scale the surface points
            mins = surface_coords.min(axis=0).astype(np.float32)
            maxs = surface_coords.max(axis=0).astype(np.float32)
            center = (mins + maxs) / 2.0
            dims = (maxs - mins)
            longest = np.max(dims) if np.max(dims) > 0 else 1.0

            # Translate and scale
            centred = (surface_coords.astype(np.float32) - center) / longest
            pts = np.column_stack((centred[:, 2], centred[:, 1], centred[:, 0]))
            
            return surface_coords, pts, colours

        except Exception as e:
            raise e

    def _extract_segment_voxels(self, p_value: int, extra_coords: np.ndarray | None = None):
        """Return full-volume voxel coordinates for a given colour and prepare them for OpenGL display.
        The logic mirrors _extract_and_process_boundary but uses the entire mask rather than just the
        surface voxels.  Down-sampling and point-limit handling are preserved.
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (raw_coords[k,i,j], pts[x,y,z] centred/scaled, colours[r,g,b])
        """
        try:
            # Down-sample factor specified by the UI
            factor = self._downsample_spin.value()

            # Build boolean mask for the selected colour
            mask = (self.image_uploaded == p_value)
            if factor > 1:
                mask = mask[::factor, ::factor, ::factor]

            # Convert to (k,i,j) coordinate list (depth, height, width)
            coords = np.column_stack(np.nonzero(mask))
            if coords.size == 0:
                raise ValueError(f"Colour {p_value} contains no voxels.")

            # Re-scale back up if we down-sampled
            if factor > 1:
                coords = coords * factor

            # Append any extra coordinates (e.g., boundary/connection voxels)
            if extra_coords is not None and extra_coords.size > 0:
                coords = np.vstack((coords, extra_coords))

            # Limit number of points if requested (after merge)
            max_pts = self._max_points_spin.value() * 1000
            if coords.shape[0] > max_pts:
                sel = np.random.choice(coords.shape[0], max_pts, replace=False)
                coords = coords[sel]

            # Assign colours – default colour for segment, black for extras
            rgb_norm = np.array(self.rgb_colors[p_value], dtype=np.float32) / 255.0
            base_colours = np.tile(rgb_norm, (coords.shape[0], 1)).astype(np.float32)
            if extra_coords is not None and extra_coords.size > 0:
                # Determine which entries correspond to extras (last ones before optional shuffle)
                extra_len = extra_coords.shape[0]
                base_colours[-extra_len:] = (0.0, 0.0, 0.0)  # black
            colours = base_colours

            # Centre & scale into ±0.5 cube (same scheme as boundary helper)
            mins = coords.min(axis=0).astype(np.float32)
            maxs = coords.max(axis=0).astype(np.float32)
            center = (mins + maxs) / 2.0
            dims = (maxs - mins)
            longest = np.max(dims) if np.max(dims) > 0 else 1.0
            centred = (coords.astype(np.float32) - center) / longest
            pts = np.column_stack((centred[:, 2], centred[:, 1], centred[:, 0]))

            return coords, pts, colours
        except Exception as e:
            raise e

    # ------------------------------------------------------------------
    # Point cloud generation helpers
    # ------------------------------------------------------------------
    def _create_voxel_plot(self, *, canvas=None, full_image: bool = False):
        if canvas is None:
            canvas = self._main_canvas
            
        if self.image_uploaded is None:
            return

        if not full_image:
            # Placeholder case - show empty
            canvas.set_points(np.empty((0, 3)), np.empty((0, 3)))
            return

        try:
            # Extract boundaries for all colors using the boundary extractor
            extractor = BoundaryExtractor()
            all_coords = []
            all_colors = []

            # Apply downsampling factor and choose image source
            factor = self._downsample_spin.value()
            
            # Choose which image to use based on legacy edge detection setting
            if self._use_legacy_edges_checkbox.isChecked() and self.image_uploaded_edge is not None:
                base_image = self.image_uploaded_edge
                print("Using legacy edge-detected image for full visualization")
            else:
                base_image = self.image_uploaded
                print("Using fast boundary extraction for full visualization")
                
            work_image = base_image[::factor, ::factor, ::factor] if factor > 1 else base_image

            # Get all unique colors (excluding background)
            unique_colors = np.unique(work_image)
            unique_colors = unique_colors[unique_colors != 0]  # Remove background

            for color_idx in unique_colors:
                if color_idx >= len(self.rgb_colors):
                    continue  # Skip invalid color indices

                # Extract surface for this color
                surface_coords = extractor.extract_surface_coordinates(work_image, color_idx)
                
                if surface_coords.size == 0:
                    continue

                # Scale coordinates back if downsampled
                if factor > 1:
                    surface_coords = surface_coords * factor

                # Get color for these coordinates
                color_rgb = np.array(self.rgb_colors[color_idx], dtype=np.float32) / 255.0
                point_colors = np.tile(color_rgb, (surface_coords.shape[0], 1))

                all_coords.append(surface_coords)
                all_colors.append(point_colors)

            if not all_coords:
                canvas.set_points(np.empty((0, 3)), np.empty((0, 3)))
                return

            # Combine all coordinates and colors
            coords = np.vstack(all_coords)
            colours = np.vstack(all_colors)

            # Apply point limit
            max_pts = self._max_points_spin.value() * 1000
            if coords.shape[0] > max_pts:
                sel = np.random.choice(coords.shape[0], max_pts, replace=False)
                coords = coords[sel]
                colours = colours[sel]

            # Centre and scale to fit within ±0.5 cube
            if coords.size > 0:
                mins = coords.min(axis=0).astype(np.float32)
                maxs = coords.max(axis=0).astype(np.float32)
                center = (mins + maxs) / 2.0
                dims = (maxs - mins)
                longest = np.max(dims) if np.max(dims) > 0 else 1.0

                centred = (coords.astype(np.float32) - center) / longest
                pts = np.column_stack((centred[:, 2], centred[:, 1], centred[:, 0]))

                canvas.set_points(pts, colours)
            else:
                canvas.set_points(np.empty((0, 3)), np.empty((0, 3)))

        except Exception as e:
            print(f"Error in boundary-based plot creation: {e}")
            # Fallback to empty display
            canvas.set_points(np.empty((0, 3)), np.empty((0, 3)))

    def _save_color_data(self, p_value: int, surface_coords: np.ndarray):
        """Save the color data using both text and TIFF export functions"""
        if self.target_folder is None:
            QtWidgets.QMessageBox.warning(self, "No Target Folder", 
                                        "Please select a target folder first using the 'Target Folder' button.")
            return
            
        if p_value not in self._current_data_results:
            QtWidgets.QMessageBox.warning(self, "Data Not Ready", 
                                        "Metadata calculation is still in progress. Please wait and try again.")
            return
        
        try:
            # Get the data result for this color
            data_result = self._current_data_results[p_value]

            # Retrieve edited values from UI and update data_result
            if hasattr(self, '_metadata_editors') and p_value in self._metadata_editors:
                editor_map = self._metadata_editors[p_value]
                for tag, attr in DataInImage.tag_dict.items():
                    if tag not in editor_map:
                        continue
                    txt = editor_map[tag].text().strip()
                    if txt == "":
                        continue  # Skip empty edits

                    # Best-effort type conversion: int -> float -> tuple -> str
                    new_val: object = txt
                    try:
                        # First try strict int
                        new_val = int(txt)
                    except ValueError:
                        try:
                            # Next try float
                            f_val = float(txt)
                            # If the float is integral (e.g., 12.0) store as int
                            new_val = int(f_val) if f_val.is_integer() else f_val
                        except ValueError:
                            # Try tuple syntax "(1,2,3)" or "(1.0,2.0)"
                            if txt.startswith("(") and txt.endswith(")"):
                                try:
                                    parts = [p.strip() for p in txt[1:-1].split(',') if p.strip()]
                                    tuple_vals = []
                                    for p in parts:
                                        try:
                                            iv = int(p)
                                            tuple_vals.append(iv)
                                        except ValueError:
                                            tuple_vals.append(float(p))
                                    new_val = tuple(tuple_vals)
                                except Exception:
                                    new_val = txt  # keep raw string
                            else:
                                new_val = txt  # keep raw string
                    setattr(data_result, attr, new_val)

                # Persist the edited object back to cache so subsequent saves or
                # other routines use the modified values.
                self._current_data_results[p_value] = data_result
            
            # ------------------------------------------------------------------
            # Determine which 3-D matrix to save:
            #    • Preferred: the full spine_matrix produced by Geneate_Estimations
            #      (contains 0/1/2 values – interior, contact, background)
            #    • Fallback: simple mask derived from image_uploaded
            # ------------------------------------------------------------------
            if hasattr(data_result, "spine_matrix") and data_result.spine_matrix is not None:
                color_mask = data_result.spine_matrix  # already numpy array
            else:
                # Legacy fallback – will miss contact voxels
                color_mask = (self.image_uploaded == p_value).astype(int)
            
            # Generate base filename
            base_name = f"color_P{p_value}"
            
            # Save as CSV file (full matrix + metadata)
            csv_path = os.path.join(self.target_folder, f"{base_name}.csv")
            Export_Spine_as_Text(csv_path, color_mask, data_result)
            
            # Save as TIFF file
            tiff_path = os.path.join(self.target_folder, f"{base_name}.tiff")
            Export_Spine_as_tiff(tiff_path, color_mask, data_result)
            
            self._set_status(f"Color P{p_value} saved to {self.target_folder}")
            QtWidgets.QMessageBox.information(self, "Save Complete", 
                                            f"Color P{p_value} data saved successfully:\n"
                                            f"• CSV file: {base_name}.csv\n"
                                            f"• TIFF file: {base_name}.tiff")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", 
                                         f"Failed to save color P{p_value} data:\n{str(e)}")

    def _create_color_tab_ui(self, p_value: int, surface_coords, pts, colours):
        """Create the UI for a color tab"""
        # Create tab widget for this color
        color_widget = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(color_widget)

        # ------------ Metadata panel (left) ------------
        meta_scroll = QtWidgets.QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_container = QtWidgets.QWidget()
        meta_form = QtWidgets.QFormLayout(meta_container)
        meta_scroll.setWidget(meta_container)
        meta_scroll.setFixedWidth(260)

        # ------------------------------------------------------------
        # 1) Quick metadata (dimensions, voxel count, color) – instant
        # ------------------------------------------------------------
        label_widgets = {}
        for tag in DataInImage.tag_dict.keys():
            attr_label = QtWidgets.QLabel(tag)
            val_label = QtWidgets.QLabel("…")
            attr_label.setMinimumWidth(150)
            meta_form.addRow(attr_label, val_label)
            label_widgets[tag] = val_label

        # Quick calculations using NumPy (based on surface data)
        surface_voxel_count = surface_coords.shape[0] if 'surface_coords' in locals() else 0
        if surface_voxel_count > 0:
            dims = (surface_coords.max(axis=0) - surface_coords.min(axis=0) + 1).astype(int)
            quick = DataInImage()
            quick.num_layers = int(dims[0])
            quick.height = int(dims[1])
            quick.width = int(dims[2])
            quick.spine_color = self.rgb_colors[p_value]
            quick.surface = surface_voxel_count  # surface area in voxels

            # Update quick fields
            for tag, attr in DataInImage.tag_dict.items():
                val = getattr(quick, attr)
                if val not in (None, ""):
                    label_widgets[tag].setText(str(val))

        # Background re-estimation disabled – Geneate_Estimations has already
        # populated all fields.  Uncomment below if you need independent
        # verification in a future build.

        # Add save button below metadata
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_color_data(p_value, surface_coords))
        meta_form.addRow("", save_btn)  # Empty label, just the button

        # ------------ 3-D canvas (right) ------------
        canvas = QtAdvanced3DCanvas(color_widget)
        canvas.set_points(pts, colours)

        hbox.addWidget(meta_scroll)
        hbox.addWidget(canvas, 1)

        # Add tab to main tab widget
        tab_title = f"Color P{p_value}"
        self._tab_widget.addTab(color_widget, tab_title)
        self._color_tabs[p_value] = color_widget
        
        # Switch to the new tab
        self._tab_widget.setCurrentWidget(color_widget)


# Convenience function for main.py

def launch_qt_viewer():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = QtTIFFViewer3D()
    viewer.show()
    return app.exec() 