import sys
import os
from pathlib import Path
import numpy as np
import tifffile as tiff
from PySide6 import QtWidgets, QtCore, QtGui
from qt_opengl_canvas import QtAdvanced3DCanvas
from PIL import Image
import mmap, io, threading
from data_in_image import DataInImage
from estimator import Estimator
from boundary_extractor import extract_surface_only, BoundaryExtractor
from Extract_Figures_FV import Export_Spine_as_Text, Export_Spine_as_tiff


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

        # Main canvas area with tabs
        self._tab_widget = QtWidgets.QTabWidget()
        self._main_canvas = QtAdvanced3DCanvas(self)
        self._tab_widget.addTab(self._main_canvas, "Full Image")
        main_layout.addWidget(self._tab_widget, 3)  # 3/4 of space

        # ------------------------------------------------------------------
        # Right-hand control panel
        # ------------------------------------------------------------------
        control_widget = QtWidgets.QWidget()
        control_widget.setFixedWidth(300)
        main_layout.addWidget(control_widget, 1)  # 1/4 of space

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

        # File open & performance settings
        open_btn = QtWidgets.QPushButton("Open TIFF …")
        open_btn.clicked.connect(self._open_file_dialog)
        color_layout.addWidget(open_btn)

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
        perf_form.addRow("Downsample", self._downsample_spin)
        perf_form.addRow("Max points (k)", self._max_points_spin)
        perf_form.addRow("Chunk size", self._chunk_size_spin)
        perf_form.addRow("Background preprocessing", self._preprocess_checkbox)
        
        # Connect performance settings to cache clearing
        self._downsample_spin.valueChanged.connect(self._clear_caches)
        self._max_points_spin.valueChanged.connect(self._clear_caches)
        
        color_layout.addWidget(perf_group)

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
        nav_row.addWidget(self._next_btn)
        color_layout.addLayout(nav_row)

        color_layout.addStretch()

        # ----------------- Internal State -----------------
        self.image_uploaded = None
        self.rgb_colors = None
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

    def _load_tiff(self, path: str):
        self._set_status("Loading …")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # Use PIL to access palette & frames efficiently
            with open(path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                img = Image.open(io.BytesIO(mm.read()))

                if img.mode != 'P':
                    raise ValueError('The image is not in indexed colour mode (P).')

                frames = []
                for frame in range(img.n_frames):
                    img.seek(frame)
                    frames.append(np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1]))
                self.image_uploaded = np.stack(frames, axis=0)

                palette = img.getpalette()
                ncol = len(palette) // 3
                self.rgb_colors = [tuple(palette[i*3:i*3+3]) for i in range(ncol)]

                mm.close()

            unique_indices = np.unique(self.image_uploaded)
            self.color_index_list = [(int(idx), self.rgb_colors[int(idx)]) for idx in unique_indices if idx != 0 and idx < len(self.rgb_colors)]

            # Build colour panel & initial full-image plot
            self._build_color_panel()
            self._create_voxel_plot(full_image=True)
            
            # Preprocess boundaries for faster loading (optional - can be toggled)
            if self._preprocess_checkbox.isChecked():
                self._preprocess_boundaries_async()

            self._set_status(os.path.basename(path) + " loaded")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
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
        # Recompute the list of currently checked colours to avoid sync issues
        self.selected_p_values = [idx for idx, cb in self._color_checkboxes.items() if cb.isChecked()]

        if not self.selected_p_values:
            self._set_status("No colors selected")
            return

        if self.current_color_index >= len(self.selected_p_values):
            self._set_status("All selected colours processed")
            self.current_color_index = 0  # Reset for next cycle
            return

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
        
        self.current_color_index += 1

    def _open_single_color_tab(self, p_value: int):
        # Extract boundary/surface voxels only using fast boundary detection
        try:
            # Check cache first
            cache_key = f"{p_value}_{self._downsample_spin.value()}"
            if cache_key in self._boundary_cache:
                surface_coords, pts, colours = self._boundary_cache[cache_key]
                self._set_status(f"Using cached data for Color P{p_value}")
            else:
                # Extract boundary data
                surface_coords, pts, colours = self._extract_and_process_boundary(p_value)
                # Cache the result
                self._boundary_cache[cache_key] = (surface_coords, pts, colours)
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Boundary Extraction Error", 
                                         f"Failed to extract boundaries: {str(e)}")
            return

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

        # ------------------------------------------------------------
        # 2) Full estimation in background to fill remaining fields
        # ------------------------------------------------------------
        def heavy_calc():
            dii = DataInImage()
            est = Estimator(self.image_uploaded, self.rgb_colors)
            result = est.run_estimations(p_value, dii)
            
            # Store the result for this p_value
            self._current_data_results[p_value] = result

            def apply():
                for tag, attr in DataInImage.tag_dict.items():
                    label_widgets[tag].setText(str(getattr(result, attr)))
            QtCore.QMetaObject.invokeMethod(self, apply, QtCore.Qt.QueuedConnection)

        threading.Thread(target=heavy_calc, daemon=True).start()

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

    def _extract_and_process_boundary(self, p_value: int):
        """Extract and process boundary data for a specific color"""
        try:
            # Apply downsampling to the image first if requested
            factor = self._downsample_spin.value()
            
            # Use the faster BoundaryExtractor class
            extractor = BoundaryExtractor()
            if factor > 1:
                downsampled_img = self.image_uploaded[::factor, ::factor, ::factor]
                surface_coords = extractor.extract_surface_coordinates(downsampled_img, p_value)
                # Scale coordinates back up
                surface_coords = surface_coords * factor
            else:
                surface_coords = extractor.extract_surface_coordinates(self.image_uploaded, p_value)

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

            # Apply downsampling factor
            factor = self._downsample_spin.value()
            work_image = self.image_uploaded[::factor, ::factor, ::factor] if factor > 1 else self.image_uploaded

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
            
            # Create 3D matrix from surface coordinates for export
            # We need to reconstruct a 3D matrix from the surface coordinates
            # For now, we'll create a simplified matrix based on the original image
            color_mask = (self.image_uploaded == p_value).astype(int)
            
            # Generate base filename
            base_name = f"color_P{p_value}"
            
            # Save as text file
            text_path = os.path.join(self.target_folder, f"{base_name}.txt")
            Export_Spine_as_Text(text_path, color_mask, data_result)
            
            # Save as TIFF file
            tiff_path = os.path.join(self.target_folder, f"{base_name}.tiff")
            Export_Spine_as_tiff(tiff_path, color_mask, data_result)
            
            self._set_status(f"Color P{p_value} saved to {self.target_folder}")
            QtWidgets.QMessageBox.information(self, "Save Complete", 
                                            f"Color P{p_value} data saved successfully:\n"
                                            f"• Text file: {base_name}.txt\n"
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

        # ------------------------------------------------------------
        # 2) Full estimation in background to fill remaining fields
        # ------------------------------------------------------------
        def heavy_calc():
            dii = DataInImage()
            est = Estimator(self.image_uploaded, self.rgb_colors)
            result = est.run_estimations(p_value, dii)
            
            # Store the result for this p_value
            self._current_data_results[p_value] = result

            def apply():
                for tag, attr in DataInImage.tag_dict.items():
                    label_widgets[tag].setText(str(getattr(result, attr)))
            QtCore.QMetaObject.invokeMethod(self, apply, QtCore.Qt.QueuedConnection)

        threading.Thread(target=heavy_calc, daemon=True).start()

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