import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from opengl_canvas import VoxelOpenGLCanvas
import numpy as np
import gc
import os
from threading import Thread
from data_in_image import DataInImage
from estimator import Estimator
# Import helper wrappers from legacy module
from Extract_Figures_FV_Classes import Plotting, DataExport


class TIFFViewerUI:
    """
    Class that handles all the UI tasks and interactions.
    """

    def __init__(self, root, viewer):
        self.root = root
        self.viewer = viewer
        self.metadata_entries = {}
        
        # Performance settings
        self.downsample_factor = tk.IntVar(value=1)  # For downsampling large datasets
        self.max_points = tk.IntVar(value=100000)  # Maximum points to render
        self.quality_level = tk.IntVar(value=3)  # Quality level for rendering (1-5)
        self.chunk_size = tk.IntVar(value=100)  # Chunk size for processing
        
        # Status bar for progress updates
        self.status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize frames
        self.welcome_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.viewer_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.single_color_frame = tk.Frame(self.root, bg="#f0f0f0")

        # Show the welcome frame initially
        self.show_welcome_frame()

    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def show_welcome_frame(self):
        """Show the welcome frame with performance settings"""
        self.clear_frames()
        self.welcome_frame.pack(fill=tk.BOTH, expand=True)

        # Welcome content
        welcome_label = tk.Label(self.welcome_frame, 
                               text="Welcome to the 3D TIFF Viewer",
                               font=("Helvetica", 20, "bold"),
                               bg="#f0f0f0")
        welcome_label.pack(pady=20)

        # Performance settings frame
        settings_frame = tk.LabelFrame(self.welcome_frame, 
                                     text="Performance Settings",
                                     font=("Helvetica", 12),
                                     bg="#f0f0f0")
        settings_frame.pack(pady=10, padx=10, fill="x")

        # Downsample factor
        tk.Label(settings_frame, 
                text="Downsample Factor:",
                font=("Helvetica", 10),
                bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5)
        
        tk.Scale(settings_frame,
                from_=1, to=10,
                orient="horizontal",
                variable=self.downsample_factor,
                length=200,
                bg="#f0f0f0").grid(row=0, column=1, padx=5, pady=5)

        # Quality level
        tk.Label(settings_frame,
                text="Render Quality:",
                font=("Helvetica", 10),
                bg="#f0f0f0").grid(row=1, column=0, padx=5, pady=5)
        
        tk.Scale(settings_frame,
                from_=1, to=5,
                orient="horizontal",
                variable=self.quality_level,
                length=200,
                bg="#f0f0f0").grid(row=1, column=1, padx=5, pady=5)

        # Max points
        tk.Label(settings_frame,
                text="Max Points (thousands):",
                font=("Helvetica", 10),
                bg="#f0f0f0").grid(row=2, column=0, padx=5, pady=5)
        
        tk.Scale(settings_frame,
                from_=10, to=1000,
                orient="horizontal",
                variable=self.max_points,
                length=200,
                bg="#f0f0f0").grid(row=2, column=1, padx=5, pady=5)

        # Chunk size
        tk.Label(settings_frame,
                text="Processing Chunk Size:",
                font=("Helvetica", 10),
                bg="#f0f0f0").grid(row=3, column=0, padx=5, pady=5)
        
        tk.Scale(settings_frame,
                from_=50, to=500,
                orient="horizontal",
                variable=self.chunk_size,
                length=200,
                bg="#f0f0f0").grid(row=3, column=1, padx=5, pady=5)

        # Help text
        help_text = "Higher downsample = faster but less detailed\n"
        help_text += "Lower quality = faster rendering\n"
        help_text += "Lower max points = better performance\n"
        help_text += "Larger chunk size = faster but more memory"
        
        tk.Label(settings_frame,
                text=help_text,
                font=("Helvetica", 8),
                bg="#f0f0f0",
                fg="gray").grid(row=4, column=0, columnspan=2, pady=5)

        # Upload button
        upload_button = tk.Button(self.welcome_frame,
                                text="Upload TIFF File",
                                command=self.upload_file,
                                font=("Helvetica", 14),
                                bg="#4a90e2",
                                fg="white",
                                relief=tk.FLAT)
        upload_button.pack(pady=10)

        # Save directory selection
        self.save_directory = tk.StringVar()
        save_directory_label = tk.Label(self.welcome_frame,
                                      text="Save Directory:",
                                      font=("Helvetica", 14),
                                      bg="#f0f0f0")
        save_directory_label.pack(pady=5)
        
        save_directory_entry = tk.Entry(self.welcome_frame,
                                      textvariable=self.save_directory,
                                      font=("Helvetica", 12),
                                      width=50)
        save_directory_entry.pack(pady=5)
        
        browse_button = tk.Button(self.welcome_frame,
                                text="Browse",
                                command=self.browse_directory,
                                font=("Helvetica", 12),
                                bg="#4a90e2",
                                fg="white",
                                relief=tk.FLAT)
        browse_button.pack(pady=5)

    def show_viewer_frame(self):
        """Show the viewer frame and set up the initial view"""
        self.clear_frames()
        self.viewer_frame.pack(fill=tk.BOTH, expand=True)
        self.add_navbar(self.viewer_frame)

        print("Setting up viewer frame with OpenGL canvas...")

        # ------------------------------------------------------------------
        # OpenGL canvas setup (replaces the Matplotlib figure)
        # ------------------------------------------------------------------
        try:
            self.gl_canvas = VoxelOpenGLCanvas(self.viewer_frame, width=600, height=600)
            self.gl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)
            print("✓ OpenGL canvas created successfully")

            # Make canvas focusable and give it focus
            self.gl_canvas.focus_set()
            self.gl_canvas.bind("<Button-1>", lambda e: self.gl_canvas.focus_set())
            
            # Bind mouse wheel event for zoom functionality directly on the canvas
            self.gl_canvas.bind("<MouseWheel>", self.on_scroll)
            print("✓ OpenGL canvas events bound")

        except Exception as e:
            print(f"✗ Error creating OpenGL canvas: {e}")
            import traceback
            traceback.print_exc()
            # Create a fallback label
            error_label = tk.Label(self.viewer_frame, 
                                 text=f"OpenGL Error: {str(e)}\nCheck console for details",
                                 bg="red", fg="white", font=("Arial", 12))
            error_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)
            return

        # Create a frame for displaying all unique colors
        self.color_frame = tk.Frame(self.viewer_frame, relief=tk.RAISED, borderwidth=2, bg="#f0f0f0")
        self.color_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Make the color frame scrollable
        self.color_canvas = tk.Canvas(self.color_frame, bg="#f0f0f0")
        self.scrollbar = tk.Scrollbar(self.color_frame, orient="vertical", command=self.color_canvas.yview)
        self.scrollable_frame = tk.Frame(self.color_canvas, bg="#f0f0f0")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.color_canvas.configure(
                scrollregion=self.color_canvas.bbox("all")
            )
        )

        self.color_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.color_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.color_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel events for scrolling - only to the color frame
        self.color_canvas.bind_all("<MouseWheel>", self.on_mousewheel_color_frame)

        # Reset and recreate checkbuttons list
        self.checkbuttons = []

        # Create a title label for the color selection area
        color_frame_title = tk.Label(self.scrollable_frame, text="Select Colors:", font=("Helvetica", 14, "bold"),
                                     bg="#f0f0f0")
        color_frame_title.pack(pady=10, padx=5)

        # Populate color frame with checkbuttons
        print("Populating color frame...")
        self.populate_color_frame()

        # Create a frame for the select/unselect all buttons
        select_frame = tk.Frame(self.scrollable_frame, bg="#f0f0f0")
        select_frame.pack(pady=10)

        # Create a "Select All" button
        select_all_button = tk.Button(select_frame, text="Select All", command=self.select_all_colors,
                                      font=("Helvetica", 12), bg="#4a90e2", fg="white", relief=tk.FLAT)
        select_all_button.pack(side=tk.LEFT, padx=5)

        # Create an "Unselect All" button
        unselect_all_button = tk.Button(select_frame, text="Unselect All", command=self.unselect_all_colors,
                                        font=("Helvetica", 12), bg="#4a90e2", fg="white", relief=tk.FLAT)
        unselect_all_button.pack(side=tk.LEFT, padx=5)

        # Create an "Apply" button
        apply_button = tk.Button(self.scrollable_frame, text="Apply", command=self.apply_selections,
                                 font=("Helvetica", 12), bg="#4a90e2", fg="white", relief=tk.FLAT)
        apply_button.pack(pady=10)

        # Create the initial voxel plot - delay this to ensure canvas is ready
        print("Scheduling initial plot creation...")
        self.root.after(100, lambda: self.create_voxel_plot(full_image=True))

    def on_mousewheel_color_frame(self, event):
        """
        Scroll the color frame using the mouse wheel.
        """
        if self.color_canvas.winfo_containing(event.x_root, event.y_root) in [self.color_canvas, self.scrollable_frame]:
            self.color_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def show_single_color_frame(self):
        """Show single color frame with optimized rendering"""
        self.clear_frames()
        self.single_color_frame.pack(fill=tk.BOTH, expand=True)
        self.metadata_entries = {}

        for widget in self.single_color_frame.winfo_children():
            widget.destroy()

        self.add_navbar(self.single_color_frame)

        # Extract current color data
        p_value = self.viewer.selected_p_values[self.viewer.current_color_index]
        self.viewer.current_p_value = p_value
        self.viewer.current_spine_number = p_value
        rgb_color = self.viewer.rgb_colors[p_value]
        rgb_normalized = np.array(rgb_color, dtype=np.float32) / 255.0

        # Get mask efficiently using boolean indexing
        mask = self.viewer.image_uploaded == p_value

        # Apply downsampling if enabled
        if self.downsample_factor.get() > 1:
            factor = self.downsample_factor.get()
            mask = mask[::factor, ::factor, ::factor]

        total_points = np.sum(mask)

        # Extract coordinates once
        coords = np.argwhere(mask)

        # Down-sample to the user-defined maximum
        max_pts = self.max_points.get() * 1000
        if coords.shape[0] > max_pts:
            sel = np.random.choice(coords.shape[0], max_pts, replace=False)
            coords = coords[sel]

        # Build colours array (all voxels share the same colour for this view)
        colours = np.tile(rgb_normalized, (coords.shape[0], 1)).astype(np.float32)

        # Build point array (x, y, z)
        points = np.column_stack((coords[:, 2], coords[:, 1], coords[:, 0])).astype(np.float32)

        # ------------------------------------------------------------------
        # OpenGL canvas – pack it on the left side (similar to the old mpl canvas)
        # ------------------------------------------------------------------
        self.single_gl_canvas = VoxelOpenGLCanvas(self.single_color_frame, width=600, height=600)
        self.single_gl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)

        self.single_gl_canvas.set_points(points, colours)

        # Make canvas focusable and give it focus
        self.single_gl_canvas.focus_set()
        self.single_gl_canvas.bind("<Button-1>", lambda e: self.single_gl_canvas.focus_set())

        # Mouse-wheel zoom
        self.single_gl_canvas.bind("<MouseWheel>", lambda e: self.single_gl_canvas.zoom(1.1 if e.delta < 0 else 1/1.1))

        # Create metadata panel
        self._create_metadata_panel(p_value, mask)

        # NOTE: Removed matplotlib scatter plot to prevent separate window opening
        # All visualization is now handled by the OpenGL canvas within the application

        # Force garbage collection
        gc.collect()

        self.update_status("Ready")

    def _create_metadata_panel(self, p_value, mask):
        """Create metadata panel with optimized estimation processing"""
        metadata_frame = tk.Frame(self.single_color_frame, bg="#f0f0f0", relief=tk.RAISED, borderwidth=2)
        metadata_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        # Make the metadata frame scrollable
        self.metadata_canvas = tk.Canvas(metadata_frame, bg="#f0f0f0", width=300)  # Set fixed width
        scrollbar = tk.Scrollbar(metadata_frame, orient="vertical", command=self.metadata_canvas.yview)
        self.scrollable_metadata_frame = tk.Frame(self.metadata_canvas, bg="#f0f0f0")

        self.scrollable_metadata_frame.bind(
            "<Configure>",
            lambda e: self.metadata_canvas.configure(
                scrollregion=self.metadata_canvas.bbox("all")
            )
        )

        self.metadata_canvas.create_window((0, 0), window=self.scrollable_metadata_frame, anchor="nw")
        self.metadata_canvas.configure(yscrollcommand=scrollbar.set)

        self.metadata_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel events for scrolling - only to the metadata frame
        self.metadata_canvas.bind_all("<MouseWheel>", self.on_mousewheel_metadata_frame)

        # Create title label in the scrollable frame
        metadata_label = tk.Label(self.scrollable_metadata_frame, 
                                text="Metadata:", 
                                font=("Helvetica", 14, "bold"),
                                bg="#f0f0f0")
        metadata_label.pack(pady=5)

        # Status label for estimation progress
        status_label = tk.Label(self.scrollable_metadata_frame, 
                              text="Calculating estimations...",
                              font=("Helvetica", 10), 
                              bg="#f0f0f0")
        status_label.pack(pady=5)

        # Run estimations in background thread
        def run_estimations():
            try:
                # Create a new DataInImage instance for this color
                data_in_spine = DataInImage()
                
                # Copy existing metadata if available
                if self.viewer.data_in_image:
                    data_in_spine.__dict__.update(self.viewer.data_in_image.__dict__)
                
                # Create estimator and run calculations
                estimator = Estimator(self.viewer.image_uploaded, self.viewer.rgb_colors)
                single_color_data = estimator.run_estimations(p_value, data_in_spine)
                
                # Update UI in main thread
                self.root.after(0, lambda: self._update_metadata_ui(single_color_data, self.scrollable_metadata_frame, mask))
                self.root.after(0, lambda: status_label.destroy())
            except Exception as e:
                self.root.after(0, lambda: status_label.config(
                    text=f"Error calculating estimations: {str(e)}",
                    fg="red"
                ))
                print(f"Error in estimation calculation: {e}")

        Thread(target=run_estimations, daemon=True).start()

    def _update_metadata_ui(self, single_color_data, frame, mask):
        """Update metadata UI with estimation results"""
        # Clear any existing widgets in the frame
        for widget in frame.winfo_children():
            widget.destroy()

        # Create title label again since we cleared all widgets
        metadata_label = tk.Label(frame, 
                                text="Metadata:", 
                                font=("Helvetica", 14, "bold"),
                                bg="#f0f0f0")
        metadata_label.pack(pady=5)

        single_color_data.print_data()

        # Store the data and mask for saving
        self.viewer.single_color_data = single_color_data
        self.viewer.current_mask = mask

        # Create metadata entries
        for key, attr in single_color_data.tag_dict.items():
            # Create a frame for each metadata item
            item_frame = tk.Frame(frame, bg="#f0f0f0")
            item_frame.pack(fill='x', padx=5, pady=2)

            # Label for the metadata item
            metadata_label = tk.Label(item_frame, 
                                    text=f"{key}:", 
                                    font=("Helvetica", 12),
                                    bg="#f0f0f0",
                                    anchor='w')
            metadata_label.pack(side='top', fill='x')

            # Entry for the metadata value
            default_value = getattr(single_color_data, attr)
            var = tk.StringVar(value=str(default_value))
            metadata_entry = tk.Entry(item_frame, 
                                    textvariable=var, 
                                    font=("Helvetica", 12),
                                    width=30)
            metadata_entry.pack(side='top', fill='x')
            self.metadata_entries[attr] = var

        # Add buttons frame
        button_frame = tk.Frame(frame, bg="#f0f0f0")
        button_frame.pack(pady=10)

        # Add buttons
        save_button = tk.Button(button_frame, 
                              text="Save",
                              command=self.save_current_mask,
                              font=("Helvetica", 12), 
                              bg="#4a90e2", 
                              fg="white", 
                              relief=tk.FLAT)
        save_button.pack(side=tk.LEFT, padx=5)

        prev_button = tk.Button(button_frame, 
                              text="Previous",
                              command=self.previous_single_color,
                              font=("Helvetica", 12), 
                              bg="#4a90e2", 
                              fg="white", 
                              relief=tk.FLAT)
        prev_button.pack(side=tk.LEFT, padx=5)

        next_button_text = "Finish" if self.viewer.current_color_index == len(
            self.viewer.selected_p_values) - 1 else "Next"
        next_button = tk.Button(button_frame, 
                              text=next_button_text,
                              command=self.next_single_color,
                              font=("Helvetica", 12), 
                              bg="#4a90e2", 
                              fg="white", 
                              relief=tk.FLAT)
        next_button.pack(side=tk.LEFT, padx=5)

    def on_mousewheel_metadata_frame(self, event):
        """
        Scroll the metadata frame using the mouse wheel.
        """
        if event.widget.winfo_containing(event.x_root, event.y_root) in [self.metadata_canvas, self.scrollable_metadata_frame]:
            self.metadata_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def save_current_mask(self):
        """
        Save the current mask as a multi-layered TIFF file with metadata when the "SAVE" button is pressed.
        """
        if self.save_directory.get():
            # Before saving, update self.viewer.single_color_data with modified values
            attr_value_dict = {}
            for attr in self.viewer.single_color_data.tag_dict.values():
                var = self.metadata_entries.get(attr)
                if var:
                    value_str = var.get()
                    attr_value_dict[attr] = value_str
            # Update self.viewer.single_color_data with the modified values
            self.viewer.single_color_data.update_from_strings(attr_value_dict)

            base_name = os.path.splitext(os.path.basename(self.viewer.image_path))[0]
            save_dir = self.save_directory.get()
            p_value = self.viewer.current_p_value

            # Paths
            tiff_save_path = os.path.join(save_dir, f"{base_name}_color_{p_value}_data.tiff")
            txt_save_path = os.path.join(save_dir, f"{base_name}_color_{p_value}_data.csv")

            # Legacy export functions (ensures wrappers are used)
            de = DataExport()
            labeled_matrix = self.viewer.current_mask.astype(int)
            try:
                de.export_spine_as_text(txt_save_path, labeled_matrix, self.viewer.single_color_data)
                de.export_spine_as_tiff(tiff_save_path, labeled_matrix, self.viewer.single_color_data)
            except Exception as err:
                print(f"Legacy export failed: {err}")

            # Still call optimized saver for large data sets
            try:
                self.viewer.save_data_as_tiff(tiff_save_path, self.viewer.single_color_data)
            except Exception:
                pass
            print(f"Saved data (legacy + optimized) to {save_dir}")
        else:
            messagebox.showinfo("Save Directory Not Set", "Please set a save directory before saving.")

    def clear_frames(self):
        """Remove all frames and clean up bindings and resources"""
        # Clean up OpenGL canvases
        if hasattr(self, 'gl_canvas') and self.gl_canvas is not None:
            self.gl_canvas.destroy()
            self.gl_canvas = None
        if hasattr(self, 'single_gl_canvas') and self.single_gl_canvas is not None:
            self.single_gl_canvas.destroy()
            self.single_gl_canvas = None

        # Unbind all mousewheel events before clearing frames
        try:
            self.color_canvas.unbind_all("<MouseWheel>")
        except AttributeError:
            pass
        try:
            self.metadata_canvas.unbind_all("<MouseWheel>")
        except AttributeError:
            pass

        # Remove all frames from the root window
        for frame in (self.welcome_frame, self.viewer_frame, self.single_color_frame):
            frame.pack_forget()

        # Clear metadata entries
        self.metadata_entries.clear()
        
        # Force garbage collection
        gc.collect()

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_directory.set(directory)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff"), ("All files", "*.*")])
        if file_path:
            self.viewer.initialize_viewer(file_path)
            self.show_viewer_frame()

    def add_navbar(self, window):
        """
        Add a navigation bar to the specified window.
        """
        navbar = tk.Frame(window, bg="#4a4a4a")
        navbar.pack(side=tk.TOP, fill=tk.X)
        nav_title = tk.Label(navbar, text="3D TIFF Viewer", font=("Helvetica", 16, "bold"), fg="white", bg="#4a4a4a")
        nav_title.pack(side=tk.LEFT, padx=20)

    def on_scroll(self, event):
        """
        Mouse-wheel zoom handler for the OpenGL canvas.
        """
        direction = 1 if event.delta < 0 else -1  # Windows gives delta multiples of 120
        base_scale = 1.1
        scale_factor = base_scale if direction > 0 else 1 / base_scale

        if hasattr(self, 'gl_canvas') and self.gl_canvas is not None:
            self.gl_canvas.zoom(scale_factor)

    def create_voxel_plot(self, full_image=False):
        """Create (or update) the point-cloud representation on the OpenGL canvas."""

        if not hasattr(self, "gl_canvas") or self.gl_canvas is None:
            print("Warning: OpenGL canvas not yet initialized")
            return  # Canvas not yet initialised

        self.update_status("Creating plot…")
        print(f"Creating voxel plot, full_image={full_image}")

        try:
            # Check if we have valid image data
            if not hasattr(self.viewer, 'image_uploaded') or self.viewer.image_uploaded is None:
                print("Error: No image data loaded")
                self.update_status("Error: No image data loaded")
                return
                
            if not hasattr(self.viewer, 'rgb_colors') or self.viewer.rgb_colors is None:
                print("Error: No color data loaded")
                self.update_status("Error: No color data loaded")
                return

            print(f"Image shape: {self.viewer.image_uploaded.shape}")
            print(f"Available colors: {len(self.viewer.rgb_colors)}")

            if full_image:
                # Create mask for all non-zero voxels
                mask = self.viewer.image_uploaded > 0
                print(f"Full image mask: {np.sum(mask)} voxels")
            else:
                mask = full_image  # Not used but keep structure

            # Down-sample if requested
            if self.downsample_factor.get() > 1:
                f = self.downsample_factor.get()
                print(f"Downsampling by factor {f}")
                mask = mask[::f, ::f, ::f]
                print(f"After downsampling: {np.sum(mask)} voxels")

            # Extract coordinates of all non-zero voxels
            coords = np.argwhere(mask)
            print(f"Found {len(coords)} coordinate points")

            if len(coords) == 0:
                print("Warning: No voxels found to display")
                self.update_status("Warning: No voxels found to display")
                # Set empty data to clear the canvas
                self.gl_canvas.set_points(np.empty((0, 3)), np.empty((0, 3)))
                return

            # Limit the number of points for performance
            max_pts = self.max_points.get() * 1000
            if coords.shape[0] > max_pts:
                print(f"Limiting points from {coords.shape[0]} to {max_pts}")
                sel = np.random.choice(coords.shape[0], max_pts, replace=False)
                coords = coords[sel]

            # Map voxel indices to RGB colours (0-255 ➔ 0-1)
            if self.downsample_factor.get() > 1:
                # Need to map back to original coordinates for color lookup
                f = self.downsample_factor.get()
                original_coords = coords * f
                # Ensure we don't go out of bounds
                original_coords = np.minimum(original_coords, 
                                           np.array(self.viewer.image_uploaded.shape) - 1)
                voxel_indices = self.viewer.image_uploaded[
                    original_coords[:,0], original_coords[:,1], original_coords[:,2]]
            else:
                voxel_indices = self.viewer.image_uploaded[coords[:,0], coords[:,1], coords[:,2]]

            print(f"Voxel indices range: {voxel_indices.min()} to {voxel_indices.max()}")
            
            # Ensure all indices are valid
            valid_indices = (voxel_indices >= 0) & (voxel_indices < len(self.viewer.rgb_colors))
            if not np.all(valid_indices):
                print(f"Warning: {np.sum(~valid_indices)} invalid color indices found")
                coords = coords[valid_indices]
                voxel_indices = voxel_indices[valid_indices]

            # Get RGB colors
            colour_rgb = np.array([self.viewer.rgb_colors[idx] for idx in voxel_indices], 
                                dtype=np.float32) / 255.0

            # Re-order coordinates to (x, y, z) - swap z and x for proper orientation
            points = np.column_stack((coords[:, 2], coords[:, 1], coords[:, 0])).astype(np.float32)

            print(f"Final data: {len(points)} points, {len(colour_rgb)} colors")
            print(f"Points range: {points.min(axis=0)} to {points.max(axis=0)}")
            print(f"Colors range: {colour_rgb.min(axis=0)} to {colour_rgb.max(axis=0)}")

            # Set the data in the OpenGL canvas
            self.gl_canvas.set_points(points, colour_rgb)
            self.update_status("Plot created successfully")
            print("✓ Voxel plot created successfully")

        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            self.update_status(error_msg)

    def populate_color_frame(self):
        """
        Populate the frame with the unique colors extracted from the image.
        """
        for idx, color in self.viewer.color_index_list:
            rgb_color = tuple(color)
            hex_color = f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'
            var = tk.BooleanVar(value=False)  # Set the initial value to False for all colors
            color_button = tk.Checkbutton(self.scrollable_frame, bg=hex_color, width=25, height=2, variable=var,
                                          command=lambda p=idx, v=var: self.on_color_select(p, v))
            color_button.p_value = idx  # Store p_value in the button for easy access
            color_button.pack(pady=5, padx=5, anchor='w')
            self.checkbuttons.append((color_button, var))

    def on_color_select(self, p_value, var):
        """
        Handle the color selection and store the P value.
        If the color is already selected, deselect it.
        """
        if var.get():
            if p_value not in self.viewer.selected_p_values:
                self.viewer.selected_p_values.append(p_value)
        else:
            if p_value in self.viewer.selected_p_values:
                self.viewer.selected_p_values.remove(p_value)

    def select_all_colors(self):
        for btn, var in self.checkbuttons:
            if not var.get():
                var.set(True)
                self.on_color_select(btn.p_value, var)

    def unselect_all_colors(self):
        for btn, var in self.checkbuttons:
            if var.get():
                var.set(False)
                self.on_color_select(btn.p_value, var)

    def apply_selections(self):
        """
        Apply the selected colors and print the P values.
        """
        print("Selected P values:", self.viewer.selected_p_values)
        self.viewer.current_color_index = 0
        if self.viewer.selected_p_values:
            self.display_next_color()
        else:
            messagebox.showinfo("No Colors Selected", "Please select colors and apply them first.")

    def display_next_color(self):
        """
        Display the next color from the selected colors.
        Ensures proper cleanup before displaying new color.
        """
        if not self.viewer.selected_p_values:
            messagebox.showinfo("No Colors Selected", "Please select colors and apply them first.")
            return

        if self.viewer.current_color_index < len(self.viewer.selected_p_values):
            # Clean up any existing resources
            if hasattr(self, 'single_gl_canvas') and self.single_gl_canvas is not None:
                self.single_gl_canvas.destroy()
                self.single_gl_canvas = None
            
            # Clear metadata entries
            self.metadata_entries.clear()
            
            # Force garbage collection
            gc.collect()

            # Switch to the single color frame
            self.show_single_color_frame()

            # Increment the index after displaying the color
            self.viewer.current_color_index += 1
        else:
            # Exit the program when no more colors to display
            self.root.destroy()

    def next_single_color(self):
        """
        Display the next color.
        Properly cleans up resources before transitioning.
        """
        # Clean up current figure and canvas
        if hasattr(self, 'single_gl_canvas') and self.single_gl_canvas is not None:
            self.single_gl_canvas.destroy()
            self.single_gl_canvas = None
            
        # Clear metadata entries
        self.metadata_entries.clear()
        
        # Force garbage collection
        gc.collect()

        if self.viewer.current_color_index < len(self.viewer.selected_p_values):
            self.display_next_color()
        else:
            # Exit the program when 'Finish' is clicked
            self.root.destroy()

    def previous_single_color(self):
        """
        Display the previous color.
        If it's the first color plot, return to the full figure window.
        Properly cleans up resources before transitioning.
        """
        # Clean up current figure and canvas
        if hasattr(self, 'single_gl_canvas') and self.single_gl_canvas is not None:
            self.single_gl_canvas.destroy()
            self.single_gl_canvas = None
        
        # Clear metadata entries
        self.metadata_entries.clear()
        
        # Force garbage collection
        gc.collect()

        if self.viewer.current_color_index > 1:
            # Go back two steps (one for the current view, one for the previous)
            self.viewer.current_color_index -= 2
            self.display_next_color()
        else:
            # If it's the first color plot, return to the full figure window
            self.viewer.current_color_index = 0
            # Keep the selected values but return to selection view
            saved_selections = self.viewer.selected_p_values.copy()
            
            # Show viewer frame will recreate the figure and canvas
            self.show_viewer_frame()
            
            # Restore the checkbutton states
            for btn, var in self.checkbuttons:
                if btn.p_value in saved_selections:
                    var.set(True)
                    self.on_color_select(btn.p_value, var)
                else:
                    var.set(False)
            
            # Restore the selected p_values
            self.viewer.selected_p_values = saved_selections
            
            # Create the initial voxel plot with all colors
            self.create_voxel_plot(full_image=True)
            
            # OpenGL canvas will redraw automatically via create_voxel_plot 