import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import io
import gc
from threading import Thread
from queue import Queue
import mmap
from concurrent.futures import ThreadPoolExecutor

from optimized_logic import Estimator, save_data_as_tiff_optimized

from Extract_Figures_FV_Classes import DataExport, Utilities


class DataInImage:
    """
    Class for handling metadata of 3D figure properties.
    """
    tag_dict = {
        'Number_of_Layers': 'num_layers',
        'Image_Height': 'height',
        'Image_Width': 'width',
        'X_Resolution': 'x_resolution',
        'Y_Resolution': 'y_resolution',
        'Z_Resolution': 'z_resolution',
        'Resolution_Unit': 'resolution_unit',
        'Volume': 'volume',
        'Volume_unit': 'volume_unit',
        'Surface': 'surface',
        'Surface_unit': 'surface_unit',
        'L': 'L',
        'd': 'd',
        'Spine_Color': 'spine_color',
        'Connection_Point': 'point_connect',
        'Spine_Middle_Point': 'point_middle',
        'Spine_Far_Point': 'point_far',
        'Connection_Is_Inner': 'point_connect_value',
        'Description': 'description'
    }

    def __init__(self):
        self.num_layers = None
        self.height = None
        self.width = None
        self.x_resolution = None
        self.y_resolution = None
        self.z_resolution = None
        self.resolution_unit = "nm"
        self.volume = None
        self.volume_unit = "um3"
        self.surface = None
        self.surface_unit = "um2"
        self.L = None
        self.d = None
        self.spine_color = (255, 0, 0)
        self.point_connect = (None, None, None)
        self.point_middle = (None, None, None)
        self.point_far = (None, None, None)
        self.point_connect_value = None
        self.description = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Attribute {key} not found in the class.")

    def update_from_strings(self, attr_value_dict):
        """
        Update attributes from a dictionary of strings, converting to the appropriate type.
        """
        for attr, value_str in attr_value_dict.items():
            if not hasattr(self, attr):
                continue  # Skip unknown attributes
            # Determine the type of the attribute
            current_value = getattr(self, attr)
            attr_type = type(current_value)
            if current_value is None:
                # Assume default types based on attribute name
                if attr in ['num_layers', 'height', 'width', 'point_connect_value']:
                    attr_type = int
                elif attr in ['x_resolution', 'y_resolution', 'z_resolution', 'volume', 'surface', 'L', 'd']:
                    attr_type = float
                elif attr in ['point_connect', 'point_middle', 'point_far', 'spine_color']:
                    attr_type = tuple
                else:
                    attr_type = str

            if attr_type is tuple:
                # Handle tuples, e.g., "(1, 2, 3)"
                try:
                    # Remove parentheses and split by comma
                    value = tuple(map(float, value_str.strip('()').split(',')))
                    # If original tuple contains ints, convert to ints
                    if all(isinstance(x, int) for x in current_value or []):
                        value = tuple(map(int, value))
                except ValueError:
                    value = current_value  # Keep original value if parsing fails
            elif attr_type is int:
                try:
                    value = int(float(value_str))  # Convert via float to handle inputs like "1.0"
                except ValueError:
                    value = current_value  # Keep original value if parsing fails
            elif attr_type is float:
                try:
                    value = float(value_str)
                except ValueError:
                    value = current_value  # Keep original value if parsing fails
            else:
                # Keep as string
                value = value_str
            setattr(self, attr, value)

    def print_data(self):
        data = self.tag_dict
        for key, value in data.items():
            print(f"{key}: {getattr(self, value)}")


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

        # Matplotlib figure setup
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Embed the matplotlib figure into Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viewer_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)

        # Bind mouse scroll event for zoom functionality
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

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

        # Create the initial voxel plot
        self.create_voxel_plot(full_image=True)

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

        # Optimize visualization based on data size and performance settings
        total_points = np.sum(mask)
        use_scatter = total_points > self.max_points.get() * 1000

        # Create figure with optimized settings based on quality level
        quality_factor = 6 - self.quality_level.get()  # Convert quality level to scaling factor
        dpi = max(80, 100 // quality_factor)  # Adjust DPI based on quality
        self.single_fig = plt.figure(figsize=(8, 6), dpi=dpi)
        self.single_ax = self.single_fig.add_subplot(111, projection='3d')

        if use_scatter:
            # Memory-efficient coordinate extraction
            coords = np.argwhere(mask)
            if len(coords) > self.max_points.get() * 1000:
                idx = np.random.choice(len(coords),
                                     self.max_points.get() * 1000,
                                     replace=False)
                coords = coords[idx]

            # Adjust point size and alpha based on quality level
            point_size = max(5, 15 // quality_factor)
            alpha = max(0.3, 0.8 / quality_factor)

            # Optimized scatter plot
            self.single_ax.scatter(coords[:, 2],
                                 coords[:, 1],
                                 coords[:, 0],
                                 c=[rgb_normalized],
                                 s=point_size,
                                 alpha=alpha)
        else:
            # Optimized voxel plot
            # Only create colors array for visible voxels
            colors = np.zeros(mask.shape + (4,), dtype=np.float32)
            colors[mask] = np.append(rgb_normalized, 0.7)  # Use 0.7 alpha

            # Process in chunks if needed
            if total_points > 50000:
                chunk_size = self.chunk_size.get()
                for start_idx in range(0, len(mask), chunk_size):
                    end_idx = min(start_idx + chunk_size, len(mask))
                    chunk_mask = mask[start_idx:end_idx]
                    chunk_colors = colors[start_idx:end_idx]
                    
                    self.single_ax.voxels(chunk_mask,
                                        facecolors=chunk_colors,
                                        edgecolor=None)  # Remove edges for better visibility
                    
                    # Update progress
                    progress = (end_idx / len(mask)) * 100
                    self.update_status(f"Rendering voxels... {progress:.1f}%")
            else:
                self.single_ax.voxels(mask,
                                    facecolors=colors,
                                    edgecolor=None)  # Remove edges for better visibility

        self.single_ax.set_title('3D Voxel Plot')
        self.single_ax.set_xlabel('X axis')
        self.single_ax.set_ylabel('Y axis')
        self.single_ax.set_zlabel('Z axis')
        self.single_ax.set_box_aspect([1, 1, 1])

        # Embed figure with optimized canvas
        self.single_canvas = FigureCanvasTkAgg(self.single_fig, master=self.single_color_frame)
        self.single_canvas.draw()
        self.single_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add zoom functionality
        self.single_canvas.mpl_connect("scroll_event",
                                     lambda event: self.on_scroll_single(event, self.single_ax, self.single_canvas))

        # Create metadata panel
        self._create_metadata_panel(p_value, mask)

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

            # Save the mask as a layered TIFF file with metadata
            tiff_save_path = os.path.join(save_dir, f"{base_name}_color_{p_value}_data.tiff")
            self.viewer.save_data_as_tiff(tiff_save_path, self.viewer.single_color_data)
            print(f"Saved data TIFF with metadata to {tiff_save_path}")
        else:
            messagebox.showinfo("Save Directory Not Set", "Please set a save directory before saving.")

    def clear_frames(self):
        """Remove all frames and clean up bindings and resources"""
        # Clean up matplotlib figures and canvases
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            self.fig = None
        if hasattr(self, 'single_fig'):
            plt.close(self.single_fig)
            self.single_fig = None
            
        # Clean up canvases
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if hasattr(self, 'single_canvas'):
            self.single_canvas.get_tk_widget().destroy()
            self.single_canvas = None

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
        Handle zooming in and out of the 3D plot using the mouse scroll wheel.
        """
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            # No need to handle other events
            return

        # Get the current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        # Compute the new limits
        self.ax.set_xlim([xlim[0] * scale_factor, xlim[1] * scale_factor])
        self.ax.set_ylim([ylim[0] * scale_factor, ylim[1] * scale_factor])
        self.ax.set_zlim([zlim[0] * scale_factor, zlim[1] * scale_factor])

        self.canvas.draw()

    def create_voxel_plot(self, full_image=False):
        """Create an optimized voxel plot"""
        if not hasattr(self, 'ax') or self.ax is None:
            # If ax doesn't exist, create a new figure and axis
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            # Clear existing plot
            self.ax.clear()

        self.update_status("Creating plot...")

        if full_image:
            try:
                # Get mask with memory-efficient boolean indexing
                mask = self.viewer.image_uploaded > 0
                
                # Create color array with proper RGBA values
                colors = np.zeros(mask.shape + (4,), dtype=np.float32)
                unique_values = np.unique(self.viewer.image_uploaded[mask])
                
                for idx in unique_values:
                    if idx == 0:  # Skip background
                        continue
                    color_mask = self.viewer.image_uploaded == idx
                    rgb_color = np.array(self.viewer.rgb_colors[idx], dtype=np.float32) / 255.0
                    colors[color_mask] = np.append(rgb_color, 1.0)  # Full opacity
                
                # Downsample if needed
                if self.downsample_factor.get() > 1:
                    factor = self.downsample_factor.get()
                    mask = mask[::factor, ::factor, ::factor]
                    colors = colors[::factor, ::factor, ::factor]

                # Use scatter plot for very large datasets
                if np.sum(mask) > self.max_points.get() * 1000:
                    coords = np.argwhere(mask)
                    if len(coords) > self.max_points.get() * 1000:
                        idx = np.random.choice(len(coords),
                                             self.max_points.get() * 1000,
                                             replace=False)
                        coords = coords[idx]
                    
                    # Get colors for scatter plot
                    point_colors = colors[coords[:, 0], coords[:, 1], coords[:, 2], :3]
                    
                    quality_factor = 6 - self.quality_level.get()
                    point_size = max(5, 15 // quality_factor)
                    alpha = max(0.3, 0.8 / quality_factor)
                    
                    self.ax.scatter(coords[:, 2],
                                  coords[:, 1],
                                  coords[:, 0],
                                  c=point_colors,
                                  s=point_size,
                                  alpha=alpha)
                else:
                    # Use voxels for smaller datasets
                    self.ax.voxels(mask,
                                 facecolors=colors,
                                 edgecolor=None,  # Remove edges for better visibility
                                 alpha=0.8)  # Slightly transparent

                self.ax.set_title('3D Voxel Plot')
                self.ax.set_xlabel('X axis')
                self.ax.set_ylabel('Y axis')
                self.ax.set_zlabel('Z axis')
                self.ax.set_box_aspect([1, 1, 1])
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                self.update_status(f"Error creating plot: {str(e)}")
                return

        if hasattr(self, 'canvas'):
            self.canvas.draw()
        self.update_status("Ready")

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
            if hasattr(self, 'single_fig'):
                plt.close(self.single_fig)
                self.single_fig = None
            if hasattr(self, 'single_canvas'):
                self.single_canvas.get_tk_widget().destroy()
                self.single_canvas = None
            
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
        if hasattr(self, 'single_fig'):
            plt.close(self.single_fig)
            self.single_fig = None
        if hasattr(self, 'single_canvas'):
            self.single_canvas.get_tk_widget().destroy()
            self.single_canvas = None
            
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
        if hasattr(self, 'single_fig'):
            plt.close(self.single_fig)
            self.single_fig = None
        if hasattr(self, 'single_canvas'):
            self.single_canvas.get_tk_widget().destroy()
            self.single_canvas = None
        
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
            
            # Force a redraw of the canvas
            if hasattr(self, 'canvas'):
                self.canvas.draw()

    def on_scroll_single(self, event, ax, canvas):
        """
        Handle zooming in and out of the 3D plot using the mouse scroll wheel for the single color windows.
        """
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            # No need to handle other events
            return

        # Get the current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Compute the new limits
        ax.set_xlim([xlim[0] * scale_factor, xlim[1] * scale_factor])
        ax.set_ylim([ylim[0] * scale_factor, ylim[1] * scale_factor])
        ax.set_zlim([zlim[0] * scale_factor, zlim[1] * scale_factor])

        canvas.draw()


class TIFFViewer3D:
    """
    Main class that handles the loading and processing of the TIFF images.
    """

    def __init__(self):
        # Initialize variables
        self.rgb_colors = None
        self.image_uploaded = None
        self.selected_p_values = []
        self.current_color_index = 0
        self.current_spine_number = None
        self._processing_queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize metadata handling
        self.data_in_image = DataInImage()
        self.single_color_data = None
        self.current_mask = None

        # Create the Tkinter window
        self.root = tk.Tk()
        self.root.title("3D TIFF Viewer with Color Selection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Initialize UI
        self.ui = TIFFViewerUI(self.root, self)

        # Start background processing thread
        self._start_background_processing()

        # Start the Tkinter event loop
        self.root.mainloop()

    def _start_background_processing(self):
        """Start background processing thread"""
        def process_queue():
            while True:
                try:
                    func, args, kwargs = self._processing_queue.get()
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Background processing error: {e}")
                finally:
                    self._processing_queue.task_done()
                    gc.collect()

        thread = Thread(target=process_queue, daemon=True)
        thread.start()

    def initialize_viewer(self, image_path):
        """Initialize the viewer with optimized TIFF loading"""
        self.image_path = image_path
        self.ui.update_status("Loading TIFF file...")
        
        # Reset metadata for new file
        self.data_in_image = DataInImage()
        self.single_color_data = None
        self.current_mask = None
        
        try:
            # Memory-mapped file reading
            with open(image_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                img = Image.open(io.BytesIO(mm.read()))

                if img.mode != 'P':
                    raise ValueError('The image is not in indexed color mode (P mode).')

                if not hasattr(img, "n_frames") or img.n_frames < 1:
                    raise ValueError('The TIFF file does not contain multiple frames.')

                # Initialize basic metadata
                self.data_in_image.num_layers = img.n_frames
                self.data_in_image.height = img.size[1]
                self.data_in_image.width = img.size[0]
                
                # Get color palette and ensure it exists
                palette = img.getpalette()
                if not palette:
                    raise ValueError('No color palette found in the image.')
                
                # Process color palette
                num_colors = len(palette) // 3
                self.rgb_colors = []
                for i in range(num_colors):
                    r = palette[i * 3]
                    g = palette[i * 3 + 1]
                    b = palette[i * 3 + 2]
                    if r is not None and g is not None and b is not None:
                        self.rgb_colors.append((r, g, b))
                    else:
                        self.rgb_colors.append((0, 0, 0))  # Default to black for invalid colors
                
                # Ensure we have at least one color
                if not self.rgb_colors:
                    raise ValueError('No valid colors found in the palette.')

                # Convert to numpy array for faster processing
                self.rgb_colors_normalized = np.array(self.rgb_colors, dtype=np.float32) / 255.0

                # Process frames in chunks
                chunk_size = self.ui.chunk_size.get()
                total_frames = img.n_frames
                layers = []
                
                for start_frame in range(0, total_frames, chunk_size):
                    end_frame = min(start_frame + chunk_size, total_frames)
                    chunk_layers = []
                    
                    for frame in range(start_frame, end_frame):
                        img.seek(frame)
                        frame_data = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
                        chunk_layers.append(frame_data)
                        
                        # Update progress
                        progress = (frame + 1) / total_frames * 100
                        self.ui.update_status(f"Loading frames... {progress:.1f}%")
                    
                    # Stack chunk and append to layers
                    chunk_array = np.stack(chunk_layers, axis=0)
                    layers.append(chunk_array)
                    
                    # Force garbage collection after each chunk
                    gc.collect()

                # Stack all chunks
                self.image_uploaded = np.concatenate(layers, axis=0)
                
                # Clean up
                mm.close()
                del layers
                gc.collect()

                # Get unique indices and create color index list
                unique_indices = np.unique(self.image_uploaded)
                self.color_index_list = []
                for idx in unique_indices:
                    if idx == 0:  # Skip background
                        continue
                    if idx < len(self.rgb_colors):
                        self.color_index_list.append((idx, self.rgb_colors[idx]))
                    else:
                        print(f"Warning: Color index {idx} out of range")

                if not self.color_index_list:
                    raise ValueError('No valid color indices found in the image.')

                self.ui.update_status("TIFF file loaded successfully")

        except Exception as e:
            self.ui.update_status("Error loading TIFF file")
            tk.messagebox.showerror(
                "File Error",
                f"The selected file could not be read as a TIFF image.\nError: {str(e)}"
            )
            return

        self.selected_p_values = []

    def _create_colors_mapped(self):
        """Create color mapping with memory optimization"""
        try:
            # Calculate shape and preallocate memory
            shape = self.image_uploaded.shape + (4,)
            self.colors_mapped = np.zeros(shape, dtype=np.float32)
            
            # Process in chunks
            chunk_size = self.ui.chunk_size.get()
            for i in range(0, len(self.rgb_colors), chunk_size):
                chunk = slice(i, min(i + chunk_size, len(self.rgb_colors)))
                
                for idx in range(chunk.start, chunk.stop):
                    if idx == 0:
                        continue
                    
                    # Process each color index efficiently
                    mask = (self.image_uploaded == idx)
                    if np.any(mask):
                        rgb_normalized = self.rgb_colors_normalized[idx]
                        self.colors_mapped[mask] = np.append(rgb_normalized, 1.0)
                
                # Update progress
                progress = (i + chunk_size) / len(self.rgb_colors) * 100
                self.ui.update_status(f"Creating color mapping... {progress:.1f}%")
                
                # Clean up after each chunk
                gc.collect()
            
            self.ui.update_status("Color mapping complete")
            
        except Exception as e:
            self.ui.update_status(f"Error creating color mapping: {str(e)}")

    def save_data_as_tiff(self, save_path, data_in_spine):
        """Save TIFF with optimized memory usage"""
        save_data_as_tiff_optimized(save_path, self.current_mask, data_in_spine, self.ui.update_status)

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types"""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        else:
            return obj


# Run the 3D TIFF Viewer
if __name__ == "__main__":
    viewer = TIFFViewer3D()

""""
Maybe More file restrictions and checkups
verifiying P mode or choosing P or other mode?
Better UI ideas ? More beautiful ?
More Error handeling for the Modyfing estimation fields maybe ?
Stress test the program 

 Only use functions from the extract file: break that file into classes and single use classes. 
 Use for Save tiff file 



Start CLI part ?

"""