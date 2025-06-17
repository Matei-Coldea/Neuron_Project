#!/usr/bin/env python3
"""
Complete 3D TIFF Viewer
A comprehensive standalone application for viewing 3D TIFF images with full capabilities.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL.GL import *
from OpenGL.GLU import *
import tifffile
from PIL import Image
import json
import os
import threading
import time

class Complete3DTIFFViewer(OpenGLFrame):
    """Complete 3D TIFF Viewer with all necessary capabilities"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Data storage
        self.tiff_data = None
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.float32)
        self.original_points = None
        self.color_palette = None
        self.unique_values = None
        
        # View parameters
        self.scale = 1.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.translation_z = 0.0
        self.camera_distance = 5.0
        
        # Display settings
        self.point_size = 3.0
        self.show_axes = True
        self.show_grid = True
        self.show_bounding_box = True
        self.background_color = (0.1, 0.1, 0.1, 1.0)
        self.lighting_enabled = False
        
        # Interaction
        self.last_mouse_pos = None
        self.mouse_button = None
        
        # Performance
        self.max_points = 500000
        self.downsample_factor = 1
        
        # Initialize
        self.setup_bindings()
        
    def setup_bindings(self):
        """Setup mouse and keyboard bindings"""
        # Mouse events
        self.bind("<Button-1>", self.on_mouse_down)
        self.bind("<Button-2>", self.on_mouse_down)
        self.bind("<Button-3>", self.on_mouse_down)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<B2-Motion>", self.on_mouse_drag)
        self.bind("<B3-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<ButtonRelease-2>", self.on_mouse_up)
        self.bind("<ButtonRelease-3>", self.on_mouse_up)
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<Double-Button-1>", self.reset_view)
        
        # Keyboard events
        self.bind("<Key>", self.on_key_press)
        self.focus_set()
        
        # Window events
        self.bind("<Configure>", self.on_resize)
        
    def initgl(self):
        """Initialize OpenGL"""
        try:
            print("Initializing OpenGL...")
            
            # Basic OpenGL setup
            glClearColor(*self.background_color)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Point rendering
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            glPointSize(self.point_size)
            
            # Blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Setup viewport and projection
            self.setup_projection()
            
            print("✓ OpenGL initialized successfully")
            
        except Exception as e:
            print(f"OpenGL initialization error: {e}")
            
    def setup_projection(self):
        """Setup projection matrix"""
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def redraw(self):
        """Main rendering function"""
        try:
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Setup camera
            glLoadIdentity()
            gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)
            
            # Apply transformations
            glTranslatef(self.translation_x, self.translation_y, self.translation_z)
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)
            glRotatef(self.rotation_z, 0, 0, 1)
            glScalef(self.scale, self.scale, self.scale)
            
            # Draw coordinate system
            if self.show_axes:
                self.draw_axes()
                
            if self.show_grid:
                self.draw_grid()
                
            if self.show_bounding_box and len(self.points) > 0:
                self.draw_bounding_box()
            
            # Draw data points
            if len(self.points) > 0:
                self.draw_points()
            else:
                self.draw_placeholder()
                
        except Exception as e:
            print(f"Rendering error: {e}")
            
    def draw_axes(self):
        """Draw coordinate axes"""
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        
        # Y-axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        
        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        
        glEnd()
        
    def draw_grid(self):
        """Draw reference grid"""
        glColor3f(0.3, 0.3, 0.3)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        # Grid in XY plane
        for i in range(-5, 6):
            # Horizontal lines
            glVertex3f(-5.0, i, 0.0)
            glVertex3f(5.0, i, 0.0)
            # Vertical lines
            glVertex3f(i, -5.0, 0.0)
            glVertex3f(i, 5.0, 0.0)
            
        glEnd()
        
    def draw_bounding_box(self):
        """Draw bounding box around data"""
        if len(self.points) == 0:
            return
            
        min_coords = self.points.min(axis=0)
        max_coords = self.points.max(axis=0)
        
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        
        # Bottom face
        glVertex3f(min_coords[0], min_coords[1], min_coords[2])
        glVertex3f(max_coords[0], min_coords[1], min_coords[2])
        glVertex3f(max_coords[0], min_coords[1], min_coords[2])
        glVertex3f(max_coords[0], max_coords[1], min_coords[2])
        glVertex3f(max_coords[0], max_coords[1], min_coords[2])
        glVertex3f(min_coords[0], max_coords[1], min_coords[2])
        glVertex3f(min_coords[0], max_coords[1], min_coords[2])
        glVertex3f(min_coords[0], min_coords[1], min_coords[2])
        
        # Top face
        glVertex3f(min_coords[0], min_coords[1], max_coords[2])
        glVertex3f(max_coords[0], min_coords[1], max_coords[2])
        glVertex3f(max_coords[0], min_coords[1], max_coords[2])
        glVertex3f(max_coords[0], max_coords[1], max_coords[2])
        glVertex3f(max_coords[0], max_coords[1], max_coords[2])
        glVertex3f(min_coords[0], max_coords[1], max_coords[2])
        glVertex3f(min_coords[0], max_coords[1], max_coords[2])
        glVertex3f(min_coords[0], min_coords[1], max_coords[2])
        
        # Vertical edges
        glVertex3f(min_coords[0], min_coords[1], min_coords[2])
        glVertex3f(min_coords[0], min_coords[1], max_coords[2])
        glVertex3f(max_coords[0], min_coords[1], min_coords[2])
        glVertex3f(max_coords[0], min_coords[1], max_coords[2])
        glVertex3f(max_coords[0], max_coords[1], min_coords[2])
        glVertex3f(max_coords[0], max_coords[1], max_coords[2])
        glVertex3f(min_coords[0], max_coords[1], min_coords[2])
        glVertex3f(min_coords[0], max_coords[1], max_coords[2])
        
        glEnd()
        
    def draw_points(self):
        """Draw the point cloud"""
        glPointSize(self.point_size)
        glBegin(GL_POINTS)
        
        for i, (point, color) in enumerate(zip(self.points, self.colors)):
            glColor3f(*color)
            glVertex3f(*point)
            
        glEnd()
        
    def draw_placeholder(self):
        """Draw placeholder when no data is loaded"""
        glColor3f(0.5, 0.5, 0.5)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        
        # Draw a simple pattern
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) + abs(j) <= 2:
                    glVertex3f(i * 0.2, j * 0.2, 0.0)
                    
        glEnd()
        
    def load_tiff_file(self, filepath):
        """Load and process TIFF file"""
        try:
            print(f"Loading TIFF file: {filepath}")
            
            # Load TIFF data
            if filepath.lower().endswith('.tiff') or filepath.lower().endswith('.tif'):
                # Try tifffile first
                try:
                    self.tiff_data = tifffile.imread(filepath)
                    print(f"Loaded with tifffile: shape {self.tiff_data.shape}")
                except:
                    # Fallback to PIL
                    with Image.open(filepath) as img:
                        frames = []
                        try:
                            while True:
                                frames.append(np.array(img))
                                img.seek(img.tell() + 1)
                        except EOFError:
                            pass
                        self.tiff_data = np.stack(frames, axis=0)
                        print(f"Loaded with PIL: shape {self.tiff_data.shape}")
            else:
                # Try as regular image
                img = Image.open(filepath)
                self.tiff_data = np.array(img)
                if len(self.tiff_data.shape) == 2:
                    self.tiff_data = self.tiff_data[np.newaxis, ...]
                print(f"Loaded as image: shape {self.tiff_data.shape}")
            
            # Process the data
            self.process_tiff_data()
            
            # Update display
            self.after_idle(self.redraw)
            
            return True
            
        except Exception as e:
            print(f"Error loading TIFF file: {e}")
            messagebox.showerror("Error", f"Failed to load TIFF file:\n{str(e)}")
            return False
            
    def process_tiff_data(self):
        """Process TIFF data into point cloud"""
        if self.tiff_data is None:
            return
            
        print("Processing TIFF data...")
        
        # Handle different data types and shapes
        data = self.tiff_data.copy()
        
        # Ensure 3D
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        elif len(data.shape) == 4:
            # Take first channel if RGBA
            data = data[..., 0]
            
        print(f"Data shape: {data.shape}, dtype: {data.dtype}")
        print(f"Data range: {data.min()} to {data.max()}")
        
        # Get unique values for color mapping
        self.unique_values = np.unique(data)
        print(f"Unique values: {len(self.unique_values)} ({self.unique_values[:10]}...)")
        
        # Create color palette
        self.create_color_palette()
        
        # Extract non-zero coordinates
        if data.max() > 0:
            coords = np.argwhere(data > 0)
        else:
            # If all zeros, create a sample
            coords = np.argwhere(data >= data.min())
            
        print(f"Found {len(coords)} non-zero voxels")
        
        if len(coords) == 0:
            print("No data points found")
            return
            
        # Downsample if too many points
        if len(coords) > self.max_points:
            step = len(coords) // self.max_points
            coords = coords[::step]
            print(f"Downsampled to {len(coords)} points")
            
        # Get values at coordinates
        values = data[coords[:, 0], coords[:, 1], coords[:, 2]]
        
        # Convert coordinates to world space (swap and normalize)
        points = coords[:, [2, 1, 0]].astype(np.float32)  # (x, y, z)
        
        # Normalize to [-1, 1] range
        if len(points) > 0:
            center = points.mean(axis=0)
            points -= center
            max_range = np.abs(points).max()
            if max_range > 0:
                points /= max_range
                
        # Map values to colors
        colors = np.zeros((len(points), 3), dtype=np.float32)
        for i, value in enumerate(values):
            if value in self.color_palette:
                colors[i] = self.color_palette[value]
            else:
                # Default color
                colors[i] = [0.5, 0.5, 0.5]
                
        self.points = points
        self.colors = colors
        self.original_points = coords
        
        print(f"Processed {len(self.points)} points")
        print(f"Point range: {self.points.min(axis=0)} to {self.points.max(axis=0)}")
        
    def create_color_palette(self):
        """Create color palette for unique values"""
        self.color_palette = {}
        
        if len(self.unique_values) <= 1:
            self.color_palette[self.unique_values[0]] = [1.0, 1.0, 1.0]
            return
            
        # Create rainbow colors
        for i, value in enumerate(self.unique_values):
            if value == 0:
                self.color_palette[value] = [0.0, 0.0, 0.0]  # Black for background
            else:
                # Rainbow mapping
                hue = (i - 1) / max(1, len(self.unique_values) - 2)
                rgb = self.hsv_to_rgb(hue, 1.0, 1.0)
                self.color_palette[value] = rgb
                
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        import colorsys
        return list(colorsys.hsv_to_rgb(h, s, v))
        
    # Event handlers
    def on_mouse_down(self, event):
        """Handle mouse press"""
        self.last_mouse_pos = (event.x, event.y)
        self.mouse_button = event.num
        self.focus_set()
        
    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if self.last_mouse_pos is None:
            return
            
        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]
        
        if self.mouse_button == 1:  # Left button - rotate
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
        elif self.mouse_button == 2:  # Middle button - pan
            self.translation_x += dx * 0.01
            self.translation_y -= dy * 0.01
        elif self.mouse_button == 3:  # Right button - zoom
            self.camera_distance += dy * 0.1
            self.camera_distance = max(0.5, min(self.camera_distance, 20.0))
            
        self.last_mouse_pos = (event.x, event.y)
        self.after_idle(self.redraw)
        
    def on_mouse_up(self, event):
        """Handle mouse release"""
        self.last_mouse_pos = None
        self.mouse_button = None
        
    def on_mouse_wheel(self, event):
        """Handle mouse wheel"""
        factor = 1.1 if event.delta > 0 else 1/1.1
        self.scale *= factor
        self.scale = max(0.1, min(self.scale, 10.0))
        self.after_idle(self.redraw)
        
    def on_key_press(self, event):
        """Handle key press"""
        key = event.keysym.lower()
        
        if key == 'r':
            self.reset_view()
        elif key == 'a':
            self.show_axes = not self.show_axes
        elif key == 'g':
            self.show_grid = not self.show_grid
        elif key == 'b':
            self.show_bounding_box = not self.show_bounding_box
        elif key == 'plus' or key == 'equal':
            self.point_size = min(self.point_size + 1, 20)
            glPointSize(self.point_size)
        elif key == 'minus':
            self.point_size = max(self.point_size - 1, 1)
            glPointSize(self.point_size)
            
        self.after_idle(self.redraw)
        
    def on_resize(self, event):
        """Handle window resize"""
        self.setup_projection()
        self.after_idle(self.redraw)
        
    def reset_view(self, event=None):
        """Reset view to default"""
        self.scale = 1.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.translation_z = 0.0
        self.camera_distance = 5.0
        self.after_idle(self.redraw)


class TIFFViewerApp:
    """Main application window"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Complete 3D TIFF Viewer")
        self.root.geometry("1200x800")
        
        self.setup_ui()
        self.viewer = None
        
    def setup_ui(self):
        """Setup user interface"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open TIFF...", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self.reset_view)
        view_menu.add_command(label="Fit to Screen", command=self.fit_to_screen)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # File controls
        ttk.Button(control_frame, text="Open TIFF File", 
                  command=self.open_file).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # View controls
        ttk.Label(control_frame, text="View Controls:").pack(anchor=tk.W)
        ttk.Button(control_frame, text="Reset View", 
                  command=self.reset_view).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Fit to Screen", 
                  command=self.fit_to_screen).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Display settings
        ttk.Label(control_frame, text="Display Settings:").pack(anchor=tk.W)
        
        self.show_axes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Axes", 
                       variable=self.show_axes_var,
                       command=self.update_display).pack(anchor=tk.W)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Grid", 
                       variable=self.show_grid_var,
                       command=self.update_display).pack(anchor=tk.W)
        
        self.show_bbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Bounding Box", 
                       variable=self.show_bbox_var,
                       command=self.update_display).pack(anchor=tk.W)
        
        # Point size
        ttk.Label(control_frame, text="Point Size:").pack(anchor=tk.W, pady=(10, 0))
        self.point_size_var = tk.DoubleVar(value=3.0)
        point_size_scale = ttk.Scale(control_frame, from_=1.0, to=10.0, 
                                   variable=self.point_size_var,
                                   command=self.update_point_size)
        point_size_scale.pack(fill=tk.X)
        
        # Performance settings
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Performance:").pack(anchor=tk.W)
        
        ttk.Label(control_frame, text="Max Points (K):").pack(anchor=tk.W)
        self.max_points_var = tk.IntVar(value=500)
        max_points_scale = ttk.Scale(control_frame, from_=50, to=2000, 
                                   variable=self.max_points_var,
                                   command=self.update_max_points)
        max_points_scale.pack(fill=tk.X)
        
        # Instructions
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        instructions = """Mouse Controls:
• Left: Rotate
• Middle: Pan  
• Right: Zoom
• Wheel: Scale
• Double-click: Reset

Keyboard:
• R: Reset view
• A: Toggle axes
• G: Toggle grid
• B: Toggle bbox
• +/-: Point size"""
        
        ttk.Label(control_frame, text=instructions, 
                 font=("Courier", 8)).pack(anchor=tk.W)
        
        # 3D Viewer
        viewer_frame = ttk.LabelFrame(main_frame, text="3D View", padding=5)
        viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        try:
            self.viewer = Complete3DTIFFViewer(viewer_frame, width=800, height=600)
            self.viewer.pack(fill=tk.BOTH, expand=True)
            print("✓ 3D Viewer created successfully")
        except Exception as e:
            print(f"✗ Error creating 3D viewer: {e}")
            error_label = ttk.Label(viewer_frame, 
                                  text=f"OpenGL Error:\n{str(e)}\n\nPlease check:\n• Graphics drivers\n• PyOpenGL installation",
                                  foreground="red")
            error_label.pack(expand=True)
            
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Open a TIFF file to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def open_file(self):
        """Open TIFF file dialog"""
        filetypes = [
            ("TIFF files", "*.tiff *.tif"),
            ("All image files", "*.tiff *.tif *.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Open TIFF File",
            filetypes=filetypes
        )
        
        if filepath and self.viewer:
            self.status_var.set("Loading TIFF file...")
            self.root.update()
            
            # Load in background thread
            def load_file():
                success = self.viewer.load_tiff_file(filepath)
                if success:
                    self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
                    # Update max points if needed
                    if hasattr(self.viewer, 'points') and len(self.viewer.points) > 0:
                        self.status_var.set(f"Loaded: {os.path.basename(filepath)} ({len(self.viewer.points)} points)")
                else:
                    self.status_var.set("Failed to load file")
                    
            threading.Thread(target=load_file, daemon=True).start()
            
    def reset_view(self):
        """Reset 3D view"""
        if self.viewer:
            self.viewer.reset_view()
            
    def fit_to_screen(self):
        """Fit data to screen"""
        if self.viewer and len(self.viewer.points) > 0:
            # Calculate appropriate scale
            max_coord = np.abs(self.viewer.points).max()
            if max_coord > 0:
                self.viewer.scale = 1.0 / max_coord
                self.viewer.after_idle(self.viewer.redraw)
                
    def update_display(self):
        """Update display settings"""
        if self.viewer:
            self.viewer.show_axes = self.show_axes_var.get()
            self.viewer.show_grid = self.show_grid_var.get()
            self.viewer.show_bounding_box = self.show_bbox_var.get()
            self.viewer.after_idle(self.viewer.redraw)
            
    def update_point_size(self, value):
        """Update point size"""
        if self.viewer:
            self.viewer.point_size = float(value)
            glPointSize(self.viewer.point_size)
            self.viewer.after_idle(self.viewer.redraw)
            
    def update_max_points(self, value):
        """Update max points setting"""
        if self.viewer:
            self.viewer.max_points = int(float(value)) * 1000
            
    def run(self):
        """Run the application"""
        print("Starting Complete 3D TIFF Viewer...")
        print("✓ Application ready")
        self.root.mainloop()


def main():
    """Main entry point"""
    print("=== Complete 3D TIFF Viewer ===")
    
    # Check dependencies
    try:
        import tkinter
        import numpy
        import OpenGL
        import tifffile
        import PIL
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install numpy PyOpenGL pyopengltk tifffile Pillow")
        return 1
        
    try:
        app = TIFFViewerApp()
        app.run()
        return 0
    except Exception as e:
        print(f"✗ Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 