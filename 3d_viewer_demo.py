#!/usr/bin/env python3
"""
Enhanced 3D TIFF Viewer Demo

This demo showcases all the advanced 3D viewer capabilities including:
- Multiple rendering modes
- Advanced lighting and materials
- Interactive controls
- Performance optimizations
- Export and screenshot functionality
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from opengl_canvas import Advanced3DCanvas

class Enhanced3DViewerDemo:
    """
    Demo application for the enhanced 3D viewer
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced 3D TIFF Viewer - Demo")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        # Create main layout
        self.create_layout()
        
        # Generate sample data
        self.generate_sample_data()
        
        # Start the application
        self.root.mainloop()
    
    def create_layout(self):
        """Create the main application layout"""
        # Create main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create toolbar
        self.create_toolbar(main_container)
        
        # Create content area
        content_frame = tk.Frame(main_container, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create 3D canvas
        self.canvas = Advanced3DCanvas(content_frame, width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create control panel
        self.create_control_panel(content_frame)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready - Use mouse to interact, keyboard shortcuts available", 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#f0f0f0")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_toolbar(self, parent):
        """Create toolbar with main actions"""
        toolbar = tk.Frame(parent, bg="#e0e0e0", height=40)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        toolbar.pack_propagate(False)
        
        # File operations
        tk.Button(toolbar, text="Load TIFF", command=self.load_tiff_file,
                 bg="#4a90e2", fg="white", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Button(toolbar, text="Generate Sample", command=self.generate_sample_data,
                 bg="#4a90e2", fg="white", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Separator
        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # View operations
        tk.Button(toolbar, text="Reset View", command=self.canvas.reset_view,
                 bg="#e2e2e2", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Button(toolbar, text="Fit Screen", command=self.canvas.fit_to_screen,
                 bg="#e2e2e2", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Separator
        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Export operations
        tk.Button(toolbar, text="Screenshot", command=self.canvas.screenshot,
                 bg="#e2e2e2", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Help
        tk.Button(toolbar, text="Help", command=self.show_help,
                 bg="#e2e2e2", relief=tk.FLAT, padx=10).pack(side=tk.RIGHT, padx=2, pady=5)
        
        # FPS display
        self.fps_label = tk.Label(toolbar, text="FPS: 0.0", bg="#e0e0e0")
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=5)
        self.update_fps()
    
    def create_control_panel(self, parent):
        """Create comprehensive control panel"""
        # Main control frame
        control_frame = tk.Frame(parent, relief=tk.RAISED, borderwidth=2, bg="#f0f0f0", width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)
        
        # Make it scrollable
        control_canvas = tk.Canvas(control_frame, bg="#f0f0f0")
        scrollbar = tk.Scrollbar(control_frame, orient="vertical", command=control_canvas.yview)
        scrollable_frame = tk.Frame(control_canvas, bg="#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        control_canvas.bind("<MouseWheel>", lambda e: control_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        
        # Add control sections
        self.create_render_controls(scrollable_frame)
        self.create_view_controls(scrollable_frame)
        self.create_lighting_controls(scrollable_frame)
        self.create_display_controls(scrollable_frame)
        self.create_interaction_controls(scrollable_frame)
        self.create_data_controls(scrollable_frame)
    
    def create_render_controls(self, parent):
        """Create rendering controls"""
        frame = tk.LabelFrame(parent, text="Rendering Mode", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0", padx=5, pady=5)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Render mode buttons in a grid
        modes_frame = tk.Frame(frame, bg="#f0f0f0")
        modes_frame.pack(fill="x")
        
        self.current_mode = tk.StringVar(value="points")
        
        modes = [
            ("Points", "points"),
            ("Wireframe", "wireframe"),
            ("Surface", "surface"),
            ("Volumetric", "volumetric")
        ]
        
        for i, (text, mode) in enumerate(modes):
            row, col = i // 2, i % 2
            tk.Radiobutton(modes_frame, text=text, variable=self.current_mode, value=mode,
                          command=lambda m=mode: self.canvas.set_render_mode(m),
                          bg="#f0f0f0").grid(row=row, column=col, sticky="w", padx=5)
        
        # Point size control
        point_frame = tk.Frame(frame, bg="#f0f0f0")
        point_frame.pack(fill="x", pady=5)
        tk.Label(point_frame, text="Point Size:", bg="#f0f0f0").pack(side="left")
        self.point_scale = tk.Scale(point_frame, from_=1, to=20, orient="horizontal",
                                  bg="#f0f0f0", command=lambda v: self.canvas.set_point_size(float(v)))
        self.point_scale.set(4)
        self.point_scale.pack(side="right", fill="x", expand=True)
    
    def create_view_controls(self, parent):
        """Create view controls"""
        frame = tk.LabelFrame(parent, text="View Controls", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0", padx=5, pady=5)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Projection toggle
        proj_frame = tk.Frame(frame, bg="#f0f0f0")
        proj_frame.pack(fill="x", pady=2)
        tk.Label(proj_frame, text="Projection:", bg="#f0f0f0").pack(side="left")
        self.proj_var = tk.StringVar(value="Perspective")
        tk.Button(proj_frame, textvariable=self.proj_var, 
                 command=self.toggle_projection, bg="#e2e2e2", relief=tk.FLAT).pack(side="right")
        
        # Camera distance
        cam_frame = tk.Frame(frame, bg="#f0f0f0")
        cam_frame.pack(fill="x", pady=2)
        tk.Label(cam_frame, text="Distance:", bg="#f0f0f0").pack(side="left")
        self.camera_scale = tk.Scale(cam_frame, from_=0.5, to=20.0, resolution=0.1,
                                   orient="horizontal", bg="#f0f0f0",
                                   command=self.on_camera_distance_change)
        self.camera_scale.set(5.0)
        self.camera_scale.pack(side="right", fill="x", expand=True)
        
        # Field of view
        fov_frame = tk.Frame(frame, bg="#f0f0f0")
        fov_frame.pack(fill="x", pady=2)
        tk.Label(fov_frame, text="FOV:", bg="#f0f0f0").pack(side="left")
        self.fov_scale = tk.Scale(fov_frame, from_=10, to=120, orient="horizontal",
                                bg="#f0f0f0", command=self.on_fov_change)
        self.fov_scale.set(45)
        self.fov_scale.pack(side="right", fill="x", expand=True)
    
    def create_lighting_controls(self, parent):
        """Create lighting controls"""
        frame = tk.LabelFrame(parent, text="Lighting", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0", padx=5, pady=5)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Toggle lighting
        tk.Button(frame, text="Toggle Lighting", command=self.canvas.toggle_lighting,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(fill="x", pady=2)
        
        # Light components
        for name, default in [("Ambient", 0.3), ("Diffuse", 0.7), ("Specular", 1.0)]:
            light_frame = tk.Frame(frame, bg="#f0f0f0")
            light_frame.pack(fill="x", pady=2)
            tk.Label(light_frame, text=f"{name}:", bg="#f0f0f0").pack(side="left")
            scale = tk.Scale(light_frame, from_=0.0, to=1.0, resolution=0.1,
                           orient="horizontal", bg="#f0f0f0",
                           command=lambda v, n=name.lower(): self.on_light_change(n, v))
            scale.set(default)
            scale.pack(side="right", fill="x", expand=True)
    
    def create_display_controls(self, parent):
        """Create display controls"""
        frame = tk.LabelFrame(parent, text="Display Options", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0", padx=5, pady=5)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Display toggles
        self.axes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Show Axes", variable=self.axes_var,
                      command=self.canvas.toggle_axes, bg="#f0f0f0").pack(anchor="w")
        
        self.grid_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Show Grid", variable=self.grid_var,
                      command=self.canvas.toggle_grid, bg="#f0f0f0").pack(anchor="w")
        
        self.bbox_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Show Bounding Box", variable=self.bbox_var,
                      command=self.toggle_bbox, bg="#f0f0f0").pack(anchor="w")
        
        # Background color
        bg_frame = tk.Frame(frame, bg="#f0f0f0")
        bg_frame.pack(fill="x", pady=2)
        tk.Label(bg_frame, text="Background:", bg="#f0f0f0").pack(side="left")
        tk.Button(bg_frame, text="Color", command=self.choose_background_color,
                 bg="#e2e2e2", relief=tk.FLAT).pack(side="right")
    
    def create_interaction_controls(self, parent):
        """Create interaction controls"""
        frame = tk.LabelFrame(parent, text="Interaction", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0", padx=5, pady=5)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Selection controls
        selection_frame = tk.Frame(frame, bg="#f0f0f0")
        selection_frame.pack(fill="x", pady=2)
        tk.Button(selection_frame, text="Selection Mode", command=self.canvas.toggle_selection_mode,
                 bg="#4a90e2", fg="white", relief=tk.FLAT, width=12).pack(side="left", padx=1)
        tk.Button(selection_frame, text="Clear", command=self.canvas.clear_selection,
                 bg="#e2e2e2", relief=tk.FLAT, width=8).pack(side="right", padx=1)
        
        # Animation
        tk.Button(frame, text="Toggle Animation", command=self.canvas.toggle_animation,
                 bg="#e2e2e2", relief=tk.FLAT).pack(fill="x", pady=2)
        
        # Mouse sensitivity
        sens_frame = tk.Frame(frame, bg="#f0f0f0")
        sens_frame.pack(fill="x", pady=2)
        tk.Label(sens_frame, text="Sensitivity:", bg="#f0f0f0").pack(side="left")
        self.sensitivity_scale = tk.Scale(sens_frame, from_=0.1, to=3.0, resolution=0.1,
                                        orient="horizontal", bg="#f0f0f0",
                                        command=self.on_sensitivity_change)
        self.sensitivity_scale.set(1.0)
        self.sensitivity_scale.pack(side="right", fill="x", expand=True)
    
    def create_data_controls(self, parent):
        """Create data manipulation controls"""
        frame = tk.LabelFrame(parent, text="Data", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0", padx=5, pady=5)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Data generation
        tk.Button(frame, text="Generate Sphere", command=lambda: self.generate_sample_data("sphere"),
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(fill="x", pady=1)
        tk.Button(frame, text="Generate Cube", command=lambda: self.generate_sample_data("cube"),
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(fill="x", pady=1)
        tk.Button(frame, text="Generate Random", command=lambda: self.generate_sample_data("random"),
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(fill="x", pady=1)
        
        # Data info
        self.data_info = tk.Label(frame, text="No data loaded", bg="#f0f0f0", font=("Helvetica", 9))
        self.data_info.pack(fill="x", pady=5)
    
    # Event handlers
    def toggle_projection(self):
        """Toggle projection mode"""
        self.canvas.toggle_projection()
        current_mode = "Orthographic" if self.canvas._projection_mode == "orthographic" else "Perspective"
        self.proj_var.set(current_mode)
    
    def on_camera_distance_change(self, value):
        """Handle camera distance change"""
        self.canvas._camera_distance = float(value)
        self.canvas.after_idle(self.canvas.redraw)
    
    def on_fov_change(self, value):
        """Handle FOV change"""
        self.canvas._fov = float(value)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas._setup_projection(width, height)
        self.canvas.after_idle(self.canvas.redraw)
    
    def on_light_change(self, component, value):
        """Handle lighting changes"""
        val = float(value)
        if component == "ambient":
            self.canvas._ambient_light = [val, val, val, 1.0]
        elif component == "diffuse":
            self.canvas._diffuse_light = [val, val, val, 1.0]
        elif component == "specular":
            self.canvas._specular_light = [val, val, val, 1.0]
        
        self.canvas._setup_lighting()
        self.canvas.after_idle(self.canvas.redraw)
    
    def toggle_bbox(self):
        """Toggle bounding box"""
        self.canvas._show_bounding_box = self.bbox_var.get()
        self.canvas.after_idle(self.canvas.redraw)
    
    def choose_background_color(self):
        """Choose background color"""
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Choose Background Color")
        if color[0]:
            r, g, b = [c/255.0 for c in color[0]]
            self.canvas.set_background_color((r, g, b, 1.0))
    
    def on_sensitivity_change(self, value):
        """Handle mouse sensitivity change"""
        self.canvas._mouse_sensitivity = float(value)
    
    def update_fps(self):
        """Update FPS display"""
        fps = self.canvas.get_fps()
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.root.after(1000, self.update_fps)
    
    # Data generation and loading
    def generate_sample_data(self, shape="sphere"):
        """Generate sample data for demonstration"""
        n_points = 5000
        
        if shape == "sphere":
            # Generate points on a sphere
            phi = np.random.uniform(0, 2*np.pi, n_points)
            costheta = np.random.uniform(-1, 1, n_points)
            theta = np.arccos(costheta)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            # Add some noise for interest
            noise = np.random.normal(0, 0.1, (n_points, 3))
            points = np.column_stack([x, y, z]) + noise
            
        elif shape == "cube":
            # Generate points in a cube with higher density on edges
            points = []
            # Face points
            for _ in range(n_points // 6):
                # Each face
                for face in range(6):
                    if face < 2:  # x faces
                        x = (-1) ** face
                        y = np.random.uniform(-1, 1)
                        z = np.random.uniform(-1, 1)
                    elif face < 4:  # y faces
                        x = np.random.uniform(-1, 1)
                        y = (-1) ** (face - 2)
                        z = np.random.uniform(-1, 1)
                    else:  # z faces
                        x = np.random.uniform(-1, 1)
                        y = np.random.uniform(-1, 1)
                        z = (-1) ** (face - 4)
                    points.append([x, y, z])
            
            points = np.array(points[:n_points])
            
        else:  # random
            # Generate random points in a cube
            points = np.random.uniform(-1, 1, (n_points, 3))
        
        # Generate colors based on position
        colors = np.zeros((len(points), 3))
        colors[:, 0] = (points[:, 0] + 1) / 2  # Red based on X
        colors[:, 1] = (points[:, 1] + 1) / 2  # Green based on Y
        colors[:, 2] = (points[:, 2] + 1) / 2  # Blue based on Z
        
        # Set the data
        self.canvas.set_points(points.astype(np.float32), colors.astype(np.float32))
        
        # Update info
        self.data_info.config(text=f"Generated {len(points)} points ({shape})")
        self.status_bar.config(text=f"Generated {shape} with {len(points)} points")
    
    def load_tiff_file(self):
        """Load a TIFF file"""
        filename = filedialog.askopenfilename(
            title="Select TIFF File",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Import the TIFF viewer to load the file
                from tiff_viewer_3d import TIFFViewer3D
                
                # This is a simplified version - in practice you'd integrate with the full TIFF loader
                messagebox.showinfo("TIFF Loading", 
                                  f"TIFF file selected: {os.path.basename(filename)}\n\n"
                                  "Note: Full TIFF integration requires the complete viewer.\n"
                                  "For now, generating sample data.")
                self.generate_sample_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load TIFF file:\n{str(e)}")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
Enhanced 3D TIFF Viewer - Help

MOUSE CONTROLS:
• Left click + drag: Rotate view
• Middle click + drag: Pan view  
• Right click + drag: Zoom
• Mouse wheel: Zoom in/out
• Double-click: Reset view

KEYBOARD SHORTCUTS:
• R: Reset view
• P: Points rendering mode
• W: Wireframe rendering mode
• S: Surface rendering mode
• V: Volumetric rendering mode
• A: Toggle coordinate axes
• G: Toggle grid
• L: Toggle lighting
• O: Toggle projection mode
• F: Fit to screen
• C: Take screenshot
• Space: Toggle animation
• Escape: Clear selection

RENDERING MODES:
• Points: Display as individual points
• Wireframe: Display surface wireframe
• Surface: Display filled surfaces with lighting
• Volumetric: Display with transparency effects

FEATURES:
• Interactive lighting controls
• Multiple projection modes
• Selection and highlighting
• Performance monitoring
• Export capabilities
• Customizable appearance
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - 3D Viewer")
        help_window.geometry("500x600")
        help_window.configure(bg="#f0f0f0")
        
        # Make it non-resizable
        help_window.resizable(False, False)
        
        # Center the window
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(help_window, bg="#f0f0f0")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, bg="white", font=("Courier", 10))
        scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        tk.Button(help_window, text="Close", command=help_window.destroy,
                 bg="#4a90e2", fg="white", relief=tk.FLAT, padx=20).pack(pady=10)

def main():
    """Main entry point"""
    try:
        # Check for required dependencies
        import OpenGL
        from PIL import Image
        
        # Create and run the demo
        app = Enhanced3DViewerDemo()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install PyOpenGL PyOpenGL_accelerate pyopengltk Pillow numpy")
        return 1
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 