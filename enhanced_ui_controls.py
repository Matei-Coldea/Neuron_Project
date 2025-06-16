import tkinter as tk
from tkinter import ttk, colorchooser
import numpy as np

class Enhanced3DControls:
    """
    Enhanced UI controls for the 3D viewer with comprehensive settings panel.
    """
    
    def __init__(self, parent, canvas):
        self.parent = parent
        self.canvas = canvas  # Reference to Advanced3DCanvas
        
        # Create control panel
        self.create_control_panel()
        
    def create_control_panel(self):
        """Create comprehensive control panel"""
        # Main control frame
        self.control_frame = tk.Frame(self.parent, relief=tk.RAISED, borderwidth=2, bg="#f0f0f0")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Make it scrollable
        self.control_canvas = tk.Canvas(self.control_frame, bg="#f0f0f0", width=300)
        scrollbar = tk.Scrollbar(self.control_frame, orient="vertical", command=self.control_canvas.yview)
        self.scrollable_control_frame = tk.Frame(self.control_canvas, bg="#f0f0f0")
        
        self.scrollable_control_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )
        
        self.control_canvas.create_window((0, 0), window=self.scrollable_control_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        self.control_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Add control sections
        self._create_view_controls()
        self._create_render_controls()
        self._create_lighting_controls()
        self._create_display_controls()
        self._create_interaction_controls()
        self._create_performance_controls()
        self._create_export_controls()
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def _create_view_controls(self):
        """Create view control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="View Controls", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Reset view button
        tk.Button(frame, text="Reset View", command=self.canvas.reset_view,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Fit to screen button
        tk.Button(frame, text="Fit to Screen", command=self.canvas.fit_to_screen,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Projection mode
        projection_frame = tk.Frame(frame, bg="#f0f0f0")
        projection_frame.pack(fill="x", pady=2)
        tk.Label(projection_frame, text="Projection:", bg="#f0f0f0").pack(side="left")
        tk.Button(projection_frame, text="Toggle Proj", command=self.canvas.toggle_projection,
                 bg="#e2e2e2", relief=tk.FLAT).pack(side="right")
        
        # Camera distance
        cam_frame = tk.Frame(frame, bg="#f0f0f0")
        cam_frame.pack(fill="x", pady=2)
        tk.Label(cam_frame, text="Camera Distance:", bg="#f0f0f0").pack(side="left")
        self.camera_scale = tk.Scale(cam_frame, from_=0.5, to=20.0, resolution=0.1,
                                   orient="horizontal", bg="#f0f0f0",
                                   command=self._on_camera_distance_change)
        self.camera_scale.set(5.0)
        self.camera_scale.pack(side="right")
        
        # FOV control
        fov_frame = tk.Frame(frame, bg="#f0f0f0")
        fov_frame.pack(fill="x", pady=2)
        tk.Label(fov_frame, text="Field of View:", bg="#f0f0f0").pack(side="left")
        self.fov_scale = tk.Scale(fov_frame, from_=10, to=120, orient="horizontal", 
                                bg="#f0f0f0", command=self._on_fov_change)
        self.fov_scale.set(45)
        self.fov_scale.pack(side="right")
        
    def _create_render_controls(self):
        """Create rendering control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="Rendering", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Render mode buttons
        mode_frame = tk.Frame(frame, bg="#f0f0f0")
        mode_frame.pack(fill="x", pady=2)
        
        tk.Button(mode_frame, text="Points", command=lambda: self.canvas.set_render_mode('points'),
                 bg="#e2e2e2", relief=tk.FLAT, width=8).grid(row=0, column=0, padx=1)
        tk.Button(mode_frame, text="Wireframe", command=lambda: self.canvas.set_render_mode('wireframe'),
                 bg="#e2e2e2", relief=tk.FLAT, width=8).grid(row=0, column=1, padx=1)
        tk.Button(mode_frame, text="Surface", command=lambda: self.canvas.set_render_mode('surface'),
                 bg="#e2e2e2", relief=tk.FLAT, width=8).grid(row=1, column=0, padx=1)
        tk.Button(mode_frame, text="Volumetric", command=lambda: self.canvas.set_render_mode('volumetric'),
                 bg="#e2e2e2", relief=tk.FLAT, width=8).grid(row=1, column=1, padx=1)
        
        # Point size
        point_frame = tk.Frame(frame, bg="#f0f0f0")
        point_frame.pack(fill="x", pady=2)
        tk.Label(point_frame, text="Point Size:", bg="#f0f0f0").pack(side="left")
        self.point_scale = tk.Scale(point_frame, from_=1, to=20, orient="horizontal",
                                  bg="#f0f0f0", command=self._on_point_size_change)
        self.point_scale.set(4)
        self.point_scale.pack(side="right")
        
        # Background color
        bg_frame = tk.Frame(frame, bg="#f0f0f0")
        bg_frame.pack(fill="x", pady=2)
        tk.Label(bg_frame, text="Background:", bg="#f0f0f0").pack(side="left")
        tk.Button(bg_frame, text="Choose Color", command=self._choose_background_color,
                 bg="#e2e2e2", relief=tk.FLAT).pack(side="right")
        
    def _create_lighting_controls(self):
        """Create lighting control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="Lighting", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Toggle lighting
        tk.Button(frame, text="Toggle Lighting", command=self.canvas.toggle_lighting,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Ambient light
        ambient_frame = tk.Frame(frame, bg="#f0f0f0")
        ambient_frame.pack(fill="x", pady=2)
        tk.Label(ambient_frame, text="Ambient:", bg="#f0f0f0").pack(side="left")
        self.ambient_scale = tk.Scale(ambient_frame, from_=0.0, to=1.0, resolution=0.1,
                                    orient="horizontal", bg="#f0f0f0",
                                    command=self._on_ambient_change)
        self.ambient_scale.set(0.3)
        self.ambient_scale.pack(side="right")
        
        # Diffuse light
        diffuse_frame = tk.Frame(frame, bg="#f0f0f0")
        diffuse_frame.pack(fill="x", pady=2)
        tk.Label(diffuse_frame, text="Diffuse:", bg="#f0f0f0").pack(side="left")
        self.diffuse_scale = tk.Scale(diffuse_frame, from_=0.0, to=1.0, resolution=0.1,
                                    orient="horizontal", bg="#f0f0f0",
                                    command=self._on_diffuse_change)
        self.diffuse_scale.set(0.7)
        self.diffuse_scale.pack(side="right")
        
        # Specular light
        specular_frame = tk.Frame(frame, bg="#f0f0f0")
        specular_frame.pack(fill="x", pady=2)
        tk.Label(specular_frame, text="Specular:", bg="#f0f0f0").pack(side="left")
        self.specular_scale = tk.Scale(specular_frame, from_=0.0, to=1.0, resolution=0.1,
                                     orient="horizontal", bg="#f0f0f0",
                                     command=self._on_specular_change)
        self.specular_scale.set(1.0)
        self.specular_scale.pack(side="right")
        
    def _create_display_controls(self):
        """Create display control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="Display", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Toggle buttons
        tk.Button(frame, text="Toggle Axes", command=self.canvas.toggle_axes,
                 bg="#e2e2e2", relief=tk.FLAT).pack(pady=1, fill="x")
        tk.Button(frame, text="Toggle Grid", command=self.canvas.toggle_grid,
                 bg="#e2e2e2", relief=tk.FLAT).pack(pady=1, fill="x")
        
        # Show bounding box
        self.bbox_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Show Bounding Box", variable=self.bbox_var,
                      command=self._on_bbox_toggle, bg="#f0f0f0").pack(pady=1, fill="x")
        
    def _create_interaction_controls(self):
        """Create interaction control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="Interaction", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Selection mode
        tk.Button(frame, text="Toggle Selection", command=self.canvas.toggle_selection_mode,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Clear selection
        tk.Button(frame, text="Clear Selection", command=self.canvas.clear_selection,
                 bg="#e2e2e2", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Export selection
        tk.Button(frame, text="Export Selection", command=self.canvas.export_selection,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Animation
        tk.Button(frame, text="Toggle Animation", command=self.canvas.toggle_animation,
                 bg="#e2e2e2", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Mouse sensitivity
        sens_frame = tk.Frame(frame, bg="#f0f0f0")
        sens_frame.pack(fill="x", pady=2)
        tk.Label(sens_frame, text="Mouse Sensitivity:", bg="#f0f0f0").pack(side="left")
        self.sensitivity_scale = tk.Scale(sens_frame, from_=0.1, to=3.0, resolution=0.1,
                                        orient="horizontal", bg="#f0f0f0",
                                        command=self._on_sensitivity_change)
        self.sensitivity_scale.set(1.0)
        self.sensitivity_scale.pack(side="right")
        
    def _create_performance_controls(self):
        """Create performance control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="Performance", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # FPS display
        self.fps_label = tk.Label(frame, text="FPS: 0.0", bg="#f0f0f0")
        self.fps_label.pack(pady=2)
        
        # Update FPS periodically
        self._update_fps()
        
        # VBO toggle
        self.vbo_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Use VBO", variable=self.vbo_var,
                      command=self._on_vbo_toggle, bg="#f0f0f0").pack(pady=1, fill="x")
        
        # Frustum culling
        self.culling_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Frustum Culling", variable=self.culling_var,
                      command=self._on_culling_toggle, bg="#f0f0f0").pack(pady=1, fill="x")
        
    def _create_export_controls(self):
        """Create export control section"""
        frame = tk.LabelFrame(self.scrollable_control_frame, text="Export", 
                            font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Screenshot
        tk.Button(frame, text="Take Screenshot", command=self.canvas.screenshot,
                 bg="#4a90e2", fg="white", relief=tk.FLAT).pack(pady=2, fill="x")
        
        # Help button
        tk.Button(frame, text="Show Keyboard Shortcuts", command=self._show_help,
                 bg="#e2e2e2", relief=tk.FLAT).pack(pady=2, fill="x")
        
    # Event handlers
    def _on_camera_distance_change(self, value):
        """Handle camera distance change"""
        self.canvas._camera_distance = float(value)
        self.canvas.after_idle(self.canvas.redraw)
        
    def _on_fov_change(self, value):
        """Handle FOV change"""
        self.canvas._fov = float(value)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas._setup_projection(width, height)
        self.canvas.after_idle(self.canvas.redraw)
        
    def _on_point_size_change(self, value):
        """Handle point size change"""
        self.canvas.set_point_size(float(value))
        
    def _choose_background_color(self):
        """Choose background color"""
        color = colorchooser.askcolor(title="Choose Background Color")
        if color[0]:  # User selected a color
            # Convert RGB to normalized values
            r, g, b = [c/255.0 for c in color[0]]
            self.canvas.set_background_color((r, g, b, 1.0))
            
    def _on_ambient_change(self, value):
        """Handle ambient light change"""
        val = float(value)
        self.canvas._ambient_light = [val, val, val, 1.0]
        self.canvas._setup_lighting()
        self.canvas.after_idle(self.canvas.redraw)
        
    def _on_diffuse_change(self, value):
        """Handle diffuse light change"""
        val = float(value)
        self.canvas._diffuse_light = [val, val, val, 1.0]
        self.canvas._setup_lighting()
        self.canvas.after_idle(self.canvas.redraw)
        
    def _on_specular_change(self, value):
        """Handle specular light change"""
        val = float(value)
        self.canvas._specular_light = [val, val, val, 1.0]
        self.canvas._setup_lighting()
        self.canvas.after_idle(self.canvas.redraw)
        
    def _on_bbox_toggle(self):
        """Handle bounding box toggle"""
        self.canvas._show_bounding_box = self.bbox_var.get()
        self.canvas.after_idle(self.canvas.redraw)
        
    def _on_sensitivity_change(self, value):
        """Handle mouse sensitivity change"""
        self.canvas._mouse_sensitivity = float(value)
        
    def _on_vbo_toggle(self):
        """Handle VBO toggle"""
        self.canvas._use_vbo = self.vbo_var.get()
        
    def _on_culling_toggle(self):
        """Handle frustum culling toggle"""
        self.canvas._frustum_culling = self.culling_var.get()
        
    def _update_fps(self):
        """Update FPS display"""
        fps = self.canvas.get_fps()
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        # Update every second
        self.parent.after(1000, self._update_fps)
        
    def _show_help(self):
        """Show keyboard shortcuts help"""
        help_text = """
Keyboard Shortcuts:

R - Reset view
P - Points mode
W - Wireframe mode
S - Surface mode
V - Volumetric mode
A - Toggle axes
G - Toggle grid
L - Toggle lighting
O - Toggle projection
F - Fit to screen
C - Take screenshot
Space - Toggle animation
Escape - Clear selection

Mouse Controls:
Left click + drag - Rotate
Middle click + drag - Pan
Right click + drag - Zoom
Mouse wheel - Zoom
Double click - Reset view
        """
        
        help_window = tk.Toplevel(self.parent)
        help_window.title("Keyboard Shortcuts")
        help_window.geometry("400x400")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Make it read-only 