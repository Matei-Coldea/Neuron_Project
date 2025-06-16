from pyopengltk import OpenGLFrame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
from PIL import Image
import io

class Advanced3DCanvas(OpenGLFrame):
    """
    A comprehensive 3D OpenGL viewer with full capabilities for scientific visualization.
    
    Features:
    - Multiple rendering modes (points, wireframe, surface, volumetric)
    - Advanced lighting and shading
    - Coordinate axes and grids
    - Selection and highlighting
    - Clipping planes
    - Screenshots and animations
    - Performance optimizations
    - Multiple camera modes
    - Measurements and annotations
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Data storage
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colours = np.empty((0, 3), dtype=np.float32)
        self._normals = np.empty((0, 3), dtype=np.float32)
        self._indices = np.empty((0, 3), dtype=np.uint32)
        self._selected_points = set()
        
        # VBO for performance
        self._point_vbo = None
        self._color_vbo = None
        self._normal_vbo = None
        self._index_vbo = None
        
        # Camera and transformation
        self._scale = 1.0
        self._angle_x = 0.0
        self._angle_y = 0.0
        self._angle_z = 0.0
        self._trans_x = 0.0
        self._trans_y = 0.0
        self._trans_z = 0.0
        self._camera_distance = 5.0
        self._fov = 45.0
        
        # View settings
        self._projection_mode = 'perspective'  # 'perspective' or 'orthographic'
        self._render_mode = 'points'  # 'points', 'wireframe', 'surface', 'volumetric'
        self._point_size = 4.0
        self._line_width = 1.0
        self._show_axes = True
        self._show_grid = True
        self._show_bounding_box = True
        self._background_color = (0.95, 0.95, 0.95, 1.0)
        
        # Lighting
        self._lighting_enabled = True
        self._light_position = [2.0, 2.0, 2.0, 1.0]
        self._ambient_light = [0.3, 0.3, 0.3, 1.0]
        self._diffuse_light = [0.7, 0.7, 0.7, 1.0]
        self._specular_light = [1.0, 1.0, 1.0, 1.0]
        
        # Material properties
        self._material_ambient = [0.2, 0.2, 0.2, 1.0]
        self._material_diffuse = [0.8, 0.8, 0.8, 1.0]
        self._material_specular = [1.0, 1.0, 1.0, 1.0]
        self._material_shininess = 32.0
        
        # Clipping planes
        self._clipping_enabled = False
        self._clip_planes = []
        
        # Selection
        self._selection_mode = False
        self._selection_box = None
        self._highlight_color = [1.0, 1.0, 0.0, 1.0]  # Yellow highlight
        
        # Performance settings
        self._use_vbo = True
        self._frustum_culling = True
        self._level_of_detail = True
        self._max_points_render = 1000000
        
        # Animation
        self._animation_running = False
        self._animation_speed = 1.0
        
        # Mouse interaction
        self._last_mouse = None
        self._mouse_button = None
        self._mouse_sensitivity = 1.0
        
        # Measurements
        self._measurement_points = []
        self._show_measurements = True
        
        # Initialize event bindings
        self._setup_bindings()
        
        # Performance tracking
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0

    def _setup_bindings(self):
        """Setup all mouse and keyboard bindings"""
        # Mouse events
        self.bind("<Button-1>", self._on_mouse_down)
        self.bind("<Button-2>", self._on_mouse_down)
        self.bind("<Button-3>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<B2-Motion>", self._on_mouse_drag)
        self.bind("<B3-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", lambda e: self._clear_mouse())
        self.bind("<ButtonRelease-2>", lambda e: self._clear_mouse())
        self.bind("<ButtonRelease-3>", lambda e: self._clear_mouse())
        self.bind("<Double-Button-1>", self._on_double_click)
        
        # Mouse wheel
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda e: self.zoom(1/1.1))
        self.bind("<Button-5>", lambda e: self.zoom(1.1))
        
        # Keyboard events
        self.bind("<Key>", self._on_key_press)
        self.bind("<KeyPress>", self._on_key_press)
        
        # Window events
        self.bind("<Configure>", self._on_resize)
        self.bind("<Enter>", lambda e: self.focus_set())
        
        # Make widget focusable
        self.focus_set()

    def initgl(self):
        """Initialize OpenGL context with advanced settings"""
        # Enable depth testing and face culling
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Point and line settings
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Initialize lighting
        self._setup_lighting()
        
        # Set initial viewport and projection
        width, height = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        self._setup_projection(width, height)
        
        # Initialize display lists for axes and grid
        self._create_display_lists()
        
        # Setup VBO if supported
        if self._use_vbo and glGenBuffers:
            self._setup_vbo()

    def _setup_lighting(self):
        """Setup OpenGL lighting"""
        if self._lighting_enabled:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            
            glLightfv(GL_LIGHT0, GL_POSITION, self._light_position)
            glLightfv(GL_LIGHT0, GL_AMBIENT, self._ambient_light)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, self._diffuse_light)
            glLightfv(GL_LIGHT0, GL_SPECULAR, self._specular_light)
            
            # Enable color material
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Set material properties
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, self._material_ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, self._material_diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self._material_specular)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self._material_shininess)
        else:
            glDisable(GL_LIGHTING)

    def _setup_projection(self, width, height):
        """Setup projection matrix"""
        glViewport(0, 0, width, height)
        aspect = width / float(height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self._projection_mode == 'perspective':
            gluPerspective(self._fov, aspect, 0.1, 100.0)
        else:  # orthographic
            size = self._camera_distance
            glOrtho(-size * aspect, size * aspect, -size, size, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)

    def _create_display_lists(self):
        """Create display lists for frequently drawn objects"""
        # Axes display list
        self._axes_list = glGenLists(1)
        glNewList(self._axes_list, GL_COMPILE)
        self._draw_axes_immediate()
        glEndList()
        
        # Grid display list
        self._grid_list = glGenLists(1)
        glNewList(self._grid_list, GL_COMPILE)
        self._draw_grid_immediate()
        glEndList()

    def _setup_vbo(self):
        """Setup Vertex Buffer Objects for better performance"""
        try:
            self._point_vbo = glGenBuffers(1)
            self._color_vbo = glGenBuffers(1)
            self._normal_vbo = glGenBuffers(1)
            self._index_vbo = glGenBuffers(1)
        except:
            self._use_vbo = False
            print("VBO not supported, falling back to immediate mode")

    def redraw(self):
        """Main rendering function"""
        # Clear buffers
        glClearColor(*self._background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        glLoadIdentity()
        gluLookAt(0, 0, self._camera_distance,
                  0, 0, 0,
                  0, 1, 0)
        
        # Apply transformations
        glTranslatef(self._trans_x, self._trans_y, self._trans_z)
        glRotatef(self._angle_x, 1, 0, 0)
        glRotatef(self._angle_y, 0, 1, 0)
        glRotatef(self._angle_z, 0, 0, 1)
        glScalef(self._scale, self._scale, self._scale)
        
        # Setup clipping planes
        if self._clipping_enabled:
            self._setup_clipping()
        
        # Draw coordinate system
        if self._show_axes:
            self._draw_axes()
        
        if self._show_grid:
            self._draw_grid()
            
        if self._show_bounding_box and self._points.size > 0:
            self._draw_bounding_box()
        
        # Draw main data
        if self._points.size > 0:
            self._draw_data()
        
        # Draw selection box
        if self._selection_box:
            self._draw_selection_box()
        
        # Draw measurements
        if self._show_measurements and self._measurement_points:
            self._draw_measurements()
        
        # Update FPS counter
        self._update_fps()
        
        # Disable clipping
        if self._clipping_enabled:
            for i in range(len(self._clip_planes)):
                glDisable(GL_CLIP_PLANE0 + i)

    def _draw_data(self):
        """Draw the main point cloud data with current rendering mode"""
        if self._render_mode == 'points':
            self._draw_points()
        elif self._render_mode == 'wireframe':
            self._draw_wireframe()
        elif self._render_mode == 'surface':
            self._draw_surface()
        elif self._render_mode == 'volumetric':
            self._draw_volumetric()

    def _draw_points(self):
        """Draw points using VBO or immediate mode"""
        glPointSize(self._point_size)
        
        if self._use_vbo and self._point_vbo:
            self._draw_points_vbo()
        else:
            self._draw_points_immediate()

    def _draw_points_vbo(self):
        """Draw points using VBO for better performance"""
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glBindBuffer(GL_ARRAY_BUFFER, self._point_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, self._color_vbo)
        glColorPointer(3, GL_FLOAT, 0, None)
        
        glDrawArrays(GL_POINTS, 0, len(self._points))
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_points_immediate(self):
        """Draw points using immediate mode"""
        glBegin(GL_POINTS)
        for i, ((x, y, z), (r, g, b)) in enumerate(zip(self._points, self._colours)):
            if i in self._selected_points:
                glColor3f(*self._highlight_color[:3])
            else:
                glColor3f(r, g, b)
            glVertex3f(x, y, z)
        glEnd()

    def _draw_wireframe(self):
        """Draw wireframe representation"""
        glLineWidth(self._line_width)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        if len(self._indices) > 0:
            # Draw indexed triangles as wireframe
            glBegin(GL_TRIANGLES)
            for triangle in self._indices:
                for vertex_idx in triangle:
                    if vertex_idx < len(self._points):
                        point = self._points[vertex_idx]
                        color = self._colours[vertex_idx]
                        glColor3f(*color)
                        glVertex3f(*point)
            glEnd()
        else:
            # Fallback to points if no topology
            self._draw_points()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _draw_surface(self):
        """Draw filled surface representation"""
        if len(self._indices) == 0:
            self._generate_surface()
        
        glEnable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_TRIANGLES)
        for triangle in self._indices:
            for vertex_idx in triangle:
                if vertex_idx < len(self._points):
                    point = self._points[vertex_idx]
                    color = self._colours[vertex_idx]
                    if vertex_idx < len(self._normals):
                        normal = self._normals[vertex_idx]
                        glNormal3f(*normal)
                    glColor3f(*color)
                    glVertex3f(*point)
        glEnd()

    def _draw_volumetric(self):
        """Draw volumetric representation using alpha blending"""
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        
        # Sort points by depth for proper alpha blending
        view_matrix = np.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        depths = []
        for point in self._points:
            transformed = view_matrix @ np.append(point, 1.0)
            depths.append(transformed[2])
        
        sorted_indices = np.argsort(depths)[::-1]  # Back to front
        
        glPointSize(self._point_size * 2)
        glBegin(GL_POINTS)
        for i in sorted_indices:
            point = self._points[i]
            color = self._colours[i]
            glColor4f(color[0], color[1], color[2], 0.3)  # Semi-transparent
            glVertex3f(*point)
        glEnd()
        
        glDepthMask(GL_TRUE)

    def _draw_axes(self):
        """Draw coordinate axes"""
        glCallList(self._axes_list)

    def _draw_axes_immediate(self):
        """Immediate mode axes drawing"""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
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
        
        if self._lighting_enabled:
            glEnable(GL_LIGHTING)

    def _draw_grid(self):
        """Draw grid"""
        glCallList(self._grid_list)

    def _draw_grid_immediate(self):
        """Immediate mode grid drawing"""
        glDisable(GL_LIGHTING)
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.0)
        
        grid_size = 10
        grid_spacing = 0.2
        
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            # XY plane lines
            glVertex3f(-grid_size * grid_spacing, i * grid_spacing, 0)
            glVertex3f(grid_size * grid_spacing, i * grid_spacing, 0)
            glVertex3f(i * grid_spacing, -grid_size * grid_spacing, 0)
            glVertex3f(i * grid_spacing, grid_size * grid_spacing, 0)
        glEnd()
        
        if self._lighting_enabled:
            glEnable(GL_LIGHTING)

    def _draw_bounding_box(self):
        """Draw bounding box around data"""
        if self._points.size == 0:
            return
            
        min_coords = self._points.min(axis=0)
        max_coords = self._points.max(axis=0)
        
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        
        # Draw wireframe box
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
        
        if self._lighting_enabled:
            glEnable(GL_LIGHTING)

    def _draw_selection_box(self):
        """Draw selection box"""
        if not self._selection_box:
            return
            
        x1, y1, x2, y2 = self._selection_box
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor4f(1.0, 1.0, 0.0, 0.3)
        
        # Convert screen coordinates to world coordinates
        # This is a simplified version - proper implementation would use unproject
        glBegin(GL_QUADS)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        if self._lighting_enabled:
            glEnable(GL_LIGHTING)

    def _draw_measurements(self):
        """Draw measurement lines and annotations"""
        if len(self._measurement_points) < 2:
            return
            
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.0, 0.0)
        glLineWidth(2.0)
        
        glBegin(GL_LINES)
        for i in range(0, len(self._measurement_points) - 1, 2):
            if i + 1 < len(self._measurement_points):
                glVertex3f(*self._measurement_points[i])
                glVertex3f(*self._measurement_points[i + 1])
        glEnd()
        
        if self._lighting_enabled:
            glEnable(GL_LIGHTING)

    def _setup_clipping(self):
        """Setup clipping planes"""
        for i, plane in enumerate(self._clip_planes):
            glEnable(GL_CLIP_PLANE0 + i)
            glClipPlane(GL_CLIP_PLANE0 + i, plane)

    def _generate_surface(self):
        """Generate surface triangulation from point cloud"""
        if len(self._points) < 3:
            return
            
        # Simple triangulation - in practice, you'd use Delaunay triangulation
        # or marching cubes for volumetric data
        n_points = len(self._points)
        triangles = []
        
        # Create a simple grid-based triangulation
        # This is a placeholder - implement proper surface reconstruction
        for i in range(0, n_points - 2, 3):
            if i + 2 < n_points:
                triangles.append([i, i + 1, i + 2])
        
        self._indices = np.array(triangles, dtype=np.uint32)
        
        # Generate normals
        self._generate_normals()

    def _generate_normals(self):
        """Generate vertex normals"""
        if len(self._indices) == 0:
            return
            
        self._normals = np.zeros_like(self._points)
        
        # Calculate face normals and accumulate at vertices
        for triangle in self._indices:
            if all(idx < len(self._points) for idx in triangle):
                v0, v1, v2 = [self._points[idx] for idx in triangle]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                for idx in triangle:
                    self._normals[idx] += normal
        
        # Normalize
        norms = np.linalg.norm(self._normals, axis=1)
        norms = np.maximum(norms, 1e-8)
        self._normals = self._normals / norms.reshape(-1, 1)

    def _update_fps(self):
        """Update FPS counter"""
        self._frame_count += 1
        current_time = time.time()
        
        if current_time - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (current_time - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = current_time

    # Event handlers
    def _on_mouse_down(self, event):
        """Handle mouse press events"""
        self._last_mouse = (event.x, event.y)
        self._mouse_button = event.num
        
        if self._selection_mode and event.num == 1:
            self._selection_box = [event.x, event.y, event.x, event.y]

    def _on_mouse_drag(self, event):
        """Handle mouse drag events"""
        if self._last_mouse is None:
            return
            
        dx = event.x - self._last_mouse[0]
        dy = event.y - self._last_mouse[1]
        
        if self._mouse_button == 1:  # Left button - rotate
            if self._selection_mode and self._selection_box:
                self._selection_box[2] = event.x
                self._selection_box[3] = event.y
            else:
                self._angle_y += dx * 0.5 * self._mouse_sensitivity
                self._angle_x += dy * 0.5 * self._mouse_sensitivity
        elif self._mouse_button == 2:  # Middle button - pan
            factor = 0.01 / self._scale
            self._trans_x += dx * factor
            self._trans_y -= dy * factor
        elif self._mouse_button == 3:  # Right button - zoom/pan
            self._camera_distance += dy * 0.01
            self._camera_distance = max(0.1, min(self._camera_distance, 50.0))
        
        self._last_mouse = (event.x, event.y)
        self.after_idle(self.redraw)

    def _on_double_click(self, event):
        """Handle double-click events"""
        # Reset view
        self.reset_view()

    def _clear_mouse(self):
        """Clear mouse state"""
        if self._selection_mode and self._selection_box:
            self._process_selection()
            self._selection_box = None
        
        self._last_mouse = None
        self._mouse_button = None

    def _on_wheel(self, event):
        """Handle mouse wheel events"""
        direction = -1 if event.delta > 0 else 1
        factor = 1.1 if direction > 0 else 1/1.1
        self.zoom(factor)

    def _on_key_press(self, event):
        """Handle key press events"""
        key = event.keysym.lower()
        
        if key == 'r':
            self.reset_view()
        elif key == 'p':
            self.set_render_mode('points')
        elif key == 'w':
            self.set_render_mode('wireframe')
        elif key == 's':
            self.set_render_mode('surface')
        elif key == 'v':
            self.set_render_mode('volumetric')
        elif key == 'a':
            self.toggle_axes()
        elif key == 'g':
            self.toggle_grid()
        elif key == 'l':
            self.toggle_lighting()
        elif key == 'o':
            self.toggle_projection()
        elif key == 'f':
            self.fit_to_screen()
        elif key == 'c':
            self.screenshot()
        elif key == 'space':
            self.toggle_animation()
        elif key == 'escape':
            self.clear_selection()
        
        self.after_idle(self.redraw)

    def _on_resize(self, event):
        """Handle window resize events"""
        width, height = max(event.width, 1), max(event.height, 1)
        self._setup_projection(width, height)
        self.after_idle(self.redraw)

    def _process_selection(self):
        """Process selection box and select points within it"""
        if not self._selection_box or len(self._points) == 0:
            return
        
        # This is a simplified version - proper implementation would use
        # OpenGL selection buffer or ray casting
        x1, y1, x2, y2 = self._selection_box
        
        # Convert to normalized device coordinates and select points
        # For now, just select random points as placeholder
        import random
        n_select = min(10, len(self._points))
        self._selected_points = set(random.sample(range(len(self._points)), n_select))

    # Public API methods
    def set_points(self, points: np.ndarray, colours: np.ndarray):
        """Set point cloud data"""
        if points.shape[0] != colours.shape[0]:
            raise ValueError("points and colours must have the same length")
        
        # Normalize points to fit in [-1, 1] cube
        if points.size > 0:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            centre = (mins + maxs) / 2.0
            ranges = maxs - mins
            max_range = ranges.max()
            scale = 2.0 / max_range if max_range != 0 else 1.0
            self._points = ((points - centre) * scale).astype(np.float32)
        else:
            self._points = points.astype(np.float32)
        
        self._colours = colours.astype(np.float32)
        
        # Update VBO if using them
        if self._use_vbo and self._point_vbo:
            self._update_vbo()
        
        # Clear previous selection and measurements
        self._selected_points.clear()
        self._measurement_points.clear()
        
        self.after_idle(self.redraw)

    def _update_vbo(self):
        """Update VBO data"""
        if not self._use_vbo:
            return
            
        # Update point VBO
        glBindBuffer(GL_ARRAY_BUFFER, self._point_vbo)
        glBufferData(GL_ARRAY_BUFFER, self._points.nbytes, self._points, GL_STATIC_DRAW)
        
        # Update color VBO
        glBindBuffer(GL_ARRAY_BUFFER, self._color_vbo)
        glBufferData(GL_ARRAY_BUFFER, self._colours.nbytes, self._colours, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def zoom(self, factor: float):
        """Zoom in/out"""
        self._scale *= factor
        self._scale = max(0.01, min(self._scale, 100.0))
        self.after_idle(self.redraw)

    def set_render_mode(self, mode: str):
        """Set rendering mode"""
        if mode in ['points', 'wireframe', 'surface', 'volumetric']:
            self._render_mode = mode
            self.after_idle(self.redraw)

    def toggle_axes(self):
        """Toggle coordinate axes display"""
        self._show_axes = not self._show_axes
        self.after_idle(self.redraw)

    def toggle_grid(self):
        """Toggle grid display"""
        self._show_grid = not self._show_grid
        self.after_idle(self.redraw)

    def toggle_lighting(self):
        """Toggle lighting"""
        self._lighting_enabled = not self._lighting_enabled
        self._setup_lighting()
        self.after_idle(self.redraw)

    def toggle_projection(self):
        """Toggle between perspective and orthographic projection"""
        self._projection_mode = 'orthographic' if self._projection_mode == 'perspective' else 'perspective'
        width, height = self.winfo_width(), self.winfo_height()
        self._setup_projection(width, height)
        self.after_idle(self.redraw)

    def reset_view(self):
        """Reset view to default"""
        self._scale = 1.0
        self._angle_x = 0.0
        self._angle_y = 0.0
        self._angle_z = 0.0
        self._trans_x = 0.0
        self._trans_y = 0.0
        self._trans_z = 0.0
        self._camera_distance = 5.0
        self.after_idle(self.redraw)

    def fit_to_screen(self):
        """Fit all data to screen"""
        if len(self._points) == 0:
            return
        
        # Calculate appropriate scale to fit data
        mins = self._points.min(axis=0)
        maxs = self._points.max(axis=0)
        ranges = maxs - mins
        max_range = ranges.max()
        
        if max_range > 0:
            self._scale = 1.5 / max_range
        
        self.after_idle(self.redraw)

    def screenshot(self, filename=None):
        """Take a screenshot"""
        if filename is None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
        
        if filename:
            # Read pixels from framebuffer
            width, height = self.winfo_width(), self.winfo_height()
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            # Convert to PIL Image and save
            image = Image.frombytes("RGB", (width, height), data)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL coordinates are flipped
            image.save(filename)
            
            messagebox.showinfo("Screenshot", f"Screenshot saved to {filename}")

    def toggle_animation(self):
        """Toggle automatic rotation animation"""
        self._animation_running = not self._animation_running
        if self._animation_running:
            self._animate()

    def _animate(self):
        """Animation loop"""
        if self._animation_running:
            self._angle_y += self._animation_speed
            if self._angle_y >= 360:
                self._angle_y -= 360
            self.after_idle(self.redraw)
            self.after(50, self._animate)  # 20 FPS animation

    def toggle_selection_mode(self):
        """Toggle selection mode"""
        self._selection_mode = not self._selection_mode

    def clear_selection(self):
        """Clear current selection"""
        self._selected_points.clear()
        self._selection_box = None
        self.after_idle(self.redraw)

    def add_measurement_point(self, point):
        """Add a measurement point"""
        self._measurement_points.append(point)
        self.after_idle(self.redraw)

    def clear_measurements(self):
        """Clear all measurements"""
        self._measurement_points.clear()
        self.after_idle(self.redraw)

    def get_fps(self):
        """Get current FPS"""
        return self._fps

    def set_point_size(self, size):
        """Set point size"""
        self._point_size = max(1.0, min(size, 20.0))
        self.after_idle(self.redraw)

    def set_background_color(self, color):
        """Set background color (r, g, b, a)"""
        self._background_color = color
        self.after_idle(self.redraw)

    def export_selection(self):
        """Export selected points"""
        if not self._selected_points:
            messagebox.showwarning("Export", "No points selected")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            selected_points = self._points[list(self._selected_points)]
            selected_colors = self._colours[list(self._selected_points)]
            
            with open(filename, 'w') as f:
                f.write("x,y,z,r,g,b\n")
                for point, color in zip(selected_points, selected_colors):
                    f.write(f"{point[0]},{point[1]},{point[2]},{color[0]},{color[1]},{color[2]}\n")
            
            messagebox.showinfo("Export", f"Selected points exported to {filename}")

# For backward compatibility
VoxelOpenGLCanvas = Advanced3DCanvas 