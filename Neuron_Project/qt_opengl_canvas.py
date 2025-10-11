from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QMouseEvent, QWheelEvent, QVector3D
from PySide6.QtCore import Qt
from OpenGL.GL import *  # noqa: F401
from OpenGL.GLU import *  # noqa: F401
import numpy as np
import math
import sys
import platform


class QtAdvanced3DCanvas(QOpenGLWidget):
    """A simplified PySide6 replacement for Advanced3DCanvas.

    Only a subset of the full feature-set is currently implemented:
        • perspective / orthographic projection
        • point cloud rendering via vertex arrays
        • basic mouse interaction – rotate (LMB), pan (MMB), zoom (wheel)
    The public API (methods used by the existing viewer code) is preserved
    so that higher-level modules can switch from the tkinter/pyopengltk
    implementation to this Qt one with minimal changes.
    
    WINDOWS COMPATIBILITY:
        • Automatic fallback for Intel integrated graphics
        • Safe mode rendering for driver issues
        • Multiple error recovery layers
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        # Data storage
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colours = np.empty((0, 3), dtype=np.float32)

        # Camera controls
        self._angle = QVector3D(0.0, 0.0, 0.0)
        self._translation = QVector3D(0.0, 0.0, -5.0)
        # Zoom boundaries (in camera space Z). Negative values: closer to zero is nearer
        self._zoom_near = kwargs.get("zoom_near", -0.5)   # closest allowed
        self._zoom_far = kwargs.get("zoom_far", -6.0)   # farthest allowed
        self._last_pos = None
        self._point_size = kwargs.get("point_size", 6.0)

        # Render options
        self._background_color = kwargs.get("background_color", (0.95, 0.95, 0.95, 1.0))
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Windows compatibility flags
        self._safe_mode = False  # Fallback rendering mode
        self._intel_gpu = False  # Intel integrated graphics detected
        self._opengl_version = None
        self._initialization_failed = False

    # ------------------------------------------------------------------
    # API parity helpers – only the subset actually used by the viewer.
    # ------------------------------------------------------------------
    def set_points(self, points: np.ndarray, colours: np.ndarray):
        """Assign a new point cloud – both arrays must be float32."""
        self._points = points.astype(np.float32, copy=False)
        self._colours = colours.astype(np.float32, copy=False)
        self.fit_to_screen()
        self.update()

    # Maintain naming compatibility
    def set_point_size(self, size: float):
        self._point_size = size
        self.update()

    def zoom(self, factor: float):
        new_z = self._translation.z() * factor
        # Enforce boundaries so the point cloud isn't lost or clipped
        if new_z > self._zoom_near:
            new_z = self._zoom_near
        elif new_z < self._zoom_far:
            new_z = self._zoom_far
        self._translation.setZ(new_z)
        self.update()

    # ------------------------------------------------------------------
    # Qt OpenGL overrides
    # ------------------------------------------------------------------
    def initializeGL(self):
        """Initialize OpenGL with EXTREME Windows compatibility and fallback modes."""
        initialization_successful = False
        
        try:
            # First, try to get basic OpenGL info
            version_string = glGetString(GL_VERSION)
            if version_string:
                version_string = version_string.decode('utf-8') if isinstance(version_string, bytes) else version_string
                self._opengl_version = version_string
                print(f"OpenGL Version: {version_string}")
            
            # Check renderer (GPU info)
            renderer = glGetString(GL_RENDERER)
            if renderer:
                renderer = renderer.decode('utf-8') if isinstance(renderer, bytes) else renderer
                print(f"OpenGL Renderer: {renderer}")
                
                # Detect Intel integrated graphics (common issue on Windows)
                if 'intel' in renderer.lower():
                    self._intel_gpu = True
                    print("⚠️  Intel integrated GPU detected - enabling safe mode")
                    self._safe_mode = True
                    
                    if platform.system() == 'Windows':
                        print("   Windows + Intel GPU: Using conservative rendering settings")
            
            # Get vendor info
            vendor = glGetString(GL_VENDOR)
            if vendor:
                vendor = vendor.decode('utf-8') if isinstance(vendor, bytes) else vendor
                print(f"OpenGL Vendor: {vendor}")
            
            # Basic OpenGL setup with error checking
            try:
                glEnable(GL_DEPTH_TEST)
            except Exception as e:
                print(f"⚠️  Could not enable depth testing: {e}")
                self._safe_mode = True
            
            # Set background color
            try:
                r, g, b, a = self._background_color
                glClearColor(r, g, b, a)
            except Exception as e:
                print(f"⚠️  Could not set background color: {e}")
                glClearColor(0.95, 0.95, 0.95, 1.0)  # Hardcoded fallback
            
            # Windows-specific optimizations
            if platform.system() == 'Windows':
                # Enable multisampling only if not Intel GPU
                if not self._intel_gpu:
                    try:
                        glEnable(GL_MULTISAMPLE)
                        print("✓ Multisampling enabled (better quality)")
                    except:
                        pass  # Not critical
                
                # Enable point smoothing for better rendering
                try:
                    glEnable(GL_POINT_SMOOTH)
                    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
                except:
                    pass  # Not critical
                
                # Set blend function for transparency (if needed)
                try:
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                except:
                    pass
            
            initialization_successful = True
            print("✓ OpenGL initialization successful")
            
        except Exception as e:
            print(f"⚠️  OpenGL initialization error: {e}")
            print("Attempting fallback initialization...")
            self._safe_mode = True
            self._initialization_failed = True
            
            # FALLBACK MODE - Minimal OpenGL setup
            try:
                # Try absolute minimum
                glClearColor(0.95, 0.95, 0.95, 1.0)
                glEnable(GL_DEPTH_TEST)
                print("✓ Fallback initialization successful (safe mode)")
                initialization_successful = True
            except Exception as fallback_error:
                print(f"✗ CRITICAL: Fallback initialization failed: {fallback_error}")
                
                if platform.system() == 'Windows':
                    print("\n" + "="*70)
                    print("WINDOWS OPENGL CRITICAL ERROR")
                    print("="*70)
                    print("Your graphics drivers may be outdated or incompatible.")
                    print("\nImmediate actions:")
                    print("1. Update graphics drivers from manufacturer:")
                    print("   - Intel: https://www.intel.com/content/www/us/en/download-center/")
                    print("   - NVIDIA: https://www.nvidia.com/Download/index.aspx")
                    print("   - AMD: https://www.amd.com/en/support")
                    print("2. Restart your computer after installing drivers")
                    print("3. Run this application as Administrator")
                    print("4. Check Windows Update for additional driver updates")
                    print("="*70 + "\n")
                else:
                    sys.stderr.write("OpenGL initialization failed completely.\n")
                    sys.stderr.write("Please update your graphics drivers.\n")
        
        # Log final status
        if not initialization_successful:
            print("⚠️  Application may have rendering issues")
        elif self._safe_mode:
            print("⚠️  Running in safe mode (reduced quality but more stable)")

    def resizeGL(self, w: int, h: int):
        """Handle window resize with error checking."""
        try:
            h = max(h, 1)
            glViewport(0, 0, w, h)
            self._set_projection(w, h)
        except Exception as e:
            if not hasattr(self, '_resize_error_logged'):
                print(f"OpenGL resize error: {e}")
                self._resize_error_logged = True

    def paintGL(self):
        """Render the 3D scene with EXTREME error handling and fallback modes."""
        try:
            # Clear buffers - wrapped in try-catch for maximum robustness
            try:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            except Exception as e:
                if not hasattr(self, '_clear_error_logged'):
                    print(f"⚠️  glClear error: {e} - attempting recovery")
                    self._clear_error_logged = True
                # Try alternative clear
                try:
                    glClear(GL_COLOR_BUFFER_BIT)
                except:
                    pass  # Continue anyway

            # Setup matrices
            try:
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
            except Exception as e:
                if not hasattr(self, '_matrix_error_logged'):
                    print(f"⚠️  Matrix setup error: {e}")
                    self._matrix_error_logged = True
                return  # Can't proceed without matrix setup

            # Camera transform - each operation wrapped for safety
            try:
                glTranslatef(self._translation.x(), self._translation.y(), self._translation.z())
                glRotatef(self._angle.x(), 1, 0, 0)
                glRotatef(self._angle.y(), 0, 1, 0)
                glRotatef(self._angle.z(), 0, 0, 1)
            except Exception as e:
                if not hasattr(self, '_transform_error_logged'):
                    print(f"⚠️  Transform error: {e}")
                    self._transform_error_logged = True
                # Continue with identity transform

            # Draw points - multiple fallback strategies
            if self._points.size > 0:
                try:
                    # Set point size
                    point_size = self._point_size if not self._safe_mode else min(self._point_size, 4.0)
                    glPointSize(point_size)
                except:
                    glPointSize(3.0)  # Fallback size
                
                # SAFE MODE: Render in smaller batches to avoid driver timeout
                if self._safe_mode or platform.system() == 'Windows':
                    self._paint_safe_mode()
                else:
                    # Normal mode
                    self._paint_normal_mode()
                    
        except Exception as e:
            # Ultimate fallback - log error and try to continue
            if not hasattr(self, '_paint_error_logged'):
                print(f"⚠️  OpenGL rendering error: {e}")
                if platform.system() == 'Windows':
                    print("   Windows: This may indicate graphics driver issues.")
                    print("   Enabling safe mode for stability...")
                    self._safe_mode = True
                self._paint_error_logged = True
    
    def _paint_normal_mode(self):
        """Normal rendering mode - all points at once."""
        try:
            glBegin(GL_POINTS)
            try:
                for (x, y, z), (r, g, b) in zip(self._points, self._colours):
                    glColor3f(r, g, b)
                    glVertex3f(x, y, z)
            finally:
                glEnd()  # Ensure glEnd is always called
        except Exception as e:
            print(f"⚠️  Normal mode rendering failed: {e}, switching to safe mode")
            self._safe_mode = True
            self._paint_safe_mode()
    
    def _paint_safe_mode(self):
        """Safe mode rendering - render in smaller batches to avoid driver timeout."""
        try:
            # Render in batches of 1000 points (Windows graphics drivers can timeout on large operations)
            batch_size = 1000
            num_points = len(self._points)
            
            for start_idx in range(0, num_points, batch_size):
                end_idx = min(start_idx + batch_size, num_points)
                
                try:
                    glBegin(GL_POINTS)
                    try:
                        for i in range(start_idx, end_idx):
                            x, y, z = self._points[i]
                            r, g, b = self._colours[i]
                            glColor3f(r, g, b)
                            glVertex3f(x, y, z)
                    finally:
                        glEnd()
                except Exception as batch_error:
                    # Skip this batch and continue
                    if start_idx == 0:  # Only log on first batch
                        print(f"⚠️  Batch rendering issue: {batch_error}")
                    continue
                    
        except Exception as e:
            print(f"⚠️  Safe mode rendering failed: {e}")

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_pos is None:
            return
        dx = event.position().x() - self._last_pos.x()
        dy = event.position().y() - self._last_pos.y()

        if event.buttons() & Qt.LeftButton:
            self._angle.setX(self._angle.x() + dy)
            self._angle.setY(self._angle.y() + dx)
        elif event.buttons() & Qt.MiddleButton:
            # Panning
            self._translation.setX(self._translation.x() + dx * 0.01)
            self._translation.setY(self._translation.y() - dy * 0.01)
        self._last_pos = event.position()
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.zoom(factor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_projection(self, w: int, h: int):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h
        gluPerspective(45.0, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def fit_to_screen(self):
        """Place the camera so the entire point cloud is comfortably inside view."""
        if self._points.size == 0:
            return
        mins = self._points.min(axis=0)
        maxs = self._points.max(axis=0)
        center = (mins + maxs) / 2.0
        diag = np.linalg.norm(maxs - mins)
        # Move translation so center is at origin and camera back enough
        self._translation.setX(-center[0])
        self._translation.setY(-center[1])
        # Heuristic distance: half diag / tan(fov/2)
        distance = diag * 1.2 / (2 * math.tan(math.radians(45.0) / 2))
        self._translation.setZ(-max(distance, 0.5))
        # Ensure within zoom boundaries
        if self._translation.z() > self._zoom_near:
            self._translation.setZ(self._zoom_near)
        elif self._translation.z() < self._zoom_far:
            self._translation.setZ(self._zoom_far)


# For backwards compatibility with existing imports
VoxelOpenGLCanvas = QtAdvanced3DCanvas 