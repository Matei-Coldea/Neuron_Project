from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QMouseEvent, QWheelEvent, QVector3D
from PySide6.QtCore import Qt
from OpenGL.GL import *  # noqa: F401
from OpenGL.GLU import *  # noqa: F401
import numpy as np
import math


class QtAdvanced3DCanvas(QOpenGLWidget):
    """A simplified PySide6 replacement for Advanced3DCanvas.

    Only a subset of the full feature-set is currently implemented:
        • perspective / orthographic projection
        • point cloud rendering via vertex arrays
        • basic mouse interaction – rotate (LMB), pan (MMB), zoom (wheel)
    The public API (methods used by the existing viewer code) is preserved
    so that higher-level modules can switch from the tkinter/pyopengltk
    implementation to this Qt one with minimal changes.
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
        glEnable(GL_DEPTH_TEST)
        r, g, b, a = self._background_color
        glClearColor(r, g, b, a)

    def resizeGL(self, w: int, h: int):
        h = max(h, 1)
        glViewport(0, 0, w, h)
        self._set_projection(w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera transform
        glTranslatef(self._translation.x(), self._translation.y(), self._translation.z())
        glRotatef(self._angle.x(), 1, 0, 0)
        glRotatef(self._angle.y(), 0, 1, 0)
        glRotatef(self._angle.z(), 0, 0, 1)

        # Draw points – simple immediate mode is sufficient for now
        if self._points.size > 0:
            glPointSize(self._point_size)
            glBegin(GL_POINTS)
            for (x, y, z), (r, g, b) in zip(self._points, self._colours):
                glColor3f(r, g, b)
                glVertex3f(x, y, z)
            glEnd()

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