#!/usr/bin/env python3
"""
Simple OpenGL Canvas Test
This script tests if the enhanced OpenGL canvas works correctly by displaying a simple point cloud.
"""

import tkinter as tk
import numpy as np
from opengl_canvas import VoxelOpenGLCanvas

def test_opengl_canvas():
    """Test the OpenGL canvas with a simple point cloud"""
    
    # Create root window
    root = tk.Tk()
    root.title("OpenGL Canvas Test")
    root.geometry("800x600")
    
    try:
        # Create OpenGL canvas
        canvas = VoxelOpenGLCanvas(root, width=600, height=500)
        canvas.pack(padx=10, pady=10, expand=True, fill='both')
        
        # Create test data - a simple cube of points
        print("Creating test data...")
        n = 20  # Points per dimension
        x, y, z = np.meshgrid(np.linspace(-1, 1, n), 
                             np.linspace(-1, 1, n), 
                             np.linspace(-1, 1, n))
        
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)
        
        # Create colors - rainbow gradient based on distance from center
        distances = np.sqrt(np.sum(points**2, axis=1))
        max_dist = distances.max()
        colors = np.zeros((len(points), 3), dtype=np.float32)
        
        for i, dist in enumerate(distances):
            # Create rainbow colors based on distance
            hue = (dist / max_dist) * 360  # 0 to 360 degrees
            if hue < 120:
                colors[i] = [1.0, hue/120.0, 0.0]  # Red to yellow
            elif hue < 240:
                colors[i] = [(240-hue)/120.0, 1.0, 0.0]  # Yellow to green
            else:
                colors[i] = [0.0, 1.0, (hue-240)/120.0]  # Green to blue
        
        print(f"Created {len(points)} points with colors")
        
        # Set the points in the canvas
        canvas.set_points(points, colors)
        
        # Add some control instructions
        instructions = tk.Label(root, 
                              text="Controls:\n" +
                                   "Left Mouse: Rotate\n" +
                                   "Middle Mouse: Pan\n" +
                                   "Right Mouse/Wheel: Zoom\n" +
                                   "Double-click: Reset view\n" +
                                   "R: Reset, P: Points, W: Wireframe, S: Surface\n" +
                                   "A: Toggle Axes, G: Toggle Grid, L: Toggle Lighting",
                              justify=tk.LEFT,
                              bg="lightgray",
                              font=("Arial", 8))
        instructions.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        print("OpenGL canvas created successfully!")
        print("If you see a 3D point cloud, the canvas is working correctly.")
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        print(f"Error creating OpenGL canvas: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error in a message box
        try:
            tk.messagebox.showerror("OpenGL Error", 
                                  f"Failed to create OpenGL canvas:\n{str(e)}\n\n" +
                                  "This might be due to:\n" +
                                  "1. Missing OpenGL drivers\n" +
                                  "2. PyOpenGL not installed correctly\n" +
                                  "3. Graphics card compatibility issues")
        except:
            pass
        
        root.destroy()

if __name__ == "__main__":
    print("Testing OpenGL Canvas...")
    print("This will create a window with a 3D point cloud if everything works correctly.")
    test_opengl_canvas() 