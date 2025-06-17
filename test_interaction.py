#!/usr/bin/env python3
"""
Test OpenGL Canvas Interaction
This script tests if the OpenGL canvas properly handles mouse and keyboard interaction.
"""

import tkinter as tk
import numpy as np
from opengl_canvas import VoxelOpenGLCanvas

def test_interaction():
    """Test OpenGL canvas interaction"""
    
    root = tk.Tk()
    root.title("OpenGL Interaction Test")
    root.geometry("800x600")
    
    # Create instructions
    instructions = tk.Label(root, 
                          text="OpenGL Canvas Interaction Test\n\n" +
                               "Controls:\n" +
                               "• Left Mouse: Rotate (drag to rotate the view)\n" +
                               "• Middle Mouse: Pan (drag to move the view)\n" +
                               "• Right Mouse/Wheel: Zoom (drag or scroll to zoom)\n" +
                               "• Double-click: Reset view\n" +
                               "• R: Reset view\n" +
                               "• P: Points mode, W: Wireframe, S: Surface\n" +
                               "• A: Toggle Axes, G: Toggle Grid, L: Toggle Lighting\n\n" +
                               "If you can rotate/zoom the view, interaction is working!",
                          justify=tk.LEFT,
                          bg="lightblue",
                          font=("Arial", 10),
                          padx=10,
                          pady=10)
    instructions.pack(side=tk.TOP, fill=tk.X)
    
    # Create OpenGL canvas
    canvas = VoxelOpenGLCanvas(root, width=700, height=500)
    canvas.pack(padx=10, pady=10, expand=True, fill='both')
    
    # Create test data - a colorful 3D structure
    print("Creating test data...")
    
    # Create a spiral of points
    t = np.linspace(0, 4*np.pi, 200)
    x = np.cos(t) * np.linspace(0.1, 1, 200)
    y = np.sin(t) * np.linspace(0.1, 1, 200)
    z = np.linspace(-1, 1, 200)
    
    points = np.column_stack([x, y, z]).astype(np.float32)
    
    # Create rainbow colors
    colors = np.zeros((len(points), 3), dtype=np.float32)
    for i in range(len(points)):
        hue = (i / len(points)) * 360
        if hue < 120:
            colors[i] = [1.0, hue/120.0, 0.0]  # Red to yellow
        elif hue < 240:
            colors[i] = [(240-hue)/120.0, 1.0, 0.0]  # Yellow to green
        else:
            colors[i] = [0.0, 1.0, (hue-240)/120.0]  # Green to blue
    
    # Set the data
    canvas.set_points(points, colors)
    
    # Make sure canvas gets focus
    canvas.focus_set()
    
    # Add a status label
    status = tk.Label(root, text="Try interacting with the 3D view above!", 
                     bg="lightgreen", font=("Arial", 12))
    status.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def update_status():
        """Update status with current view info"""
        fps = canvas.get_fps()
        status.config(text=f"FPS: {fps:.1f} | Try mouse controls and keyboard shortcuts!")
        root.after(1000, update_status)  # Update every second
    
    # Start status updates
    root.after(1000, update_status)
    
    print("Test window created. Try interacting with the 3D view:")
    print("- Drag with left mouse to rotate")
    print("- Drag with middle mouse to pan") 
    print("- Use mouse wheel to zoom")
    print("- Press 'R' to reset view")
    print("- Press 'A' to toggle axes")
    
    root.mainloop()

if __name__ == "__main__":
    print("Testing OpenGL Canvas Interaction...")
    test_interaction() 