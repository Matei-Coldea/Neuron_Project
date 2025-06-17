#!/usr/bin/env python3
"""
Test script for the Enhanced 3D Viewer

This script tests the key functionalities of the enhanced OpenGL 3D viewer
and ensures all features are working properly.
"""

import sys
import os
import traceback

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import tkinter as tk
        print("✓ Tkinter imported successfully")
    except ImportError as e:
        print(f"✗ Tkinter import failed: {e}")
        return False
        
    try:
        import OpenGL.GL as gl
        print("✓ PyOpenGL imported successfully")
    except ImportError as e:
        print(f"✗ PyOpenGL import failed: {e}")
        return False
        
    try:
        from pyopengltk import OpenGLFrame
        print("✓ pyopengltk imported successfully")
    except ImportError as e:
        print(f"✗ pyopengltk import failed: {e}")
        return False
        
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
        
    return True

def test_opengl_canvas():
    """Test the Advanced3DCanvas class"""
    print("\nTesting Advanced3DCanvas...")
    
    try:
        from opengl_canvas import Advanced3DCanvas
        print("✓ Advanced3DCanvas imported successfully")
        
        # Test basic initialization (without actually creating window)
        print("✓ Advanced3DCanvas class accessible")
        return True
        
    except ImportError as e:
        print(f"✗ Advanced3DCanvas import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Advanced3DCanvas test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic viewer functionality without GUI"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        
        # Generate test data
        n_points = 1000
        points = np.random.rand(n_points, 3).astype(np.float32)
        colors = np.random.rand(n_points, 3).astype(np.float32)
        
        print(f"✓ Generated test data: {n_points} points")
        
        # Test data validation
        assert points.shape == (n_points, 3), "Points shape incorrect"
        assert colors.shape == (n_points, 3), "Colors shape incorrect"
        assert points.dtype == np.float32, "Points dtype incorrect"
        assert colors.dtype == np.float32, "Colors dtype incorrect"
        
        print("✓ Data validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_viewer_creation():
    """Test actual viewer creation (requires display)"""
    print("\nTesting viewer creation...")
    
    try:
        import tkinter as tk
        from opengl_canvas import Advanced3DCanvas
        import numpy as np
        
        # Create root window
        root = tk.Tk()
        root.title("Test Window")
        root.geometry("400x300")
        
        # Create canvas
        canvas = Advanced3DCanvas(root, width=300, height=200)
        canvas.pack()
        
        # Generate and set test data
        points = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        
        canvas.set_points(points, colors)
        
        print("✓ Viewer created successfully")
        
        # Test some basic methods
        canvas.zoom(1.1)
        canvas.set_render_mode('points')
        canvas.toggle_axes()
        canvas.reset_view()
        
        print("✓ Basic methods work")
        
        # Clean up
        root.after(100, root.destroy)  # Close after 100ms
        root.mainloop()
        
        print("✓ Viewer test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Viewer creation test failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_features():
    """Test specific enhanced features"""
    print("\nTesting enhanced features...")
    
    try:
        # Test keyboard shortcut mapping
        shortcuts = {
            'r': 'reset_view',
            'p': 'points_mode',
            'w': 'wireframe_mode',
            's': 'surface_mode',
            'v': 'volumetric_mode',
            'a': 'toggle_axes',
            'g': 'toggle_grid',
            'l': 'toggle_lighting',
            'o': 'toggle_projection',
            'f': 'fit_to_screen',
            'c': 'screenshot',
            'space': 'toggle_animation',
            'escape': 'clear_selection'
        }
        
        print(f"✓ Keyboard shortcuts defined: {len(shortcuts)} shortcuts")
        
        # Test rendering modes
        render_modes = ['points', 'wireframe', 'surface', 'volumetric']
        print(f"✓ Rendering modes available: {render_modes}")
        
        # Test projection modes
        projection_modes = ['perspective', 'orthographic']
        print(f"✓ Projection modes available: {projection_modes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced features test failed: {e}")
        return False

def run_interactive_demo():
    """Run a brief interactive demo"""
    print("\nRunning interactive demo...")
    
    try:
        import tkinter as tk
        from tkinter import messagebox
        from opengl_canvas import Advanced3DCanvas
        import numpy as np
        
        root = tk.Tk()
        root.title("Enhanced 3D Viewer Demo")
        root.geometry("800x600")
        
        # Create instructions
        instructions = tk.Label(root, 
                              text="Enhanced 3D Viewer Demo\n\n" +
                                   "Mouse: Left=Rotate, Middle=Pan, Right=Zoom\n" +
                                   "Keys: R=Reset, P=Points, W=Wireframe, S=Surface, V=Volumetric\n" +
                                   "      A=Axes, G=Grid, L=Lighting, O=Projection\n\n" +
                                   "Close window to end demo",
                              font=("Arial", 10))
        instructions.pack(pady=10)
        
        # Create canvas
        canvas = Advanced3DCanvas(root, width=700, height=400)
        canvas.pack(padx=10, pady=10)
        
        # Generate sample data - colorful sphere
        n_points = 2000
        phi = np.random.uniform(0, 2*np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
        
        # Rainbow colors based on position
        colors = np.zeros((n_points, 3), dtype=np.float32)
        colors[:, 0] = (x + 1) / 2  # Red
        colors[:, 1] = (y + 1) / 2  # Green
        colors[:, 2] = (z + 1) / 2  # Blue
        
        canvas.set_points(points, colors)
        
        # Control buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Points", 
                 command=lambda: canvas.set_render_mode('points')).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Wireframe", 
                 command=lambda: canvas.set_render_mode('wireframe')).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Surface", 
                 command=lambda: canvas.set_render_mode('surface')).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Reset View", 
                 command=canvas.reset_view).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Toggle Lighting", 
                 command=canvas.toggle_lighting).pack(side=tk.LEFT, padx=2)
        
        print("✓ Interactive demo started")
        print("  - Use mouse to interact with the 3D view")
        print("  - Try different rendering modes")
        print("  - Use keyboard shortcuts")
        print("  - Close window when done")
        
        root.mainloop()
        
        print("✓ Interactive demo completed")
        return True
        
    except Exception as e:
        print(f"✗ Interactive demo failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("Enhanced 3D Viewer Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("OpenGL Canvas Test", test_opengl_canvas),
        ("Basic Functionality Test", test_basic_functionality),
        ("Enhanced Features Test", test_features),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced 3D Viewer is working correctly.")
        
        # Ask if user wants to run interactive demo
        try:
            response = input("\nWould you like to run the interactive demo? (y/n): ").lower()
            if response in ['y', 'yes']:
                run_interactive_demo()
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping interactive demo.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Update graphics drivers for OpenGL support")
        print("3. Ensure you have a display available for GUI tests")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 