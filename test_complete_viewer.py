#!/usr/bin/env python3
"""
Test Complete 3D TIFF Viewer
Comprehensive testing script for the 3D TIFF viewer application.
"""

import numpy as np
import tkinter as tk
from tkinter import messagebox
import sys
import os
import traceback

def test_dependencies():
    """Test all required dependencies"""
    print("=== Testing Dependencies ===")
    
    dependencies = [
        ("tkinter", "tkinter"),
        ("numpy", "numpy"),
        ("OpenGL.GL", "PyOpenGL"),
        ("pyopengltk", "pyopengltk"),
        ("tifffile", "tifffile"),
        ("PIL", "Pillow")
    ]
    
    missing = []
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✓ All dependencies available")
    return True

def test_opengl():
    """Test OpenGL functionality"""
    print("\n=== Testing OpenGL ===")
    
    try:
        from OpenGL.GL import *
        from pyopengltk import OpenGLFrame
        
        # Create test window
        root = tk.Tk()
        root.title("OpenGL Test")
        root.geometry("400x300")
        
        class TestOpenGL(OpenGLFrame):
            def initgl(self):
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glEnable(GL_DEPTH_TEST)
                
            def redraw(self):
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glColor3f(1.0, 1.0, 1.0)
                glBegin(GL_TRIANGLES)
                glVertex3f(0.0, 0.5, 0.0)
                glVertex3f(-0.5, -0.5, 0.0)
                glVertex3f(0.5, -0.5, 0.0)
                glEnd()
        
        canvas = TestOpenGL(root, width=300, height=200)
        canvas.pack(expand=True, fill='both')
        
        # Test for a short time
        root.after(1000, root.destroy)
        root.mainloop()
        
        print("✓ OpenGL test passed")
        return True
        
    except Exception as e:
        print(f"✗ OpenGL test failed: {e}")
        return False

def create_test_tiff():
    """Create a test TIFF file"""
    print("\n=== Creating Test TIFF ===")
    
    try:
        import tifffile
        
        # Create 3D test data
        depth, height, width = 20, 50, 50
        data = np.zeros((depth, height, width), dtype=np.uint8)
        
        # Create some interesting patterns
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    # Create a sphere
                    center = np.array([width//2, height//2, depth//2])
                    pos = np.array([x, y, z])
                    distance = np.linalg.norm(pos - center)
                    
                    if distance < 15:
                        data[z, y, x] = int(distance * 255 / 15)
                    
                    # Add some noise
                    if np.random.random() < 0.1:
                        data[z, y, x] = np.random.randint(50, 255)
        
        # Save test TIFF
        test_file = "test_data.tiff"
        tifffile.imwrite(test_file, data)
        
        print(f"✓ Created test TIFF: {test_file}")
        print(f"  Shape: {data.shape}")
        print(f"  Data range: {data.min()} to {data.max()}")
        print(f"  Non-zero voxels: {np.sum(data > 0)}")
        
        return test_file
        
    except Exception as e:
        print(f"✗ Failed to create test TIFF: {e}")
        return None

def test_tiff_processor():
    """Test the TIFF processor"""
    print("\n=== Testing TIFF Processor ===")
    
    try:
        from enhanced_tiff_processor import EnhancedTIFFProcessor
        
        # Create test data
        test_data = np.random.randint(0, 5, (10, 20, 20), dtype=np.uint8)
        
        processor = EnhancedTIFFProcessor()
        
        # Test point cloud extraction
        points, colors = processor.extract_point_cloud(test_data, max_points=1000)
        
        print(f"✓ Extracted {len(points)} points")
        print(f"  Points shape: {points.shape}")
        print(f"  Colors shape: {colors.shape}")
        print(f"  Point range: {points.min(axis=0)} to {points.max(axis=0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ TIFF processor test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_viewer():
    """Test the complete viewer application"""
    print("\n=== Testing Complete Viewer ===")
    
    try:
        from complete_3d_tiff_viewer import TIFFViewerApp
        
        print("Creating viewer application...")
        app = TIFFViewerApp()
        
        # Test for a short time
        app.root.after(2000, app.root.destroy)
        
        print("✓ Viewer application created successfully")
        print("Starting test run...")
        
        app.run()
        
        print("✓ Viewer test completed")
        return True
        
    except Exception as e:
        print(f"✗ Viewer test failed: {e}")
        traceback.print_exc()
        return False

def run_interactive_test():
    """Run interactive test with user"""
    print("\n=== Interactive Test ===")
    
    try:
        # Create test TIFF
        test_file = create_test_tiff()
        if not test_file:
            return False
        
        # Start viewer
        from complete_3d_tiff_viewer import TIFFViewerApp
        
        app = TIFFViewerApp()
        
        # Show instructions
        instructions = f"""
Interactive Test Instructions:

1. The 3D TIFF Viewer should now be open
2. Click "Open TIFF File" and select: {test_file}
3. You should see a 3D visualization of a sphere
4. Test the controls:
   • Left mouse: Rotate
   • Middle mouse: Pan
   • Right mouse: Zoom
   • Mouse wheel: Scale
   • Double-click: Reset view
5. Try the checkboxes and sliders
6. Close the window when done

The test file will be cleaned up automatically.
"""
        
        print(instructions)
        
        # Run the application
        app.run()
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✓ Cleaned up test file: {test_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Interactive test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔬 Complete 3D TIFF Viewer Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("OpenGL", test_opengl),
        ("TIFF Processor", test_tiff_processor),
        ("Complete Viewer", test_complete_viewer)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The 3D TIFF viewer is ready to use.")
        
        # Ask if user wants interactive test
        try:
            response = input("\nRun interactive test? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_interactive_test()
        except KeyboardInterrupt:
            print("\nSkipping interactive test.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        # Show troubleshooting tips
        print("\n🔧 Troubleshooting Tips:")
        if not results.get("Dependencies", True):
            print("• Install missing dependencies with pip")
        if not results.get("OpenGL", True):
            print("• Update graphics drivers")
            print("• Try: pip install --upgrade PyOpenGL PyOpenGL-accelerate")
        if not results.get("TIFF Processor", True):
            print("• Check tifffile installation: pip install tifffile")
        if not results.get("Complete Viewer", True):
            print("• Check all dependencies are installed")
            print("• Try running with administrator privileges")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main()) 