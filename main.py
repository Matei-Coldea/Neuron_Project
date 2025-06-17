#!/usr/bin/env python3
"""
3D TIFF Viewer Application

This is the main entry point for the 3D TIFF Viewer application.
It provides functionality for viewing, analyzing, and processing 3D TIFF images
with color-based spine analysis and metadata management.

Author: [Your Name]
Version: 1.0
"""

import sys
import traceback

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required_modules = [
        ("tkinter", "tkinter (usually included with Python)"),
        ("numpy", "numpy"),
        ("OpenGL.GL", "PyOpenGL"),
        ("pyopengltk", "pyopengltk"),
        ("PIL", "Pillow"),
        ("tifffile", "tifffile"),
    ]
    
    missing = []
    for module, description in required_modules:
        try:
            __import__(module)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description}")
            missing.append(description)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Please install them with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies available")
    return True

def main():
    """
    Main entry point for the 3D TIFF Viewer application.
    """
    print("=== 3D TIFF Viewer Starting ===")
    print("Enhanced with full 3D OpenGL capabilities")
    print("Controls: Left mouse=rotate, Middle=pan, Right/wheel=zoom")
    print("Keyboard: R=reset, P/W/S=render modes, A/G/L=toggle features")
    print()
    
    # Check dependencies first
    if not check_dependencies():
        input("Press Enter to exit...")
        return 1
    
    try:
        print("Importing TIFFViewer3D...")
        from tiff_viewer_3d import TIFFViewer3D
        print("✓ TIFFViewer3D imported successfully")
        
        print("Creating TIFFViewer3D instance...")
        # Create and run the TIFF viewer
        viewer = TIFFViewer3D()
        print("✓ TIFFViewer3D created successfully")
        print("✓ Application window should now be visible")
        print("✓ Upload a TIFF file to begin 3D visualization")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install missing dependencies:")
        print("pip install -r requirements.txt")
        print("\nFor troubleshooting, run:")
        print("python test_fixed_viewer.py")
        input("Press Enter to exit...")
        return 1
    except Exception as e:
        print(f"✗ Error starting application: {e}")
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Run 'python test_fixed_viewer.py' to test OpenGL")
        print("2. Run 'python debug_main.py' for detailed diagnostics")
        print("3. Check TROUBLESHOOTING.md for common solutions")
        print("4. Ensure graphics drivers are up to date")
        print("5. Try running as administrator (Windows)")
        input("Press Enter to exit...")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 