
import sys
import traceback

def check_dependencies():
   
    print("Checking dependencies...")
    
    required_modules = [
        ("PySide6", "PySide6"),
        ("numpy", "numpy"),
        ("OpenGL.GL", "PyOpenGL"),
        ("PIL", "Pillow"),
        ("tifffile", "tifffile"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
        ("matplotlib", "matplotlib"),
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
        print("Importing QtTIFFViewer3D...")
        from qt_tiff_viewer import launch_qt_viewer
        print("✓ QtTIFFViewer3D imported successfully")

        print("Launching Qt viewer...")
        launch_qt_viewer()
        print("✓ Qt viewer closed")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install missing dependencies:")
        print("pip install -r requirements.txt")
        print("\nFor troubleshooting, run:")
        print("python -m Neuron_Project.main")
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


# some functions from the file are really slow, should we still use them ?