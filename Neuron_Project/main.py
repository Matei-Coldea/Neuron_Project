
import sys
import traceback
import platform
import os

# CRITICAL: Set matplotlib backend BEFORE any imports (especially for VS Code debug mode)
# This must happen at module level, before matplotlib_config or any other imports
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('PYTHONUNBUFFERED', '1')

# Additional Windows environment setup for debug mode
if platform.system() == 'Windows':
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
    os.environ.setdefault('QT_OPENGL', 'desktop')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '0')

# CRITICAL: Configure matplotlib BEFORE any other imports that might use it
# This prevents PySide6 + matplotlib Qt backend conflicts on Windows
try:
    import matplotlib_config
except ImportError:
    print("⚠️  matplotlib_config not found, matplotlib may have issues")
except Exception as e:
    print(f"⚠️  matplotlib_config error: {e}, continuing anyway...")

def initialize_windows_environment():
    """Initialize Windows-specific environment before anything else."""
    if platform.system() != 'Windows':
        return True
    
    try:
        # Import Windows setup module
        from windows_setup import initialize_windows
        setup = initialize_windows()
        return True
    except Exception as e:
        print(f"⚠️  Windows initialization warning: {e}")
        print("Continuing anyway...")
        return True

def check_opengl_support():
    """Check if OpenGL is available on the system."""
    try:
        from OpenGL.GL import glGetString, GL_VERSION, GL_RENDERER
        # Try to get version info (this will fail if OpenGL isn't properly set up)
        print("\n=== OpenGL Support Check ===")
        print(f"Platform: {platform.system()} {platform.release()}")
        
        if platform.system() == 'Windows':
            print("Windows detected - checking for common issues...")
            print("Note: If you have dual GPUs, ensure Python is using the dedicated GPU")
            print("Note: Update your graphics drivers if you encounter crashes")
        
        return True
    except Exception as e:
        print(f"\n⚠️  OpenGL check warning: {e}")
        print("This may indicate graphics driver issues.")
        if platform.system() == 'Windows':
            print("\nWindows troubleshooting:")
            print("1. Update graphics drivers (Intel/NVIDIA/AMD)")
            print("2. Run as administrator")
            print("3. Check Windows Update for driver updates")
            print("4. Disable display scaling for this app")
        return True  # Don't block startup, just warn

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
    print(f"Platform: {platform.system()} {platform.release()}")
    print("Enhanced with full 3D OpenGL capabilities")
    print("Controls: Left mouse=rotate, Middle=pan, Right/wheel=zoom")
    print("Keyboard: R=reset, P/W/S=render modes, A/G/L=toggle features")
    print()
    
    # Windows-specific initialization (must be first)
    if platform.system() == 'Windows':
        print("\n[Windows] Running compatibility checks...")
        initialize_windows_environment()
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        return 1
    
    # Check OpenGL support
    check_opengl_support()
    
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
        print("\n=== Platform-Specific Troubleshooting ===")
        
        system = platform.system()
        if system == 'Windows':
            print("\nWindows-specific solutions:")
            print("1. Update graphics drivers (NVIDIA/AMD/Intel)")
            print("2. Run as administrator")
            print("3. Right-click app → Properties → Compatibility → Disable display scaling")
            print("4. Check Windows Update for driver updates")
            print("5. If dual GPU: Force use of dedicated GPU in graphics settings")
            print("6. Try: pip install --upgrade --force-reinstall PySide6")
        elif system == 'Linux':
            print("\nLinux-specific solutions:")
            print("1. Install OpenGL libraries: sudo apt-get install libgl1-mesa-glx libglu1-mesa")
            print("2. Check: glxinfo | grep 'OpenGL version'")
            print("3. Ensure X11 is running (if using Wayland, try XWayland)")
            print("4. Update Mesa drivers")
        elif system == 'Darwin':  # macOS
            print("\nmacOS-specific solutions:")
            print("1. Ensure Xcode Command Line Tools are installed")
            print("2. Check System Preferences → Security & Privacy")
            print("3. Try: pip install --upgrade --force-reinstall PySide6")
        
        print("\nGeneral troubleshooting:")
        print("• Ensure graphics drivers are up to date")
        print("• Try reinstalling dependencies: pip install -r requirements.txt --force-reinstall")
        print("• Check if running in virtual environment")
        
        input("\nPress Enter to exit...")
        return 1
    
    return 0


if __name__ == "__main__":
    # VS Code debug mode compatibility
    # Ensure environment is set even if .env file wasn't loaded
    if 'MPLBACKEND' not in os.environ or os.environ['MPLBACKEND'] != 'Agg':
        print("⚠️  MPLBACKEND not set (VS Code debug mode), setting now...")
        os.environ['MPLBACKEND'] = 'Agg'
        
        # Force matplotlib to use Agg if already imported
        try:
            import matplotlib
            if matplotlib.get_backend() != 'Agg':
                matplotlib.use('Agg', force=True)
                print("✓ Forced matplotlib backend to Agg")
        except:
            pass
    
    exit(main()) 


# some functions from the file are really slow, should we still use them ?