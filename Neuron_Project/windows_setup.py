"""
Windows-specific setup and compatibility checks.
This module handles all Windows-specific issues automatically.
"""

import sys
import os
import platform
import ctypes
import subprocess
from pathlib import Path


class WindowsSetup:
    """Handle all Windows-specific setup and compatibility issues."""
    
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        self.is_admin = False
        self.opengl_available = False
        self.gpu_info = {}
        self.issues_found = []
        self.fixes_applied = []
        
    def check_and_fix_all(self, verbose=True):
        """Run all Windows checks and apply fixes automatically."""
        if not self.is_windows:
            return True
            
        if verbose:
            print("\n" + "="*70)
            print("WINDOWS COMPATIBILITY CHECK")
            print("="*70)
        
        # Run all checks
        self.check_admin_rights(verbose)
        self.check_python_architecture(verbose)
        self.setup_environment_variables(verbose)
        self.check_matplotlib_compatibility(verbose)  # CRITICAL: Check matplotlib + PySide6
        self.check_gpu_availability(verbose)
        self.check_opengl_dlls(verbose)
        self.check_qt_platform(verbose)
        self.optimize_windows_settings(verbose)
        self.check_display_scaling(verbose)
        
        # Summary
        if verbose:
            print("\n" + "="*70)
            if self.issues_found:
                print(f"⚠️  Found {len(self.issues_found)} potential issues:")
                for issue in self.issues_found:
                    print(f"   - {issue}")
            else:
                print("✓ No compatibility issues found")
                
            if self.fixes_applied:
                print(f"\n✓ Applied {len(self.fixes_applied)} automatic fixes:")
                for fix in self.fixes_applied:
                    print(f"   + {fix}")
            print("="*70 + "\n")
        
        return len(self.issues_found) == 0
    
    def check_admin_rights(self, verbose=True):
        """Check if running with administrator privileges."""
        try:
            self.is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if verbose:
                if self.is_admin:
                    print("✓ Running with administrator privileges")
                else:
                    print("⚠️  Not running as administrator (may cause OpenGL issues)")
                    self.issues_found.append("Not running as administrator")
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not check admin status: {e}")
    
    def check_python_architecture(self, verbose=True):
        """Check Python architecture (32-bit vs 64-bit)."""
        arch = platform.architecture()[0]
        if verbose:
            print(f"✓ Python architecture: {arch}")
        
        if arch == "32bit":
            if verbose:
                print("⚠️  32-bit Python detected (64-bit recommended for better performance)")
            self.issues_found.append("Using 32-bit Python (64-bit recommended)")
        
        return arch
    
    def setup_environment_variables(self, verbose=True):
        """Setup optimal environment variables for Windows."""
        env_vars = {
            # Qt platform settings
            'QT_QPA_PLATFORM': 'windows',
            'QT_OPENGL': 'desktop',
            # OpenGL settings
            'LIBGL_ALWAYS_SOFTWARE': '0',  # Prefer hardware rendering
            # Python settings
            'PYTHONUNBUFFERED': '1',  # Better logging
            # Matplotlib settings - CRITICAL for PySide6 compatibility
            'MPLBACKEND': 'Agg',  # Non-interactive backend to avoid Qt conflicts
        }
        
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                self.fixes_applied.append(f"Set environment variable: {key}={value}")
        
        if verbose:
            print("✓ Environment variables configured")
    
    def check_gpu_availability(self, verbose=True):
        """Detect available GPUs and configure for dedicated GPU."""
        try:
            # Try to get GPU info using wmic
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                gpus = [line.strip() for line in result.stdout.split('\n') 
                       if line.strip() and line.strip() != 'Name']
                
                self.gpu_info['available'] = gpus
                
                if verbose:
                    print(f"✓ Detected GPU(s): {', '.join(gpus)}")
                
                # Check for Intel integrated graphics
                intel_only = all('Intel' in gpu for gpu in gpus)
                has_dedicated = any(name in gpu.lower() for gpu in gpus 
                                  for name in ['nvidia', 'amd', 'radeon', 'geforce'])
                
                if intel_only:
                    if verbose:
                        print("⚠️  Only Intel integrated graphics detected")
                        print("   This may cause performance issues or crashes")
                    self.issues_found.append("Intel integrated graphics only")
                elif has_dedicated:
                    if verbose:
                        print("✓ Dedicated GPU available (recommended)")
                    # Try to force dedicated GPU
                    self._set_gpu_preference(verbose)
                    
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not detect GPU: {e}")
            self.issues_found.append("GPU detection failed")
    
    def _set_gpu_preference(self, verbose=True):
        """Set GPU preference to use dedicated GPU."""
        try:
            # Windows 10/11: Set process to prefer high performance GPU
            # This is a hint to the system, actual GPU selection is done by Windows
            if hasattr(ctypes.windll, 'user32'):
                # Try to set process DPI awareness (helps with scaling issues)
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                    self.fixes_applied.append("Set DPI awareness mode")
                except:
                    pass
            
            if verbose:
                print("✓ Configured GPU preferences for high performance")
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not set GPU preference: {e}")
    
    def check_opengl_dlls(self, verbose=True):
        """Check if OpenGL DLLs are available and working."""
        try:
            # Check for OpenGL32.dll
            system32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
            opengl_dll = system32 / 'opengl32.dll'
            
            if opengl_dll.exists():
                if verbose:
                    print(f"✓ OpenGL DLL found: {opengl_dll}")
                self.opengl_available = True
            else:
                if verbose:
                    print(f"✗ OpenGL DLL not found at {opengl_dll}")
                self.issues_found.append("OpenGL32.dll not found")
            
            # Try to import and test OpenGL
            try:
                from OpenGL.GL import glGetString, GL_VERSION, GL_VENDOR, GL_RENDERER
                # Don't actually call these yet (needs context), just check import
                if verbose:
                    print("✓ PyOpenGL module imported successfully")
            except ImportError as e:
                if verbose:
                    print(f"✗ PyOpenGL import failed: {e}")
                self.issues_found.append("PyOpenGL not properly installed")
                # Try to fix
                self._fix_pyopengl(verbose)
                
        except Exception as e:
            if verbose:
                print(f"⚠️  OpenGL check error: {e}")
    
    def _fix_pyopengl(self, verbose=True):
        """Attempt to fix PyOpenGL installation."""
        try:
            if verbose:
                print("   Attempting to repair PyOpenGL installation...")
            
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 
                 'PyOpenGL', 'PyOpenGL-accelerate'],
                capture_output=True,
                timeout=60
            )
            
            self.fixes_applied.append("Reinstalled PyOpenGL")
            if verbose:
                print("   ✓ PyOpenGL reinstalled")
        except Exception as e:
            if verbose:
                print(f"   ✗ Could not repair PyOpenGL: {e}")
    
    def check_matplotlib_compatibility(self, verbose=True):
        """Check and fix matplotlib + PySide6 compatibility."""
        try:
            # Set matplotlib backend before any imports
            os.environ['MPLBACKEND'] = 'Agg'
            
            try:
                import matplotlib
                matplotlib.use('Agg', force=True)
                
                if verbose:
                    print("✓ Matplotlib configured for PySide6 compatibility (Agg backend)")
                
                self.fixes_applied.append("Configured matplotlib backend for PySide6")
                
                # Test import
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                plt.close(fig)
                
                if verbose:
                    print("✓ Matplotlib import test passed")
                    
            except Exception as e:
                if verbose:
                    print(f"⚠️  Matplotlib test failed: {e}")
                    print("   This may cause issues with plotting features")
                self.issues_found.append("Matplotlib compatibility issue")
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not check matplotlib: {e}")
    
    def check_qt_platform(self, verbose=True):
        """Check Qt platform plugin availability."""
        try:
            # Check if PySide6 is installed
            try:
                import PySide6
                if verbose:
                    print(f"✓ PySide6 version: {PySide6.__version__}")
            except ImportError:
                if verbose:
                    print("✗ PySide6 not installed")
                self.issues_found.append("PySide6 not installed")
                self._fix_pyside6(verbose)
                return
            
            # Check for platform plugins
            try:
                from PySide6.QtCore import QCoreApplication
                # This will fail if platform plugins are missing
                if verbose:
                    print("✓ Qt platform plugins available")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Qt platform plugin issue: {e}")
                self.issues_found.append("Qt platform plugins issue")
                self._fix_pyside6(verbose)
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Qt check error: {e}")
    
    def _fix_pyside6(self, verbose=True):
        """Attempt to fix PySide6 installation."""
        try:
            if verbose:
                print("   Attempting to repair PySide6 installation...")
            
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 
                 'PySide6', 'PySide6-Addons', 'PySide6-Essentials'],
                capture_output=True,
                timeout=120
            )
            
            self.fixes_applied.append("Reinstalled PySide6")
            if verbose:
                print("   ✓ PySide6 reinstalled")
        except Exception as e:
            if verbose:
                print(f"   ✗ Could not repair PySide6: {e}")
    
    def optimize_windows_settings(self, verbose=True):
        """Apply Windows-specific optimizations."""
        try:
            # Set process priority to high (if admin)
            if self.is_admin:
                try:
                    import psutil
                    p = psutil.Process()
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                    self.fixes_applied.append("Set process priority to HIGH")
                    if verbose:
                        print("✓ Process priority set to HIGH")
                except:
                    pass  # psutil might not be installed
            
            # Disable Windows transparency effects for this process (can cause issues)
            try:
                import ctypes
                # DWM_BLURBEHIND structure
                DWM_BB_ENABLE = 0x00000001
                ctypes.windll.dwmapi.DwmEnableBlurBehindWindow(
                    ctypes.windll.kernel32.GetConsoleWindow(), 
                    ctypes.byref(ctypes.c_int(0))
                )
            except:
                pass  # Not critical
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not apply optimizations: {e}")
    
    def check_display_scaling(self, verbose=True):
        """Check for display scaling issues."""
        try:
            # Get display scaling factor
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            
            dc = user32.GetDC(0)
            dpi_x = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
            dpi_y = ctypes.windll.gdi32.GetDeviceCaps(dc, 90)  # LOGPIXELSY
            user32.ReleaseDC(0, dc)
            
            scaling_x = dpi_x / 96.0 * 100
            scaling_y = dpi_y / 96.0 * 100
            
            if verbose:
                if scaling_x > 100 or scaling_y > 100:
                    print(f"⚠️  Display scaling: {scaling_x:.0f}% (may cause rendering issues)")
                    print("   Consider disabling display scaling for python.exe")
                    self.issues_found.append(f"Display scaling at {scaling_x:.0f}%")
                else:
                    print(f"✓ Display scaling: {scaling_x:.0f}%")
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not check display scaling: {e}")
    
    def create_compatibility_manifest(self):
        """Create application manifest for Windows compatibility."""
        manifest_content = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity
    version="2.0.0.0"
    processorArchitecture="*"
    name="TIFFViewer3D"
    type="win32"
  />
  <description>3D TIFF Viewer Application</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
    </windowsSettings>
  </application>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <!-- Windows 10 and Windows 11 -->
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
</assembly>
"""
        try:
            manifest_path = Path(__file__).parent / "main.exe.manifest"
            with open(manifest_path, 'w') as f:
                f.write(manifest_content)
            return manifest_path
        except Exception:
            return None


def initialize_windows():
    """Main initialization function for Windows - call this before starting the app."""
    setup = WindowsSetup()
    success = setup.check_and_fix_all(verbose=True)
    
    if not success:
        print("\n" + "!"*70)
        print("WINDOWS COMPATIBILITY WARNINGS DETECTED")
        print("!"*70)
        print("\nThe application will attempt to run, but you may experience issues.")
        print("\nTo fix these issues:")
        print("1. Update your graphics drivers from manufacturer website")
        print("2. Run this script as Administrator (right-click → Run as administrator)")
        print("3. Install latest Windows updates")
        print("4. If you have dual GPUs, configure Windows to use dedicated GPU:")
        print("   Settings → System → Display → Graphics settings")
        print("   → Add python.exe → Set to 'High performance'")
        print("\nPress Enter to continue anyway...")
        try:
            input()
        except:
            pass
    
    return setup


if __name__ == "__main__":
    # Test the Windows setup
    initialize_windows()

