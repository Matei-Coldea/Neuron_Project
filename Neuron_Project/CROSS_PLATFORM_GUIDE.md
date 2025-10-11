# 3D TIFF Viewer - Cross-Platform Installation Guide

This guide ensures the application runs smoothly on Windows, macOS, and Linux.

## Quick Start

### Windows

**Option 1: Using Batch Script (Recommended)**
```cmd
run_windows.bat
```

**Option 2: Using PowerShell**
```powershell
powershell -ExecutionPolicy Bypass -File run_windows.ps1
```

**Option 3: Manual**
```cmd
venv\Scripts\activate
python main.py
```

### macOS / Linux

**Option 1: Using Shell Script (Recommended)**
```bash
./run_unix.sh
```

**Option 2: Manual**
```bash
source venv/bin/activate
python main.py
```

---

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **Graphics drivers** (must be up to date)
- **OpenGL 2.1 or higher**

### Step 1: Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Platform-Specific Issues & Solutions

### Windows

#### Issue 1: Application Crashes Immediately
**Causes:**
- Outdated graphics drivers
- Display scaling issues
- Dual GPU configuration

**Solutions:**
1. **Update Graphics Drivers:**
   - Intel: [Intel Driver & Support Assistant](https://www.intel.com/content/www/us/en/support/detect.html)
   - NVIDIA: [GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/)
   - AMD: [AMD Driver Auto-Detect](https://www.amd.com/en/support)

2. **Disable Display Scaling:**
   - Right-click `python.exe` in `venv\Scripts\`
   - Properties → Compatibility tab
   - Check "Override high DPI scaling behavior"
   - Select "Application" from dropdown

3. **Force Dedicated GPU (Dual GPU Systems):**
   - Open Windows Settings → System → Display → Graphics settings
   - Browse to your `python.exe` in the venv folder
   - Set to "High performance"

4. **Run as Administrator:**
   - Right-click `run_windows.bat`
   - Select "Run as administrator"

#### Issue 2: Import Error or Module Not Found
**Solution:**
```cmd
venv\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

#### Issue 3: OpenGL Error
**Solution:**
1. Check OpenGL version:
   ```cmd
   python -c "from OpenGL.GL import *; import ctypes; print(glGetString(GL_VERSION))"
   ```
2. If error, update graphics drivers
3. Ensure you're using dedicated GPU (not integrated Intel)

---

### macOS

#### Issue 1: Command Not Found - python
**Solution:**
```bash
# Use python3 instead
python3 -m venv venv
source venv/bin/activate
python3 main.py
```

#### Issue 2: Xcode Command Line Tools Missing
**Solution:**
```bash
xcode-select --install
```

#### Issue 3: Permission Denied
**Solution:**
```bash
chmod +x run_unix.sh
./run_unix.sh
```

#### Issue 4: Security Warning
**Solution:**
1. Go to System Preferences → Security & Privacy
2. Click "Open Anyway" for the blocked application
3. Or run: `sudo spctl --master-disable` (not recommended)

#### Issue 5: Application Appears But OpenGL Fails
**Solution:**
- macOS uses older OpenGL version (4.1)
- Ensure PySide6 is latest version:
  ```bash
  pip install --upgrade PySide6
  ```

---

### Linux

#### Issue 1: No Display or "Cannot open display"
**Causes:**
- X11 not running
- DISPLAY variable not set
- Running over SSH without X forwarding

**Solutions:**
1. **Check X11:**
   ```bash
   echo $DISPLAY
   # Should show something like :0 or :1
   ```

2. **Start X11:**
   ```bash
   startx
   ```

3. **For SSH:**
   ```bash
   ssh -X user@host
   # or
   export DISPLAY=:0
   ```

4. **For Wayland:**
   ```bash
   # Install XWayland
   sudo apt-get install xwayland
   ```

#### Issue 2: OpenGL Libraries Missing
**Solution:**

**Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglu1-mesa
sudo apt-get install mesa-utils  # For glxinfo
```

**Fedora/RHEL:**
```bash
sudo dnf install mesa-libGL mesa-libGLU
```

**Arch:**
```bash
sudo pacman -S mesa glu
```

#### Issue 3: Verify OpenGL Installation
```bash
glxinfo | grep "OpenGL version"
# Should show version 2.1 or higher
```

#### Issue 4: Python Tkinter Missing (if applicable)
```bash
# Debian/Ubuntu
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter
```

#### Issue 5: Qt Platform Plugin Error
**Error:** `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`

**Solution:**
```bash
# Install missing libraries
sudo apt-get install libxcb-xinerama0 libxcb-cursor0

# Or reinstall Qt
pip install --upgrade --force-reinstall PySide6
```

---

## Verifying Installation

### Test 1: Check Python and Packages
```bash
python --version
python -c "import PySide6; print('PySide6:', PySide6.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import OpenGL; print('PyOpenGL: OK')"
```

### Test 2: Check OpenGL (All Platforms)
```bash
python -c "from PySide6.QtWidgets import QApplication; from qt_opengl_canvas import QtAdvanced3DCanvas; app = QApplication([]); canvas = QtAdvanced3DCanvas(); print('OpenGL test passed')"
```

### Test 3: Run Main Application
```bash
python main.py
```

---

## Performance Optimization

### For Large TIFF Files

1. **Increase Downsampling:**
   - In the UI: Open Figure → Performance Settings → Downsample
   - Set to 2-4 for very large files

2. **Reduce Max Points:**
   - Performance Settings → Max points (k)
   - Lower value = faster rendering

3. **Disable Background Preprocessing:**
   - Uncheck "Background preprocessing" for faster initial load

### For Better Quality

1. **Reduce Downsampling to 1**
2. **Increase Max Points to 500+**
3. **Use Legacy Edge Detection** (slower but more accurate)

---

## Common Error Messages

### "numpy.dtype size changed"
**Windows/Linux:**
```bash
pip install --upgrade numpy scikit-image --no-cache-dir
```

### "Failed to create OpenGL context"
- Update graphics drivers
- Check OpenGL version compatibility
- Try different Qt platform plugin (Windows):
  ```cmd
  set QT_QPA_PLATFORM=windows
  python main.py
  ```

### "QMetaObject.invokeMethod called with wrong argument types"
This has been fixed in the latest version. If you still see it:
```bash
pip install --upgrade PySide6
```

---

## Development Notes

### Adding New Features
- All path handling uses `os.path.normpath()` for cross-platform compatibility
- Qt signals/slots are preferred over direct thread calls
- OpenGL errors are caught and logged with platform-specific hints

### Testing on Multiple Platforms
1. Test on Windows 10/11 with both integrated and dedicated GPU
2. Test on macOS 12+ (Monterey or later)
3. Test on Ubuntu 20.04+ and Fedora 35+

---

## Getting Help

### Diagnostic Information
When reporting issues, include:
```bash
python --version
pip list | grep -i "pyside6\|numpy\|opengl"
# Windows: wmic path win32_VideoController get name
# macOS: system_profiler SPDisplaysDataType
# Linux: lspci | grep VGA
```

### Application Logs
Run with verbose output:
```bash
python main.py 2>&1 | tee app_log.txt
```

---

## License & Credits

See main README.md for license and attribution information.

