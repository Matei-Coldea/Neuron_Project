# 3D TIFF Viewer Troubleshooting Guide

## Problem: Program doesn't display any figure

This troubleshooting guide will help you diagnose and fix issues where the 3D TIFF viewer application starts but doesn't show any 3D visualization.

## Step 1: Check Dependencies

First, ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

If you encounter issues, try installing dependencies individually:

```bash
pip install numpy matplotlib Pillow tifffile
pip install PyOpenGL PyOpenGL-accelerate
pip install pyopengltk
```

## Step 2: Run the Simple Test

Test the OpenGL canvas functionality independently:

```bash
python test_opengl_simple.py
```

**Expected Result:** A window should open showing a colorful 3D point cloud with rainbow colors.

**If this fails:** The issue is with OpenGL setup. See "OpenGL Issues" section below.

**If this works:** The issue is with data loading or main application integration.

## Step 3: Run the Debug Script

Run the comprehensive debug script:

```bash
python debug_main.py
```

This will test:
- All imports
- Basic OpenGL functionality  
- Main application startup

## Step 4: Check for Common Issues

### OpenGL Issues

**Symptoms:**
- Error messages about OpenGL
- Black/empty canvas
- Application crashes when trying to display 3D

**Solutions:**
1. **Update Graphics Drivers**
   - Windows: Update through Device Manager or manufacturer website
   - Linux: Install mesa-utils, update graphics drivers
   - macOS: Keep system updated

2. **Check OpenGL Version**
   ```python
   from OpenGL.GL import *
   import tkinter as tk
   from pyopengltk import OpenGLFrame
   
   root = tk.Tk()
   canvas = OpenGLFrame(root)
   canvas.pack()
   root.after(100, lambda: print(f"OpenGL Version: {glGetString(GL_VERSION)}"))
   root.after(200, root.destroy)
   root.mainloop()
   ```

3. **Try Software Rendering (if hardware fails)**
   ```bash
   # On Linux/macOS
   export LIBGL_ALWAYS_SOFTWARE=1
   python main.py
   
   # On Windows (PowerShell)
   $env:LIBGL_ALWAYS_SOFTWARE=1
   python main.py
   ```

### Data Loading Issues

**Symptoms:**
- Window opens but canvas is blank
- No error messages
- Console shows "Setting points: 0 points"

**Solutions:**
1. **Check TIFF file format**
   - Ensure file is a valid multi-frame TIFF
   - Check that file contains indexed color data (palette mode)

2. **Verify file path**
   - Make sure file path is correct
   - Check file permissions

3. **Test with sample data**
   - Use the test script to verify canvas works
   - Try with a different TIFF file

### Memory Issues

**Symptoms:**
- Application starts but becomes unresponsive
- System runs out of memory
- Very slow performance

**Solutions:**
1. **Adjust performance settings**
   - Increase downsample factor (1-10)
   - Reduce max points (10k-100k)
   - Lower render quality (1-3)

2. **Check file size**
   - Large TIFF files may need downsampling
   - Consider processing in chunks

### UI Integration Issues

**Symptoms:**
- Canvas appears but no data is shown
- Controls don't work
- Canvas doesn't respond to mouse

**Solutions:**
1. **Check data flow**
   - Look for console output from `set_points` method
   - Verify that `create_voxel_plot` is being called

2. **Test individual components**
   ```python
   # Test canvas independently
   python test_opengl_simple.py
   
   # Test main application
   python debug_main.py
   ```

## Step 5: Enable Debug Mode

Add debugging to see what's happening:

1. **Edit main.py** to add debug output:
```python
print("Starting application...")
try:
    viewer = TIFFViewer3D()
    print("TIFFViewer3D created successfully")
except Exception as e:
    print(f"Error creating viewer: {e}")
    import traceback
    traceback.print_exc()
```

2. **Check console output** for error messages and debug information.

## Step 6: Platform-Specific Issues

### Windows
- Ensure Visual C++ Redistributable is installed
- Try running as administrator
- Check Windows graphics settings (prefer high-performance GPU)

### Linux
- Install required packages: `sudo apt-get install python3-opengl mesa-utils`
- Check X11 forwarding if using SSH
- Verify display environment: `echo $DISPLAY`

### macOS
- Install Xcode command line tools: `xcode-select --install`
- Check system OpenGL support
- May need to adjust security settings for GPU access

## Step 7: Alternative Solutions

If OpenGL continues to fail, you can fall back to matplotlib visualization:

1. **Temporary workaround:** Use the original matplotlib-based visualization by reverting to older version
2. **Software rendering:** Force software OpenGL rendering (slower but more compatible)
3. **Remote display:** Use VNC or remote desktop if running on a server

## Getting Help

If none of these solutions work, please provide:

1. **System information:**
   - Operating system and version
   - Python version
   - Graphics card information
   - OpenGL version (if available)

2. **Error messages:**
   - Complete error output from console
   - Results from debug_main.py
   - Any crash logs

3. **File information:**
   - TIFF file size and properties
   - Sample file (if possible)

4. **Steps to reproduce:**
   - Exact commands run
   - Expected vs actual behavior

## Quick Fix Checklist

✅ **Try these quick fixes first:**

1. `pip install --upgrade PyOpenGL PyOpenGL-accelerate pyopengltk`
2. Update graphics drivers
3. Run `python test_opengl_simple.py`
4. Try with smaller TIFF file
5. Increase downsample factor in performance settings
6. Check console for error messages
7. Try software rendering mode

---

**Still having issues?** The enhanced 3D viewer includes extensive debugging output. Check the console for detailed error messages and diagnostic information. 