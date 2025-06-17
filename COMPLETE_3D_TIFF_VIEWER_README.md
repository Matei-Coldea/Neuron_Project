# Complete 3D TIFF Viewer

A comprehensive, standalone 3D TIFF image viewer with full visualization capabilities, built with Python, OpenGL, and Tkinter.

## 🌟 Features

### Core Capabilities
- **Full 3D Visualization**: Interactive 3D rendering of TIFF image stacks
- **Multiple Format Support**: Handles various TIFF formats (standard, scientific, multi-frame)
- **Real-time Interaction**: Smooth rotation, panning, and zooming
- **Color Palette Support**: Automatic detection and use of embedded color palettes
- **Performance Optimization**: Efficient handling of large datasets with downsampling

### Visualization Features
- **Point Cloud Rendering**: High-quality 3D point cloud visualization
- **Coordinate System**: RGB-colored axes (X=red, Y=green, Z=blue)
- **Reference Grid**: Optional grid for spatial reference
- **Bounding Box**: Visual bounds around your data
- **Multiple View Modes**: Perspective and orthographic projection

### Interactive Controls
- **Mouse Controls**:
  - Left button: Rotate view
  - Middle button: Pan view
  - Right button: Zoom in/out
  - Mouse wheel: Scale objects
  - Double-click: Reset view
- **Keyboard Shortcuts**:
  - `R`: Reset view
  - `A`: Toggle axes
  - `G`: Toggle grid
  - `B`: Toggle bounding box
  - `+/-`: Adjust point size

### Performance Features
- **Smart Downsampling**: Automatic point reduction for large datasets
- **Configurable Limits**: Adjustable maximum point counts
- **Memory Optimization**: Efficient memory usage for large files
- **Background Processing**: Non-blocking file loading

## 📋 Requirements

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.7 or higher
- **Graphics**: OpenGL 2.1+ compatible graphics card
- **Memory**: 4GB RAM minimum (8GB+ recommended for large files)

### Python Dependencies
```bash
pip install numpy PyOpenGL pyopengltk tifffile Pillow
```

## 🚀 Installation

### Quick Install
1. **Download the files**:
   - `complete_3d_tiff_viewer.py`
   - `enhanced_tiff_processor.py`
   - `test_complete_viewer.py`

2. **Install dependencies**:
   ```bash
   pip install numpy PyOpenGL pyopengltk tifffile Pillow
   ```

3. **Test the installation**:
   ```bash
   python test_complete_viewer.py
   ```

4. **Run the viewer**:
   ```bash
   python complete_3d_tiff_viewer.py
   ```

### Detailed Installation

#### Step 1: Python Environment
Ensure you have Python 3.7+ installed:
```bash
python --version
```

#### Step 2: Install Dependencies
```bash
# Core dependencies
pip install numpy

# OpenGL support
pip install PyOpenGL PyOpenGL-accelerate pyopengltk

# Image processing
pip install tifffile Pillow

# Optional: for better performance
pip install numba
```

#### Step 3: Graphics Drivers
- **Windows**: Update graphics drivers through Device Manager or manufacturer website
- **macOS**: Graphics drivers are typically up-to-date
- **Linux**: Install appropriate drivers for your graphics card

#### Step 4: Verify Installation
Run the test suite:
```bash
python test_complete_viewer.py
```

## 📖 Usage

### Basic Usage

1. **Start the application**:
   ```bash
   python complete_3d_tiff_viewer.py
   ```

2. **Open a TIFF file**:
   - Click "Open TIFF File" button
   - Or use File → Open TIFF... menu

3. **Interact with the 3D view**:
   - Drag with left mouse to rotate
   - Drag with middle mouse to pan
   - Use mouse wheel to zoom
   - Double-click to reset view

### Advanced Features

#### Performance Tuning
- **Max Points**: Adjust the maximum number of points to render
- **Point Size**: Change point size for better visibility
- **Display Options**: Toggle axes, grid, and bounding box

#### File Formats Supported
- **Standard TIFF**: Single and multi-frame TIFF files
- **Scientific TIFF**: Files with embedded metadata
- **Indexed Color TIFF**: Files with color palettes
- **Grayscale TIFF**: 8-bit and 16-bit grayscale images

### Example Workflow

1. **Load your TIFF file**
2. **Adjust performance settings** if needed (for large files)
3. **Explore your data** using mouse controls
4. **Customize the view** with display options
5. **Use keyboard shortcuts** for quick adjustments

## 🔧 Configuration

### Performance Settings
```python
# In the application, adjust these settings:
max_points = 500000      # Maximum points to render
point_size = 3.0         # Size of rendered points
downsample_factor = 1    # Downsampling factor for large files
```

### Display Settings
```python
show_axes = True         # Show coordinate axes
show_grid = True         # Show reference grid
show_bounding_box = True # Show data bounds
background_color = (0.1, 0.1, 0.1, 1.0)  # Background color (RGBA)
```

## 🐛 Troubleshooting

### Common Issues

#### "OpenGL Error" or Black Screen
**Cause**: Graphics driver or OpenGL compatibility issues
**Solutions**:
1. Update graphics drivers
2. Try: `pip install --upgrade PyOpenGL PyOpenGL-accelerate`
3. Run with administrator privileges (Windows)
4. Check OpenGL version: should be 2.1+

#### "Failed to load TIFF file"
**Cause**: Unsupported TIFF format or corrupted file
**Solutions**:
1. Verify file is a valid TIFF
2. Try opening with another image viewer first
3. Check file permissions
4. For large files, increase available memory

#### Slow Performance
**Cause**: Large dataset or insufficient hardware
**Solutions**:
1. Reduce "Max Points" setting
2. Increase downsampling factor
3. Close other applications
4. Use a computer with better graphics card

#### Application Crashes
**Cause**: Memory issues or missing dependencies
**Solutions**:
1. Run the test suite: `python test_complete_viewer.py`
2. Check all dependencies are installed
3. Monitor memory usage
4. Try with smaller files first

### Error Messages

#### "Missing dependency: [package]"
```bash
pip install [package]
```

#### "Could not load TIFF: [error]"
- Check file format and integrity
- Try converting to standard TIFF format
- Verify file permissions

#### "OpenGL context creation failed"
- Update graphics drivers
- Check OpenGL support
- Try running on different hardware

## 🧪 Testing

### Run Test Suite
```bash
python test_complete_viewer.py
```

### Test Components
1. **Dependencies Test**: Verifies all required packages
2. **OpenGL Test**: Tests graphics functionality
3. **TIFF Processor Test**: Tests file loading capabilities
4. **Complete Viewer Test**: Tests full application
5. **Interactive Test**: Manual testing with sample data

### Create Test Data
The test suite automatically creates sample TIFF files for testing.

## 📊 Performance Guidelines

### File Size Recommendations
- **Small files** (< 50MB): No optimization needed
- **Medium files** (50-500MB): Adjust max points to 100K-500K
- **Large files** (> 500MB): Use downsampling, limit to 50K-100K points

### Hardware Recommendations
- **Minimum**: Intel HD Graphics, 4GB RAM
- **Recommended**: Dedicated graphics card, 8GB+ RAM
- **Optimal**: Gaming/workstation graphics card, 16GB+ RAM

## 🔬 Technical Details

### Architecture
- **Frontend**: Tkinter GUI with custom OpenGL canvas
- **Rendering**: OpenGL-based 3D visualization
- **Data Processing**: NumPy-based efficient array operations
- **File Handling**: Multiple backend support (tifffile, PIL)

### Data Flow
1. **File Loading**: Multi-format TIFF loader
2. **Data Processing**: Conversion to 3D point cloud
3. **Rendering**: OpenGL-based visualization
4. **Interaction**: Real-time view manipulation

### Supported Data Types
- **8-bit unsigned**: Standard grayscale/color
- **16-bit unsigned**: High dynamic range
- **32-bit float**: Scientific data
- **Boolean**: Binary masks

## 🤝 Contributing

### Development Setup
1. Clone/download the source files
2. Install development dependencies
3. Run tests to verify setup
4. Make your changes
5. Test thoroughly

### Code Structure
- `complete_3d_tiff_viewer.py`: Main application
- `enhanced_tiff_processor.py`: TIFF file handling
- `test_complete_viewer.py`: Test suite

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

### Getting Help
1. **Run the test suite** to identify issues
2. **Check the troubleshooting section** above
3. **Verify your system meets requirements**
4. **Try with sample data** first

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Graphics card information
- Error messages (full traceback)
- Sample file (if possible)

## 🎯 Quick Start Checklist

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install numpy PyOpenGL pyopengltk tifffile Pillow`)
- [ ] Graphics drivers updated
- [ ] Test suite passes (`python test_complete_viewer.py`)
- [ ] Application starts (`python complete_3d_tiff_viewer.py`)
- [ ] Sample TIFF file loads successfully
- [ ] 3D interaction works (mouse controls)

## 🌈 Example Use Cases

### Scientific Imaging
- **Microscopy**: Visualize 3D cell structures
- **Medical Imaging**: Explore volumetric scan data
- **Materials Science**: Analyze 3D material structures

### Data Analysis
- **Quality Control**: Visual inspection of 3D data
- **Pattern Recognition**: Identify structures in 3D space
- **Measurement**: Spatial analysis of features

### Education
- **Teaching Tool**: Demonstrate 3D data concepts
- **Research**: Explore scientific datasets
- **Visualization**: Create compelling 3D presentations

---

**Ready to explore your 3D TIFF data? Start with the test suite and then dive into your own datasets!** 🚀 