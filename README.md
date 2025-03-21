# 3D TIFF Viewer

## Introduction

The 3D TIFF Viewer is a sophisticated application designed for visualizing and analyzing 3D TIFF images, with a particular focus on neuron spine analysis. This tool allows users to load multilayer TIFF files, visualize them in 3D, select specific colors (representing different structures), and perform various calculations on the selected structures.

## Installation

### Requirements
- Python 3.7+
- Required packages:
  - tkinter
  - numpy
  - matplotlib
  - Pillow (PIL)
  - tifffile

### Installing Dependencies
```bash
pip install numpy matplotlib pillow tifffile
```

Note: tkinter usually comes with Python installation, but if needed:
- Windows: Included with Python
- macOS: `brew install python-tk`
- Linux: `sudo apt-get install python3-tk`

### Running the Application
1. Clone or download this repository
2. Navigate to the directory containing the code
3. Run the application:
```bash
python neuron3.py
```

## Core Functionality

- Load and visualize multilayered TIFF files in 3D
- Extract and display unique colors from the TIFF palette
- Select specific colors/structures for detailed analysis
- Calculate metadata for selected structures (volume, dimensions, etc.)
- Save selected structures as separate TIFF files with metadata

## Performance Settings

The application includes several performance optimization settings that allow it to handle large datasets efficiently:

### Downsample Factor
- Reduces the resolution of the displayed 3D data by skipping voxels
- Higher values (1-10) result in faster rendering but lower detail
- Particularly useful for very large datasets that would otherwise be too memory-intensive

### Render Quality
- Controls the visual quality of the 3D rendering (1-5)
- Affects point size, transparency, and DPI of the rendering
- Lower quality settings are faster and use less memory

### Max Points (thousands)
- Limits the maximum number of points/voxels to render (10,000-1,000,000)
- When a structure contains more points than this limit, a random subset is selected
- Prevents memory overflows when visualizing large structures

### Processing Chunk Size
- Controls how many slices/frames are processed at once (50-500)
- Larger chunk sizes process data faster but require more memory
- Smaller chunk sizes are more memory-efficient but slower

## Technical Implementation Details

### Memory Optimization
- Uses memory mapping for efficient file loading without loading the entire file into memory
- Processes data in chunks to prevent memory overflow
- Implements garbage collection after intensive operations
- Uses boolean indexing and vectorized operations for efficient array handling

### Multithreading
- Runs intensive calculations in background threads to keep the UI responsive
- Uses ThreadPoolExecutor for parallel estimation processing
- Implements a processing queue for background tasks

### Visualization Techniques
- Automatically switches between voxel and scatter plot rendering based on data size
- Uses scatter plots for very large datasets and voxel plots for smaller ones
- Optimizes rendering parameters based on the quality setting

### UI Design
- Implements a three-stage workflow: welcome screen → full image view → individual structure analysis
- Uses scrollable frames for metadata and color selection to handle large datasets
- Provides real-time progress updates via a status bar

## Usage Guide

1. **Launch the application**
   - Adjust performance settings on the welcome screen based on your system capabilities

2. **Upload a TIFF File**
   - Click "Upload TIFF File" and select your multilayered TIFF image
   - The application supports indexed color (P mode) TIFF files with multiple layers

3. **Select Colors for Analysis**
   - In the viewer screen, use the color panel on the right to select structures of interest
   - Use "Select All" or "Unselect All" buttons as needed
   - Click "Apply" to proceed to detailed analysis

4. **Structure Analysis**
   - The application will show each selected structure individually
   - Review and modify metadata values as needed
   - Use "Previous" and "Next" buttons to navigate between structures
   - Click "Save" to export the current structure as a TIFF file with metadata

5. **Navigation and Visualization**
   - Use the mouse wheel to zoom in and out of 3D visualizations
   - The application will automatically optimize rendering based on structure size

## Usage Recommendations

1. For very large datasets, start with higher downsample factor (4-8) and lower quality (1-2)
2. Adjust the max points setting based on your computer's memory capacity
3. For detailed analysis of individual structures, reduce the downsample factor and increase quality
4. Use larger chunk sizes on machines with more RAM, smaller chunk sizes on memory-limited systems

## Troubleshooting

- **Memory Issues**: If the application crashes or becomes unresponsive, try:
  - Increasing the downsample factor
  - Reducing the max points
  - Reducing the chunk size
  - Closing other memory-intensive applications

- **Slow Rendering**: If visualization is too slow:
  - Reduce rendering quality
  - Increase downsample factor
  - Set a lower max points value

- **File Loading Problems**: If TIFF files fail to load:
  - Ensure the file is in indexed color (P mode)
  - Check that the file contains multiple layers/frames
  - Verify the file has a valid color palette
