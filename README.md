# 3D TIFF Viewer

A modular Python application for viewing, analyzing, and processing 3D TIFF images with color-based spine analysis and metadata management.

## File Structure

The application has been broken down into the following modular files:

### Core Modules

- **`data_in_image.py`** - Contains the `DataInImage` class for handling metadata of 3D figure properties
- **`estimator.py`** - Contains the `Estimator` class with all estimation functions for spine analysis
- **`tiff_viewer_ui.py`** - Contains the `TIFFViewerUI` class that handles all UI tasks and interactions
- **`tiff_viewer_3d.py`** - Contains the `TIFFViewer3D` main class for loading and processing TIFF images

### Application Entry Points

- **`main.py`** - Main entry point to run the application
- **`Stable_version_see_through.py`** - Original monolithic file (kept for reference)

### Configuration Files

- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python main.py
```

### Using Individual Modules

You can also import and use individual classes:

```python
from data_in_image import DataInImage
from estimator import Estimator
from tiff_viewer_ui import TIFFViewerUI
from tiff_viewer_3d import TIFFViewer3D

# Example usage
data = DataInImage()
# ... configure data ...

estimator = Estimator(image_data, rgb_colors)
# ... run estimations ...
```

## Features

- **3D TIFF Visualization**: Interactive 3D plotting of TIFF images
- **Color-based Analysis**: Select and analyze specific colors/spines
- **Performance Optimization**: Configurable downsampling, quality settings, and chunked processing
- **Metadata Management**: Comprehensive metadata handling and export
- **Memory Efficiency**: Optimized for large datasets with garbage collection and memory mapping

## Module Dependencies

```
main.py
└── tiff_viewer_3d.py
    ├── data_in_image.py
    └── tiff_viewer_ui.py
        ├── data_in_image.py
        └── estimator.py
```

## Performance Settings

The application includes several performance optimization features:

- **Downsample Factor**: Reduce data size for faster processing
- **Render Quality**: Adjust rendering quality vs. speed
- **Max Points**: Limit maximum points for visualization
- **Chunk Size**: Control memory usage during processing

## Notes

- The original `Stable_version_see_through.py` file is preserved for reference
- All functionality from the original file has been maintained
- The modular structure improves maintainability and allows for easier testing and extension
- Each module has clear separation of concerns and minimal dependencies 