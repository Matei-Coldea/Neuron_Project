#!/usr/bin/env python3
"""
Enhanced TIFF Processor
Advanced TIFF file handling with metadata extraction and format support.
"""

import numpy as np
import tifffile
from PIL import Image
import json
import os
from typing import Tuple, Dict, Any, Optional, List

class EnhancedTIFFProcessor:
    """Advanced TIFF processor with comprehensive format support"""
    
    def __init__(self):
        self.metadata = {}
        self.color_palette = None
        self.original_shape = None
        self.data_type = None
        
    def load_tiff(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load TIFF file with comprehensive format support
        
        Returns:
            tuple: (data_array, metadata_dict)
        """
        print(f"Loading TIFF: {filepath}")
        
        # Try different loading methods
        data, metadata = None, {}
        
        # Method 1: tifffile (best for scientific TIFF)
        try:
            data = tifffile.imread(filepath)
            metadata = {'method': 'tifffile'}
            print("✓ Loaded with tifffile")
        except Exception as e:
            print(f"tifffile failed: {e}")
            
        # Method 2: PIL (good for standard TIFF)
        if data is None:
            try:
                frames = []
                with Image.open(filepath) as img:
                    metadata = {'format': img.format, 'mode': img.mode}
                    
                    # Get palette if available
                    if hasattr(img, 'getpalette') and img.getpalette():
                        palette = np.array(img.getpalette()).reshape(-1, 3) / 255.0
                        self.color_palette = palette
                        metadata['has_palette'] = True
                        metadata['palette_size'] = len(palette)
                        
                    # Load all frames for multi-frame TIFF
                    try:
                        frame_count = 0
                        while True:
                            frames.append(np.array(img))
                            img.seek(img.tell() + 1)
                            frame_count += 1
                    except EOFError:
                        pass
                        
                data = np.stack(frames, axis=0) if len(frames) > 1 else frames[0]
                if len(data.shape) == 2:
                    data = data[np.newaxis, ...]  # Add depth dimension
                else:
                    data = data[0]
                metadata['frame_count'] = frame_count
                print("✓ Loaded with PIL")
            except Exception as e:
                print(f"PIL failed: {e}")
                
        # Method 3: Raw numpy (fallback)
        if data is None:
            try:
                data, metadata = self._load_raw(filepath)
                print("✓ Loaded as raw data")
            except Exception as e:
                print(f"Raw loading failed: {e}")
                raise ValueError(f"Could not load TIFF file: {filepath}")
                
        # Process and validate data
        data = self._process_data(data)
        metadata = self._enhance_metadata(metadata, filepath)
        
        self.original_shape = data.shape
        self.data_type = data.dtype
        self.metadata = metadata
        
        print(f"Final data shape: {data.shape}, dtype: {data.dtype}")
        print(f"Data range: {data.min()} to {data.max()}")
        
        return data, metadata
        
    def _load_with_tifffile(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """Load using tifffile library"""
        # Load data
        data = tifffile.imread(filepath)
        
        # Extract metadata
        metadata = {}
        try:
            with tifffile.TiffFile(filepath) as tif:
                # Basic metadata
                metadata['software'] = getattr(tif.pages[0], 'software', None)
                metadata['description'] = getattr(tif.pages[0], 'description', None)
                metadata['datetime'] = getattr(tif.pages[0], 'datetime', None)
                
                # Image properties
                page = tif.pages[0]
                metadata['width'] = page.imagewidth
                metadata['height'] = page.imagelength
                metadata['bitspersample'] = getattr(page, 'bitspersample', None)
                metadata['samplesperpixel'] = getattr(page, 'samplesperpixel', None)
                metadata['photometric'] = getattr(page, 'photometric', None)
                
                # Resolution
                if hasattr(page, 'tags'):
                    if 'XResolution' in page.tags:
                        metadata['x_resolution'] = page.tags['XResolution'].value
                    if 'YResolution' in page.tags:
                        metadata['y_resolution'] = page.tags['YResolution'].value
                    if 'ResolutionUnit' in page.tags:
                        metadata['resolution_unit'] = page.tags['ResolutionUnit'].value
                        
                # Color palette for indexed images
                if hasattr(page, 'colormap') and page.colormap is not None:
                    self.color_palette = np.array(page.colormap).reshape(-1, 3) / 65535.0
                    metadata['has_palette'] = True
                    metadata['palette_size'] = len(self.color_palette)
                    
        except Exception as e:
            print(f"Metadata extraction failed: {e}")
            
        return data, metadata
        
    def _load_with_pil(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """Load using PIL"""
        metadata = {}
        frames = []
        
        with Image.open(filepath) as img:
            # Extract basic metadata
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['width'] = img.width
            metadata['height'] = img.height
            
            # Get palette if available
            if hasattr(img, 'getpalette') and img.getpalette():
                palette = np.array(img.getpalette()).reshape(-1, 3) / 255.0
                self.color_palette = palette
                metadata['has_palette'] = True
                metadata['palette_size'] = len(palette)
                
            # Load all frames for multi-frame TIFF
            try:
                frame_count = 0
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
                    frame_count += 1
            except EOFError:
                pass
                
            metadata['frame_count'] = frame_count
            
        # Stack frames into 3D array
        if len(frames) == 1:
            data = frames[0]
            if len(data.shape) == 2:
                data = data[np.newaxis, ...]  # Add depth dimension
        else:
            data = np.stack(frames, axis=0)
            
        return data, metadata
        
    def _load_raw(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """Load as raw binary data (last resort)"""
        # This is a very basic fallback
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            
        # Try to interpret as image data
        # This is highly speculative and may not work
        size = int(np.sqrt(len(raw_data) / 4))  # Assume square, 32-bit
        data = np.frombuffer(raw_data[:size*size*4], dtype=np.uint8)
        data = data.reshape(size, size)
        data = data[np.newaxis, ...]  # Add depth dimension
        
        metadata = {
            'method': 'raw',
            'file_size': len(raw_data),
            'estimated_size': size
        }
        
        return data, metadata
        
    def _process_data(self, data: np.ndarray) -> np.ndarray:
        """Process and normalize data"""
        # Ensure we have a 3D array
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        elif len(data.shape) == 4:
            # Handle RGBA or multi-channel
            if data.shape[-1] in [3, 4]:
                # Take first channel or convert to grayscale
                if data.shape[-1] == 3:
                    data = np.mean(data, axis=-1)
                else:  # RGBA
                    data = data[..., 0]  # Take red channel
            else:
                # Assume first dimension is channels
                data = data[0]
                
        # Ensure 3D: (depth, height, width)
        if len(data.shape) != 3:
            raise ValueError(f"Cannot process data with shape {data.shape}")
            
        # Convert to appropriate dtype
        if data.dtype == np.bool_:
            data = data.astype(np.uint8)
        elif data.dtype in [np.float32, np.float64]:
            # Normalize float data to 0-255 range
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)
                
        return data
        
    def _enhance_metadata(self, metadata: Dict, filepath: str) -> Dict:
        """Enhance metadata with additional information"""
        # Add file information
        metadata['filepath'] = filepath
        metadata['filename'] = os.path.basename(filepath)
        metadata['file_size'] = os.path.getsize(filepath)
        
        # Add processing information
        metadata['processor'] = 'EnhancedTIFFProcessor'
        metadata['has_color_palette'] = self.color_palette is not None
        
        if self.original_shape:
            metadata['original_shape'] = self.original_shape
            metadata['dimensions'] = len(self.original_shape)
            
        return metadata
        
    def extract_point_cloud(self, data: np.ndarray, 
                           max_points: int = 500000,
                           downsample_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract point cloud from 3D data
        
        Returns:
            tuple: (points, colors) as numpy arrays
        """
        print(f"Extracting point cloud (max_points={max_points}, downsample={downsample_factor})")
        
        # Apply downsampling
        if downsample_factor > 1:
            data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            print(f"Downsampled to shape: {data.shape}")
            
        # Find non-zero coordinates
        coords = np.argwhere(data > 0)
        
        if len(coords) == 0:
            print("No non-zero voxels found, using all data")
            coords = np.argwhere(data >= data.min())
            
        print(f"Found {len(coords)} voxels")
        
        # Limit number of points
        if len(coords) > max_points:
            indices = np.random.choice(len(coords), max_points, replace=False)
            coords = coords[indices]
            print(f"Randomly sampled {len(coords)} points")
            
        # Get values at coordinates
        values = data[coords[:, 0], coords[:, 1], coords[:, 2]]
        
        # Convert coordinates to world space (z, y, x) -> (x, y, z)
        points = coords[:, [2, 1, 0]].astype(np.float32)
        
        # Normalize points to [-1, 1] range
        if len(points) > 0:
            center = points.mean(axis=0)
            points -= center
            max_range = np.abs(points).max()
            if max_range > 0:
                points /= max_range
                
        # Generate colors
        colors = self._generate_colors(values)
        
        print(f"Generated {len(points)} points with colors")
        return points, colors
        
    def _generate_colors(self, values: np.ndarray) -> np.ndarray:
        """Generate colors for values"""
        unique_values = np.unique(values)
        colors = np.zeros((len(values), 3), dtype=np.float32)
        
        if self.color_palette is not None and len(self.color_palette) > 0:
            # Use existing palette
            for i, value in enumerate(values):
                if value < len(self.color_palette):
                    colors[i] = self.color_palette[value]
                else:
                    colors[i] = [0.5, 0.5, 0.5]  # Default gray
        else:
            # Generate rainbow colors
            if len(unique_values) == 1:
                colors[:] = [1.0, 1.0, 1.0]  # White for single value
            else:
                for i, value in enumerate(values):
                    # Map value to color
                    if value == 0:
                        colors[i] = [0.1, 0.1, 0.1]  # Dark for background
                    else:
                        # Rainbow mapping
                        value_idx = np.where(unique_values == value)[0][0]
                        hue = value_idx / max(1, len(unique_values) - 1)
                        colors[i] = self._hsv_to_rgb(hue, 1.0, 1.0)
                        
        return colors
        
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> List[float]:
        """Convert HSV to RGB"""
        import colorsys
        return list(colorsys.hsv_to_rgb(h, s, v))
        
    def get_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive data statistics"""
        unique_values = np.unique(data)
        
        stats = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min_value': float(data.min()),
            'max_value': float(data.max()),
            'mean_value': float(data.mean()),
            'std_value': float(data.std()),
            'unique_values': len(unique_values),
            'non_zero_voxels': int(np.sum(data > 0)),
            'total_voxels': int(data.size),
            'sparsity': float(np.sum(data > 0) / data.size),
            'memory_usage_mb': float(data.nbytes / 1024 / 1024)
        }
        
        # Value distribution
        if len(unique_values) <= 20:
            value_counts = {}
            for value in unique_values:
                count = int(np.sum(data == value))
                value_counts[str(value)] = count
            stats['value_distribution'] = value_counts
            
        return stats
        
    def save_processed_data(self, data: np.ndarray, filepath: str, 
                          metadata: Optional[Dict] = None):
        """Save processed data with metadata"""
        try:
            # Prepare metadata
            if metadata is None:
                metadata = self.metadata.copy()
                
            # Add processing info
            metadata['processed'] = True
            metadata['processor_version'] = '1.0'
            
            # Save as TIFF with metadata
            tifffile.imwrite(filepath, data, 
                           description=json.dumps(metadata, indent=2),
                           metadata=metadata)
            print(f"Saved processed data to: {filepath}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            # Fallback: save as numpy array
            fallback_path = filepath.replace('.tiff', '.npy').replace('.tif', '.npy')
            np.save(fallback_path, data)
            print(f"Saved as numpy array: {fallback_path}")


def test_processor():
    """Test the TIFF processor"""
    processor = EnhancedTIFFProcessor()
    
    # Create test data
    test_data = np.random.randint(0, 5, (10, 20, 20), dtype=np.uint8)
    test_data[test_data == 0] = 0  # Ensure some zeros
    
    print("Testing TIFF processor...")
    
    # Test point cloud extraction
    points, colors = processor.extract_point_cloud(test_data, max_points=1000)
    print(f"Extracted {len(points)} points")
    
    # Test statistics
    stats = processor.get_statistics(test_data)
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("✓ Processor test completed")


if __name__ == "__main__":
    test_processor() 