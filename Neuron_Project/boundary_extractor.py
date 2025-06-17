"""
Boundary Extractor Module

Fast boundary detection for 3D voxel data using modern scipy/skimage algorithms.
This module provides efficient alternatives to the legacy boundary detection
embedded in Extract_Figures_FV.py.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.segmentation import find_boundaries


class BoundaryExtractor:
    """
    Fast boundary extraction for 3D voxel data using vectorized operations.
    """
    
    def __init__(self):
        pass
    
    def extract_surface_voxels(self, image_3d, color_value, method='outer'):
        """
        Extract surface/boundary voxels for a specific color using skimage.
        
        Args:
            image_3d (np.ndarray): 3D image array
            color_value (int): The color/label value to extract boundaries for
            method (str): 'outer', 'inner', or 'thick' boundary detection
            
        Returns:
            np.ndarray: Boolean mask of boundary voxels
        """
        # Create binary mask for the target color
        mask = (image_3d == color_value)
        
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        
        # Use skimage's find_boundaries - much faster than manual loops
        boundaries = find_boundaries(mask, mode=method, background=0)
        
        return boundaries
    
    def extract_surface_coordinates(self, image_3d, color_value, method='outer'):
        """
        Extract coordinates of surface voxels.
        
        Args:
            image_3d (np.ndarray): 3D image array
            color_value (int): The color/label value to extract boundaries for
            method (str): Boundary detection method
            
        Returns:
            np.ndarray: Array of (z, y, x) coordinates of boundary voxels
        """
        boundaries = self.extract_surface_voxels(image_3d, color_value, method)
        return np.argwhere(boundaries)
    
    def extract_6_connectivity_surface(self, image_3d, color_value):
        """
        Extract surface using 6-connectivity (faces only, not edges/corners).
        This mimics the legacy algorithm's approach but uses vectorized operations.
        
        Args:
            image_3d (np.ndarray): 3D image array
            color_value (int): The color/label value to extract boundaries for
            
        Returns:
            np.ndarray: Boolean mask of surface voxels
        """
        mask = (image_3d == color_value)
        
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        
        # Create structuring element for 6-connectivity (faces only)
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[1, 1, 1] = True  # center
        struct[0, 1, 1] = True  # front/back
        struct[2, 1, 1] = True
        struct[1, 0, 1] = True  # top/bottom
        struct[1, 2, 1] = True
        struct[1, 1, 0] = True  # left/right
        struct[1, 1, 2] = True
        
        # Erode the mask - surface voxels are those that disappear
        eroded = binary_erosion(mask, structure=struct)
        surface = mask & ~eroded
        
        return surface
    
    def extract_connection_points(self, image_3d, color_value, neighbor_colors=None):
        """
        Extract voxels that are on the boundary AND adjacent to other colored regions.
        This identifies connection points between different structures.
        
        Args:
            image_3d (np.ndarray): 3D image array
            color_value (int): The target color
            neighbor_colors (list): List of colors to check for connections.
                                  If None, checks for any non-zero, non-target color.
            
        Returns:
            tuple: (connection_mask, boundary_mask) as boolean arrays
        """
        mask = (image_3d == color_value)
        
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool)
        
        # Get surface voxels
        surface = self.extract_6_connectivity_surface(image_3d, color_value)
        
        # Check which surface voxels are adjacent to other colors
        connection_points = np.zeros_like(mask, dtype=bool)
        
        # 6-connectivity offsets
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        
        surface_coords = np.argwhere(surface)
        
        for z, y, x in surface_coords:
            for dz, dy, dx in offsets:
                nz, ny, nx = z + dz, y + dy, x + dx
                
                # Check bounds
                if (0 <= nz < image_3d.shape[0] and 
                    0 <= ny < image_3d.shape[1] and 
                    0 <= nx < image_3d.shape[2]):
                    
                    neighbor_val = image_3d[nz, ny, nx]
                    
                    # Check if neighbor is a connection point
                    if neighbor_colors is None:
                        # Any non-zero, non-target color is a connection
                        if neighbor_val != 0 and neighbor_val != color_value:
                            connection_points[z, y, x] = True
                            break
                    else:
                        # Check specific neighbor colors
                        if neighbor_val in neighbor_colors:
                            connection_points[z, y, x] = True
                            break
        
        return connection_points, surface
    
    def get_surface_statistics(self, image_3d, color_value):
        """
        Get quick statistics about the surface of a colored region.
        
        Args:
            image_3d (np.ndarray): 3D image array
            color_value (int): The color/label value
            
        Returns:
            dict: Statistics including surface area, volume, surface-to-volume ratio
        """
        mask = (image_3d == color_value)
        volume = np.sum(mask)
        
        if volume == 0:
            return {
                'volume': 0,
                'surface_area': 0,
                'surface_to_volume_ratio': 0,
                'surface_coordinates': np.array([])
            }
        
        surface = self.extract_surface_voxels(image_3d, color_value)
        surface_area = np.sum(surface)
        surface_coords = np.argwhere(surface)
        
        return {
            'volume': volume,
            'surface_area': surface_area,
            'surface_to_volume_ratio': surface_area / volume if volume > 0 else 0,
            'surface_coordinates': surface_coords
        }


# Convenience functions for quick access
def extract_boundaries_fast(image_3d, color_value, method='outer'):
    """
    Quick function to extract boundaries for a specific color.
    
    Args:
        image_3d (np.ndarray): 3D image array
        color_value (int): Color to extract boundaries for
        method (str): Boundary detection method
        
    Returns:
        np.ndarray: Coordinates of boundary voxels
    """
    extractor = BoundaryExtractor()
    return extractor.extract_surface_coordinates(image_3d, color_value, method)


def extract_surface_only(image_3d, color_value):
    """
    Extract only surface voxels (not the full volume) for display.
    
    Args:
        image_3d (np.ndarray): 3D image array
        color_value (int): Color to extract surface for
        
    Returns:
        np.ndarray: Coordinates of surface voxels only
    """
    extractor = BoundaryExtractor()
    return extractor.extract_surface_coordinates(image_3d, color_value, method='outer')


if __name__ == "__main__":
    # Example usage
    print("Boundary Extractor Module")
    print("Use: from boundary_extractor import BoundaryExtractor, extract_boundaries_fast") 