import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import tifffile as tiff

class Estimator:
    """
    Class that contains all the estimation functions.
    Optimized for performance while maintaining the same calculation logic.
    """

    def __init__(self, image_data, rgb_colors):
        self.image_data = image_data
        self.rgb_colors = rgb_colors
        # Pre-calculate coordinates for reuse
        self._coords_cache = {}
        # Cache for common calculations
        self._calculation_cache = {}

    def run_estimations(self, spine_number, data_in_spine):
        """
        Run estimations for the selected color and populate all fields in DataInImage.
        """
        # Clear caches for new spine
        self._coords_cache = {}
        self._calculation_cache = {}
        
        data_in_spine.spine_color = self.rgb_colors[spine_number]
        self.generate_estimations(spine_number, data_in_spine)
        return data_in_spine

    def generate_estimations(self, spine_number, data_in_spine):
        """
        Generate estimations for the given spine number.
        Optimized to minimize redundant calculations.
        """
        # Set the color of the spine
        self.set_spine_color(spine_number, data_in_spine)

        # Extract the mask and subvolume - cached for reuse
        mask, spine_3D = self.extract_spine_mask(spine_number)
        if mask is None or spine_3D is None:
            print(f"No voxels found for spine number {spine_number}")
            return data_in_spine

        # Calculate dimensions using cached coordinates
        self.calculate_dimensions(mask, data_in_spine)

        # Cache resolution estimate as it's used multiple times
        resolution_estimate = self.estimate_resolution(data_in_spine)
        self._calculation_cache['resolution_estimate'] = resolution_estimate

        # Run estimations in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all estimation tasks
            volume_future = executor.submit(self.estimate_volume, spine_3D, resolution_estimate, data_in_spine)
            surface_future = executor.submit(self.estimate_surface_area, spine_3D, resolution_estimate, data_in_spine)
            length_future = executor.submit(self.estimate_length_L, spine_3D, resolution_estimate, data_in_spine)
            diameter_future = executor.submit(self.estimate_diameter, spine_3D, resolution_estimate, data_in_spine)

            # Wait for all tasks to complete
            volume_future.result()
            surface_future.result()
            length_future.result()
            diameter_future.result()

        # Set description (quick operation, no need for parallel)
        self.set_description(data_in_spine)

    def set_spine_color(self, spine_number, data_in_spine):
        """
        Set the color of the spine in data_in_spine.
        """
        spine_color = self.rgb_colors[spine_number]
        data_in_spine.spine_color = spine_color

    def extract_spine_mask(self, spine_number):
        """
        Extract the mask and subvolume containing the spine.
        Optimized with caching and efficient array operations.
        """
        if spine_number in self._coords_cache:
            return self._coords_cache[spine_number]

        # Use boolean indexing for efficiency
        mask = self.image_data == spine_number
        coords = np.argwhere(mask)
        
        if coords.size == 0:
            self._coords_cache[spine_number] = (None, None)
            return None, None

        # Use vectorized operations for min/max
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        min_k, min_i, min_j = min_coords
        max_k, max_i, max_j = max_coords

        # Extract subvolume efficiently using views
        spine_3D = mask[min_k:max_k + 1, min_i:max_i + 1, min_j:max_j + 1]
        
        # Cache the result
        self._coords_cache[spine_number] = (mask, spine_3D)
        return mask, spine_3D

    def calculate_dimensions(self, mask, data_in_spine):
        """
        Calculate the dimensions of the spine and update data_in_spine.
        Uses cached coordinates when possible.
        """
        if 'dimensions' in self._calculation_cache:
            di, dj, dk = self._calculation_cache['dimensions']
        else:
            coords = np.argwhere(mask)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            min_k, min_i, min_j = min_coords
            max_k, max_i, max_j = max_coords
            di = max_i - min_i + 1
            dj = max_j - min_j + 1
            dk = max_k - min_k + 1
            self._calculation_cache['dimensions'] = (di, dj, dk)

        data_in_spine.height = di
        data_in_spine.width = dj
        data_in_spine.num_layers = dk

    def estimate_resolution(self, data_in_spine):
        """
        Estimate the resolution based on the maximum dimension.
        """
        dmax = max(data_in_spine.height, data_in_spine.width, data_in_spine.num_layers)
        if dmax > 2:
            resolution_estimate = 1000.0 / (dmax - 2)
        else:
            resolution_estimate = 1.0
        data_in_spine.x_resolution = resolution_estimate
        data_in_spine.y_resolution = resolution_estimate
        data_in_spine.z_resolution = resolution_estimate
        data_in_spine.resolution_unit = "nm"
        return resolution_estimate

    def estimate_volume(self, spine_3D, resolution_estimate, data_in_spine):
        """
        Estimate the volume of the spine and update data_in_spine.
        """
        spine_3D_volume_estimate = np.sum(spine_3D)
        volume_voxel = resolution_estimate ** 3
        data_in_spine.volume = spine_3D_volume_estimate * volume_voxel * 1e-9  # nm^3 to um^3
        data_in_spine.volume_unit = "um3"

    def estimate_surface_area(self, spine_3D, resolution_estimate, data_in_spine):
        """
        Estimate the surface area of the spine and update data_in_spine.
        Currently set to None (placeholder for future implementation).
        """
        data_in_spine.surface = None  # Placeholder
        data_in_spine.surface_unit = "um2"

    def estimate_length_L(self, spine_3D, resolution_estimate, data_in_spine):
        """
        Estimate the length L of the spine and update data_in_spine.
        """
        dk_connect, di_connect, dj_connect = self.find_mid_point_by_arithmetic_mean(spine_3D)
        mk, mi, mj, L = self.find_far_point(dk_connect, di_connect, dj_connect, spine_3D)
        data_in_spine.L = L * resolution_estimate / 1000.0  # nm to um

        # Points
        data_in_spine.point_connect = (int(dj_connect), int(di_connect), int(dk_connect))
        data_in_spine.point_far = (int(mj), int(mi), int(mk))
        point_middle = ((dj_connect + mj) / 2, (di_connect + mi) / 2, (dk_connect + mk) / 2)
        data_in_spine.point_middle = (int(point_middle[0]), int(point_middle[1]), int(point_middle[2]))

        data_in_spine.point_connect_value = int(spine_3D[int(dk_connect), int(di_connect), int(dj_connect)])

    def estimate_diameter(self, spine_3D, resolution_estimate, data_in_spine):
        """
        Estimate the diameter of the spine and update data_in_spine.
        Currently set to None (placeholder for future implementation).
        """
        data_in_spine.d = None  # Placeholder

    def set_description(self, data_in_spine):
        """
        Set the description field in data_in_spine.
        """
        data_in_spine.description = f"Estimations for selected color with RGB {data_in_spine.spine_color}"

    def find_mid_point_by_arithmetic_mean(self, matrix):
        """
        Find the arithmetic mean point of the voxels in the matrix.
        Optimized using numpy's efficient array operations.
        """
        coords = np.argwhere(matrix)
        if coords.size == 0:
            return 0, 0, 0
        # Use numpy's mean for better performance
        mean_coords = np.mean(coords, axis=0)
        return mean_coords[0], mean_coords[1], mean_coords[2]

    def find_far_point(self, dk_connect, di_connect, dj_connect, matrix):
        """
        Find the farthest point from the given connection point in the matrix.
        Optimized using vectorized operations.
        """
        coords = np.argwhere(matrix)
        if coords.size == 0:
            return 0, 0, 0, 0

        # Vectorized distance calculation
        given_point = np.array([dk_connect, di_connect, dj_connect])
        distances = np.linalg.norm(coords - given_point, axis=1)
        max_distance_idx = np.argmax(distances)
        max_distance = distances[max_distance_idx]
        farthest_point = coords[max_distance_idx]
        
        return tuple(farthest_point) + (max_distance,)


def _convert_numpy_types(obj):
    """Convert NumPy types to Python native types"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj


def save_data_as_tiff_optimized(save_path, current_mask, data_in_spine, update_status_callback=None):
    """Save TIFF with optimized memory usage"""
    try:
        if update_status_callback:
            update_status_callback("Preparing data for saving...")
        
        mask = np.zeros_like(current_mask, dtype=np.uint8)
        mask[current_mask] = 255

        metadata_dict = {}
        for tag, attr in data_in_spine.tag_dict.items():
            value = getattr(data_in_spine, attr)
            if value is not None:
                metadata_dict[tag] = _convert_numpy_types(value)

        if update_status_callback:
            update_status_callback("Saving TIFF file...")
        
        chunk_size = min(100, mask.shape[0])
        with tiff.TiffWriter(save_path) as tif:
            for i in range(0, mask.shape[0], chunk_size):
                end = min(i + chunk_size, mask.shape[0])
                chunk = mask[i:end]
                if i == 0:
                    tif.write(chunk, photometric='minisblack', description=json.dumps(metadata_dict))
                else:
                    tif.write(chunk, photometric='minisblack')
                
                if update_status_callback:
                    progress = (end / mask.shape[0]) * 100
                    update_status_callback(f"Saving... {progress:.1f}%")

        if update_status_callback:
            update_status_callback(f"Saved TIFF file to {save_path}")
            
    except Exception as e:
        if update_status_callback:
            update_status_callback("Error saving TIFF file")
        print(f"Failed to save TIFF with metadata: {e}") 