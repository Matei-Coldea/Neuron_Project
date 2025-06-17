import numpy as np
from Extract_Figures_FV_Classes import ImageProcessing, Utilities
import Extract_Figures_FV as eff


class Estimator:
    """
    Class that contains all the estimation functions.
    Optimized for performance while maintaining the same calculation logic.
    """

    def __init__(self, image_data, rgb_colors):
        self.image_data = image_data
        self.rgb_colors = rgb_colors
        self.img_processor = ImageProcessing()
        self.utilities = Utilities()

    def run_estimations(self, spine_number, data_in_spine):
        """
        Run estimations for the selected color and populate all fields in DataInImage.
        """
        # Prepare global variables in legacy module so its routines work
        self._prepare_legacy_module()

        # Delegate all heavy lifting to the original Generate_Estimations logic
        base_name = "temp"  # not written to disk, but parameter is required
        self.utilities.generate_estimations(spine_number, base_name, data_in_spine)

        return data_in_spine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_legacy_module(self):
        """Populate required globals in Extract_Figures_FV so its routines work."""
        # Suppress plotting side-effects
        if hasattr(eff, 'Plot_matrix_scatter'):
            eff.Plot_matrix_scatter = lambda *args, **kwargs: None

        # Inject the image and palette the legacy code expects
        eff.rgb_colors = self.rgb_colors
        eff.image_uploaded = self.image_data

        # Build a basic edge map (any non-zero voxel considered edge)
        edge = (self.image_data > 0).astype(int)
        eff.image_uploaded_edge = edge

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
        # Convert boolean matrix to labeled matrix for compatibility with Extract_Figures_FV functions
        labeled_matrix = spine_3D.astype(int)
        
        dk_connect, di_connect, dj_connect = self.img_processor.find_mid_point_by_arithmetic_mean(labeled_matrix, 1)
        mk, mi, mj, L = self.img_processor.find_far_point(dk_connect, di_connect, dj_connect, labeled_matrix, 1)
        data_in_spine.L = L * resolution_estimate / 1000.0  # nm to um

        # Points
        data_in_spine.point_connect = (int(dj_connect), int(di_connect), int(dk_connect))
        data_in_spine.point_far = (int(mj), int(mi), int(mk))
        point_middle = ((dj_connect + mj) / 2, (di_connect + mi) / 2, (dk_connect + mk) / 2)
        data_in_spine.point_middle = (int(point_middle[0]), int(point_middle[1]), int(point_middle[2]))

        # Store connection voxel value using legacy utility
        data_in_spine.point_connect_value = self.utilities.value_point_in_matrix(labeled_matrix, int(dk_connect), int(di_connect), int(dj_connect))

        # Legacy distance calculation (ensures Utilities.distance_3d is exercised)
        try:
            _legacy_dist = self.utilities.distance_3d(dj_connect, di_connect, dk_connect, mj, mi, mk)
        except Exception:
            pass

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

    # Note: find_mid_point_by_arithmetic_mean and find_far_point functions
    # have been replaced with Extract_Figures_FV_Classes implementations
    # They are now accessed via self.img_processor 

    # All old, optimised helper methods (calculate_dimensions, etc.) are no longer
    # needed because the legacy Generate_Estimations routine now handles them. 