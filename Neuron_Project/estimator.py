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

        # Internal caches for coordinate and calculation reuse
        self._coords_cache = {}
        self._calculation_cache = {}

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
        # Suppress plotting side-effects using legacy helper (falls back to monkey-patch)
        if hasattr(eff, 'set_matplotlib_display'):
            try:
                eff.set_matplotlib_display(False)
            except Exception:
                pass

        # Extra guard – ensure heavyweight scatter routine is a no-op if still reached
        if hasattr(eff, 'Plot_matrix_scatter'):
            eff.Plot_matrix_scatter = lambda *args, **kwargs: None

        # Inject the image and palette the legacy code expects
        eff.rgb_colors = self.rgb_colors
        eff.image_uploaded = self.image_data

        # Build a basic edge map (any non-zero voxel considered edge)
        edge = (self.image_data > 0).astype(int)
        eff.image_uploaded_edge = edge

    # ------------------------------------------------------------------
    # NOTE: All fine-grained helper methods that previously performed their own
    # numeric work (dimension, resolution, volume, etc.) have been removed.  All
    # estimations are now delegated to Extract_Figures_FV via
    # Utilities.generate_estimations inside run_estimations().
    # ------------------------------------------------------------------ 