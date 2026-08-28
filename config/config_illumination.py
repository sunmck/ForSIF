from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SunlitFractionConfig:
    # Spectral ranges [nm]
    red_range: Tuple[float, float] = (650.0, 680.0)
    nir_range: Tuple[float, float] = (780.0, 900.0)
    red_edge_search_range: Tuple[float, float] = (680.0, 750.0)
    red_edge_half_width_nm: float = 2.5
    red_edge_wavelength_nm: Optional[float] = None

    # NDCSI and forest use
    ndcsi_low_quantile: float = 0.01
    ndcsi_high_quantile: float = 0.99
    sun_ndcsi_quantile: float = 0.85
    shade_ndcsi_quantile: float = 0.15
    forest_fraction_min: float = 0.50

    # Known artificial structures are excluded only from endmember learning.
    # The two-endmember model is still applied to those forest pixels so RMSE
    # can be inspected as a diagnostic.
    endmember_scaffold_fraction_max: float = 0.0

    # Treatment polygons used only for endmember learning
    treatment_field: str = "treatment"
    plot_id_field: str = "PLOTID"
    polygon_all_touched: bool = False

    # Final endmember support
    endmember_support_size: int = 20
    max_endmember_per_plot: int = 3
    endmember_min_spacing_pixels: int = 1
    min_candidate_spectra: int = 20

    # Diagnostics / sampling
    max_red_edge_samples: int = 3000
    max_diagnostic_spectra: int = 5000
    random_seed: int = 42

    # Two-endmember constrained unmixing
    unmixing_range: Tuple[float, float] = (430.0, 1000.0) # clipt the uoter wavelengths as they sometimes are invalid in the DUAL data
    exclude_ranges: Tuple[Tuple[float, float], ...] = ()
    block_rows: int = 64


@dataclass(frozen=True)
class FQEIlluminationQCConfig:
    sunlit_veg_min: Optional[float] = None
    unmix_rmse_max: Optional[float] = None


ILLUMINATION_CONFIG = SunlitFractionConfig()

# Leave disabled until the illumination retrieval has been validated.
FQE_ILLUMINATION_QC = FQEIlluminationQCConfig()

# forest_fraction and scaffold_fraction are produced separately by
# 01_run_formask.py.
ILLUMINATION_PRODUCTS_TO_SAVE = (
    "f_sun_veg",
    "rmse",
    "NDCSI",
)