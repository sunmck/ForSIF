from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.config_downscaling import OUT_ROOT


FOREST_NDSM_RASTER = Path(
    r"E:/Proj1_Pfynwald_Data/HyPlant/coregistration/LiDAR/"
    r"Pfynwald_20240823_nDSM_0_5m_coreg.tif"
)

SCAFFOLD_VECTOR = Path(
    r"E:/Proj1_Pfynwald_Data/General/04_Scaffolds/"
    r"schaffolds_v2.gpkg"
)

FOREST_MASK_PROFILE = "SFMNN"
FOREST_MASK_OUT_ROOT = OUT_ROOT / "forestmask"


@dataclass(frozen=True)
class ForestMaskConfig:
    height_threshold_m: float = 7.0
    valid_fraction_min: float = 0.75

    # Fraction -> binary mask thresholds
    forest_binary_fraction_threshold: float = 0.5
    scaffold_binary_fraction_threshold: float = 0.0

    overwrite: bool = True


FOREST_MASK_CONFIG = ForestMaskConfig()


def forest_fraction_path():
    return FOREST_MASK_OUT_ROOT / "forest_fraction.tif"


def scaffold_fraction_path():
    return FOREST_MASK_OUT_ROOT / "scaffold_fraction.tif"


def forest_mask_path():
    return FOREST_MASK_OUT_ROOT / "forest_mask.tif"


def scaffold_mask_path():
    return FOREST_MASK_OUT_ROOT / "scaffold_mask.tif"