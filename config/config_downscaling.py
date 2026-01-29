from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


# ---------- Define directories ----------
OUT_ROOT = Path(r"E:/Pfynwald/Results/ForSIF")
DATA_ROOT = Path(r"E:/Pfynwald")
PLOTS_DIRNAME = "plots"

# ---------- Global constants ----------

# CRS
DEFAULT_RASTER_CRS = "EPSG:32632"

# HyPlant metadata
REFL_SCALE = 10000.0
FILL_VALUE = 15000.0
NODATA_OUT = -999.0

# PAR conversion constants (PPFD -> irradiance)
# 1 µmol photons m-2 s-1 ≈ 0.218 W m-2
PAR_UMOL_TO_W = 0.218

# Ranges (nm)
RED_RANGE = (665, 680)
NIR_RANGE = (795, 810)
VIS_RANGE = (400, 700)

# FCVI
NIR_FCVI_RANGE = (768, 773)

# MTCI
MTCI_NIR_RANGE = (746.5, 761.5)
MTCI_REDEDGE_RANGE = (699, 719)
MTCI_RED_RANGE = (673.5, 688.5)

# saR2F
NIR_saR2F_RANGE = NIR_FCVI_RANGE
RED_saR2F_RANGE = (436, 440)
BLUE_saR2F_RANGE = (673, 677)

# thresholds
DEFAULT_NDVI_THRESHOLD = 0.5
DEFAULT_FCVI_THRESHOLD = 0.18

@dataclass(frozen=True)
class SceneConfig:
    date: str
    sif_files: Dict[str, Path]
    toc_refl_files: Dict[str, Path]
    par_umol_m2_s: float


@dataclass(frozen=True)
class ProfileConfig:
    name: str

    crowns_shp: Path
    treatment_areas_shp: Path

    hdr_path_for_wavelengths: Path

    sif_o2a_band: int
    sif_to_sif760_factor: float # factor to convert SFMNN o2a band (at 737 nm) to 770 nm

    sif_scale_factor: float = 1.0 # factor to undo integer encoding (e.g., *100 for SFM)

    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD
    fcvi_threshold: float = DEFAULT_FCVI_THRESHOLD

    fesc_methods: Tuple[str, ...] = ("nirv", "fcvi", "sar2f")
    scenes: Tuple[SceneConfig, ...] = ()


def get_profiles():
    R = DATA_ROOT

    crowns = Path(
        "E:/Pfynwald/Data/General/03_Tree_crowns/"
        "Tree_crowns_from_RGB_corr_GPS_manuallyedited_pine.shp"
    )
    treatments = Path(
        "E:/Pfynwald/Data/General/02_Treatment_boundaries/"
        "PFY_IRR_STOP_CH1903_LV95.shp"
    )

    hdr = Path(
        "E:/Pfynwald/Data/HyPlant/original/TOC_REFL/"
        "20240823-PHY-1305-1340-L1-W-DUAL_radiance_img_atm_pol-rect.hdr"
    )

    #  PAR values stored as µmol m-2 s-1
    PAR_20230617 = 1961.0
    PAR_20240613 = 2140.0
    PAR_20240823 = 1642.5

    toc_20230617 = {
        "L2_W": R / "Data/Hyplant/coregistration/TOC_REFL/20230617-PHY-1124-1360-L2-W-DUAL-rect_img_atm_pol_coreg_resampled.tif",
        "L1_E": R / "Data/Hyplant/coregistration/TOC_REFL/20230617-PHY-1117-1360-L1-E-DUAL-rect_img_atm_pol_coreg_resampled.tif",
    }
    toc_20240613 = {
        "L1_W": R / "Data/Hyplant/coregistration/TOC_REFL/20240613-PHY-1143-1340-L1-W-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
        "L2_W": R / "Data/Hyplant/coregistration/TOC_REFL/20240613-PHY-1149-1340-L2-W-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
        "L2_E": R / "Data/Hyplant/coregistration/TOC_REFL/20240613-PHY-1154-1340-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
        "L1_E": R / "Data/Hyplant/coregistration/TOC_REFL/20240613-PHY-1200-1340-L1-E-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
    }
    toc_20240823 = {
        "L1_W": R / "Data/Hyplant/coregistration/TOC_REFL/20240823-PHY-1305-1340-L1-W-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
        "L2_W": R / "Data/Hyplant/coregistration/TOC_REFL/20240823-PHY-1259-1340-L2-W-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
        "L2_E": R / "Data/Hyplant/coregistration/TOC_REFL/20240823-PHY-1253-1340-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
        "L1_E": R / "Data/Hyplant/coregistration/TOC_REFL/20240823-PHY-1248-1340-L1-E-DUAL_radiance_img_atm_pol-rect_coreg_resampled.tif",
    }

    sfmnn_scene_20230617 = SceneConfig(
        date="20230617",
        sif_files={
            "L2_W": R / "Data/Hyplant/coregistration/SIF_SFMNN/20230617-PHY-1124-1360-L2-W-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_SFMNN/20230617-PHY-1117-1360-L1-E-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20230617,
        par_umol_m2_s=PAR_20230617,
    )

    sfmnn_scene_20240613 = SceneConfig(
        date="20240613",
        sif_files={
            "L1_W": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240613-PHY-1143-1340-L1-W-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L2_W": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240613-PHY-1149-1340-L2-W-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L2_E": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240613-PHY-1154-1340-L2-E-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240613-PHY-1200-1340-L1-E-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20240613,
        par_umol_m2_s=PAR_20240613,
    )

    sfmnn_scene_20240823 = SceneConfig(
        date="20240823",
        sif_files={
            "L1_W": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240823-PHY-1305-1340-L1-W-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L2_W": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240823-PHY-1259-1340-L2-W-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L2_E": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240823-PHY-1253-1340-L2-E-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_SFMNN/20240823-PHY-1248-1340-L1-E-FLUO_radiance_EmSFMNN_rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20240823,
        par_umol_m2_s=PAR_20240823,
    )

    sfm_scene_20230617 = SceneConfig(
        date="20230617",
        sif_files={
            "L2_W": R / "Data/Hyplant/coregistration/SIF_SFM/20230617-PHY-1124-1360-L2-W-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20230617,
        par_umol_m2_s=PAR_20230617,
    )

    sfm_scene_20240613 = SceneConfig(
        date="20240613",
        sif_files={
            "L1_W": R / "Data/Hyplant/coregistration/SIF_SFM/20240613-PHY-1143-1340-L1-W-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
            "L2_W": R / "Data/Hyplant/coregistration/SIF_SFM/20240613-PHY-1149-1340-L2-W-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
            "L2_E": R / "Data/Hyplant/coregistration/SIF_SFM/20240613-PHY-1154-1340-L2-E-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_SFM/20240613-PHY-1200-1340-L1-E-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20240613,
        par_umol_m2_s=PAR_20240613,
    )

    sfm_scene_20240823 = SceneConfig(
        date="20240823",
        sif_files={
            "L1_W": R / "Data/Hyplant/coregistration/SIF_SFM/20240823-PHY-1305-1340-L1-W-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
            "L2_W": R / "Data/Hyplant/coregistration/SIF_SFM/20240823-PHY-1259-1340-L2-W-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
            "L2_E": R / "Data/Hyplant/coregistration/SIF_SFM/20240823-PHY-1253-1340-L2-E-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_SFM/20240823-PHY-1248-1340-L1-E-FLUO_radiance_SFM_ALL-rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20240823,
        par_umol_m2_s=PAR_20240823,
    )

    ifld_scene_20240613 = SceneConfig(
        date="20240613",
        sif_files={
            "L1_W": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240613-PHY-1143-1340-L1-W-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
            "L2_W": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240613-PHY-1149-1340-L2-W-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
            "L2_E": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240613-PHY-1154-1340-L2-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240613-PHY-1200-1340-L1-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20240613,
        par_umol_m2_s=PAR_20240613,
    )

    ifld_scene_20240823 = SceneConfig(
        date="20240823",
        sif_files={
            "L2_W": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240823-PHY-1259-1340-L2-W-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
            "L2_E": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240823-PHY-1253-1340-L2-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
            "L1_E": R / "Data/Hyplant/coregistration/SIF_iFLD/FS_iFLD_20240823-PHY-1248-1340-L1-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_resampled.tif",
        },
        toc_refl_files=toc_20240823,
        par_umol_m2_s=PAR_20240823,
    )

    return {
        "SFMNN": ProfileConfig(
            name="SFMNN",
            crowns_shp=crowns,
            treatment_areas_shp=treatments,
            hdr_path_for_wavelengths=hdr,
            sif_o2a_band=11,
            sif_to_sif760_factor=0.516, # convert SIF 737nm --> SIF 760nm
            sif_scale_factor=1.0, 
            scenes=(sfmnn_scene_20230617, sfmnn_scene_20240613, sfmnn_scene_20240823),
        ),
        "SFM": ProfileConfig(
            name="SFM",
            crowns_shp=crowns,
            treatment_areas_shp=treatments,
            hdr_path_for_wavelengths=hdr,
            sif_o2a_band=2,
            sif_to_sif760_factor=1.0,
            sif_scale_factor=100.0, # undo integer encoding (*100)
            scenes=(sfm_scene_20230617, sfm_scene_20240613, sfm_scene_20240823),
        ),
        "iFLD": ProfileConfig(
            name="iFLD",
            crowns_shp=crowns,
            treatment_areas_shp=treatments,
            hdr_path_for_wavelengths=hdr,
            sif_o2a_band=3,
            sif_to_sif760_factor=1.0,
            sif_scale_factor=1.0,
            scenes=(ifld_scene_20240613, ifld_scene_20240823),
        ),
    }
