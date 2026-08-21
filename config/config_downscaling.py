from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------- Define directories ----------
OUT_ROOT = Path(r"C:/Users/avinn/OneDrive - Forschungszentrum Jülich GmbH/Proj1_Pfynwald/Results/ForSIF") 
DATA_ROOT = Path(r"E:/Proj1_Pfynwald_Data")
PLOTS_DIRNAME = "plots"

# ---------- Global constants ----------

DEFAULT_RASTER_CRS = "EPSG:32632"

REFL_SCALE = 10000.0
FILL_VALUE = 15000.0
NODATA_OUT = -999.0

# PPFD -> irradiance
PAR_UMOL_TO_W = 0.218

# ---------- Spectral ranges ----------
RED_RANGE = (665, 680)
NIR_RANGE = (795, 810)
VIS_RANGE = (400, 700)

# PRI
PRI_531_RANGE = (528.5, 533.5)   # 531 ± 2.5
PRI_570_RANGE = (567.5, 572.5)   # 570 ± 2.5

# WBI
WBI_NIR1_RANGE = (890.0, 905.0)
WBI_NIR2_RANGE = (955.0, 970.0)

# FCVI
NIR_FCVI_RANGE = (767.5, 772.5)

# MTCI
MTCI_NIR_RANGE = (746.5, 761.5)
MTCI_REDEDGE_RANGE = (699, 719)
MTCI_RED_RANGE = (673.5, 688.5)

# saR2F
NIR_saR2F_RANGE = NIR_FCVI_RANGE
BLUE_saR2F_RANGE = (435.5, 440.5)
RED_saR2F_RANGE = (672.5, 677.5)

# thresholds
DEFAULT_NDVI_THRESHOLD = 0.5
DEFAULT_FCVI_THRESHOLD = 0.18


# ---------- Data structures ----------
@dataclass(frozen=True)
class FlightPair:
    flight_id: str
    sif_file: Path
    toc_refl_file: Path
    par_umol_m2_s: Optional[float] = None


@dataclass(frozen=True)
class SceneConfig:
    date: str
    flights: Tuple[FlightPair, ...]
    par_umol_m2_s: Optional[float] = None


@dataclass(frozen=True)
class ProfileConfig:
    name: str

    treatment_areas_shp: Path

    hdr_path_for_wavelengths: Path

    sif_o2a_band: int
    sif_to_sif760_factor: float
    sif_scale_factor: float = 1.0

    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD
    fcvi_threshold: float = DEFAULT_FCVI_THRESHOLD

    fesc_methods: Tuple[str, ...] = ("nirv", "fcvi", "sar2f")
    scenes: Tuple[SceneConfig, ...] = ()


def get_profiles():
    R = DATA_ROOT

    treatments = Path(
        "E:/Proj1_Pfynwald_Data/General/02_Treatment_boundaries/"
        "PFY_IRR_STOP_CH1903_LV95.shp"
    )

    hdr = Path(
        "E:/Proj1_Pfynwald_Data/HyPlant/original/TOC_REFL/"
        "20240823-PHY-1305-1340-L1-W-DUAL_radiance_img_atm_pol-rect.hdr"
    )


    # ---------- PAR data ----------

    # TODO: check whether these measured PAR values are sunlit or shaded
    #  PAR values stored as µmol m-2 s-1
    PAR_BY_DATE = {
        "20230617": 1961.0,
        "20240613": 2140.0,
        "20240823": 1642.5,
        "20260529": None,
        "20260805": None,
    }


    # ---------- Remote sensing data ----------

    TOC_DIR = R / "Hyplant/coregistration/TOC_REFL/shared_grid"
    SFMNN_DIR = R / "Hyplant/coregistration/SIF_SFMNN/shared_grid"
    SFM_DIR = R / "Hyplant/coregistration/SIF_SFM/shared_grid"
    IFLD_DIR = R / "Hyplant/coregistration/SIF_iFLD/shared_grid"

    # TOC reflectance

    TOC_REFL_BY_DATE = {

        "20230617": {
            "1117_L1_E":
                TOC_DIR /
                "20230617-PHY-1117-1360-L1-E-DUAL-rect_img_atm_pol_coreg_shared.tif",

            "1124_L2_W":
                TOC_DIR /
                "20230617-PHY-1124-1360-L2-W-DUAL-rect_img_atm_pol_coreg_shared.tif",
        },


        "20240613": {
            "1143_L1_W":
                TOC_DIR /
                "20240613-PHY-1143-1340-L1-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1149_L2_W":
                TOC_DIR /
                "20240613-PHY-1149-1340-L2-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1154_L2_E":
                TOC_DIR /
                "20240613-PHY-1154-1340-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1200_L1_E":
                TOC_DIR /
                "20240613-PHY-1200-1340-L1-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",
        },


        "20240823": {
            "1248_L1_E":
                TOC_DIR /
                "20240823-PHY-1248-1340-L1-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1253_L2_E":
                TOC_DIR /
                "20240823-PHY-1253-1340-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1259_L2_W":
                TOC_DIR /
                "20240823-PHY-1259-1340-L2-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1305_L1_W":
                TOC_DIR /
                "20240823-PHY-1305-1340-L1-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",
        },


        "20260529": {
            "1113_L2_E":
                TOC_DIR /
                "20260529-PHY-1113-1360-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1125_L1_W":
                TOC_DIR /
                "20260529-PHY-1125-1360-L1-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1131_L2_E":
                TOC_DIR /
                "20260529-PHY-1131-1360-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1120_L1_W":
                TOC_DIR /
                "20260529-PHY-1120-1360-L1-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",
        },


        "20260805": {
            "1014_L2_E":
                TOC_DIR /
                "20260805-PHY-1014-1340-L2-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1008_L1_E":
                TOC_DIR /
                "20260805-PHY-1008-1340-L1-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1020_L2_W":
                TOC_DIR /
                "20260805-PHY-1020-1340-L2-W-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",

            "1026_L1_E":
                TOC_DIR /
                "20260805-PHY-1026-1340-L1-E-DUAL_radiance_img_atm_pol-rect_coreg_shared.tif",
        },
    }


    # Helper function
    def make_scene(
        date: str,
        sif_files: Dict[str, Path],
        flight_par: Optional[Dict[str, float]] = None,
    ) -> SceneConfig:

        if date not in TOC_REFL_BY_DATE:
            raise KeyError(f"No TOC reflectance data configured for {date}")

        toc_files = TOC_REFL_BY_DATE[date]

        missing_toc = set(sif_files) - set(toc_files)

        if missing_toc:
            raise ValueError(
                f"{date}: SIF flights without matching TOC: "
                f"{sorted(missing_toc)}"
            )

        flight_par = flight_par or {}

        unknown_par = set(flight_par) - set(sif_files)

        if unknown_par:
            raise ValueError(
                f"{date}: PAR values provided for unknown flights: "
                f"{sorted(unknown_par)}"
            )

        flights = tuple(
            FlightPair(
                flight_id=flight_id,
                sif_file=sif_file,
                toc_refl_file=toc_files[flight_id],
                par_umol_m2_s=flight_par.get(flight_id),
            )
            for flight_id, sif_file in sif_files.items()
        )

        if not flights:
            raise ValueError(f"{date}: no matched flights configured")

        return SceneConfig(
            date=date,
            flights=flights,
            par_umol_m2_s=PAR_BY_DATE[date],
        )


    # SIF SFMNN
    sfmnn_scene_20230617 = make_scene(
        "20230617",
        {
            "1117_L1_E":
                SFMNN_DIR /
                "20230617-PHY-1117-1360-L1-E-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1124_L2_W":
                SFMNN_DIR /
                "20230617-PHY-1124-1360-L2-W-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked
        },
    )

    sfmnn_scene_20240613 = make_scene(
        "20240613",
        {
            "1143_L1_W":
                SFMNN_DIR /
                "20240613-PHY-1143-1340-L1-W-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1149_L2_W":
                SFMNN_DIR /
                "20240613-PHY-1149-1340-L2-W-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1154_L2_E":
                SFMNN_DIR /
                "20240613-PHY-1154-1340-L2-E-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1200_L1_E":
                SFMNN_DIR /
                "20240613-PHY-1200-1340-L1-E-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked
        },
    )

    sfmnn_scene_20240823 = make_scene(
        "20240823",
        {
            "1248_L1_E":
                SFMNN_DIR /
                "20240823-PHY-1248-1340-L1-E-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1253_L2_E":
                SFMNN_DIR /
                "20240823-PHY-1253-1340-L2-E-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1259_L2_W":
                SFMNN_DIR /
                "20240823-PHY-1259-1340-L2-W-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked

            "1305_L1_W":
                SFMNN_DIR /
                "20240823-PHY-1305-1340-L1-W-FLUO_radiance_EmSFMNN_rect_coreg_shared.tif", # quality checked
        },
    )

    sfmnn_scene_20260529 = make_scene(
        "20260529",
        {
            "1113_L2_E":
                SFMNN_DIR /
                "20260529-PHY-1113-1360-L2-E-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked

            "1125_L1_W":
                SFMNN_DIR /
                "20260529-PHY-1125-1360-L1-W-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked

            "1131_L2_E":
                SFMNN_DIR /
                "20260529-PHY-1131-1360-L2-E-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked

            "1120_L1_W":
                SFMNN_DIR /
                "20260529-PHY-1120-1360-L1-W-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked
        },
    )

    sfmnn_scene_20260805 = make_scene(
        "20260805",
        {
            "1014_L2_E":
                SFMNN_DIR /
                "20260805-PHY-1014-1340-L2-E-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked

            "1026_L1_E":
                SFMNN_DIR /
                "20260805-PHY-1026-1340-L1-E-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked

            "1008_L1_E":
                SFMNN_DIR /
                "20260805-PHY-1008-1340-L1-E-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked

            "1020_L2_W":
                SFMNN_DIR /
                "20260805-PHY-1020-1340-L2-W-FLUO_radiance_EmSFMNN-rect_coreg_shared.tif", # quality checked
        },
    )

    # SIF SFM 

    sfm_scene_20230617 = make_scene(
        "20230617",
        {
            "1117_L1_E":
                SFM_DIR /
                "20230617-PHY-1117-1360-L1-E-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked

            "1124_L2_W":
                SFM_DIR /
                "20230617-PHY-1124-1360-L2-W-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked
        },
    )

    sfm_scene_20240613 = make_scene(
        "20240613",
        {
            "1143_L1_W":
                SFM_DIR /
                "20240613-PHY-1143-1340-L1-W-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked

            # TODO: shift 1149_L2_W 

            "1154_L2_E":
                SFM_DIR /
                "20240613-PHY-1154-1340-L2-E-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked

            "1200_L1_E":
                SFM_DIR /
                "20240613-PHY-1200-1340-L1-E-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked
        },
    )

    sfm_scene_20240823 = make_scene(
        "20240823",
        {
            "1248_L1_E":
                SFM_DIR /
                "20240823-PHY-1248-1340-L1-E-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked

            "1253_L2_E":
                SFM_DIR /
                "20240823-PHY-1253-1340-L2-E-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked

            "1259_L2_W":
                SFM_DIR /
                "20240823-PHY-1259-1340-L2-W-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked

            "1305_L1_W":
                SFM_DIR /
                "20240823-PHY-1305-1340-L1-W-FLUO_radiance_SFM_ALL-rect_coreg_shared.tif", # quality checked
        },
    )

    # SIF iFLD

    ifld_scene_20230617 = make_scene(
        "20230617",
        {
            "1117_L1_E":
                IFLD_DIR /
                "FS_iFLD_20230617-PHY-1117-1360-L1-E-FLUO_radiance_deconv_i1FIXDEM_V5_noborder-rect_coreg_shared.tif", # quality checked

            # 1124_L2_W excluded
        },
    )


    ifld_scene_20240613 = make_scene(
        "20240613",
        {
            "1143_L1_W":
                IFLD_DIR /
                "FS_iFLD_20240613-PHY-1143-1340-L1-W-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked

            # TODO: shift 1149_L2_W 

            "1154_L2_E":
                IFLD_DIR /
                "FS_iFLD_20240613-PHY-1154-1340-L2-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked

            "1200_L1_E":
                IFLD_DIR /
                "FS_iFLD_20240613-PHY-1200-1340-L1-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked
        },
    )


    ifld_scene_20240823 = make_scene(
        "20240823",
        {
            "1248_L1_E":
                IFLD_DIR /
                "FS_iFLD_20240823-PHY-1248-1340-L1-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked

            "1253_L2_E":
                IFLD_DIR /
                "FS_iFLD_20240823-PHY-1253-1340-L2-E-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked

            "1259_L2_W":
                IFLD_DIR /
                "FS_iFLD_20240823-PHY-1259-1340-L2-W-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked

            "1305_L1_W":
                IFLD_DIR /
                "FS_iFLD_20240823-PHY-1305-1340-L1-W-FLUO_radiance_deconv_i1FIXDEM_V5-rect_coreg_shared.tif", # quality checked
        },
    )

    # ---------- Profiles ----------

    return {
        "SFMNN": ProfileConfig(
            name="SFMNN",
            treatment_areas_shp=treatments,
            hdr_path_for_wavelengths=hdr,
            sif_o2a_band=11,
            sif_to_sif760_factor=0.516,
            sif_scale_factor=1.0,
            scenes=(
                sfmnn_scene_20230617,
                sfmnn_scene_20240613,
                sfmnn_scene_20240823,
                sfmnn_scene_20260529,
                sfmnn_scene_20260805,
            ),
        ),

        "SFM": ProfileConfig(
            name="SFM",
            treatment_areas_shp=treatments,
            hdr_path_for_wavelengths=hdr,
            sif_o2a_band=2,
            sif_to_sif760_factor=1.0,
            sif_scale_factor=100.0,
            scenes=(
                sfm_scene_20230617,
                sfm_scene_20240613,
                sfm_scene_20240823,
            ),
        ),

        "iFLD": ProfileConfig(
            name="iFLD",
            treatment_areas_shp=treatments,
            hdr_path_for_wavelengths=hdr,
            sif_o2a_band=3,
            sif_to_sif760_factor=1.0,
            sif_scale_factor=1.0,
            scenes=(
                ifld_scene_20230617,
                ifld_scene_20240613,
                ifld_scene_20240823,
            ),
        ),
    }