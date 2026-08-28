from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from pyproj import CRS as PyprojCRS
from rasterio.crs import CRS as RasterioCRS
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject, transform_bounds
from sklearn.mixture import GaussianMixture

from config.config_downscaling import NODATA_OUT, OUT_ROOT, get_profiles
from config.config_formask import (
    FOREST_MASK_CONFIG,
    FOREST_NDSM_RASTER,
    SCAFFOLD_VECTOR,
    forest_fraction_path,
    scaffold_fraction_path,
)
from config.config_illumination import ILLUMINATION_CONFIG


# ---------- Run settings ----------

@dataclass(frozen=True)
class UAVIlluminationValidationConfig:
    # UAV multispectral bands, 1-based raster indices
    red_band: int = 6       # 668 nm
    nir_band: int = 10      # 842 nm

    # Optional vegetation QA. Keep None initially.
    ndvi_min: float | None = None

    # Two-component GMM on log(NIR)
    gmm_components: int = 2
    random_seed: int = 42
    max_gmm_samples: int = 500_000

    # Final 2 m comparison filtering
    forest_fraction_min: float = ILLUMINATION_CONFIG.forest_fraction_min
    scaffold_fraction_max: float = 0.0
    uav_valid_crown_fraction_min: float = 0.80

    # Polygon rasterization
    polygon_all_touched: bool = False
    scaffold_all_touched: bool = True


VALIDATION_DATE = "20260805"
FLIGHT_ID = "1008_L1_E"
PROFILE_NAME = "SFMNN"

UAV_REFL_FILE = Path(
    r"E:\Proj1_Pfynwald_Data\HyPlant\coregistration"
    r"\UAV_20260805"
    r"\ortho_Pfynwald_20260805_0945_MX_refl_georef_coreg.tif"
)

HYPLANT_FSUN_FILE = (
    OUT_ROOT
    / "illumination"
    / VALIDATION_DATE
    / f"f_sun_veg_{FLIGHT_ID}.tif"
)

VALIDATION_OUT_DIR = (
    OUT_ROOT
    / "illumination"
    / "validation_uav"
)

UAV_CFG = UAVIlluminationValidationConfig()


# ---------- General helpers ----------

def get_treatment_vector() -> Path:
    profiles = get_profiles()
    if PROFILE_NAME not in profiles:
        raise KeyError(
            f"Profile '{PROFILE_NAME}' not found. Available: {list(profiles)}"
        )

    path = Path(profiles[PROFILE_NAME].treatment_areas_shp)
    if not path.exists():
        raise FileNotFoundError(f"Treatment polygons not found: {path}")
    return path


def validate_input_paths(treatment_vector: Path) -> None:
    required = {
        "UAV reflectance": UAV_REFL_FILE,
        "HyPlant sunlit fraction": HYPLANT_FSUN_FILE,
        "forest nDSM": Path(FOREST_NDSM_RASTER),
        "scaffold polygons": Path(SCAFFOLD_VECTOR),
        "treatment polygons": treatment_vector,
        "2 m forest fraction": forest_fraction_path(),
        "2 m scaffold fraction": scaffold_fraction_path(),
    }

    missing = [
        (label, Path(path))
        for label, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        text = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise FileNotFoundError(f"Missing validation input files:\n{text}")


def get_horizontal_crs(crs) -> RasterioCRS:
    parsed = PyprojCRS.from_user_input(crs)

    if parsed.is_compound:
        for sub_crs in parsed.sub_crs_list:
            if sub_crs.is_projected:
                return RasterioCRS.from_wkt(sub_crs.to_wkt())
        raise ValueError(f"No projected horizontal CRS found in: {crs}")

    return RasterioCRS.from_wkt(parsed.to_wkt())


def get_grid(path: Path, label: str, horizontal_only: bool = False) -> dict:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{label} has no CRS: {path}")

        crs = get_horizontal_crs(src.crs) if horizontal_only else src.crs

        return {
            "crs": crs,
            "source_crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "shape": (src.height, src.width),
            "bounds": src.bounds,
            "nodata": src.nodata,
            "count": src.count,
        }


def check_spatial_overlap(uav_grid: dict, hyplant_grid: dict) -> None:
    uav_bounds_hyplant = transform_bounds(
        uav_grid["crs"],
        hyplant_grid["crs"],
        *uav_grid["bounds"],
        densify_pts=21,
    )

    ul, ub, ur, ut = uav_bounds_hyplant
    hl, hb, hr, ht = hyplant_grid["bounds"]

    width = min(ur, hr) - max(ul, hl)
    height = min(ut, ht) - max(ub, hb)

    if width <= 0 or height <= 0:
        raise ValueError("UAV and HyPlant rasters do not overlap.")

    print(f"Spatial overlap: {width:.1f} x {height:.1f} m")


def window_slices(window):
    r0 = int(window.row_off)
    c0 = int(window.col_off)
    r1 = r0 + int(window.height)
    c1 = c0 + int(window.width)
    return slice(r0, r1), slice(c0, c1)


def write_float_raster(path: Path, array: np.ndarray, grid: dict, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": NODATA_OUT,
        "compress": "deflate",
        "tiled": True,
    }

    data = np.where(np.isfinite(array), array, NODATA_OUT).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)


def write_uint8_raster(
    path: Path,
    array: np.ndarray,
    grid: dict,
    description: str,
    nodata: int = 255,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": "uint8",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("uint8"), 1)
        dst.set_band_description(1, description)


# ---------- 4 cm analysis mask ----------

def rasterize_vector_mask(
    vector_path: Path,
    grid: dict,
    all_touched: bool,
) -> np.ndarray:
    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError(f"Vector has no CRS: {vector_path}")

    gdf = gdf.to_crs(grid["crs"])
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError(f"No valid geometries in: {vector_path}")

    return geometry_mask(
        gdf.geometry,
        out_shape=grid["shape"],
        transform=grid["transform"],
        invert=True,
        all_touched=all_touched,
    )


def build_uav_forest_mask(ndsm_path: Path, uav_grid: dict) -> np.ndarray:
    with rasterio.open(ndsm_path) as src:
        if src.crs is None:
            raise ValueError(f"nDSM has no CRS: {ndsm_path}")

        z = src.read(1).astype("float32")
        valid = np.isfinite(z)
        if src.nodata is not None:
            valid &= z != src.nodata

        forest_native = (
            valid & (z >= FOREST_MASK_CONFIG.height_threshold_m)
        ).astype("uint8")

        forest_uav = np.zeros(uav_grid["shape"], dtype="uint8")

        reproject(
            source=forest_native,
            destination=forest_uav,
            src_transform=src.transform,
            src_crs=get_horizontal_crs(src.crs),
            dst_transform=uav_grid["transform"],
            dst_crs=uav_grid["crs"],
            resampling=Resampling.nearest,
        )

    return forest_uav.astype(bool)


def build_uav_spatial_mask(
    treatment_vector: Path,
    uav_grid: dict,
) -> np.ndarray:
    print("\n=== Building 4 cm spatial mask ===")

    forest = build_uav_forest_mask(Path(FOREST_NDSM_RASTER), uav_grid)

    treatment = rasterize_vector_mask(
        treatment_vector,
        uav_grid,
        all_touched=UAV_CFG.polygon_all_touched,
    )

    scaffold = rasterize_vector_mask(
        Path(SCAFFOLD_VECTOR),
        uav_grid,
        all_touched=UAV_CFG.scaffold_all_touched,
    )

    clean = forest & treatment & ~scaffold

    print(f"Forest pixels:        {int(forest.sum()):,}")
    print(f"Treatment pixels:     {int(treatment.sum()):,}")
    print(f"Scaffold pixels:      {int(scaffold.sum()):,}")
    print(f"Clean forest ROI:     {int(clean.sum()):,}")

    del forest, treatment, scaffold
    return clean


# ---------- UAV reflectance / GMM ----------

def read_uav_block(src, window):
    nir = src.read(UAV_CFG.nir_band, window=window, masked=True)
    nir = nir.astype("float32").filled(np.nan)

    red = None
    if UAV_CFG.ndvi_min is not None:
        red = src.read(UAV_CFG.red_band, window=window, masked=True)
        red = red.astype("float32").filled(np.nan)

    return nir, red


def valid_uav_pixels(
    nir: np.ndarray,
    red: np.ndarray | None,
    spatial_mask: np.ndarray,
) -> np.ndarray:
    valid = spatial_mask & np.isfinite(nir) & (nir > 0)

    if UAV_CFG.ndvi_min is not None:
        valid &= red is not None
        valid &= np.isfinite(red)

        den = nir + red
        ndvi_ok = np.abs(den) > 1e-12
        ndvi = np.full(nir.shape, np.nan, dtype="float32")
        ndvi[ndvi_ok] = (nir[ndvi_ok] - red[ndvi_ok]) / den[ndvi_ok]

        valid &= ndvi_ok & (ndvi >= UAV_CFG.ndvi_min)

    return valid


def count_valid_uav_pixels(spatial_mask: np.ndarray) -> int:
    total = 0

    with rasterio.open(UAV_REFL_FILE) as src:
        for _, window in src.block_windows(UAV_CFG.nir_band):
            rs, cs = window_slices(window)
            nir, red = read_uav_block(src, window)
            valid = valid_uav_pixels(nir, red, spatial_mask[rs, cs])
            total += int(valid.sum())

    return total


def sample_log_nir(spatial_mask: np.ndarray) -> np.ndarray:
    n_valid = count_valid_uav_pixels(spatial_mask)

    if n_valid < 100:
        raise ValueError(
            f"Only {n_valid} valid UAV crown pixels available for GMM fitting."
        )

    sample_size = min(UAV_CFG.max_gmm_samples, n_valid)
    rng = np.random.default_rng(UAV_CFG.random_seed)
    samples = []
    remaining_total = n_valid
    remaining_sample = sample_size

    with rasterio.open(UAV_REFL_FILE) as src:
        for _, window in src.block_windows(UAV_CFG.nir_band):
            if remaining_sample == 0:
                break

            rs, cs = window_slices(window)
            nir, red = read_uav_block(src, window)
            valid = valid_uav_pixels(nir, red, spatial_mask[rs, cs])

            values = nir[valid]
            m = values.size
            if m == 0:
                continue

            take = int(
                rng.hypergeometric(
                    ngood=m,
                    nbad=remaining_total - m,
                    nsample=remaining_sample,
                )
            )

            if take > 0:
                idx = rng.choice(m, size=take, replace=False)
                samples.append(np.log(values[idx].astype("float64")))

            remaining_total -= m
            remaining_sample -= take

    x = np.concatenate(samples)

    if x.size != sample_size:
        raise RuntimeError(
            f"GMM sampling failed: expected {sample_size}, got {x.size}."
        )

    print(f"Valid UAV crown pixels: {n_valid:,}")
    print(f"GMM sample size:        {x.size:,}")
    return x


def fit_gmm(log_nir: np.ndarray):
    if UAV_CFG.gmm_components != 2:
        raise ValueError("Binary sun/shade validation requires gmm_components=2.")

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=UAV_CFG.random_seed,
        n_init=10,
    )
    gmm.fit(log_nir[:, None])

    means = gmm.means_.ravel()
    shade_component = int(np.argmin(means))
    sun_component = int(np.argmax(means))

    # Diagnostic boundary between the two component means.
    x_grid = np.linspace(
        means[shade_component],
        means[sun_component],
        20_000,
    )
    p = gmm.predict_proba(x_grid[:, None])
    threshold_log = float(
        x_grid[np.argmin(np.abs(p[:, sun_component] - p[:, shade_component]))]
    )

    return {
        "model": gmm,
        "shade_component": shade_component,
        "sun_component": sun_component,
        "threshold_log_nir": threshold_log,
        "threshold_nir": float(np.exp(threshold_log)),
    }


def plot_gmm_diagnostic(log_nir: np.ndarray, gmm_info: dict, out_path: Path) -> None:
    gmm = gmm_info["model"]

    lo, hi = np.quantile(log_nir, [0.001, 0.999])
    x = np.linspace(lo, hi, 1000)

    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.reshape(-1))
    weights = gmm.weights_.ravel()

    plt.figure(figsize=(8, 5))
    plt.hist(log_nir, bins=150, density=True, alpha=0.35)

    total = np.zeros_like(x)
    for mean, std, weight in zip(means, stds, weights):
        density = (
            weight
            / (std * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
        )
        total += density
        plt.plot(x, density, linewidth=1.5)

    plt.plot(x, total, linewidth=2)
    plt.axvline(
        gmm_info["threshold_log_nir"],
        linestyle="--",
        linewidth=1.5,
        label="GMM decision boundary",
    )
    plt.xlabel("log(NIR 842 nm)")
    plt.ylabel("Density")
    plt.title(f"UAV sun/shade GMM | {VALIDATION_DATE}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def classify_uav(
    spatial_mask: np.ndarray,
    uav_grid: dict,
    gmm_info: dict,
) -> tuple[Path, Path, dict]:
    sunshade_path = VALIDATION_OUT_DIR / "uav_sunshade_4cm.tif"
    valid_path = VALIDATION_OUT_DIR / "uav_valid_crown_4cm.tif"

    class_profile = {
        "driver": "GTiff",
        "height": uav_grid["height"],
        "width": uav_grid["width"],
        "count": 1,
        "dtype": "uint8",
        "crs": uav_grid["crs"],
        "transform": uav_grid["transform"],
        "nodata": 255,
        "compress": "deflate",
        "tiled": True,
    }

    valid_profile = class_profile.copy()
    valid_profile["nodata"] = None

    gmm = gmm_info["model"]
    sun_component = gmm_info["sun_component"]

    n_sun = 0
    n_shade = 0

    with rasterio.open(UAV_REFL_FILE) as src, \
         rasterio.open(sunshade_path, "w", **class_profile) as dst_class, \
         rasterio.open(valid_path, "w", **valid_profile) as dst_valid:

        dst_class.set_band_description(1, "0=shaded, 1=sunlit, 255=invalid")
        dst_valid.set_band_description(1, "valid UAV crown pixel")

        for _, window in src.block_windows(UAV_CFG.nir_band):
            rs, cs = window_slices(window)
            nir, red = read_uav_block(src, window)
            valid = valid_uav_pixels(nir, red, spatial_mask[rs, cs])

            classes = np.full(nir.shape, 255, dtype="uint8")
            valid_out = valid.astype("uint8")

            if np.any(valid):
                labels = gmm.predict(
                    np.log(nir[valid].astype("float64"))[:, None]
                )
                sun = labels == sun_component
                classes[valid] = sun.astype("uint8")

                n_sun += int(sun.sum())
                n_shade += int((~sun).sum())

            dst_class.write(classes, 1, window=window)
            dst_valid.write(valid_out, 1, window=window)

    counts = {
        "sunlit_pixels": n_sun,
        "shaded_pixels": n_shade,
        "valid_pixels": n_sun + n_shade,
        "sunlit_fraction_4cm": (
            n_sun / (n_sun + n_shade)
            if (n_sun + n_shade) > 0
            else np.nan
        ),
    }

    print(f"Sunlit UAV pixels:      {n_sun:,}")
    print(f"Shaded UAV pixels:      {n_shade:,}")

    return sunshade_path, valid_path, counts


# ---------- Aggregate UAV classification to HyPlant grid ----------

def aggregate_uav_to_hyplant(
    sunshade_path: Path,
    valid_path: Path,
    hyplant_grid: dict,
) -> tuple[np.ndarray, np.ndarray]:
    f_sun = np.full(hyplant_grid["shape"], np.nan, dtype="float32")
    valid_fraction = np.full(hyplant_grid["shape"], np.nan, dtype="float32")

    with rasterio.open(sunshade_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=f_sun,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=255,
            dst_transform=hyplant_grid["transform"],
            dst_crs=hyplant_grid["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )

    with rasterio.open(valid_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=valid_fraction,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=hyplant_grid["transform"],
            dst_crs=hyplant_grid["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )

    f_sun = np.clip(f_sun, 0.0, 1.0)
    valid_fraction = np.clip(valid_fraction, 0.0, 1.0)

    write_float_raster(
        VALIDATION_OUT_DIR / "uav_f_sun_2m.tif",
        f_sun,
        hyplant_grid,
        "UAV-derived sunlit crown fraction",
    )
    write_float_raster(
        VALIDATION_OUT_DIR / "uav_valid_crown_fraction_2m.tif",
        valid_fraction,
        hyplant_grid,
        "fraction of 2 m pixel with valid UAV crown classification",
    )

    return f_sun, valid_fraction


# ---------- 2 m comparison ----------

def load_matching_raster(path: Path, grid: dict, label: str) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.crs != grid["crs"]:
            raise ValueError(f"{label} CRS does not match HyPlant grid: {path}")
        if (src.height, src.width) != grid["shape"]:
            raise ValueError(f"{label} shape does not match HyPlant grid: {path}")
        if not src.transform.almost_equals(grid["transform"]):
            raise ValueError(f"{label} transform does not match HyPlant grid: {path}")

        return src.read(1, masked=True).filled(np.nan).astype("float32")


def build_treatment_mask_2m(
    treatment_vector: Path,
    hyplant_grid: dict,
) -> np.ndarray:
    return rasterize_vector_mask(
        treatment_vector,
        hyplant_grid,
        all_touched=UAV_CFG.polygon_all_touched,
    )


def build_comparison_mask(
    treatment_vector: Path,
    hyplant_grid: dict,
    f_sun_uav: np.ndarray,
    uav_valid_fraction: np.ndarray,
):
    f_sun_hyplant = load_matching_raster(
        HYPLANT_FSUN_FILE,
        hyplant_grid,
        "HyPlant f_sun",
    )
    forest_fraction = load_matching_raster(
        forest_fraction_path(),
        hyplant_grid,
        "forest fraction",
    )
    scaffold_fraction = load_matching_raster(
        scaffold_fraction_path(),
        hyplant_grid,
        "scaffold fraction",
    )
    treatment = build_treatment_mask_2m(treatment_vector, hyplant_grid)

    forest_ok = (
        np.isfinite(forest_fraction)
        & (forest_fraction >= UAV_CFG.forest_fraction_min)
    )
    scaffold_ok = (
        np.isfinite(scaffold_fraction)
        & (
            scaffold_fraction
            <= UAV_CFG.scaffold_fraction_max + 1e-12
        )
    )
    uav_ok = (
        np.isfinite(uav_valid_fraction)
        & (
            uav_valid_fraction
            >= UAV_CFG.uav_valid_crown_fraction_min
        )
    )

    comparison = (
        treatment
        & forest_ok
        & scaffold_ok
        & uav_ok
        & np.isfinite(f_sun_hyplant)
        & np.isfinite(f_sun_uav)
    )

    comparison_out = np.full(hyplant_grid["shape"], 255, dtype="uint8")
    comparison_out[:] = 0
    comparison_out[comparison] = 1

    write_uint8_raster(
        VALIDATION_OUT_DIR / "comparison_mask_2m.tif",
        comparison_out,
        hyplant_grid,
        "1=used for UAV-HyPlant validation, 0=excluded",
        nodata=255,
    )

    print("\n=== Final 2 m comparison mask ===")
    print(f"Treatment pixels:       {int(treatment.sum()):,}")
    print(f"Forest pixels:          {int(forest_ok.sum()):,}")
    print(f"Scaffold-clean pixels:  {int(scaffold_ok.sum()):,}")
    print(f"UAV coverage pixels:    {int(uav_ok.sum()):,}")
    print(f"Comparison pixels:      {int(comparison.sum()):,}")

    return comparison, f_sun_hyplant


def validation_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    # reference = UAV, estimate = HyPlant
    error = estimate - reference
    n = reference.size

    if n < 2:
        return {
            "n": int(n),
            "bias_hyplant_minus_uav": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
            "r2": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    r = float(np.corrcoef(reference, estimate)[0, 1])
    slope, intercept = np.polyfit(reference, estimate, 1)

    return {
        "n": int(n),
        "mean_uav": float(np.mean(reference)),
        "mean_hyplant": float(np.mean(estimate)),
        "bias_hyplant_minus_uav": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson_r": r,
        "r2": float(r ** 2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def summarize_by_plot(
    treatment_vector: Path,
    hyplant_grid: dict,
    comparison_mask: np.ndarray,
    f_sun_uav: np.ndarray,
    f_sun_hyplant: np.ndarray,
) -> pd.DataFrame:
    gdf = gpd.read_file(treatment_vector)
    if gdf.crs is None:
        raise ValueError(f"Treatment vector has no CRS: {treatment_vector}")

    gdf = gdf.to_crs(hyplant_grid["crs"])

    plot_field = ILLUMINATION_CONFIG.plot_id_field
    treatment_field = ILLUMINATION_CONFIG.treatment_field

    if plot_field not in gdf.columns:
        raise KeyError(f"'{plot_field}' not found in treatment polygons.")
    if treatment_field not in gdf.columns:
        raise KeyError(f"'{treatment_field}' not found in treatment polygons.")

    treatment_names = {
        1: "control",
        2: "irrigated",
        3: "irrigation_stopped",
    }

    rows = []
    for _, rec in gdf.iterrows():
        if rec.geometry is None or rec.geometry.is_empty:
            continue

        polygon_mask = geometry_mask(
            [rec.geometry],
            out_shape=hyplant_grid["shape"],
            transform=hyplant_grid["transform"],
            invert=True,
            all_touched=UAV_CFG.polygon_all_touched,
        )

        use = comparison_mask & polygon_mask
        if not np.any(use):
            continue

        metrics = validation_metrics(
            f_sun_uav[use],
            f_sun_hyplant[use],
        )

        treatment_value = rec[treatment_field]
        try:
            treatment_label = treatment_names.get(
                int(treatment_value),
                str(treatment_value),
            )
        except (TypeError, ValueError):
            treatment_label = str(treatment_value)

        rows.append(
            {
                "plot_id": str(rec[plot_field]),
                "treatment": treatment_label,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def save_validation_scatter(
    f_sun_uav: np.ndarray,
    f_sun_hyplant: np.ndarray,
    comparison_mask: np.ndarray,
    metrics: dict,
    out_path: Path,
) -> None:
    x = f_sun_uav[comparison_mask]
    y = f_sun_hyplant[comparison_mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.35)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    plt.xlabel("UAV sunlit crown fraction")
    plt.ylabel("HyPlant f_sun_veg")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        f"{VALIDATION_DATE} | {FLIGHT_ID}\n"
        f"n={metrics['n']}, RMSE={metrics['rmse']:.3f}, "
        f"bias={metrics['bias_hyplant_minus_uav']:.3f}, "
        f"R²={metrics['r2']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pixel_table(
    comparison_mask: np.ndarray,
    f_sun_uav: np.ndarray,
    f_sun_hyplant: np.ndarray,
    hyplant_grid: dict,
    out_path: Path,
) -> None:
    rows, cols = np.nonzero(comparison_mask)

    xs, ys = rasterio.transform.xy(
        hyplant_grid["transform"],
        rows,
        cols,
        offset="center",
    )

    table = pd.DataFrame(
        {
            "row": rows,
            "col": cols,
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "f_sun_uav": f_sun_uav[rows, cols],
            "f_sun_hyplant": f_sun_hyplant[rows, cols],
        }
    )
    table["error_hyplant_minus_uav"] = (
        table["f_sun_hyplant"] - table["f_sun_uav"]
    )
    table.to_csv(out_path, index=False)


# ---------- Main ----------

def main():
    print("\n=== UAV illumination validation ===")
    print(f"Date:           {VALIDATION_DATE}")
    print(f"HyPlant flight: {FLIGHT_ID}")

    treatment_vector = get_treatment_vector()
    validate_input_paths(treatment_vector)
    VALIDATION_OUT_DIR.mkdir(parents=True, exist_ok=True)

    uav_grid = get_grid(
        UAV_REFL_FILE,
        "UAV reflectance",
        horizontal_only=True,
    )
    hyplant_grid = get_grid(
        HYPLANT_FSUN_FILE,
        "HyPlant f_sun",
        horizontal_only=False,
    )

    if uav_grid["count"] < UAV_CFG.nir_band:
        raise ValueError(
            f"UAV raster has only {uav_grid['count']} bands; "
            f"NIR band {UAV_CFG.nir_band} requested."
        )

    print(
        f"UAV grid:      {uav_grid['height']} x {uav_grid['width']} | "
        f"{abs(uav_grid['transform'].a):.4f} x "
        f"{abs(uav_grid['transform'].e):.4f} m | {uav_grid['crs']}"
    )
    print(
        f"HyPlant grid:  {hyplant_grid['height']} x {hyplant_grid['width']} | "
        f"{abs(hyplant_grid['transform'].a):.2f} x "
        f"{abs(hyplant_grid['transform'].e):.2f} m | {hyplant_grid['crs']}"
    )

    check_spatial_overlap(uav_grid, hyplant_grid)

    spatial_mask = build_uav_spatial_mask(
        treatment_vector,
        uav_grid,
    )

    print("\n=== Fitting UAV sun/shade GMM ===")
    log_nir = sample_log_nir(spatial_mask)
    gmm_info = fit_gmm(log_nir)

    gmm = gmm_info["model"]
    means = np.exp(gmm.means_.ravel())

    print(
        f"GMM NIR component means: "
        f"{means[gmm_info['shade_component']]:.6g} (shade), "
        f"{means[gmm_info['sun_component']]:.6g} (sun)"
    )
    print(
        f"Diagnostic NIR boundary: {gmm_info['threshold_nir']:.6g}"
    )

    plot_gmm_diagnostic(
        log_nir,
        gmm_info,
        VALIDATION_OUT_DIR / "uav_gmm_diagnostic.png",
    )

    print("\n=== Classifying 4 cm UAV crown pixels ===")
    sunshade_path, valid_path, class_counts = classify_uav(
        spatial_mask,
        uav_grid,
        gmm_info,
    )

    del spatial_mask

    print("\n=== Aggregating UAV classification to 2 m ===")
    f_sun_uav, uav_valid_fraction = aggregate_uav_to_hyplant(
        sunshade_path,
        valid_path,
        hyplant_grid,
    )

    comparison_mask, f_sun_hyplant = build_comparison_mask(
        treatment_vector,
        hyplant_grid,
        f_sun_uav,
        uav_valid_fraction,
    )

    if not np.any(comparison_mask):
        raise ValueError("No valid 2 m pixels remain for UAV-HyPlant comparison.")

    overall = validation_metrics(
        f_sun_uav[comparison_mask],
        f_sun_hyplant[comparison_mask],
    )

    print("\n=== UAV vs HyPlant validation ===")
    print(f"n:       {overall['n']:,}")
    print(f"Bias:    {overall['bias_hyplant_minus_uav']:.4f}")
    print(f"MAE:     {overall['mae']:.4f}")
    print(f"RMSE:    {overall['rmse']:.4f}")
    print(f"R:       {overall['pearson_r']:.4f}")
    print(f"R²:      {overall['r2']:.4f}")
    print(f"Slope:   {overall['slope']:.4f}")
    print(f"Intercept:{overall['intercept']:.4f}")

    pd.DataFrame([overall]).to_csv(
        VALIDATION_OUT_DIR / "validation_summary.csv",
        index=False,
    )

    by_plot = summarize_by_plot(
        treatment_vector,
        hyplant_grid,
        comparison_mask,
        f_sun_uav,
        f_sun_hyplant,
    )
    by_plot.to_csv(
        VALIDATION_OUT_DIR / "validation_by_plot.csv",
        index=False,
    )

    save_pixel_table(
        comparison_mask,
        f_sun_uav,
        f_sun_hyplant,
        hyplant_grid,
        VALIDATION_OUT_DIR / "validation_pixels.csv",
    )

    save_validation_scatter(
        f_sun_uav,
        f_sun_hyplant,
        comparison_mask,
        overall,
        VALIDATION_OUT_DIR / "validation_scatter.png",
    )

    metadata = {
        "validation_date": VALIDATION_DATE,
        "flight_id": FLIGHT_ID,
        "uav_reflectance_file": str(UAV_REFL_FILE),
        "hyplant_fsun_file": str(HYPLANT_FSUN_FILE),
        "forest_ndsm": str(FOREST_NDSM_RASTER),
        "scaffold_vector": str(SCAFFOLD_VECTOR),
        "treatment_vector": str(treatment_vector),
        "config": asdict(UAV_CFG),
        "forest_height_threshold_m": FOREST_MASK_CONFIG.height_threshold_m,
        "gmm": {
            "means_log_nir": gmm.means_.ravel().tolist(),
            "means_nir": np.exp(gmm.means_.ravel()).tolist(),
            "weights": gmm.weights_.ravel().tolist(),
            "std_log_nir": np.sqrt(
                gmm.covariances_.reshape(-1)
            ).tolist(),
            "shade_component": gmm_info["shade_component"],
            "sun_component": gmm_info["sun_component"],
            "diagnostic_threshold_log_nir": gmm_info["threshold_log_nir"],
            "diagnostic_threshold_nir": gmm_info["threshold_nir"],
        },
        "uav_class_counts": class_counts,
        "overall_validation": overall,
    }

    (
        VALIDATION_OUT_DIR / "validation_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved validation outputs to:\n{VALIDATION_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())