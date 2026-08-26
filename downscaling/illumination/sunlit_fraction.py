from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray  # noqa: F401 - registers the xarray .rio accessor
import xarray as xr
from rasterio.features import geometry_mask
from scipy.stats import rankdata
from shapely.geometry import Point

from config.config_downscaling import FILL_VALUE, REFL_SCALE
from config.config_illumination import SunlitFractionConfig
from downscaling.compute_downscaling_indices import open_and_scale


@dataclass
class EndmemberLibrary:
    wavelengths: np.ndarray
    sunlit: np.ndarray
    shaded: np.ndarray

    sunlit_p10: np.ndarray
    sunlit_p90: np.ndarray
    shaded_p10: np.ndarray
    shaded_p90: np.ndarray

    red_edge_wavelength_nm: float
    candidate_counts: Dict[str, int]

    def matrix(self, band_indices: np.ndarray) -> np.ndarray:
        """Return [n_wavelengths, 2] matrix in sunlit/shaded order."""
        return np.column_stack(
            [
                self.sunlit[band_indices],
                self.shaded[band_indices],
            ]
        )


def _band_indices(
    wavelengths: Sequence[float],
    spectral_range: Tuple[float, float],
) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    low, high = spectral_range
    idx = np.flatnonzero((wl >= low) & (wl <= high))
    if idx.size == 0:
        center = 0.5 * (low + high)
        idx = np.asarray([int(np.argmin(np.abs(wl - center)))])
    return idx


def _unmixing_band_indices(
    wavelengths: Sequence[float],
    cfg: SunlitFractionConfig,
) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    low, high = cfg.unmixing_range
    keep = (wl >= low) & (wl <= high)
    for ex_low, ex_high in cfg.exclude_ranges:
        keep &= ~((wl >= ex_low) & (wl <= ex_high))
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        raise ValueError(
            f"Too few bands ({idx.size}) in unmixing range {cfg.unmixing_range}."
        )
    return idx


def _mean_reflectance(cube: xr.DataArray, wavelengths, spectral_range):
    idx = _band_indices(wavelengths, spectral_range)
    return cube.isel(band=idx).mean(dim="band", skipna=True)


def _raw_ndvi(cube: xr.DataArray, wavelengths, cfg: SunlitFractionConfig):
    red = _mean_reflectance(cube, wavelengths, cfg.red_range)
    nir = _mean_reflectance(cube, wavelengths, cfg.nir_range)
    den = nir + red
    return xr.where(np.abs(den) > 1e-8, (nir - red) / den, np.nan)


def _pixel_centers(transform, rows: np.ndarray, cols: np.ndarray):
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return list(zip(xs, ys))


def _sample_mask_coordinates(
    mask: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return rows, cols
    if rows.size > max_samples:
        pick = rng.choice(rows.size, size=max_samples, replace=False)
        rows = rows[pick]
        cols = cols[pick]
    return rows, cols


def _sample_spectra(
    filepath,
    rows: np.ndarray,
    cols: np.ndarray,
    expected_bands: int,
) -> np.ndarray:
    """Sample full reflectance spectra at raster row/column positions."""
    if rows.size == 0:
        return np.empty((0, expected_bands), dtype="float32")

    with rasterio.open(filepath) as src:
        if src.count != expected_bands:
            raise ValueError(
                f"Wavelength count ({expected_bands}) does not match raster bands "
                f"({src.count}) for {filepath}."
            )
        coords = _pixel_centers(src.transform, rows, cols)
        sampled = np.asarray(list(src.sample(coords)), dtype="float32")

    invalid = (
        (sampled < 0)
        | (sampled > REFL_SCALE)
        | (sampled == FILL_VALUE)
        | ~np.isfinite(sampled)
    )
    sampled = sampled / float(REFL_SCALE)
    sampled[invalid] = np.nan
    return sampled


def _robust_spectral_stats(spectra: np.ndarray, label: str):
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        raise ValueError(f"No spectra available for {label} endmember.")

    valid_rows = np.isfinite(spectra).mean(axis=1) >= 0.98
    spectra = spectra[valid_rows]
    if spectra.shape[0] == 0:
        raise ValueError(f"No sufficiently complete spectra for {label} endmember.")

    median = np.nanmedian(spectra, axis=0)
    p10 = np.nanpercentile(spectra, 10, axis=0)
    p90 = np.nanpercentile(spectra, 90, axis=0)
    return (
        median.astype("float32"),
        p10.astype("float32"),
        p90.astype("float32"),
        spectra.shape[0],
    )


def load_fraction_raster(
    raster_path,
    template: xr.DataArray,
    name: str,
) -> xr.DataArray:
    """Load a precomputed fraction raster on the exact hyperspectral grid."""
    with rasterio.open(raster_path) as src:
        if src.crs != template.rio.crs:
            raise ValueError(f"{name} CRS does not match: {raster_path}")
        if (src.height, src.width) != template.shape:
            raise ValueError(f"{name} shape does not match: {raster_path}")
        if not src.transform.almost_equals(template.rio.transform()):
            raise ValueError(f"{name} transform does not match: {raster_path}")

        data = src.read(1, masked=True).filled(np.nan).astype("float32")

    da = xr.DataArray(
        data,
        dims=template.dims,
        coords=template.coords,
        name=name,
    )
    return da.rio.write_crs(template.rio.crs).rio.write_transform(template.rio.transform())


def detect_red_edge_wavelength(
    filepath,
    wavelengths: Sequence[float],
    vegetation_mask: np.ndarray,
    cfg: SunlitFractionConfig,
) -> float:
    """Detect a scene-specific red-edge wavelength from balanced forest spectra."""
    if cfg.red_edge_wavelength_nm is not None:
        return float(cfg.red_edge_wavelength_nm)

    rng = np.random.default_rng(cfg.random_seed)
    rows, cols = _sample_mask_coordinates(
        vegetation_mask,
        cfg.max_red_edge_samples,
        rng,
    )
    if rows.size < cfg.min_candidate_spectra:
        raise ValueError(
            f"Only {rows.size} forest pixels available for red-edge detection."
        )

    spectra = _sample_spectra(filepath, rows, cols, len(wavelengths))
    wl = np.asarray(wavelengths, dtype=float)
    idx = _band_indices(wl, cfg.red_edge_search_range)

    median = np.nanmedian(spectra[:, idx], axis=0)
    local_wl = wl[idx]
    valid = np.isfinite(median) & np.isfinite(local_wl)
    if valid.sum() < 3:
        raise ValueError("Insufficient valid bands for red-edge detection.")

    local_wl = local_wl[valid]
    median = median[valid]
    gradient = np.gradient(median, local_wl)
    return float(local_wl[int(np.nanargmax(gradient))])


def compute_ndcsi(
    cube: xr.DataArray,
    wavelengths: Sequence[float],
    raw_ndvi: xr.DataArray,
    vegetation_mask: xr.DataArray,
    red_edge_wavelength_nm: float,
    cfg: SunlitFractionConfig,
    scaling_mask: Optional[xr.DataArray] = None,
):
    """Compute NDCSI using robust red-edge reflectance limits.

    NDCSI = NDVI * (rho_RE - rho_RE_min) / (rho_RE_max - rho_RE_min)

    rho_RE_min/max are estimated from robust forest quantiles. The calculation
    mask can cover the full forest scene while the scaling mask is restricted to
    a balanced sample from the treatment plots.
    """
    re_range = (
        red_edge_wavelength_nm - cfg.red_edge_half_width_nm,
        red_edge_wavelength_nm + cfg.red_edge_half_width_nm,
    )
    rho_re = _mean_reflectance(cube, wavelengths, re_range)

    reference = vegetation_mask if scaling_mask is None else scaling_mask
    re_values = rho_re.where(reference).values
    re_values = re_values[np.isfinite(re_values)]
    if re_values.size == 0:
        raise ValueError("No valid red-edge reflectance values in NDCSI scaling mask.")

    re_min = float(np.quantile(re_values, cfg.ndcsi_low_quantile))
    re_max = float(np.quantile(re_values, cfg.ndcsi_high_quantile))
    if not np.isfinite(re_min) or not np.isfinite(re_max) or re_max <= re_min:
        raise ValueError("Invalid robust red-edge range for NDCSI.")

    ndcsi = raw_ndvi * (rho_re - re_min) / (re_max - re_min)
    ndcsi = ndcsi.clip(min=0.0, max=1.0).where(vegetation_mask)
    ndcsi.attrs.update(
        {
            "long_name": "normalized difference canopy shadow index",
            "red_edge_wavelength_nm": red_edge_wavelength_nm,
            "red_edge_low_quantile": cfg.ndcsi_low_quantile,
            "red_edge_high_quantile": cfg.ndcsi_high_quantile,
            "red_edge_reflectance_low": re_min,
            "red_edge_reflectance_high": re_max,
        }
    )
    return ndcsi, rho_re


def build_plot_masks(
    treatment_shp,
    template: xr.DataArray,
    cfg: SunlitFractionConfig,
):
    """Return one in-memory Boolean mask per treatment polygon."""
    gdf = gpd.read_file(treatment_shp)
    if gdf.crs is None:
        raise ValueError(f"Treatment shapefile has no CRS: {treatment_shp}")
    if template.rio.crs is None:
        raise ValueError("Hyperspectral raster has no CRS.")
    if cfg.treatment_field not in gdf.columns:
        raise KeyError(f"'{cfg.treatment_field}' not found in shapefile.")
    if cfg.plot_id_field not in gdf.columns:
        raise KeyError(f"'{cfg.plot_id_field}' not found in shapefile.")

    gdf = gdf.to_crs(template.rio.crs)
    treatment_names = {1: "control", 2: "irrigated", 3: "irrigation_stopped"}
    records = []

    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        mask = geometry_mask(
            [row.geometry],
            out_shape=template.shape,
            transform=template.rio.transform(),
            invert=True,
            all_touched=cfg.polygon_all_touched,
        )
        if not np.any(mask):
            continue

        code = int(row[cfg.treatment_field])
        records.append(
            {
                "plot_id": str(row[cfg.plot_id_field]),
                "treatment": treatment_names.get(code, str(code)),
                "mask": mask,
            }
        )

    if not records:
        raise ValueError("No treatment polygons overlap the hyperspectral raster.")
    return records


def balanced_forest_reference(
    raw_ndvi: xr.DataArray,
    forest_mask: np.ndarray,
    plot_records,
    cfg: SunlitFractionConfig,
) -> np.ndarray:
    """Equal forest sample from every plot for red-edge/NDCSI scaling."""
    ndvi = np.asarray(raw_ndvi.values)
    rng = np.random.default_rng(cfg.random_seed)

    plot_forest = [
        rec["mask"] & forest_mask & np.isfinite(ndvi)
        for rec in plot_records
    ]
    counts = [int(mask.sum()) for mask in plot_forest]
    if min(counts) < cfg.min_candidate_spectra:
        smallest = int(np.argmin(counts))
        rec = plot_records[smallest]
        raise ValueError(
            f"Plot {rec['plot_id']} ({rec['treatment']}) has only {counts[smallest]} "
            f"forest pixels; need at least {cfg.min_candidate_spectra}."
        )

    target = min(
        min(counts),
        max(1, cfg.max_red_edge_samples // len(plot_forest)),
    )

    balanced = np.zeros(ndvi.shape, dtype=bool)
    for mask in plot_forest:
        rows, cols = _sample_mask_coordinates(mask, target, rng)
        balanced[rows, cols] = True

    print(f"    NDCSI scaling: {target} forest pixels per plot")
    return balanced


def _local_percentiles(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.asarray([0.5], dtype="float64")
    ranks = rankdata(values, method="average")
    return (ranks - 1.0) / (values.size - 1.0)


def _spatially_distributed_support(
    records,
    label: str,
    cfg: SunlitFractionConfig,
):
    """Select the strongest candidates with per-plot caps and spatial thinning."""
    ordered = sorted(
        records,
        key=lambda r: (-r["score"], r["plot_id"], r["row"], r["col"]),
    )

    def select_with_spacing(spacing: int):
        selected = []
        per_plot = {}
        for rec in ordered:
            if per_plot.get(rec["plot_id"], 0) >= cfg.max_endmember_per_plot:
                continue

            too_close = False
            if spacing > 0:
                for prev in selected:
                    if max(
                        abs(rec["row"] - prev["row"]),
                        abs(rec["col"] - prev["col"]),
                    ) <= spacing:
                        too_close = True
                        break
            if too_close:
                continue

            selected.append(rec)
            per_plot[rec["plot_id"]] = per_plot.get(rec["plot_id"], 0) + 1
            if len(selected) >= cfg.endmember_support_size:
                break
        return selected

    selected = select_with_spacing(cfg.endmember_min_spacing_pixels)
    if (
        len(selected) < cfg.min_candidate_spectra
        and cfg.endmember_min_spacing_pixels > 0
    ):
        print(
            f"    WARNING: only {len(selected)} spatially thinned {label} support "
            "pixels; retrying without the spacing constraint"
        )
        selected = select_with_spacing(0)

    if len(selected) < cfg.min_candidate_spectra:
        raise ValueError(
            f"Only {len(selected)} final {label} support pixels available; need at "
            f"least {cfg.min_candidate_spectra}. Increase the local NDCSI tail, "
            "increase max_endmember_per_plot, or reduce min_candidate_spectra."
        )

    for rank, rec in enumerate(selected, start=1):
        rec["support_rank"] = rank

    plot_counts = {}
    for rec in selected:
        plot_counts[rec["plot_id"]] = plot_counts.get(rec["plot_id"], 0) + 1
    print(
        f"    final {label} support: {len(selected)} pixels from "
        f"{len(plot_counts)} plots (max {max(plot_counts.values())} per plot)"
    )
    return selected


def select_endmember_candidates(
    raw_ndvi: xr.DataArray,
    ndcsi: xr.DataArray,
    forest_fraction: xr.DataArray,
    scaffold_fraction: xr.DataArray,
    learning_mask: np.ndarray,
    plot_records,
    cfg: SunlitFractionConfig,
):
    """Build local NDCSI tails, then choose compact sun/shade support sets."""
    ndvi = np.asarray(raw_ndvi.values)
    ndcsi_np = np.asarray(ndcsi.values)
    forest_frac = np.asarray(forest_fraction.values)
    scaffold_frac = np.asarray(scaffold_fraction.values)

    sun_records = []
    shade_records = []
    thresholds = []
    diagnostics = {}

    for rec in plot_records:
        eligible = (
            rec["mask"]
            & learning_mask
            & np.isfinite(ndvi)
            & np.isfinite(ndcsi_np)
        )
        rows, cols = np.nonzero(eligible)
        n_veg = rows.size
        if n_veg < cfg.min_candidate_spectra:
            raise ValueError(
                f"Plot {rec['plot_id']} ({rec['treatment']}) has only {n_veg} "
                f"valid forest pixels; need at least {cfg.min_candidate_spectra}."
            )

        vals = ndcsi_np[rows, cols]
        pct = _local_percentiles(vals)
        sun_thr = float(np.quantile(vals, cfg.sun_ndcsi_quantile))
        shade_thr = float(np.quantile(vals, cfg.shade_ndcsi_quantile))

        sun_local = vals >= sun_thr
        shade_local = vals <= shade_thr

        sun_mask = np.zeros(ndvi.shape, dtype=bool)
        shade_mask = np.zeros(ndvi.shape, dtype=bool)
        sun_mask[rows[sun_local], cols[sun_local]] = True
        shade_mask[rows[shade_local], cols[shade_local]] = True

        diagnostics.setdefault(
            rec["treatment"],
            {
                "sunlit": np.zeros(ndvi.shape, dtype=bool),
                "shaded": np.zeros(ndvi.shape, dtype=bool),
            },
        )
        diagnostics[rec["treatment"]]["sunlit"] |= sun_mask
        diagnostics[rec["treatment"]]["shaded"] |= shade_mask

        for i in np.flatnonzero(sun_local):
            sun_records.append(
                {
                    "class": "sunlit",
                    "plot_id": rec["plot_id"],
                    "treatment": rec["treatment"],
                    "row": int(rows[i]),
                    "col": int(cols[i]),
                    "ndcsi": float(vals[i]),
                    "local_percentile": float(pct[i]),
                    "score": float(pct[i]),
                    "forest_fraction": float(forest_frac[rows[i], cols[i]]),
                    "scaffold_fraction": float(scaffold_frac[rows[i], cols[i]]),
                }
            )

        for i in np.flatnonzero(shade_local):
            shade_records.append(
                {
                    "class": "shaded",
                    "plot_id": rec["plot_id"],
                    "treatment": rec["treatment"],
                    "row": int(rows[i]),
                    "col": int(cols[i]),
                    "ndcsi": float(vals[i]),
                    "local_percentile": float(pct[i]),
                    "score": float(1.0 - pct[i]),
                    "forest_fraction": float(forest_frac[rows[i], cols[i]]),
                    "scaffold_fraction": float(scaffold_frac[rows[i], cols[i]]),
                }
            )

        thresholds.append(
            {
                "plot_id": rec["plot_id"],
                "treatment": rec["treatment"],
                "n_clean_forest": int(n_veg),
                "sun_ndcsi": sun_thr,
                "shade_ndcsi": shade_thr,
                "n_sun_pool": int(sun_local.sum()),
                "n_shade_pool": int(shade_local.sum()),
            }
        )
        print(
            f"    {rec['treatment']} | plot {rec['plot_id']}: "
            f"clean_forest={n_veg}, sun_pool={int(sun_local.sum())}, "
            f"shade_pool={int(shade_local.sum())}"
        )

    selected_sun = _spatially_distributed_support(
        sun_records, "sunlit", cfg
    )
    selected_shade = _spatially_distributed_support(
        shade_records, "shaded", cfg
    )

    sun_mask = np.zeros(ndvi.shape, dtype=bool)
    shade_mask = np.zeros(ndvi.shape, dtype=bool)
    for rec in selected_sun:
        sun_mask[rec["row"], rec["col"]] = True
    for rec in selected_shade:
        shade_mask[rec["row"], rec["col"]] = True

    return {
        "sunlit": sun_mask,
        "shaded": shade_mask,
        "per_plot_thresholds": thresholds,
        "diagnostic_masks": diagnostics,
        "selected_points": selected_sun + selected_shade,
        "candidate_pool_counts": {
            "sunlit": len(sun_records),
            "shaded": len(shade_records),
        },
    }


def summarize_treatment_spectra(
    filepath,
    wavelengths,
    diagnostic_masks,
    cfg: SunlitFractionConfig,
):
    """Median and 10-90% spectra for treatment/state diagnostic plots."""
    rng = np.random.default_rng(cfg.random_seed)
    stats = {}

    for treatment, states in diagnostic_masks.items():
        stats[treatment] = {}
        for state, mask in states.items():
            rows, cols = _sample_mask_coordinates(
                mask, cfg.max_diagnostic_spectra, rng
            )
            if rows.size == 0:
                continue
            spectra = _sample_spectra(filepath, rows, cols, len(wavelengths))
            median, p10, p90, n = _robust_spectral_stats(
                spectra, f"{treatment}_{state}"
            )
            stats[treatment][state] = {
                "median": median,
                "p10": p10,
                "p90": p90,
                "n": n,
            }

    return stats


def build_endmember_library(
    filepath,
    wavelengths: Sequence[float],
    candidate_masks: Dict[str, np.ndarray],
    red_edge_wavelength_nm: float,
    cfg: SunlitFractionConfig,
) -> EndmemberLibrary:
    """Build robust median sunlit and shaded endmembers from final support pixels."""
    stats = {}
    counts = {}

    for label in ("sunlit", "shaded"):
        rows, cols = np.nonzero(candidate_masks[label])
        if rows.size < cfg.min_candidate_spectra:
            raise ValueError(
                f"Only {rows.size} support pixels for {label}; "
                f"need at least {cfg.min_candidate_spectra}."
            )

        spectra = _sample_spectra(filepath, rows, cols, len(wavelengths))
        median, p10, p90, n_valid = _robust_spectral_stats(spectra, label)
        stats[label] = (median, p10, p90)
        counts[label] = int(n_valid)

    return EndmemberLibrary(
        wavelengths=np.asarray(wavelengths, dtype="float32"),
        sunlit=stats["sunlit"][0],
        shaded=stats["shaded"][0],
        sunlit_p10=stats["sunlit"][1],
        sunlit_p90=stats["sunlit"][2],
        shaded_p10=stats["shaded"][1],
        shaded_p90=stats["shaded"][2],
        red_edge_wavelength_nm=float(red_edge_wavelength_nm),
        candidate_counts=counts,
    )


def _new_output(
    template: xr.DataArray,
    values: np.ndarray,
    name: str,
    long_name: str,
):
    da = xr.DataArray(
        values.astype("float32"),
        dims=template.dims,
        coords=template.coords,
        name=name,
        attrs={"long_name": long_name},
    )
    if template.rio.crs is not None:
        da = da.rio.write_crs(template.rio.crs)
    da = da.rio.write_transform(template.rio.transform())
    return da


def unmix_two_endmembers(
    filepath,
    wavelengths: Sequence[float],
    template: xr.DataArray,
    library: EndmemberLibrary,
    cfg: SunlitFractionConfig,
    application_mask: np.ndarray,
) -> Dict[str, xr.DataArray]:
    """Retrieve sunlit fraction on the line segment joining shade and sun spectra.

    Model:
        y = shade + f_sun * (sun - shade),  with 0 <= f_sun <= 1.

    RMSE is computed after constraining f_sun to [0, 1], so non-vegetation or
    otherwise poorly represented spectra can be identified by poor fit.
    """
    wl = np.asarray(wavelengths, dtype=float)
    band_idx = _unmixing_band_indices(wl, cfg)

    sun = library.sunlit[band_idx].astype("float64")
    shade = library.shaded[band_idx].astype("float64")
    finite = np.isfinite(sun) & np.isfinite(shade)
    sun = sun[finite]
    shade = shade[finite]
    band_idx = band_idx[finite]
    if band_idx.size < 2:
        raise ValueError("Insufficient finite endmember bands for two-class unmixing.")

    direction = sun - shade
    denom = float(np.dot(direction, direction))
    if not np.isfinite(denom) or denom <= 1e-12:
        raise ValueError("Sunlit and shaded endmembers are spectrally indistinguishable.")

    height, width = template.shape
    shape = (height, width)
    f_sun = np.full(shape, np.nan, dtype="float32")
    rmse = np.full(shape, np.nan, dtype="float32")

    with rasterio.open(filepath) as src:
        raster_bands = (band_idx + 1).tolist()
        if src.count != len(wavelengths):
            raise ValueError(
                f"Wavelength count ({len(wavelengths)}) != raster band count ({src.count})."
            )

        for y0 in range(0, height, cfg.block_rows):
            y1 = min(height, y0 + cfg.block_rows)
            window = rasterio.windows.Window(0, y0, width, y1 - y0)
            raw = src.read(raster_bands, window=window).astype("float64")

            invalid = (
                (raw < 0)
                | (raw > REFL_SCALE)
                | (raw == FILL_VALUE)
                | ~np.isfinite(raw)
            )
            raw /= float(REFL_SCALE)

            block_h = y1 - y0
            Y = raw.reshape(raw.shape[0], -1)
            valid_pix = ~np.any(invalid.reshape(invalid.shape[0], -1), axis=0)
            valid_pix &= np.all(np.isfinite(Y), axis=0)
            valid_pix &= application_mask[y0:y1, :].reshape(-1)
            if not np.any(valid_pix):
                continue

            Yv = Y[:, valid_pix]
            f = direction @ (Yv - shade[:, None]) / denom
            f = np.clip(f, 0.0, 1.0)
            pred = shade[:, None] + direction[:, None] * f[None, :]
            rmse_v = np.sqrt(np.mean((Yv - pred) ** 2, axis=0))

            def place(values, target):
                flat = np.full(block_h * width, np.nan, dtype="float32")
                flat[valid_pix] = values.astype("float32")
                target[y0:y1, :] = flat.reshape(block_h, width)

            place(f, f_sun)
            place(rmse_v, rmse)

    return {
        "f_sun_veg": _new_output(
            template,
            f_sun,
            "f_sun_veg",
            "sunlit fraction along the two-endmember forest illumination line",
        ),
        "rmse": _new_output(
            template,
            rmse,
            "rmse",
            "two-endmember spectral fit RMSE",
        ),
    }


def compute_sunlit_fraction(
    filepath,
    wavelengths: Sequence[float],
    default_crs=None,
    config: Optional[SunlitFractionConfig] = None,
    treatment_areas_shp=None,
    forest_fraction_raster=None,
    scaffold_fraction_raster=None,
):
    """Learn clean sun/shade endpoints and apply them to all forest.

    Treatment polygons define the learning domain. Forest fraction defines the
    forest application domain. Known scaffold overlap is excluded from endmember
    learning and NDCSI scaling, but not from model application, so scaffold-
    affected pixels remain available for RMSE diagnostics.
    """
    cfg = config or SunlitFractionConfig()
    cube = open_and_scale(filepath)

    with rasterio.open(filepath) as src:
        source_crs = src.crs
        source_transform = src.transform

    if source_crs is None:
        if default_crs is None:
            raise ValueError(f"Raster has no CRS: {filepath}")
        source_crs = default_crs
        print(f"    WARNING: source CRS missing; assigning {default_crs}")

    cube = cube.rio.write_crs(source_crs).rio.write_transform(source_transform)
    raw_ndvi = _raw_ndvi(cube, wavelengths, cfg)
    raw_ndvi = raw_ndvi.rio.write_crs(source_crs).rio.write_transform(source_transform)

    if treatment_areas_shp is None:
        raise ValueError("treatment_areas_shp is required for endmember learning.")
    if forest_fraction_raster is None:
        raise ValueError("forest_fraction_raster is required. Run 01_run_formask.py first.")
    if scaffold_fraction_raster is None:
        raise ValueError("scaffold_fraction_raster is required. Run 01_run_formask.py first.")

    forest_fraction = load_fraction_raster(
        forest_fraction_raster, raw_ndvi, "forest_fraction"
    )
    scaffold_fraction = load_fraction_raster(
        scaffold_fraction_raster, raw_ndvi, "scaffold_fraction"
    )
    forest_np = (
        np.isfinite(forest_fraction.values)
        & (forest_fraction.values >= cfg.forest_fraction_min)
    )
    if not np.any(forest_np):
        raise ValueError("No forest pixels remain after applying forest_fraction_min.")

    scaffold_np = np.asarray(scaffold_fraction.values)
    scaffold_clean_np = (
        np.isfinite(scaffold_np)
        & (scaffold_np <= cfg.endmember_scaffold_fraction_max + 1e-12)
    )
    learning_np = forest_np & scaffold_clean_np
    if not np.any(learning_np):
        raise ValueError(
            "No clean forest pixels remain after scaffold exclusion for endmember learning."
        )

    n_scaffold_excluded = int(np.sum(forest_np & ~scaffold_clean_np))
    print(
        f"    scaffold exclusion for endmember learning: {n_scaffold_excluded} "
        f"forest pixels above fraction {cfg.endmember_scaffold_fraction_max:g}"
    )

    plot_records = build_plot_masks(treatment_areas_shp, raw_ndvi, cfg)

    # Balanced clean treatment-plot forest defines red-edge position and NDCSI scaling.
    reference_np = balanced_forest_reference(
        raw_ndvi, learning_np, plot_records, cfg
    )
    reference_mask = xr.DataArray(
        reference_np, dims=raw_ndvi.dims, coords=raw_ndvi.coords
    )
    forest_mask = xr.DataArray(
        forest_np, dims=raw_ndvi.dims, coords=raw_ndvi.coords
    )

    red_edge_nm = detect_red_edge_wavelength(
        filepath, wavelengths, reference_np, cfg
    )
    ndcsi, _ = compute_ndcsi(
        cube,
        wavelengths,
        raw_ndvi,
        forest_mask,
        red_edge_nm,
        cfg,
        scaling_mask=reference_mask,
    )

    candidates = select_endmember_candidates(
        raw_ndvi,
        ndcsi,
        forest_fraction,
        scaffold_fraction,
        learning_np,
        plot_records,
        cfg,
    )
    candidates["n_scaffold_excluded_forest"] = n_scaffold_excluded
    candidates["spectra_by_treatment"] = summarize_treatment_spectra(
        filepath,
        wavelengths,
        candidates["diagnostic_masks"],
        cfg,
    )

    library = build_endmember_library(
        filepath,
        wavelengths,
        candidates,
        red_edge_nm,
        cfg,
    )

    products = {"NDCSI": ndcsi.astype("float32")}
    products.update(
        unmix_two_endmembers(
            filepath,
            wavelengths,
            raw_ndvi,
            library,
            cfg,
            forest_np,
        )
    )

    return products, library, candidates


def save_endmember_library(
    library: EndmemberLibrary,
    out_npz: Path,
    out_json: Optional[Path] = None,
    extra_metadata: Optional[Dict] = None,
):
    """Save the two endmember spectra/statistics used for a flight retrieval."""
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        wavelengths=library.wavelengths,
        sunlit=library.sunlit,
        shaded=library.shaded,
        sunlit_p10=library.sunlit_p10,
        sunlit_p90=library.sunlit_p90,
        shaded_p10=library.shaded_p10,
        shaded_p90=library.shaded_p90,
        red_edge_wavelength_nm=np.asarray(
            [library.red_edge_wavelength_nm], dtype=float
        ),
    )

    if out_json is not None:
        meta = {
            "red_edge_wavelength_nm": library.red_edge_wavelength_nm,
            "candidate_counts": library.candidate_counts,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_endmember_points(
    selected_points,
    reference_raster,
    out_path: Path,
):
    """Save the exact support pixels used for the endmembers as QGIS-ready GPKG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_raster) as src:
        crs = src.crs
        transform = src.transform

    if crs is None:
        raise ValueError(f"Reference raster has no CRS: {reference_raster}")

    records = []
    for rec in selected_points:
        x, y = rasterio.transform.xy(
            transform,
            rec["row"],
            rec["col"],
            offset="center",
        )
        records.append(
            {
                "class": rec["class"],
                "plot_id": rec["plot_id"],
                "treatment": rec["treatment"],
                "row": int(rec["row"]),
                "col": int(rec["col"]),
                "ndcsi": float(rec["ndcsi"]),
                "local_pct": float(rec["local_percentile"]),
                "score": float(rec["score"]),
                "forest_frac": float(rec["forest_fraction"]),
                "scaffold_frac": float(rec["scaffold_fraction"]),
                "rank": int(rec["support_rank"]),
                "geometry": Point(float(x), float(y)),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    if out_path.exists():
        out_path.unlink()
    gdf.to_file(
        out_path,
        layer="endmember_pixels",
        driver="GPKG",
    )