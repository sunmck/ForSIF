from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401  # needed to register the .rio accessor

import rasterio
import re
from rasterio import features

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from scipy.stats import gaussian_kde, ttest_ind

## Date formatting for plotting
_MONTHS = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}

# Plot options
@dataclass
class PlotOptions:
    save: bool = True
    dpi: int = 300

    make_scene_boxplots: bool = True
    make_profile_weighted_stats: bool = True
    make_profile_monthly_comparisons: bool = True
    make_profile_overview_maps: bool = True
    plot_treatments_overview_maps: bool = True
    overview_dates: Tuple[str, ...] = ("20230617", "20240613", "20240823")

    # color bars
    cmap_value: str = "viridis"
    cmap_diff: str = "RdBu_r"
    cmap_overview: str = "viridis"

    # font sizes
    base_fontsize: int = 13
    title_fontsize: int = 18
    suptitle_fontsize: int = 20
    tick_fontsize: int = 12
    legend_fontsize: int = 12
    cbar_label_fontsize: int = 10
    cbar_tick_fontsize: int = 10

    percentile_clip: Tuple[float, float] = (2, 98)
    symmetric_diff: bool = True

def apply_plot_style(opts: PlotOptions):
    mpl.rcParams.update({
        "font.size": opts.base_fontsize,
        "axes.titlesize": opts.title_fontsize,
        "axes.labelsize": opts.base_fontsize,
        "xtick.labelsize": opts.tick_fontsize,
        "ytick.labelsize": opts.tick_fontsize,
        "legend.fontsize": opts.legend_fontsize,
        "figure.titlesize": opts.suptitle_fontsize,
    })


def format_date_label(yyyymmdd: str) -> str:
    s = str(yyyymmdd).strip()
    if len(s) != 8 or not s.isdigit():
        return s
    yyyy, mm, dd = s[0:4], s[4:6], s[6:8]
    return f"{int(dd):02d} {_MONTHS.get(mm, mm)} {yyyy}"


def _boldify_axes(ax):
    try:
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for lab in ax.get_xticklabels():
            lab.set_fontweight("bold")
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")
    except Exception:
        pass


def _boldify_colorbar(cb):
    try:
        cb.ax.xaxis.label.set_fontweight("bold")
        cb.ax.yaxis.label.set_fontweight("bold")
        for lab in cb.ax.get_xticklabels():
            lab.set_fontweight("bold")
        for lab in cb.ax.get_yticklabels():
            lab.set_fontweight("bold")
    except Exception:
        pass

def _bold_ticks(ax):
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")

def _bold_axis_labels(ax):
    if ax.xaxis.label is not None:
        ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label is not None:
        ax.yaxis.label.set_fontweight("bold")


def _sci_formatter(powerlimits=(-2, 2)) -> ScalarFormatter:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits(powerlimits)
    fmt.set_useOffset(True)
    return fmt


def _apply_sci_axis(ax, which: str = "y", powerlimits=(-2, 2)):
    fmt = _sci_formatter(powerlimits)
    if which == "y":
        ax.yaxis.set_major_formatter(fmt)
    elif which == "x":
        ax.xaxis.set_major_formatter(fmt)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(fig, outpath: Path, dpi: int = 300):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def mean_mosaic(stack: Sequence[xr.DataArray]) -> xr.DataArray:
    return xr.concat(list(stack), dim="scene").mean(dim="scene")


# Raster sanitization + colormaps

def _as_2d(da: xr.DataArray) -> np.ndarray:
    if da.ndim == 2:
        arr = da.values
    elif da.ndim == 3 and "band" in da.dims and da.sizes.get("band", 0) == 1:
        arr = da.squeeze("band", drop=True).values
    else:
        arr = da.squeeze().values

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    return np.asarray(arr)


def _get_cmap(name: str):
    cmap = mpl.cm.get_cmap(name).copy()
    cmap.set_bad((0, 0, 0, 0))  # transparent NaNs
    return cmap

def _sanitize_for_plotting(
    da: xr.DataArray,
    *,
    nodata_values: Sequence[float] = (
        -999, -999.0, -9999, -9999.0,
        -32768.0, 32767.0,
        65535.0, 4294967295.0,
        3.4028235e38, -3.4028235e38,  # float32 extremes sometimes used as nodata
    ),
) -> np.ndarray:
    arr = _as_2d(da).astype("float64", copy=False)

    # 1) Explicit sentinel nodata values (THIS is what we want)
    for nv in nodata_values:
        arr[np.isclose(arr, float(nv), equal_nan=False)] = np.nan

    # 2) Common nodata attrs
    for k in ("_FillValue", "fill_value", "missing_value"):
        v = da.attrs.get(k, None)
        if v is not None:
            try:
                arr[np.isclose(arr, float(v), equal_nan=False)] = np.nan
            except Exception:
                pass

    # 3) rioxarray nodata if present — BUT ONLY if it matches our known sentinels
    # (prevents accidentally masking real 0s if a file has rio.nodata = 0)
    try:
        rn = da.rio.nodata
        if rn is not None:
            rn = float(rn)
            if any(np.isclose(rn, float(nv), atol=0, rtol=0) for nv in nodata_values):
                arr[np.isclose(arr, rn, equal_nan=False)] = np.nan
    except Exception:
        pass

    return arr

def _apply_ndvi_mask(
    data: xr.DataArray,
    ndvi: Optional[xr.DataArray],
    ndvi_threshold: float,
):
    arr = _sanitize_for_plotting(data)
    if ndvi is None:
        return arr

    nd = _sanitize_for_plotting(ndvi)
    bad = (~np.isfinite(nd)) | (nd < float(ndvi_threshold))
    arr[bad] = np.nan
    return arr


def _robust_vmin_vmax(
    arrays: Sequence[np.ndarray],
    *,
    percentile_clip: Tuple[float, float],
    force_nonnegative: bool = True,
):
    vals = []
    for a in arrays:
        if a is None:
            continue
        x = a[np.isfinite(a)]
        if x.size:
            vals.append(x)
    if not vals:
        return 0.0, 1.0

    flat = np.concatenate(vals)
    lo, hi = np.nanpercentile(flat, percentile_clip)
    vmin, vmax = float(lo), float(hi)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 0:
        vmin, vmax = float(np.nanmin(flat)), float(np.nanmax(flat))

    if force_nonnegative and np.isfinite(vmin) and vmin < 0:
        vmin = 0.0
        if vmax <= vmin:
            vmax = float(np.nanmax(flat)) if np.isfinite(np.nanmax(flat)) else 1.0
            if vmax <= 0:
                vmax = 1.0

    return vmin, vmax


# Geometry helpers
def _bbox_pixels_for_polygons(ref_raster, polygons: gpd.GeoDataFrame, pad: int = 10):
    minx, miny, maxx, maxy = polygons.total_bounds
    transform = ref_raster.rio.transform()
    xmin_pix, ymin_pix = ~transform * (minx, miny)
    xmax_pix, ymax_pix = ~transform * (maxx, maxy)

    xmin_pix, xmax_pix = sorted([xmin_pix, xmax_pix])
    ymin_pix, ymax_pix = sorted([ymin_pix, ymax_pix])

    xmin_pix -= pad
    xmax_pix += pad
    ymin_pix -= pad
    ymax_pix += pad
    return transform, int(xmin_pix), int(xmax_pix), int(ymin_pix), int(ymax_pix)


def _clip_bbox(arr: np.ndarray, xmin_pix: int, xmax_pix: int, ymin_pix: int, ymax_pix: int) -> np.ndarray:
    h, w = arr.shape
    x0, x1 = sorted([xmin_pix, xmax_pix])
    y0, y1 = sorted([ymin_pix, ymax_pix])

    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return arr
    return arr[y0:y1, x0:x1]


def _draw_treatment_outlines(
    ax,
    treatment_areas: gpd.GeoDataFrame,
    treatment_color_map: Dict[int, str],
    treatments: Sequence[int],
    *,
    raster_crs=None,
    transform=None,
    pixel_space: bool = False,
    lw: float = 1.5,
):
    for t in treatments:
        area_t = treatment_areas[treatment_areas["treatment"] == t]
        if raster_crs is not None:
            try:
                area_t = area_t.to_crs(raster_crs)
            except Exception:
                pass

        patches = []
        for geom in area_t.geometry:
            if geom is None:
                continue

            def _poly_to_patch(poly):
                if pixel_space and transform is not None:
                    coords = [(~transform * (x, y)) for x, y in poly.exterior.coords]
                else:
                    coords = list(poly.exterior.coords)
                return Polygon(coords, closed=True, fill=False,
                               edgecolor=treatment_color_map.get(t, "#000000"),
                               linewidth=lw)

            if geom.geom_type == "Polygon":
                patches.append(_poly_to_patch(geom))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom:
                    patches.append(_poly_to_patch(poly))

        if patches:
            ax.add_collection(PatchCollection(patches, match_original=True))


def _empty_map_panel(ax, title: str):
    ax.set_title(title, fontweight="bold")
    ax.axis("off")


def _empty_stats_panel(ax, title: str = ""):
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)


# Crown-weighted extraction

def get_pixels_by_treatment_weighted(
    raster: xr.DataArray,
    crowns: gpd.GeoDataFrame,
    treatment_value: int,
    supersample: int = 10,
    min_weight: float = 0.5,
):
    arr = _sanitize_for_plotting(raster)
    transform = raster.rio.transform()
    out_shape = arr.shape

    crowns_treat = crowns[crowns["treatment"] == treatment_value]
    polygons = list(crowns_treat.geometry)

    ss_shape = (out_shape[0] * supersample, out_shape[1] * supersample)
    ss_transform = rasterio.Affine(
        transform.a / supersample, transform.b, transform.c,
        transform.d, transform.e / supersample, transform.f
    )

    ss_mask = features.rasterize(
        ((geom, 1) for geom in polygons if geom is not None),
        out_shape=ss_shape,
        transform=ss_transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    mask = ss_mask.reshape(out_shape[0], supersample, out_shape[1], supersample).mean(axis=(1, 3))
    pixel_mask = mask > 0
    pixels = arr[pixel_mask]
    weights = mask[pixel_mask]

    valid = np.isfinite(pixels) & np.isfinite(weights) & (weights >= min_weight)
    return pixels[valid], weights[valid], mask


# PROFILE: Weighted stats across dates
def plot_weighted_stats_across_dates(
    datasets_by_date: Dict[str, xr.DataArray],
    *,
    crowns: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    ylabel: str,
    save_path: Optional[Path] = None,
    sci_y: bool = False,
    supersample: int = 10,
    min_weight: float = 0.5,
    # font control (optional)
    suptitle_fs: int = 20,  # unused now
    title_fs: int = 15,
    label_fs: int = 14,
    tick_fs: int = 12,
    stars_fs: int = 13,
):

    # --- same label logic as plot_monthly_comparison ---
    def _pretty_label(lbl: str) -> str:
        s = (lbl or "").strip()
        if not s:
            return s
        if "[" in s and "]" in s:
            return s
        if s == "SFMNN_fqe_saR2F":
            return r"FQE [nm$^{-1}$]"
        if s == "tasi_lst":
            return "LST [°K]"
        m = re.search(r"dual_vi\s*(?:\(|__)?\s*([A-Za-z0-9_+-]+)\s*\)?", s, flags=re.IGNORECASE)
        if m:
            band = m.group(1)
            if band.lower() == "ndvi":
                return "NDVI"
            if band.lower() == "sr":
                return "SR"
            return band.upper() if band.isalpha() else band
        return s

    ylabel_pretty = _pretty_label(ylabel)

    # ---- bold helpers ----
    def _bold_ticks(ax):
        for lab in ax.get_xticklabels():
            lab.set_fontweight("bold")
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")

    def _bold_axis_labels(ax):
        if ax.xaxis.label is not None:
            ax.xaxis.label.set_fontweight("bold")
        if ax.yaxis.label is not None:
            ax.yaxis.label.set_fontweight("bold")

    # ---- date ordering / labels ----
    dates = list(datasets_by_date.keys())
    preferred = ["2023-06-17", "2024-06-13", "2024-08-23", "20230617", "20240613", "20240823"]
    dates = [d for d in preferred if d in dates] + [d for d in dates if d not in preferred]
    date_labels = [format_date_label(d) for d in dates]

    n_cols = len(dates)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 12))
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    def p_to_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    def welch_tests(data: List[np.ndarray]) -> List[Tuple[int, int, float, str]]:
        results = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if data[i].size == 0 or data[j].size == 0:
                    continue
                _, p = ttest_ind(data[i], data[j], equal_var=False)
                results.append((i, j, float(p), p_to_stars(float(p))))
        return results

    # ---- Extract once + compute global axis ranges for alignment ----
    extracted: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int, float, str]]]] = {}
    all_px: List[np.ndarray] = []

    # For global limits:
    box_data_min_global = np.nan
    box_data_max_global = np.nan
    hist_y_max_global = 0.0
    x_min, x_max = np.nan, np.nan

    # Track max number of pairwise tests (for bracket headroom)
    max_brackets = 0

    for d in dates:
        da = datasets_by_date.get(d)
        if da is None:
            continue

        data: List[np.ndarray] = []
        weights: List[np.ndarray] = []

        for t in treatments:
            px, w, _ = get_pixels_by_treatment_weighted(
                da, crowns, t, supersample=supersample, min_weight=min_weight
            )
            data.append(px)
            weights.append(w)
            all_px.append(px)

        tests = welch_tests(data)
        max_brackets = max(max_brackets, len(tests))
        extracted[d] = (data, weights, tests)

    # ---- Global x-range for hist/cdf ----
    if all_px and any(a.size > 0 for a in all_px):
        flat = np.concatenate([a for a in all_px if a.size > 0])
        x_min = float(np.nanmin(flat)) if np.isfinite(np.nanmin(flat)) else 0.0
        x_max = float(np.nanmax(flat)) if np.isfinite(np.nanmax(flat)) else 1.0
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = 0.0, 1.0
    else:
        x_min, x_max = 0.0, 1.0

    # ---- Global y-range for boxplots (use actual data range, not 0..max) ----
    if all_px and any(a.size > 0 for a in all_px):
        box_data_min_global = float(np.nanmin([np.nanmin(a) for a in all_px if a.size > 0]))
        box_data_max_global = float(np.nanmax([np.nanmax(a) for a in all_px if a.size > 0]))
        if not np.isfinite(box_data_min_global) or not np.isfinite(box_data_max_global) or box_data_max_global <= box_data_min_global:
            # fallback
            box_data_min_global, box_data_max_global = 0.0, 1.0
    else:
        box_data_min_global, box_data_max_global = 0.0, 1.0

    box_span_global = box_data_max_global - box_data_min_global
    if not np.isfinite(box_span_global) or box_span_global <= 0:
        box_span_global = max(1.0, abs(box_data_max_global) * 0.2)

    # ---- Global y-range for hist (PDF) ----
    bins = 30
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    def _safe_pdf_max(arr: np.ndarray, w: np.ndarray) -> float:
        if arr.size == 0 or w.size == 0:
            return 0.0
        ok = np.isfinite(arr) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            return 0.0
        a = arr[ok]
        ww = w[ok]

        h, _ = np.histogram(a, bins=bin_edges, weights=ww, density=True)
        hmax = float(np.nanmax(h)) if h.size else 0.0
        if not np.isfinite(hmax):
            hmax = 0.0

        kmax = 0.0
        if a.size >= 5 and np.unique(a).size >= 3:
            try:
                kde = gaussian_kde(a, weights=ww)
                xs = np.linspace(x_min, x_max, 256)
                vals = kde(xs)
                kmax = float(np.nanmax(vals)) if vals.size else 0.0
                if not np.isfinite(kmax):
                    kmax = 0.0
            except Exception:
                kmax = 0.0

        return max(hmax, kmax)

    if extracted:
        for _d, (data, weights, _) in extracted.items():
            for arr, w in zip(data, weights):
                hist_y_max_global = max(hist_y_max_global, _safe_pdf_max(arr, w))

    if not np.isfinite(hist_y_max_global) or hist_y_max_global <= 0:
        hist_y_max_global = 1.0
    hist_y_max_global *= 1.08  # headroom

    colors = [treatment_color_map[t] for t in treatments]
    x_label_pretty = ylabel_pretty

    # ---- Compute GLOBAL boxplot ylim (bottom + top) that fits data range AND brackets ----
    # Bottom: a little below min; Top: enough above max to contain max bracket stack.
    # Use consistent bracket geometry across all panels.
    bracket_pad = 0.06 * box_span_global
    bracket_step = 0.10 * box_span_global
    bracket_height = 0.60 * bracket_step
    stars_extra = 0.25 * bracket_step

    if max_brackets > 0:
        top_needed = (
            box_data_max_global
            + bracket_pad
            + (max_brackets - 1) * bracket_step
            + bracket_height
            + stars_extra
        )
    else:
        top_needed = box_data_max_global + 0.10 * box_span_global

    bottom_needed = box_data_min_global - 0.06 * box_span_global

    # If everything is positive and bottom would go below 0, keep it above 0? (optional)
    # Comment out if you want full range always.
    # if box_data_min_global >= 0:
    #     bottom_needed = max(0.0, bottom_needed)

    colors = [treatment_color_map[t] for t in treatments]

    # ---- Plot ----
    for col, (d, dlabel) in enumerate(zip(dates, date_labels)):
        ax_box, ax_hist, ax_cdf = axes[:, col]
        da = datasets_by_date.get(d)

        for ax in (ax_box, ax_hist, ax_cdf):
            ax.tick_params(axis="both", labelsize=tick_fs)
            _bold_ticks(ax)

        # aligned axes
        ax_hist.set_xlim(x_min, x_max)
        ax_cdf.set_xlim(x_min, x_max)
        ax_cdf.set_ylim(0.0, 1.0)
        ax_hist.set_ylim(0.0, hist_y_max_global)

        # IMPORTANT: boxplot y-range uses actual data range (+small padding) + bracket headroom
        ax_box.set_ylim(bottom_needed, top_needed)

        # grid
        ax_box.grid(True, linestyle="--", alpha=0.25)
        ax_hist.grid(True, linestyle="--", alpha=0.35)
        ax_cdf.grid(True, linestyle="--", alpha=0.35)

        # titles / labels
        ax_box.set_title(dlabel, fontsize=title_fs, fontweight="bold")
        ax_box.set_ylabel(ylabel_pretty, fontsize=label_fs, fontweight="bold")

        ax_hist.set_ylabel("Probability density", fontsize=label_fs, fontweight="bold")
        ax_cdf.set_ylabel("Cumulative probability", fontsize=label_fs, fontweight="bold")
        ax_hist.set_xlabel(x_label_pretty, fontsize=label_fs, fontweight="bold")
        ax_cdf.set_xlabel(x_label_pretty, fontsize=label_fs, fontweight="bold")

        _bold_axis_labels(ax_box)
        _bold_axis_labels(ax_hist)
        _bold_axis_labels(ax_cdf)

        if da is None:
            continue

        data, weights, welch_res = extracted[d]

        # --- boxplot ---
        bp = ax_box.boxplot(data, patch_artist=True)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(1.6)
        for cap in bp["caps"]:
            cap.set_linewidth(1.6)
        for median in bp["medians"]:
            median.set_linewidth(2.4)
        for flier in bp.get("fliers", []):
            flier.set_markersize(4)

        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for median, c in zip(bp["medians"], colors):
            median.set_color(c)
            median.set_linewidth(2)

        ax_box.set_xticks(range(1, len(treatment_labels) + 1))
        ax_box.set_xticklabels(treatment_labels, rotation=20, fontweight="bold")

        # significance brackets (now guaranteed to fit due to global ylim)
        tops = [np.nanmax(d_) if d_.size else np.nan for d_ in data]
        base_y = np.nanmax(tops) if np.isfinite(np.nanmax(tops)) else box_data_max_global

        for k, (i, j, _p, stars) in enumerate(welch_res):
            y = base_y + bracket_pad + k * bracket_step
            h = bracket_height
            ax_box.plot([i + 1, i + 1, j + 1, j + 1], [y, y + h, y + h, y], lw=1.2, c="black")
            ax_box.text(
                (i + j + 2) / 2, y + h * 0.8, stars,
                ha="center", va="bottom",
                fontsize=stars_fs, fontweight="bold"
            )

        # --- histogram + KDE (PDF) ---
        for arr, w, c in zip(data, weights, colors):
            if arr.size == 0 or w.size == 0:
                continue

            ok = np.isfinite(arr) & np.isfinite(w) & (w > 0)
            if not np.any(ok):
                continue

            a = arr[ok]
            ww = w[ok]

            ax_hist.hist(
                a,
                bins=bin_edges,
                weights=ww,
                alpha=0.35,
                density=True,
                color=c,
            )
            ax_hist.axvline(np.average(a, weights=ww), color=c, linestyle="--", linewidth=1)

            if a.size >= 5 and np.unique(a).size >= 3:
                try:
                    kde = gaussian_kde(a, weights=ww)
                    xs = np.linspace(x_min, x_max, 256)
                    ax_hist.plot(xs, kde(xs), color=c, linewidth=2)
                except Exception:
                    pass

        # --- CDF (0..1) ---
        for arr, w, c in zip(data, weights, colors):
            if arr.size == 0 or w.size == 0:
                continue

            ok = np.isfinite(arr) & np.isfinite(w) & (w > 0)
            if not np.any(ok):
                continue

            a = arr[ok]
            ww = w[ok]

            idx = np.argsort(a)
            arrs = a[idx]
            ws = ww[idx]
            cdf = np.cumsum(ws) / np.sum(ws)
            ax_cdf.plot(arrs, cdf, color=c, linewidth=2)

        # scientific formatting (if requested)
        if sci_y:
            _apply_sci_axis(ax_box, "y")
            _apply_sci_axis(ax_hist, "x")
            _apply_sci_axis(ax_hist, "y")
            _apply_sci_axis(ax_cdf, "x")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# PROFILE: Monthly comparison (June/Aug/Δ)

def plot_monthly_comparison(
    *,
    data_left: Optional[xr.DataArray] = None,
    data_june: Optional[xr.DataArray],
    data_aug: Optional[xr.DataArray],
    ndvi_left: Optional[xr.DataArray] = None,
    ndvi_june: Optional[xr.DataArray] = None,
    ndvi_aug: Optional[xr.DataArray] = None,
    ndvi_threshold: float = 0.5,
    treatments: Sequence[int],
    treatment_areas: gpd.GeoDataFrame,
    treatment_color_map: Dict[int, str],
    transform,
    xmin_pix: int,
    xmax_pix: int,
    ymin_pix: int,
    ymax_pix: int,
    value_label: str,
    diff_label: str,
    title_left: str = "",
    title_june: str = "",
    title_aug: str = "",
    title_diff: str = "Δ (Aug − Jun)",
    cmap_value: str = "viridis",
    cmap_diff: str = "RdBu_r",
    percentile_clip: Tuple[float, float] = (2, 98),
    symmetric_diff: bool = True,
    force_nonnegative_values: bool = True,
    sci_value: bool = False,
    save_path: Optional[Path] = None,
    opts: Optional[PlotOptions] = None,
):
    # Default opts if not provided
    opts = opts or PlotOptions()

    # --- label prettifier (so you can keep passing layer keys if you want) ---
    def _pretty_colorbar_label(lbl: str) -> str:
        s = (lbl or "").strip()
        if not s:
            return s

        # If you already passed units, don't touch it
        if "[" in s and "]" in s:
            return s

        # Known single-band keys
        if s == "SFMNN_fqe_saR2F":
            return r"FQE [nm$^{-1}$]"
        if s == "tasi_lst":
            return "LST [°K]"

        # dual_vi patterns -> just band name
        m = re.search(r"dual_vi\s*(?:\(|__)?\s*([A-Za-z0-9_+-]+)\s*\)?", s, flags=re.IGNORECASE)
        if m:
            band = m.group(1)
            if band.lower() == "ndvi":
                return "NDVI"
            if band.lower() == "sr":
                return "SR"
            return band.upper() if band.isalpha() else band

        return s

    value_label_pretty = _pretty_colorbar_label(value_label)
    # keep delta label consistent with value label
    diff_label_pretty = r"$\Delta$ " + value_label_pretty

    # --- apply NDVI masks ---
    vl = vj = va = None
    if data_left is not None:
        vl = _apply_ndvi_mask(data_left, ndvi_left, ndvi_threshold)
    if data_june is not None:
        vj = _apply_ndvi_mask(data_june, ndvi_june, ndvi_threshold)
    if data_aug is not None:
        va = _apply_ndvi_mask(data_aug, ndvi_aug, ndvi_threshold)

    # --- diff stays Aug - Jun ---
    diff = None
    if vj is not None and va is not None:
        valid = np.isfinite(vj) & np.isfinite(va)
        diff = np.full_like(vj, np.nan, dtype="float64")
        diff[valid] = va[valid] - vj[valid]

    # --- shared vmin/vmax across ALL VALUE PANELS ---
    vv = []
    for arr in (vl, vj, va):
        if arr is not None:
            vv.append(_clip_bbox(arr, xmin_pix, xmax_pix, ymin_pix, ymax_pix))

    vmin, vmax = _robust_vmin_vmax(
        vv,
        percentile_clip=percentile_clip,
        force_nonnegative=force_nonnegative_values,
    )

    # --- diff stretch ---
    if diff is not None:
        dclip = _clip_bbox(diff, xmin_pix, xmax_pix, ymin_pix, ymax_pix)
        flatd = dclip[np.isfinite(dclip)]
        if flatd.size:
            if symmetric_diff:
                _, hi = np.nanpercentile(np.abs(flatd), percentile_clip)
                m = float(hi) if np.isfinite(hi) and hi > 0 else float(np.nanmax(np.abs(flatd)))
                dvmin, dvmax = -m, m
            else:
                lo, hi = np.nanpercentile(flatd, percentile_clip)
                dvmin, dvmax = float(lo), float(hi)
        else:
            dvmin, dvmax = -1.0, 1.0
    else:
        dvmin, dvmax = -1.0, 1.0

    cmapV = _get_cmap(cmap_value)
    cmapD = _get_cmap(cmap_diff)

    # Manual layout: keep equal aspect, reduce inter-panel gaps to ~nothing
    fig_w, fig_h = 18.0, 5.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Layout constants (figure fractions)
    left_margin = 0.01
    right_margin = 0.99
    top = 0.92
    bottom = 0.22          # reserve room for colorbars
    gap = 0.002            # tiny gap between panels

    avail_w = right_margin - left_margin
    avail_h = top - bottom

    # Pixel aspect of your bbox (square pixels assumption)
    pix_w = max(1, int(abs(xmax_pix - xmin_pix)))
    pix_h = max(1, int(abs(ymax_pix - ymin_pix)))
    aspect = pix_w / pix_h  # width / height

    # Compute max axis height that fits both height and width constraints
    max_h_from_width = (avail_w - 3 * gap) * fig_w / (4.0 * aspect * fig_h)
    axes_h = min(avail_h, max_h_from_width)
    axes_w = (aspect * axes_h * fig_h) / fig_w

    # Fallback if bbox is weird
    if not np.isfinite(axes_h) or axes_h <= 0 or axes_w <= 0:
        axes_h = avail_h
        axes_w = (avail_w - 3 * gap) / 4.0

    # Vertically center the axes block in the available map area
    y0 = bottom + (avail_h - axes_h) / 2.0

    # Create 4 axes placed manually
    axes = []
    x = left_margin
    for _ in range(4):
        ax = fig.add_axes([x, y0, axes_w, axes_h])
        axes.append(ax)
        x += axes_w + gap

    axL, axJ, axA, axD = axes

    def _plot_value_panel(ax, arr, title):
        if arr is None:
            _empty_map_panel(ax, title)
            return None

        im = ax.imshow(arr, cmap=cmapV, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_aspect("equal")

        _draw_treatment_outlines(
            ax, treatment_areas, treatment_color_map, treatments,
            transform=transform, pixel_space=True
        )
        ax.set_xlim(xmin_pix, xmax_pix)
        ax.set_ylim(ymax_pix, ymin_pix)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        return im

    imL = _plot_value_panel(axL, vl, title_left)
    imJ = _plot_value_panel(axJ, vj, title_june)
    imA = _plot_value_panel(axA, va, title_aug)

    if diff is None:
        _empty_map_panel(axD, title_diff)
        imD = None
    else:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=dvmin, vmax=dvmax)
        imD = axD.imshow(diff, cmap=cmapD, norm=norm, origin="upper")
        axD.set_aspect("equal")

        _draw_treatment_outlines(
            axD, treatment_areas, treatment_color_map, treatments,
            transform=transform, pixel_space=True
        )
        axD.set_xlim(xmin_pix, xmax_pix)
        axD.set_ylim(ymax_pix, ymin_pix)
        axD.set_title(title_diff, fontweight="bold")
        axD.axis("off")

    # --- Colorbars that match the axes extents exactly ---
    fig.canvas.draw()

    value_axes = []
    value_im = None
    for ax, im in ((axL, imL), (axJ, imJ), (axA, imA)):
        if im is not None:
            value_axes.append(ax)
            if value_im is None:
                value_im = im

    def _add_cbar_under_axes(ax_list, *, height_frac=0.10, pad_frac=0.08):
        if not ax_list:
            return None

        bbs = [ax.get_position() for ax in ax_list]
        x0 = min(bb.x0 for bb in bbs)
        x1 = max(bb.x1 for bb in bbs)
        y0b = min(bb.y0 for bb in bbs)
        h = max(bb.height for bb in bbs)

        pad = pad_frac * h
        cbar_h = height_frac * h
        return fig.add_axes([x0, y0b - pad - cbar_h, x1 - x0, cbar_h])

    # Value colorbar spanning the first three panels
    if value_im is not None and value_axes:
        cax_val = _add_cbar_under_axes(value_axes, height_frac=0.10, pad_frac=0.08)
        cb1 = fig.colorbar(value_im, cax=cax_val, orientation="horizontal")
        cb1.set_label(value_label_pretty, fontsize=opts.cbar_label_fontsize, fontweight="bold")
        cb1.ax.tick_params(labelsize=opts.cbar_tick_fontsize)
        if sci_value:
            cb1.formatter = _sci_formatter()
            cb1.update_ticks()
        _boldify_colorbar(cb1)

    # Diff colorbar under the 4th panel
    if imD is not None:
        cax_dif = _add_cbar_under_axes([axD], height_frac=0.10, pad_frac=0.08)
        cb2 = fig.colorbar(imD, cax=cax_dif, orientation="horizontal")
        cb2.set_label(diff_label_pretty, fontsize=opts.cbar_label_fontsize, fontweight="bold")
        cb2.ax.tick_params(labelsize=opts.cbar_tick_fontsize)
        if sci_value:
            cb2.formatter = _sci_formatter()
            cb2.update_ticks()
        _boldify_colorbar(cb2)

    if save_path:
        _savefig(fig, save_path, dpi=opts.dpi)
    else:
        plt.show()


# PROFILE: Overview stacks (3 dates, 4 flights)

def plot_stack_overview(
    *,
    stacks_by_date: Dict[str, Sequence[xr.DataArray]],
    dates: Sequence[str] = ("20230617", "20240613", "20240823"),
    date_titles: Optional[Dict[str, str]] = None,
    flight_names_by_date: Optional[Dict[str, Sequence[str]]] = None,
    treatments: Sequence[int],
    treatment_areas: gpd.GeoDataFrame,
    treatment_color_map: Dict[int, str],
    cmap: str,
    legend_label: str,
    percentile_clip: Tuple[float, float] = (2, 98),
    force_nonnegative_values: bool = True,
    sci_legend: bool = False,
    plot_treatments: bool = True,       
    treatment_lw: float = 1.5,         
    save_path: Optional[Path] = None,
):
    if date_titles is None:
        date_titles = {d: format_date_label(d) for d in dates}

    cmapO = _get_cmap(cmap)

    nrows = 4
    if flight_names_by_date:
        row_order: List[str] = []
        for d in dates:
            names = list(flight_names_by_date.get(d, []))
            if names:
                row_order = names.copy()
                break
        seen = set(row_order)
        for d in dates:
            for nm in flight_names_by_date.get(d, []):
                if nm not in seen:
                    row_order.append(nm)
                    seen.add(nm)
        while len(row_order) < nrows:
            row_order.append("")
        row_order = row_order[:nrows]
    else:
        row_order = [""] * nrows

    ncols = len(dates)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4 * ncols, 3.7 * nrows))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.expand_dims(axes, 1)

    stack_by_name: Dict[str, Dict[str, xr.DataArray]] = {}
    for d in dates:
        stack = list(stacks_by_date.get(d, []) or [])
        if not stack:
            stack_by_name[d] = {}
            continue

        if flight_names_by_date and d in flight_names_by_date:
            names = list(flight_names_by_date.get(d, []))
        else:
            names = [f"Flight {i+1}" for i in range(len(stack))]

        m: Dict[str, xr.DataArray] = {}
        for nm, da in zip(names, stack):
            if nm not in m and da is not None:
                m[nm] = da
        stack_by_name[d] = m

    # Global stretch across all available panels (after sanitization)
    vals = []
    for d in dates:
        for da in stack_by_name.get(d, {}).values():
            a = _sanitize_for_plotting(da)
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)

    if vals:
        flat = np.concatenate(vals)
        lo, hi = np.nanpercentile(flat, percentile_clip)
        vmin, vmax = float(lo), float(hi)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 0:
            vmin, vmax = float(np.nanmin(flat)), float(np.nanmax(flat))
        if force_nonnegative_values and np.isfinite(vmin) and vmin < 0:
            vmin = 0.0
            if vmax <= 0:
                vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im_for_cbar = None

    for c, d in enumerate(dates):
        col_title = date_titles.get(d, d)
        m = stack_by_name.get(d, {})

        for r in range(nrows):
            ax = axes[r, c]
            flight_nm = row_order[r]
            da = m.get(flight_nm) if flight_nm else None

            suffix = f" — {flight_nm}" if flight_nm else ""
            ax.set_title(f"{col_title}{suffix}", fontweight="bold")

            if da is None:
                ax.set_facecolor("white")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.tick_params(axis="both", which="both", length=4, labelsize=8)
                _boldify_axes(ax)
                continue

            x = da["x"].values
            y = da["y"].values

            arr = _sanitize_for_plotting(da)
            arr_plot = np.ma.masked_invalid(arr)

            im = ax.imshow(
                arr_plot,
                cmap=cmapO,
                vmin=vmin,
                vmax=vmax,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="upper",
                interpolation="none",
            )
            if im_for_cbar is None:
                im_for_cbar = im

            if plot_treatments:
                _draw_treatment_outlines(
                    ax,
                    treatment_areas=treatment_areas,
                    treatment_color_map=treatment_color_map,
                    treatments=treatments,
                    raster_crs=da.rio.crs,
                    pixel_space=False,
                    lw=treatment_lw,
                )

            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="both", which="both", length=4, labelsize=8)
            _boldify_axes(ax)

    fig.subplots_adjust(right=0.88, wspace=0.22, hspace=0.35, top=0.98, bottom=0.05, left=0.06)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])

    if im_for_cbar is not None:
        cb = fig.colorbar(im_for_cbar, cax=cbar_ax)
        cb.set_label(legend_label, fontweight="bold")
        if sci_legend:
            cb.formatter = _sci_formatter()
            cb.update_ticks()
        _boldify_colorbar(cb)
    else:
        cbar_ax.axis("off")

    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# SCENE: Boxplots per flightline

def plot_scene_flightline_boxplots(
    *,
    stacks: Dict[str, Sequence[xr.DataArray]],
    key: str,
    crowns: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    flight_names: Sequence[str],
    title: str,
    ylabel: str,
    sci_y: bool = False,
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    stack = list(stacks.get(key, []))
    names = list(flight_names)

    while len(stack) < 4:
        stack.append(None)
    while len(names) < 4:
        names.append("")

    for i in range(4):
        ax = axes[i]
        lbl = names[i] if names[i] else f"Flight {i+1}"
        ax.set_title(lbl, fontweight="bold")
        ax.grid(True, alpha=0.35)

        if stack[i] is None:
            ax.set_ylabel(ylabel, fontweight="bold")
            if sci_y:
                _apply_sci_axis(ax, "y")
            _boldify_axes(ax)
            continue

        data = []
        for t in treatments:
            px, _, _ = get_pixels_by_treatment_weighted(
                stack[i],
                crowns,
                t,
            )
            data.append(px)

        bp = ax.boxplot(data, tick_labels=treatment_labels, patch_artist=True)
        for patch, t in zip(bp["boxes"], treatments):
            patch.set_facecolor(treatment_color_map.get(t, "#cccccc"))
            patch.set_alpha(0.45)

        ax.set_ylabel(ylabel, fontweight="bold")
        if sci_y:
            _apply_sci_axis(ax, "y")
        _boldify_axes(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


## High-level entry points

def make_scene_plots(
    *,
    out_dir: Path,
    opts: PlotOptions,
    crowns: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    stacks: Dict[str, Sequence[xr.DataArray]],
    flight_names: Sequence[str],
):
    if not opts.make_scene_boxplots:
        return

    plots_dir = _ensure_dir(out_dir / "plots")

    if "SIF760_preprocessed" in stacks:
        save_path = plots_dir / "boxplots_SIF760_preprocessed_flightlines.png" if opts.save else None
        plot_scene_flightline_boxplots(
            stacks=stacks,
            key="SIF760_preprocessed",
            crowns=crowns,
            treatments=treatments,
            treatment_labels=treatment_labels,
            treatment_color_map=treatment_color_map,
            flight_names=flight_names,
            title="SIF760 (preprocessed) — flightlines",
            ylabel=r"SIF760 [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            sci_y=False,
            save_path=save_path,
        )

    for tag in ("NIRv", "FCVI", "saR2F"):
        k = f"FQE760_{tag}"
        if k in stacks:
            save_path = plots_dir / f"boxplots_FQE760_{tag}_flightlines.png" if opts.save else None
            plot_scene_flightline_boxplots(
                stacks=stacks,
                key=k,
                crowns=crowns,
                treatments=treatments,
                treatment_labels=treatment_labels,
                treatment_color_map=treatment_color_map,
                flight_names=flight_names,
                title=f"FQE760 ({tag}) — flightlines",
                ylabel=rf"FQE ({tag}) [nm$^{{-1}}$]",
                sci_y=True,
                save_path=save_path,
            )


def make_profile_plots(
    *,
    out_dir: Path,
    opts: PlotOptions,
    ref_raster: xr.DataArray,
    crowns: gpd.GeoDataFrame,
    treatment_areas: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    means_by_date: Dict[str, Dict[str, xr.DataArray]],
    stacks_by_date: Dict[str, Dict[str, Sequence[xr.DataArray]]],
    flight_names_by_date: Dict[str, Sequence[str]],
    ndvi_by_date: Dict[str, xr.DataArray],
    ndvi_threshold: float,
):
    plots_dir = _ensure_dir(out_dir / "plots")
    transform, xmin_pix, xmax_pix, ymin_pix, ymax_pix = _bbox_pixels_for_polygons(ref_raster, treatment_areas)

    if opts.make_profile_weighted_stats:
        sif_means = {}
        for d in ("20230617", "20240613", "20240823"):
            dd = means_by_date.get(d, {})
            if "SIF760_preprocessed_mean" in dd:
                sif_means[d] = dd["SIF760_preprocessed_mean"]

        save_path = plots_dir / "stats_SIF760_preprocessed_across_dates.png" if opts.save else None
        plot_weighted_stats_across_dates(
            sif_means,
            crowns=crowns,
            treatments=treatments,
            treatment_labels=treatment_labels,
            treatment_color_map=treatment_color_map,
            ylabel=r"SIF760 [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            sci_y=False,
            save_path=save_path,
        )

        for tag in ("NIRv", "FCVI", "saR2F"):
            fqe_means = {}
            key = f"FQE760_{tag}_mean"
            for d in ("20230617", "20240613", "20240823"):
                dd = means_by_date.get(d, {})
                if key in dd:
                    fqe_means[d] = dd[key]

            save_path = plots_dir / f"stats_FQE760_{tag}_across_dates.png" if opts.save else None
            plot_weighted_stats_across_dates(
                fqe_means,
                crowns=crowns,
                treatments=treatments,
                treatment_labels=treatment_labels,
                treatment_color_map=treatment_color_map,
                ylabel=rf"FQE ({tag}) [nm$^{{-1}}$]",
                sci_y=True,
                save_path=save_path,
            )

    if opts.make_profile_monthly_comparisons:
        jun23 = "20230617"
        jun24 = "20240613"
        aug24 = "20240823"

        jun23_lbl = format_date_label(jun23)
        jun24_lbl = format_date_label(jun24)
        aug24_lbl = format_date_label(aug24)

        sif_23 = means_by_date.get(jun23, {}).get("SIF760_preprocessed_mean")
        sif_j  = means_by_date.get(jun24, {}).get("SIF760_preprocessed_mean")
        sif_a  = means_by_date.get(aug24, {}).get("SIF760_preprocessed_mean")

        save_path = plots_dir / "monthly_SIF760_preprocessed.png" if opts.save else None

        plot_monthly_comparison(
            data_left=sif_23,
            data_june=sif_j,
            data_aug=sif_a,
            ndvi_left=ndvi_by_date.get(jun23),
            ndvi_june=ndvi_by_date.get(jun24),
            ndvi_aug=ndvi_by_date.get(aug24),
            ndvi_threshold=ndvi_threshold,
            treatments=treatments,
            treatment_areas=treatment_areas,
            treatment_color_map=treatment_color_map,
            transform=transform,
            xmin_pix=xmin_pix, xmax_pix=xmax_pix,
            ymin_pix=ymin_pix, ymax_pix=ymax_pix,
            value_label=r"SIF760 [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            diff_label=r"Δ SIF760 [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            title_left=jun23_lbl,
            title_june=jun24_lbl,
            title_aug=aug24_lbl,
            title_diff="Δ (Aug − Jun 24)",
            cmap_value=opts.cmap_value,
            cmap_diff=opts.cmap_diff,
            percentile_clip=opts.percentile_clip,
            symmetric_diff=opts.symmetric_diff,
            force_nonnegative_values=True,
            sci_value=False,
            save_path=save_path,
        )

        for tag in ("NIRv", "FCVI", "saR2F"):
            key = f"FQE760_{tag}_mean"

            fqe_23 = means_by_date.get(jun23, {}).get(key)
            fqe_j  = means_by_date.get(jun24, {}).get(key)
            fqe_a  = means_by_date.get(aug24, {}).get(key)

            save_path = plots_dir / f"monthly_FQE760_{tag}.png" if opts.save else None

        plot_monthly_comparison(
            data_left=fqe_23,
            data_june=fqe_j,
            data_aug=fqe_a,
            ndvi_left=ndvi_by_date.get(jun23),
            ndvi_june=ndvi_by_date.get(jun24),
            ndvi_aug=ndvi_by_date.get(aug24),
            ndvi_threshold=ndvi_threshold,
            treatments=treatments,
            treatment_areas=treatment_areas,
            treatment_color_map=treatment_color_map,
            transform=transform,
            xmin_pix=xmin_pix, xmax_pix=xmax_pix,
            ymin_pix=ymin_pix, ymax_pix=ymax_pix,
            value_label=rf"FQE ({tag}) [nm$^{{-1}}$]",
            diff_label=rf"Δ FQE ({tag}) [nm$^{{-1}}$]",
            title_left=jun23_lbl,
            title_june=jun24_lbl,
            title_aug=aug24_lbl,
            title_diff="Δ (Aug 2024 − Jun 2024)",
            cmap_value=opts.cmap_value,
            cmap_diff=opts.cmap_diff,
            percentile_clip=opts.percentile_clip,
            symmetric_diff=opts.symmetric_diff,
            force_nonnegative_values=True,
            sci_value=True,
            save_path=save_path,
        )

    if opts.make_profile_overview_maps:
        overview_dates=opts.overview_dates

        # --- SIF ---
        sif_stacks = {}
        for d in overview_dates:
            dd = stacks_by_date.get(d, {})
            if "SIF760_preprocessed" in dd:
                sif_stacks[d] = dd["SIF760_preprocessed"]

        save_path = plots_dir / "overview_SIF760_preprocessed.png" if opts.save else None
        plot_stack_overview(
            stacks_by_date=sif_stacks,
            dates=overview_dates,
            date_titles={d: format_date_label(d) for d in overview_dates},
            flight_names_by_date=flight_names_by_date,
            treatments=treatments,
            treatment_areas=treatment_areas,
            treatment_color_map=treatment_color_map,
            cmap=opts.cmap_overview,
            legend_label=r"SIF760 [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            percentile_clip=opts.percentile_clip,
            force_nonnegative_values=True,
            sci_legend=False,
            save_path=save_path,
            plot_treatments=opts.plot_treatments_overview_maps,
        )

        # --- FQE ---
        for tag in ("NIRv", "FCVI", "saR2F"):
            fqe_stacks = {}
            for d in overview_dates:
                dd = stacks_by_date.get(d, {})
                k = f"FQE760_{tag}"
                if k in dd:
                    fqe_stacks[d] = dd[k]

            save_path = plots_dir / f"overview_FQE760_{tag}.png" if opts.save else None
            plot_stack_overview(
                stacks_by_date=fqe_stacks,
                dates=overview_dates,
                date_titles={d: format_date_label(d) for d in overview_dates},
                flight_names_by_date=flight_names_by_date,
                treatments=treatments,
                treatment_areas=treatment_areas,
                treatment_color_map=treatment_color_map,
                cmap=opts.cmap_overview,
                legend_label=rf"FQE ({tag}) [nm$^{{-1}}$]",
                percentile_clip=opts.percentile_clip,
                force_nonnegative_values=True,
                sci_legend=True,
                save_path=save_path,
                plot_treatments=opts.plot_treatments_overview_maps,
            )