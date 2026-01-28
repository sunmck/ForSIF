# plots/plots_downscaling.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401

import rasterio
from rasterio import features

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl

from scipy.stats import gaussian_kde, ttest_ind


# --------------------------
# Options
# --------------------------

@dataclass
class PlotOptions:
    save: bool = True
    dpi: int = 300

    # Scene-level: boxplots per flightline (NOT means)
    make_scene_boxplots: bool = True

    # Profile-level:
    make_profile_weighted_stats: bool = True  # across flight dates
    make_profile_monthly_comparisons: bool = True
    make_profile_overview_maps: bool = True

    # Cosmetics
    cmap_value: str = "viridis"     # for positive-only rasters (SIF/FQE)
    cmap_diff: str = "RdBu_r"       # for deltas
    cmap_overview: str = "viridis"

    percentile_clip: Tuple[float, float] = (2, 98)
    symmetric_diff: bool = True


# --------------------------
# Date formatting + bold text
# --------------------------

_MONTHS = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


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


# --------------------------
# Small helpers
# --------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(fig, outpath: Path, dpi: int = 300):
    outpath = Path(outpath)
    if outpath.parent.exists() and not outpath.parent.is_dir():
        raise RuntimeError(
            f"Plot output directory exists but is not a directory: {outpath.parent}\n"
            f"Please delete/rename that file and re-run."
        )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _as_2d(da: xr.DataArray) -> np.ndarray:
    if da.ndim == 2:
        return da.values
    if da.ndim == 3 and "band" in da.dims and da.sizes.get("band", 0) == 1:
        return da.squeeze("band", drop=True).values
    return da.squeeze().values


def _get_cmap(name: str):
    cmap = mpl.cm.get_cmap(name).copy()
    # transparent NaNs everywhere
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def _sanitize_for_plotting(
    da: xr.DataArray,
    *,
    treat_zero_as_nodata: bool = False,
    zero_eps: float = 0.0,
    nodata_values: Sequence[float] = (
        -999, -999.0, -9999, -9999.0,
        -32768.0, 32767.0,
        65535.0, 4294967295.0,
    ),
) -> np.ndarray:
    """
    Convert DataArray to 2D float and turn nodata to NaN.
    Important for SFMNN: background often ends up as 0 -> treat_zero_as_nodata=True for SIF.
    """
    arr = _as_2d(da).astype("float64", copy=False)

    for nv in nodata_values:
        arr[np.isclose(arr, float(nv), equal_nan=False)] = np.nan

    try:
        rn = da.rio.nodata
        if rn is not None:
            arr[np.isclose(arr, float(rn), equal_nan=False)] = np.nan
    except Exception:
        pass

    if treat_zero_as_nodata:
        if zero_eps and zero_eps > 0:
            arr[np.isfinite(arr) & (np.abs(arr) <= float(zero_eps))] = np.nan
        else:
            arr[np.isclose(arr, 0.0, equal_nan=False)] = np.nan

    return arr


def mean_mosaic(stack: Sequence[xr.DataArray]) -> xr.DataArray:
    return xr.concat(list(stack), dim="scene").mean(dim="scene")


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

            if pixel_space and transform is not None:
                if geom.geom_type == "Polygon":
                    coords = [(~transform * (x, y)) for x, y in geom.exterior.coords]
                    patches.append(
                        Polygon(coords, closed=True, fill=False,
                                edgecolor=treatment_color_map[t], linewidth=lw)
                    )
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom:
                        coords = [(~transform * (x, y)) for x, y in poly.exterior.coords]
                        patches.append(
                            Polygon(coords, closed=True, fill=False,
                                    edgecolor=treatment_color_map[t], linewidth=lw)
                        )
            else:
                if geom.geom_type == "Polygon":
                    patches.append(
                        Polygon(list(geom.exterior.coords), closed=True, fill=False,
                                edgecolor=treatment_color_map[t], linewidth=lw)
                    )
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom:
                        patches.append(
                            Polygon(list(poly.exterior.coords), closed=True, fill=False,
                                    edgecolor=treatment_color_map[t], linewidth=lw)
                        )

        if patches:
            ax.add_collection(PatchCollection(patches, match_original=True))


def _empty_map_panel(ax, title: str):
    ax.set_title(title, fontweight="bold")
    ax.axis("off")


def _empty_stats_panel(ax, title: str = ""):
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)
    # keep axes visible and consistent, but no text


def _apply_ndvi_mask(
    data: xr.DataArray,
    ndvi: Optional[xr.DataArray],
    ndvi_threshold: float,
    *,
    treat_zero_as_nodata: bool = False,
) -> np.ndarray:
    arr = _sanitize_for_plotting(data, treat_zero_as_nodata=treat_zero_as_nodata)
    if ndvi is None:
        return arr

    nd = _sanitize_for_plotting(ndvi, treat_zero_as_nodata=False)

    # mask if NDVI is non-finite OR below threshold
    bad = (~np.isfinite(nd)) | (nd < float(ndvi_threshold))
    arr[bad] = np.nan
    return arr


def _robust_vmin_vmax(
    arrays: Sequence[np.ndarray],
    *,
    percentile_clip: Tuple[float, float],
    force_nonnegative: bool = True,
) -> Tuple[float, float]:
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


# --------------------------
# Crown-weighted extraction
# --------------------------

def get_pixels_by_treatment_weighted(
    raster: xr.DataArray,
    crowns: gpd.GeoDataFrame,
    treatment_value: int,
    supersample: int = 10,
    min_weight: float = 0.5,
    *,
    treat_zero_as_nodata: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _sanitize_for_plotting(raster, treat_zero_as_nodata=treat_zero_as_nodata)
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
        ((geom, 1) for geom in polygons),
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

    valid = np.isfinite(pixels) & (weights >= min_weight)
    return pixels[valid], weights[valid], mask


# --------------------------
# PROFILE-LEVEL: Weighted stats across dates
# --------------------------

def plot_weighted_stats_across_dates(
    datasets_by_date: Dict[str, xr.DataArray],
    *,
    crowns: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    title: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    treat_zero_as_nodata: bool = False,
    sci_y: bool = False,
):
    """
    3xN layout: boxplot + hist+kde + CDF for each date.
    Missing dates are left empty (no "missing" text).
    """
    dates = ["20230617", "20240613", "20240823"]
    date_labels = [format_date_label(d) for d in dates]

    n_cols = len(dates)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 12))
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

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
                t, p = ttest_ind(data[i], data[j], equal_var=False)
                results.append((i, j, float(p), p_to_stars(float(p))))
        return results

    # Global x-lims (across all available)
    all_px = []
    extracted = {}

    for d in dates:
        da = datasets_by_date.get(d)
        if da is None:
            continue
        data = []
        weights = []
        for t in treatments:
            px, w, _ = get_pixels_by_treatment_weighted(
                da, crowns, t, treat_zero_as_nodata=treat_zero_as_nodata
            )
            data.append(px)
            weights.append(w)
            all_px.append(px)
        extracted[d] = (data, weights, welch_tests(data))

    if all_px:
        flat = np.concatenate(all_px)
        x_min, x_max = float(np.nanmin(flat)), float(np.nanmax(flat))
    else:
        x_min, x_max = 0.0, 1.0

    colors = [treatment_color_map[t] for t in treatments]

    for col, (d, dlabel) in enumerate(zip(dates, date_labels)):
        ax_box, ax_hist, ax_cdf = axes[:, col]
        da = datasets_by_date.get(d)

        if da is None:
            _empty_stats_panel(ax_box, dlabel)
            _empty_stats_panel(ax_hist, "")
            _empty_stats_panel(ax_cdf, "")
            ax_box.set_ylabel(ylabel, fontweight="bold")
            continue

        data, weights, welch_res = extracted[d]

        bp = ax_box.boxplot(data, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for median, c in zip(bp["medians"], colors):
            median.set_color(c)
            median.set_linewidth(2)

        ax_box.set_xticks(range(1, len(treatment_labels) + 1))
        ax_box.set_xticklabels(treatment_labels, rotation=20)
        ax_box.set_title(dlabel)
        ax_box.set_ylabel(ylabel)

        # brackets
        tops = [np.nanmax(d_) if len(d_) else np.nan for d_ in data]
        base_y = np.nanmax(tops) if np.isfinite(np.nanmax(tops)) else 0.0
        y_min, y_max = x_min, x_max
        BRACKET_PAD = 0.06 * (y_max - y_min + 1e-9)
        BRACKET_STEP = 0.10 * (y_max - y_min + 1e-9)
        for k, (i, j, p, stars) in enumerate(welch_res):
            y = base_y + BRACKET_PAD + k * BRACKET_STEP
            h = BRACKET_STEP * 0.6
            ax_box.plot([i + 1, i + 1, j + 1, j + 1], [y, y + h, y + h, y], lw=1.2, c="black")
            ax_box.text((i + j + 2) / 2, y + h * 0.8, stars, ha="center", va="bottom",
                        fontsize=12, fontweight="bold")

        ax_hist.grid(True, linestyle="--", alpha=0.35)
        ax_cdf.grid(True, linestyle="--", alpha=0.35)

        # Hist + KDE
        for arr, w, c in zip(data, weights, colors):
            if len(arr) == 0:
                continue
            ax_hist.hist(arr, bins=30, weights=w, alpha=0.35, density=True, color=c)
            ax_hist.axvline(np.average(arr, weights=w), color=c, linestyle="--", linewidth=1)
            kde = gaussian_kde(arr, weights=w)
            xs = np.linspace(np.nanmin(arr), np.nanmax(arr), 200)
            ax_hist.plot(xs, kde(xs), color=c, linewidth=2)

        ax_hist.set_xlim(x_min, x_max)

        # CDF
        for arr, w, c in zip(data, weights, colors):
            if len(arr) == 0:
                continue
            idx = np.argsort(arr)
            arrs = arr[idx]
            ws = w[idx]
            cdf = np.cumsum(ws) / np.sum(ws)
            ax_cdf.plot(arrs, cdf, color=c, linewidth=2)

        ax_cdf.set_xlim(x_min, x_max)

        if sci_y:
            _apply_sci_axis(ax_box, "y")
            _apply_sci_axis(ax_hist, "x")
            _apply_sci_axis(ax_cdf, "x")

        _boldify_axes(ax_box)
        _boldify_axes(ax_hist)
        _boldify_axes(ax_cdf)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# --------------------------
# PROFILE-LEVEL: Monthly comparison (June/Aug/Δ)
#   - value panels: viridis (positive-only)
#   - diff: RdBu, centered at 0
#   - colorbars BELOW (roomy) + scientific notation for FQE
# --------------------------

def plot_monthly_comparison(
    *,
    data_june: Optional[xr.DataArray],
    data_aug: Optional[xr.DataArray],
    ndvi_june: Optional[xr.DataArray] = None,
    ndvi_aug: Optional[xr.DataArray] = None,
    ndvi_threshold: float = 0.5,
    treat_zero_as_nodata: bool = False,
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
    title_june: str,
    title_aug: str,
    title_diff: str = "Δ (Aug − Jun)",
    cmap_value: str = "viridis",
    cmap_diff: str = "RdBu_r",
    percentile_clip: Tuple[float, float] = (2, 98),
    symmetric_diff: bool = True,
    force_nonnegative_values: bool = True,
    sci_value: bool = False,
    save_path: Optional[Path] = None,
):
    vj = None
    va = None

    if data_june is not None:
        vj = _apply_ndvi_mask(data_june, ndvi_june, ndvi_threshold, treat_zero_as_nodata=treat_zero_as_nodata)
    if data_aug is not None:
        va = _apply_ndvi_mask(data_aug, ndvi_aug, ndvi_threshold, treat_zero_as_nodata=treat_zero_as_nodata)

    diff = None
    if vj is not None and va is not None:
        valid = np.isfinite(vj) & np.isfinite(va)
        diff = np.full_like(vj, np.nan, dtype="float64")
        diff[valid] = va[valid] - vj[valid]

    # Compute vmin/vmax ONLY inside bbox (matches what you see; fixes SFMNN stretch problems)
    vv = []
    if vj is not None:
        vv.append(_clip_bbox(vj, xmin_pix, xmax_pix, ymin_pix, ymax_pix))
    if va is not None:
        vv.append(_clip_bbox(va, xmin_pix, xmax_pix, ymin_pix, ymax_pix))
    vmin, vmax = _robust_vmin_vmax(vv, percentile_clip=percentile_clip, force_nonnegative=force_nonnegative_values)

    # diff stretch (centered at 0)
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

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    # extra bottom room for horizontal colorbars
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.02, top=0.92, bottom=0.18)

    # June
    if vj is None:
        _empty_map_panel(axes[0], title_june)
        im0 = None
    else:
        im0 = axes[0].imshow(vj, cmap=cmapV, vmin=vmin, vmax=vmax, origin="upper")
        _draw_treatment_outlines(
            axes[0], treatment_areas, treatment_color_map, treatments,
            transform=transform, pixel_space=True
        )
        axes[0].set_xlim(xmin_pix, xmax_pix)
        axes[0].set_ylim(ymax_pix, ymin_pix)
        axes[0].set_title(title_june, fontweight="bold")
        axes[0].axis("off")

    # Aug
    if va is None:
        _empty_map_panel(axes[1], title_aug)
        im1 = None
    else:
        im1 = axes[1].imshow(va, cmap=cmapV, vmin=vmin, vmax=vmax, origin="upper")
        _draw_treatment_outlines(
            axes[1], treatment_areas, treatment_color_map, treatments,
            transform=transform, pixel_space=True
        )
        axes[1].set_xlim(xmin_pix, xmax_pix)
        axes[1].set_ylim(ymax_pix, ymin_pix)
        axes[1].set_title(title_aug, fontweight="bold")
        axes[1].axis("off")

    # Diff
    if diff is None:
        _empty_map_panel(axes[2], title_diff)
        im2 = None
    else:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=dvmin, vmax=dvmax)
        im2 = axes[2].imshow(diff, cmap=cmapD, norm=norm, origin="upper")
        _draw_treatment_outlines(
            axes[2], treatment_areas, treatment_color_map, treatments,
            transform=transform, pixel_space=True
        )
        axes[2].set_xlim(xmin_pix, xmax_pix)
        axes[2].set_ylim(ymax_pix, ymin_pix)
        axes[2].set_title(title_diff, fontweight="bold")
        axes[2].axis("off")

    # --- horizontal colorbars BELOW (roomy, not squished) ---
    # one for the two value maps
    im_val = im0 if im0 is not None else im1
    cax_val = fig.add_axes([0.08, 0.08, 0.58, 0.035])  # [left, bottom, width, height]
    cax_dif = fig.add_axes([0.72, 0.08, 0.22, 0.035])

    if im_val is not None:
        cb1 = fig.colorbar(im_val, cax=cax_val, orientation="horizontal")
        cb1.set_label(value_label, fontweight="bold")
        if sci_value:
            cb1.formatter = _sci_formatter()
            cb1.update_ticks()
        _boldify_colorbar(cb1)
    else:
        cax_val.axis("off")

    if im2 is not None:
        cb2 = fig.colorbar(im2, cax=cax_dif, orientation="horizontal")
        cb2.set_label(diff_label, fontweight="bold")
        if sci_value:
            cb2.formatter = _sci_formatter()
            cb2.update_ticks()
        _boldify_colorbar(cb2)
    else:
        cax_dif.axis("off")

    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# --------------------------
# PROFILE-LEVEL: Overview stacks (3 date columns, 4 flight rows)
#   - increased wspace
#   - NaNs transparent
#   - vmin/vmax computed from finite data only
# --------------------------

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
    treat_zero_as_nodata: bool = False,
    force_nonnegative_values: bool = True,
    sci_legend: bool = False,
    save_path: Optional[Path] = None,
):
    nrows = 4
    ncols = len(dates)

    if date_titles is None:
        date_titles = {d: format_date_label(d) for d in dates}

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4 * ncols, 3.7 * nrows))
    cmapO = _get_cmap(cmap)

    # robust vmin/vmax across all available rasters (finite only)
    vals = []
    for d in dates:
        stack = stacks_by_date.get(d)
        if not stack:
            continue
        for da in stack:
            a = _sanitize_for_plotting(da, treat_zero_as_nodata=treat_zero_as_nodata)
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)

    if vals:
        flat = np.concatenate(vals)
        lo, hi = np.nanpercentile(flat, percentile_clip)
        vmin, vmax = float(lo), float(hi)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 0:
            vmin, vmax = float(np.nanmin(flat)), float(np.nanmax(flat))
        if force_nonnegative_values and vmin < 0:
            vmin = 0.0
            if vmax <= 0:
                vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im_for_cbar = None

    for c, d in enumerate(dates):
        stack = stacks_by_date.get(d, None)
        col_title = date_titles.get(d, d)
        flight_names = list(flight_names_by_date.get(d, [])) if flight_names_by_date is not None else None

        for r in range(nrows):
            ax = axes[r, c]

            if (stack is None) or (r >= len(stack)):
                # leave empty but keep axes consistent (no "missing" text)
                suffix = f" — {flight_names[r]}" if (flight_names is not None and r < len(flight_names)) else ""
                ax.set_title(f"{col_title}{suffix}", fontweight="bold")
                ax.set_facecolor("white")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.tick_params(axis="both", which="both", length=4, labelsize=8)
                ax.grid(False)
                _boldify_axes(ax)
                continue

            da = stack[r]
            x = da["x"].values
            y = da["y"].values
            arr = _sanitize_for_plotting(da, treat_zero_as_nodata=treat_zero_as_nodata)

            im = ax.imshow(
                arr,
                cmap=cmapO,
                vmin=vmin,
                vmax=vmax,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="upper",
                interpolation="none",
            )
            if im_for_cbar is None:
                im_for_cbar = im

            _draw_treatment_outlines(
                ax,
                treatment_areas=treatment_areas,
                treatment_color_map=treatment_color_map,
                treatments=treatments,
                raster_crs=da.rio.crs,
                pixel_space=False,
            )

            suffix = f" — {flight_names[r]}" if (flight_names is not None and r < len(flight_names)) else ""
            ax.set_title(f"{col_title}{suffix}", fontweight="bold")

            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="both", which="both", length=4, labelsize=8)
            _boldify_axes(ax)

    # More spacing between columns so tick labels don't sit on top of maps
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


# --------------------------
# SCENE-LEVEL: Boxplots per flightline (fixed 4 slots)
# --------------------------

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
    treat_zero_as_nodata: bool = False,
    sci_y: bool = False,
    save_path: Optional[Path] = None,
):
    ncols = 4
    fig, axes = plt.subplots(1, ncols, figsize=(20, 6), sharey=True)

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
            # leave empty, keep axes (no "missing" text)
            ax.set_ylabel(ylabel, fontweight="bold")
            if sci_y:
                _apply_sci_axis(ax, "y")
            _boldify_axes(ax)
            continue

        data = []
        for t in treatments:
            px, _, _ = get_pixels_by_treatment_weighted(
                stack[i], crowns, t, treat_zero_as_nodata=treat_zero_as_nodata
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


# --------------------------
# High-level entry points
# --------------------------

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
    plots_dir = _ensure_dir(out_dir / "plots")

    if not opts.make_scene_boxplots:
        return

    # SIF flightlines
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
            ylabel=r"SIF [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            treat_zero_as_nodata=True,
            sci_y=False,
            save_path=save_path,
        )

    # FQE flightlines for each index (scientific notation)
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
                treat_zero_as_nodata=False,
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

    # ---------- STATS across dates ----------
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
            title="SIF760 (preprocessed) — weighted stats across dates",
            ylabel=r"SIF [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            treat_zero_as_nodata=True,
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
                title=f"FQE760 ({tag}) — weighted stats across dates",
                ylabel=rf"FQE ({tag}) [nm$^{{-1}}$]",
                treat_zero_as_nodata=False,
                sci_y=True,
                save_path=save_path,
            )

    # ---------- MONTHLY maps (June/Aug/delta) ----------
    if opts.make_profile_monthly_comparisons:
        june = "20240613"
        aug = "20240823"

        june_lbl = format_date_label(june)
        aug_lbl = format_date_label(aug)

        # SIF monthly maps (NDVI mask + treat 0 as nodata; positive colormap)
        sif_j = means_by_date.get(june, {}).get("SIF760_preprocessed_mean")
        sif_a = means_by_date.get(aug, {}).get("SIF760_preprocessed_mean")
        save_path = plots_dir / "monthly_SIF760_preprocessed.png" if opts.save else None

        plot_monthly_comparison(
            data_june=sif_j,
            data_aug=sif_a,
            ndvi_june=ndvi_by_date.get(june),
            ndvi_aug=ndvi_by_date.get(aug),
            ndvi_threshold=ndvi_threshold,
            treat_zero_as_nodata=True,
            treatments=treatments,
            treatment_areas=treatment_areas,
            treatment_color_map=treatment_color_map,
            transform=transform,
            xmin_pix=xmin_pix, xmax_pix=xmax_pix,
            ymin_pix=ymin_pix, ymax_pix=ymax_pix,
            value_label=r"SIF [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            diff_label=r"Δ SIF [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            title_june=june_lbl,
            title_aug=aug_lbl,
            title_diff="Δ (Aug − Jun)",
            cmap_value=opts.cmap_value,
            cmap_diff=opts.cmap_diff,
            percentile_clip=opts.percentile_clip,
            symmetric_diff=opts.symmetric_diff,
            force_nonnegative_values=True,
            sci_value=False,
            save_path=save_path,
        )

        # FQE monthly maps per index (scientific notation on colorbars)
        for tag in ("NIRv", "FCVI", "saR2F"):
            key = f"FQE760_{tag}_mean"
            fqe_j = means_by_date.get(june, {}).get(key)
            fqe_a = means_by_date.get(aug, {}).get(key)
            save_path = plots_dir / f"monthly_FQE760_{tag}.png" if opts.save else None

            plot_monthly_comparison(
                data_june=fqe_j,
                data_aug=fqe_a,
                ndvi_june=ndvi_by_date.get(june),
                ndvi_aug=ndvi_by_date.get(aug),
                ndvi_threshold=ndvi_threshold,
                treat_zero_as_nodata=False,
                treatments=treatments,
                treatment_areas=treatment_areas,
                treatment_color_map=treatment_color_map,
                transform=transform,
                xmin_pix=xmin_pix, xmax_pix=xmax_pix,
                ymin_pix=ymin_pix, ymax_pix=ymax_pix,
                value_label=rf"FQE ({tag}) [nm$^{{-1}}$]",
                diff_label=rf"Δ FQE ({tag}) [nm$^{{-1}}$]",
                title_june=june_lbl,
                title_aug=aug_lbl,
                title_diff="Δ (Aug − Jun)",
                cmap_value=opts.cmap_value,
                cmap_diff=opts.cmap_diff,
                percentile_clip=opts.percentile_clip,
                symmetric_diff=opts.symmetric_diff,
                force_nonnegative_values=True,
                sci_value=True,
                save_path=save_path,
            )

    # ---------- OVERVIEW stacks (3 date columns, 4 flights) ----------
    if opts.make_profile_overview_maps:
        sif_stacks = {}
        for d in ("20230617", "20240613", "20240823"):
            dd = stacks_by_date.get(d, {})
            if "SIF760_preprocessed" in dd:
                sif_stacks[d] = dd["SIF760_preprocessed"]

        save_path = plots_dir / "overview_SIF760_preprocessed.png" if opts.save else None
        plot_stack_overview(
            stacks_by_date=sif_stacks,
            dates=("20230617", "20240613", "20240823"),
            date_titles={d: format_date_label(d) for d in ("20230617", "20240613", "20240823")},
            flight_names_by_date=flight_names_by_date,
            treatments=treatments,
            treatment_areas=treatment_areas,
            treatment_color_map=treatment_color_map,
            cmap=opts.cmap_overview,
            legend_label=r"SIF [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
            percentile_clip=opts.percentile_clip,
            treat_zero_as_nodata=True,      # critical for SFMNN
            force_nonnegative_values=True,
            sci_legend=False,
            save_path=save_path,
        )

        for tag in ("NIRv", "FCVI", "saR2F"):
            fqe_stacks = {}
            for d in ("20230617", "20240613", "20240823"):
                dd = stacks_by_date.get(d, {})
                k = f"FQE760_{tag}"
                if k in dd:
                    fqe_stacks[d] = dd[k]

            save_path = plots_dir / f"overview_FQE760_{tag}.png" if opts.save else None
            plot_stack_overview(
                stacks_by_date=fqe_stacks,
                dates=("20230617", "20240613", "20240823"),
                date_titles={d: format_date_label(d) for d in ("20230617", "20240613", "20240823")},
                flight_names_by_date=flight_names_by_date,
                treatments=treatments,
                treatment_areas=treatment_areas,
                treatment_color_map=treatment_color_map,
                cmap=opts.cmap_overview,
                legend_label=rf"FQE ({tag}) [nm$^{{-1}}$]",
                percentile_clip=opts.percentile_clip,
                treat_zero_as_nodata=False,
                force_nonnegative_values=True,
                sci_legend=True,
                save_path=save_path,
            )
