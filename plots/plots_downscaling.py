from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401  # needed to register the .rio accessor

import rasterio
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

    cmap_value: str = "viridis"
    cmap_diff: str = "RdBu_r"
    cmap_overview: str = "viridis"

    percentile_clip: Tuple[float, float] = (2, 98)
    symmetric_diff: bool = True


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
    title: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    sci_y: bool = False,
):
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
                if data[i].size == 0 or data[j].size == 0:
                    continue
                _, p = ttest_ind(data[i], data[j], equal_var=False)
                results.append((i, j, float(p), p_to_stars(float(p))))
        return results

    all_px = []
    extracted = {}

    for d in dates:
        da = datasets_by_date.get(d)
        if da is None:
            continue
        data = []
        weights = []
        for t in treatments:
            px, w, _ = get_pixels_by_treatment_weighted(da, crowns, t)

            data.append(px)
            weights.append(w)
            all_px.append(px)
        extracted[d] = (data, weights, welch_tests(data))

    if all_px:
        flat = np.concatenate(all_px) if all_px else np.array([0.0])
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

        tops = [np.nanmax(d_) if d_.size else np.nan for d_ in data]
        base_y = np.nanmax(tops) if np.isfinite(np.nanmax(tops)) else 0.0
        y_min, y_max = x_min, x_max
        bracket_pad = 0.06 * (y_max - y_min + 1e-9)
        bracket_step = 0.10 * (y_max - y_min + 1e-9)
        for k, (i, j, _p, stars) in enumerate(welch_res):
            y = base_y + bracket_pad + k * bracket_step
            h = bracket_step * 0.6
            ax_box.plot([i + 1, i + 1, j + 1, j + 1], [y, y + h, y + h, y], lw=1.2, c="black")
            ax_box.text((i + j + 2) / 2, y + h * 0.8, stars, ha="center", va="bottom",
                        fontsize=12, fontweight="bold")

        ax_hist.grid(True, linestyle="--", alpha=0.35)
        ax_cdf.grid(True, linestyle="--", alpha=0.35)

        for arr, w, c in zip(data, weights, colors):
            if arr.size == 0 or w.size == 0:
                continue
            ax_hist.hist(arr, bins=30, weights=w, alpha=0.35, density=True, color=c)
            ax_hist.axvline(np.average(arr, weights=w), color=c, linestyle="--", linewidth=1)
            kde = gaussian_kde(arr, weights=w)
            xs = np.linspace(np.nanmin(arr), np.nanmax(arr), 200)
            ax_hist.plot(xs, kde(xs), color=c, linewidth=2)

        ax_hist.set_xlim(x_min, x_max)

        for arr, w, c in zip(data, weights, colors):
            if arr.size == 0 or w.size == 0:
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


# PROFILE: Monthly comparison (June/Aug/Δ)

def plot_monthly_comparison(
    *,
    data_june: Optional[xr.DataArray],
    data_aug: Optional[xr.DataArray],
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
        vj = _apply_ndvi_mask(
            data_june, ndvi_june, ndvi_threshold,
        )
    if data_aug is not None:
        va = _apply_ndvi_mask(
            data_aug, ndvi_aug, ndvi_threshold,
        )

    diff = None
    if vj is not None and va is not None:
        valid = np.isfinite(vj) & np.isfinite(va)
        diff = np.full_like(vj, np.nan, dtype="float64")
        diff[valid] = va[valid] - vj[valid]

    vv = []
    if vj is not None:
        vv.append(_clip_bbox(vj, xmin_pix, xmax_pix, ymin_pix, ymax_pix))
    if va is not None:
        vv.append(_clip_bbox(va, xmin_pix, xmax_pix, ymin_pix, ymax_pix))
    vmin, vmax = _robust_vmin_vmax(vv, percentile_clip=percentile_clip, force_nonnegative=force_nonnegative_values)

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

    im_val = im0 if im0 is not None else im1
    cax_val = fig.add_axes([0.08, 0.08, 0.58, 0.035])
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
            title="SIF760 (preprocessed) — weighted stats across dates",
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
                title=f"FQE760 ({tag}) — weighted stats across dates",
                ylabel=rf"FQE ({tag}) [nm$^{{-1}}$]",
                sci_y=True,
                save_path=save_path,
            )

    if opts.make_profile_monthly_comparisons:
        june = "20240613"
        aug = "20240823"

        june_lbl = format_date_label(june)
        aug_lbl = format_date_label(aug)

        sif_j = means_by_date.get(june, {}).get("SIF760_preprocessed_mean")
        sif_a = means_by_date.get(aug, {}).get("SIF760_preprocessed_mean")
        save_path = plots_dir / "monthly_SIF760_preprocessed.png" if opts.save else None

        plot_monthly_comparison(
            data_june=sif_j,
            data_aug=sif_a,
            ndvi_june=ndvi_by_date.get(june),
            ndvi_aug=ndvi_by_date.get(aug),
            ndvi_threshold=ndvi_threshold,
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
            legend_label=r"SIF [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
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