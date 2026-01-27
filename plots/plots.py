from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
from rasterio import features

from shapely.geometry import box
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Optional (used only in weighted stats plots)
from scipy.stats import gaussian_kde, ttest_ind


# --------------------------
# Plot toggles / configuration
# --------------------------

@dataclass
class PlotOptions:
    # Where figures are saved (a "plots" subfolder will be created inside run_out_dir)
    save: bool = True
    dpi: int = 300

    # Toggle plot families
    make_overview_maps: bool = True           # 4x2 SIF overview (or any stack overview)
    make_monthly_comparisons: bool = True     # June / Aug / Δ maps
    make_boxplots: bool = True                # per-treatment boxplots (4-column layout)
    make_weighted_stats: bool = True          # box+hist+cdf + Welch stars

    # Plot cosmetics
    cmap_main: str = "RdBu_r"
    cmap_overview: str = "viridis"
    tick_interval: int = 50

    # Clipping (for robust vmin/vmax)
    percentile_clip: Tuple[float, float] = (2, 98)
    symmetric_diff: bool = True


# --------------------------
# Small helpers
# --------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(fig, outpath: Path, dpi: int = 300):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _bbox_pixels_for_polygons(ref_raster, polygons: gpd.GeoDataFrame, pad: int = 10):
    """Compute pixel bbox for plotting given polygons in the same CRS as raster."""
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


def _as_2d(da: xr.DataArray) -> np.ndarray:
    """Return a 2D numpy array for plotting/statistics."""
    if da.ndim == 2:
        return da.values
    if da.ndim == 3 and "band" in da.dims and da.sizes.get("band", 0) == 1:
        return da.squeeze("band", drop=True).values
    # fallback: squeeze any singleton dims
    return da.squeeze().values


def mean_mosaic(stack: Sequence[xr.DataArray]) -> xr.DataArray:
    """Mean mosaic from already-aligned rasters."""
    return xr.concat(list(stack), dim="scene").mean(dim="scene")


# --------------------------
# Treatment-weighted extraction (from your notebook, cleaned)
# --------------------------

def get_pixels_by_treatment_weighted(
    rasters: Union[xr.DataArray, Sequence[xr.DataArray]],
    crowns: gpd.GeoDataFrame,
    treatment_value: int,
    supersample: int = 10,
    min_weight: float = 0.5,
):
    """
    Extract pixels within crowns of a treatment, weighted by fractional pixel coverage
    via supersampling rasterization.
    """
    if isinstance(rasters, (list, tuple)):
        arr = np.mean([_as_2d(r) for r in rasters], axis=0)
        transform = rasters[0].rio.transform()
        out_shape = _as_2d(rasters[0]).shape
    else:
        arr = _as_2d(rasters)
        transform = rasters.rio.transform()
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

    # fractional cover per pixel
    mask = ss_mask.reshape(out_shape[0], supersample, out_shape[1], supersample).mean(axis=(1, 3))

    pixel_mask = mask > 0
    pixels = arr[pixel_mask]
    weights = mask[pixel_mask]

    valid = np.isfinite(pixels) & (weights >= min_weight)
    return pixels[valid], weights[valid], mask


# --------------------------
# Plot 1: Weighted stats (box + hist+kde + cdf) + significance stars
# --------------------------

def plot_weighted_stats_full(
    datasets: Sequence[Tuple[xr.DataArray, str]],
    crowns: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    colors: Sequence[str],
    unit_label: str = "",
    save_path: Optional[Path] = None,
    share_x: bool = True,
):
    """
    datasets: list of (dataarray, label)
      Example labels:
        "Mean FQE760 (20240613)" or "Mean SIF760 (20240823)"
    """
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

    n_cols = len(datasets)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 12))
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    all_pixels = []
    extracted = []

    for da, label in datasets:
        data = []
        weights = []
        for t in treatments:
            px, w, _ = get_pixels_by_treatment_weighted(da, crowns, t)
            data.append(px)
            weights.append(w)
            all_pixels.append(px)
        extracted.append((label, data, weights, welch_tests(data)))

    all_pixels_flat = np.concatenate(all_pixels) if len(all_pixels) else np.array([np.nan])
    x_min, x_max = np.nanmin(all_pixels_flat), np.nanmax(all_pixels_flat)
    y_min, y_max = x_min, x_max

    BRACKET_PAD = 0.06 * (y_max - y_min + 1e-9)
    BRACKET_STEP = 0.10 * (y_max - y_min + 1e-9)

    max_brackets = max(len(wr) for _, _, _, wr in extracted) if extracted else 0
    global_ymax = y_max + BRACKET_PAD + BRACKET_STEP * max_brackets

    for col, (label, data, weights, welch_res) in enumerate(extracted):
        ax_box, ax_hist, ax_cdf = axes[:, col]

        # --- boxplot ---
        bp = ax_box.boxplot(data, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for median, c in zip(bp["medians"], colors):
            median.set_color(c)
            median.set_linewidth(2)

        ax_box.set_xticks(range(1, len(treatment_labels) + 1))
        ax_box.set_xticklabels(treatment_labels, rotation=20)
        ax_box.set_title(label)
        ax_box.set_ylabel(f"[{unit_label}]" if unit_label else "")

        # brackets
        tops = [np.nanmax(d) if len(d) else np.nan for d in data]
        base_y = np.nanmax(tops)
        for k, (i, j, p, stars) in enumerate(welch_res):
            y = base_y + BRACKET_PAD + k * BRACKET_STEP
            h = BRACKET_STEP * 0.6
            ax_box.plot([i+1, i+1, j+1, j+1], [y, y+h, y+h, y], lw=1.2, c="black")
            ax_box.text((i + j + 2) / 2, y + h * 0.8, stars, ha="center", va="bottom", fontsize=12)

        ax_box.set_ylim(y_min, global_ymax)

        # --- histogram + KDE ---
        for arr, w, lab, c in zip(data, weights, treatment_labels, colors):
            if len(arr) == 0:
                continue
            ax_hist.hist(arr, bins=30, weights=w, alpha=0.35, density=True, color=c)
            ax_hist.axvline(np.average(arr, weights=w), color=c, linestyle="--", linewidth=1)

            kde = gaussian_kde(arr, weights=w)
            xs = np.linspace(np.nanmin(arr), np.nanmax(arr), 200)
            ax_hist.plot(xs, kde(xs), color=c, linewidth=2)

        if share_x:
            ax_hist.set_xlim(x_min, x_max)
        ax_hist.grid(True, linestyle="--", alpha=0.35)

        # --- CDF ---
        for arr, w, c in zip(data, weights, colors):
            if len(arr) == 0:
                continue
            idx = np.argsort(arr)
            arrs = arr[idx]
            ws = w[idx]
            cdf = np.cumsum(ws) / np.sum(ws)
            ax_cdf.plot(arrs, cdf, color=c, linewidth=2)

        if share_x:
            ax_cdf.set_xlim(x_min, x_max)
        ax_cdf.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# --------------------------
# Plot 2: Monthly comparison maps (June / Aug / Δ)
# --------------------------

def plot_monthly_comparison(
    data_june: xr.DataArray,
    data_aug: xr.DataArray,
    treatments: Sequence[int],
    treatment_areas: gpd.GeoDataFrame,
    treatment_color_map: Dict[int, str],
    transform,
    xmin_pix: int,
    xmax_pix: int,
    ymin_pix: int,
    ymax_pix: int,
    titles=("June", "August", "Δ (Aug − Jun)"),
    cmaps=("RdBu_r", "RdBu_r", "RdBu_r"),
    value_label="Value",
    diff_label="ΔValue",
    percentile_clip=(2, 98),
    symmetric_diff=True,
    save_path: Optional[Path] = None,
):
    # Compute difference
    vj = _as_2d(data_june)
    va = _as_2d(data_aug)

    valid = np.isfinite(vj) & np.isfinite(va)
    diff = np.full_like(vj, np.nan, dtype="float64")
    diff[valid] = va[valid] - vj[valid]

    # vmin/vmax for June/Aug panels
    vmin = np.nanmin([vj, va])
    vmax = np.nanmax([vj, va])

    # vmin/vmax for diff
    flat = diff[np.isfinite(diff)]
    if flat.size:
        if symmetric_diff:
            m = np.nanmax(np.abs(flat))
            dvmin, dvmax = -m, m
        else:
            dvmin, dvmax = np.percentile(flat, percentile_clip)
    else:
        dvmin, dvmax = 0, 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    arrs = [vj, va, diff]
    vmins = [vmin, vmin, dvmin]
    vmaxs = [vmax, vmax, dvmax]

    for ax, arr, title, cmap, vmi, vma in zip(axes, arrs, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(arr, cmap=cmap, vmin=vmi, vmax=vma, origin="upper")

        # overlay treatment polygons
        for t in treatments:
            area_t = treatment_areas[treatment_areas["treatment"] == t]
            patches = []
            for geom in area_t.geometry:
                if geom.geom_type == "Polygon":
                    coords = [(~transform * (x, y)) for x, y in geom.exterior.coords]
                    patches.append(Polygon(coords, closed=True, fill=False,
                                           edgecolor=treatment_color_map[t], linewidth=1.5))
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom:
                        coords = [(~transform * (x, y)) for x, y in poly.exterior.coords]
                        patches.append(Polygon(coords, closed=True, fill=False,
                                               edgecolor=treatment_color_map[t], linewidth=1.5))
            ax.add_collection(PatchCollection(patches, match_original=True))

        ax.set_xlim(xmin_pix, xmax_pix)
        ax.set_ylim(ymax_pix, ymin_pix)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    cbar1 = fig.colorbar(axes[0].images[0], ax=axes[:2], fraction=0.046, pad=0.04)
    cbar1.set_label(value_label)
    cbar2 = fig.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label(diff_label)

    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# --------------------------
# Plot 3: Boxplots (4-column layout, supports placeholders)
# --------------------------

def plot_raster_boxplots(
    rasters: Sequence[xr.DataArray],
    raster_labels: Sequence[str],
    crowns: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    title: str = "Boxplots by Treatment",
    ylabel: str = "Value",
    save_path: Optional[Path] = None,
):
    # align to 4 slots
    aligned = []
    it = iter(rasters)
    for lbl in raster_labels:
        if lbl == "":
            aligned.append(None)
        else:
            aligned.append(next(it))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for ax, raster, lbl in zip(axes, aligned, raster_labels):
        ax.set_title(lbl)
        ax.grid(True)

        if raster is None:
            ax.set_xticks([])
            ax.set_ylabel(ylabel)
            continue

        data = []
        for t in treatments:
            px, _, _ = get_pixels_by_treatment_weighted(raster, crowns, t)
            data.append(px)

        bp = ax.boxplot(data, tick_labels=treatment_labels, patch_artist=True)
        for patch, t in zip(bp["boxes"], treatments):
            patch.set_facecolor(treatment_color_map.get(t, "#cccccc"))
            patch.set_alpha(0.45)

        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# --------------------------
# Plot 4: Overview (4x2) for any "month stacks" + treatment outlines
# --------------------------

def plot_stack_overview_4x2(
    stack_left: Sequence[xr.DataArray],
    stack_right: Sequence[xr.DataArray],
    left_title: str,
    right_title: str,
    layer_names: Sequence[str],
    treatments: Sequence[int],
    treatment_areas: gpd.GeoDataFrame,
    treatment_color_map: Dict[int, str],
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    tick_interval: int = 50,
    legend_unit: str = "",
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 16))
    titles = [left_title, right_title]
    stacks = [stack_left, stack_right]

    for col, stack in enumerate(stacks):
        for row, raster in enumerate(stack):
            ax = axes[row, col]

            x = raster["x"].values
            y = raster["y"].values
            im = ax.imshow(
                _as_2d(raster),
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="upper",
                interpolation="none",
            )

            # overlay treatment polygons
            for t in treatments:
                polys = treatment_areas[treatment_areas["treatment"] == t].to_crs(raster.rio.crs)
                patches = []
                for geom in polys.geometry:
                    if geom.geom_type == "Polygon":
                        patches.append(Polygon(list(geom.exterior.coords), closed=True, fill=False,
                                               edgecolor=treatment_color_map[t], linewidth=1.5))
                    elif geom.geom_type == "MultiPolygon":
                        for poly in geom:
                            patches.append(Polygon(list(poly.exterior.coords), closed=True, fill=False,
                                                   edgecolor=treatment_color_map[t], linewidth=1.5))
                ax.add_collection(PatchCollection(patches, match_original=True))

            ax.set_title(f"{titles[col]} - {layer_names[row]}")
            xticks = np.arange(np.floor(x.min()/tick_interval)*tick_interval,
                               np.ceil(x.max()/tick_interval)*tick_interval, tick_interval)
            yticks = np.arange(np.floor(y.min()/tick_interval)*tick_interval,
                               np.ceil(y.max()/tick_interval)*tick_interval, tick_interval)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis="both", which="both", length=5, labelsize=8)

    fig.subplots_adjust(wspace=0.05, hspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label=legend_unit)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        _savefig(fig, save_path, dpi=300)
    else:
        plt.show()


# --------------------------
# High-level: "make all plots" for a scene
# --------------------------

def make_scene_plots(
    *,
    out_dir: Path,
    opts: PlotOptions,
    ref_raster: xr.DataArray,
    crowns: gpd.GeoDataFrame,
    treatment_areas: gpd.GeoDataFrame,
    treatments: Sequence[int],
    treatment_labels: Sequence[str],
    treatment_color_map: Dict[int, str],
    # products to plot (provide what you have):
    # For monthly comparisons, pass june/aug means
    means: Dict[str, xr.DataArray],
    # optional stacks (per flightline) for overviews, etc.
    stacks: Optional[Dict[str, Sequence[xr.DataArray]]] = None,
    layer_names: Optional[Sequence[str]] = None,
):
    """
    out_dir: where rasters already got exported (e.g. E:/Pfynwald/Results/ForSIF/SFMNN/20240613)
    means: dictionary of mosaics you want plotted, e.g.
      means["SIF760_preprocessed_mean"]
      means["FQE760_NIRv_mean"]
      means["FQE760_FCVI_mean"]
      means["FQE760_saR2F_mean"]
    stacks: optional per-flightline stacks if you want overview maps
    """
    plots_dir = _ensure_dir(out_dir / "plots")

    # bbox for map plots
    transform, xmin_pix, xmax_pix, ymin_pix, ymax_pix = _bbox_pixels_for_polygons(ref_raster, treatment_areas)

    if opts.make_weighted_stats:
        # Example: plot weighted stats for the mean mosaics (FQE) if present
        weighted_targets = []
        for key in ("FQE760_NIRv_mean", "FQE760_FCVI_mean", "FQE760_saR2F_mean"):
            if key in means:
                weighted_targets.append((means[key], key))

        if weighted_targets:
            save_path = plots_dir / "weighted_stats_FQE_means.png" if opts.save else None
            plot_weighted_stats_full(
                datasets=weighted_targets,
                crowns=crowns,
                treatments=treatments,
                treatment_labels=treatment_labels,
                colors=[treatment_color_map[t] for t in treatments],
                unit_label="",
                save_path=save_path,
                share_x=True,
            )

    if opts.make_boxplots:
        # Example: boxplots of three mean mosaics (fill 4 slots, 1 empty)
        rasters = []
        labels = []
        for key, lbl in [
            ("FQE760_NIRv_mean", "FQE NIRv (mean)"),
            ("FQE760_FCVI_mean", "FQE FCVI (mean)"),
            ("FQE760_saR2F_mean", "FQE saR2F (mean)"),
        ]:
            if key in means:
                rasters.append(means[key])
                labels.append(lbl)
        # make 4 columns with an empty placeholder if needed
        while len(labels) < 4:
            labels.append("")

        if rasters:
            save_path = plots_dir / "boxplots_FQE_means.png" if opts.save else None
            plot_raster_boxplots(
                rasters=rasters,
                raster_labels=labels[:4],
                crowns=crowns,
                treatments=treatments,
                treatment_labels=treatment_labels,
                treatment_color_map=treatment_color_map,
                title="FQE mean mosaics by treatment",
                ylabel="FQE (unitless)",
                save_path=save_path,
            )

    if opts.make_monthly_comparisons:
        # Only works if you provide both June and Aug in means.
        # Recommended naming convention:
        #   FQE760_NIRv_mean_20240613 and FQE760_NIRv_mean_20240823 etc.
        # If you pass those in, we auto-plot comparisons.
        def _try_monthly(metric: str):
            k_june = f"{metric}_20240613"
            k_aug  = f"{metric}_20240823"
            if k_june in means and k_aug in means:
                save_path = plots_dir / f"monthly_{metric}.png" if opts.save else None
                plot_monthly_comparison(
                    means[k_june], means[k_aug],
                    treatments=treatments,
                    treatment_areas=treatment_areas,
                    treatment_color_map=treatment_color_map,
                    transform=transform,
                    xmin_pix=xmin_pix, xmax_pix=xmax_pix,
                    ymin_pix=ymin_pix, ymax_pix=ymax_pix,
                    titles=("June 2024", "Aug 2024", "Δ (Aug − Jun)"),
                    cmaps=(opts.cmap_main, opts.cmap_main, opts.cmap_main),
                    value_label=metric,
                    diff_label=f"Δ {metric}",
                    percentile_clip=opts.percentile_clip,
                    symmetric_diff=opts.symmetric_diff,
                    save_path=save_path,
                )

        for metric in ("FQE760_NIRv_mean", "FQE760_FCVI_mean", "FQE760_saR2F_mean"):
            _try_monthly(metric)

    if opts.make_overview_maps and stacks and layer_names:
        # Example overview for preprocessed SIF per-flightline (if you provide stacks)
        # Expect keys like: "SIF760_preprocessed_stack_20240613", "SIF760_preprocessed_stack_20240823"
        left_key = "SIF760_preprocessed_stack_20240613"
        right_key = "SIF760_preprocessed_stack_20240823"
        if left_key in stacks and right_key in stacks:
            save_path = plots_dir / "overview_SIF_preprocessed.png" if opts.save else None
            plot_stack_overview_4x2(
                stack_left=stacks[left_key],
                stack_right=stacks[right_key],
                left_title="June 2024",
                right_title="Aug 2024",
                layer_names=layer_names,
                treatments=treatments,
                treatment_areas=treatment_areas,
                treatment_color_map=treatment_color_map,
                cmap=opts.cmap_overview,
                tick_interval=opts.tick_interval,
                legend_unit="SIF760",
                save_path=save_path,
            )
