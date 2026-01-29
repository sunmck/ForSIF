from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
import xarray as xr

import rasterio
from rasterio import features

from matplotlib.ticker import ScalarFormatter

_MONTHS = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_date_label(yyyymmdd: str) -> str:
    s = str(yyyymmdd).strip()
    if len(s) != 8 or not s.isdigit():
        return s
    yyyy, mm, dd = s[0:4], s[4:6], s[6:8]
    return f"{int(dd):02d} {_MONTHS.get(mm, mm)} {yyyy}"


RETRIEVAL_ORDER_DEFAULT = ("iFLD", "SFM", "SFMNN")

RETRIEVAL_GREYS = {
    "iFLD": "#cfcfcf",
    "SFM": "#b5b5b5",
    "SFMNN": "#9b9b9b",
}

TREATMENT_LABELS_DEFAULT = ["control", "irrig.", "irrig. stopped"]
TREATMENT_COLORS_DEFAULT = {
    "control": "tab:orange",
    "irrig.": "tab:blue",
    "irrig. stopped": "tab:green",
}

POOLED_LABEL = "pooled"
POOLED_COLOR = "#7f7f7f"

SIF_YLABEL = r"SIF760 [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]"
FQE_YLABEL = r"FQE [nm$^{-1}$]"


def _normalize_sif_ylabel(ylabel: str) -> str:
    y = str(ylabel).strip()
    if y in {"SIF760", "SIF"}:
        return SIF_YLABEL
    return ylabel


def _normalize_fqe_ylabel_prefix(ylabel_prefix: str) -> str:
    y = str(ylabel_prefix).strip()
    if y in {"FQE760", "FQE"}:
        return "FQE"
    return y


def _boldify_axes(ax):
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")


def _set_sci_y(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)


def _date_header(ax, yyyymmdd: str):
    ax.text(
        0.5,
        1.04,
        format_date_label(yyyymmdd),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

def _dates_with_2023(dates: Sequence[str], include_2023: bool = True) -> List[str]:
    dates_use = [str(d) for d in dates]
    if include_2023 and "20230617" not in dates_use:
        dates_use = ["20230617"] + dates_use
    return dates_use


def _get_treatment_series(crowns: gpd.GeoDataFrame) -> pd.Series:
    if "treatment" not in crowns.columns:
        raise KeyError("Crowns GeoDataFrame has no 'treatment' column.")
    col = crowns.loc[:, "treatment"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.Series(col.values, index=crowns.index)


def _as_2d(da: xr.DataArray) -> np.ndarray:
    arr = da.values
    if da.ndim == 3 and "band" in da.dims and da.sizes.get("band", 1) == 1:
        arr = da.squeeze("band", drop=True).values
    elif da.ndim != 2:
        arr = da.squeeze().values

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    return np.asarray(arr)

def _sanitize_for_plotting(
    da: xr.DataArray,
    *,
    treat_zero_as_nodata: bool = False,
    nodata_values: Sequence[float] = (-999, -9999, -999.0, -9999.0),
) -> np.ndarray:
    arr = _as_2d(da).astype("float64", copy=False)

    for nv in nodata_values:
        arr[arr == nv] = np.nan
    if treat_zero_as_nodata:
        arr[arr == 0.0] = np.nan

    try:
        rn = da.rio.nodata
        if rn is not None:
            arr[arr == float(rn)] = np.nan
    except Exception:
        pass

    return arr

def get_pixels_weighted_in_crowns(
    raster: xr.DataArray,
    crowns: gpd.GeoDataFrame,
    treatment_value: Optional[int],
    *,
    supersample: int = 10,
    min_weight: float = 0.5,
    treat_zero_as_nodata: bool = False,
):
    arr = _sanitize_for_plotting(raster, treat_zero_as_nodata=treat_zero_as_nodata)
    transform = raster.rio.transform()
    out_shape = arr.shape

    crowns_use = crowns.reset_index(drop=True)

    if treatment_value is not None:
        tser = _get_treatment_series(crowns_use)
        crowns_use = crowns_use.loc[tser.values == treatment_value].reset_index(drop=True)

    polygons = [g for g in crowns_use.geometry if g is not None]
    if not polygons:
        return np.array([], dtype=float), np.array([], dtype=float)

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

    weights = ss_mask.reshape(out_shape[0], supersample, out_shape[1], supersample).mean(axis=(1, 3))
    pixmask = weights > 0

    values = arr[pixmask]
    w = weights[pixmask]

    valid = np.isfinite(values) & np.isfinite(w) & (w >= min_weight)
    return values[valid], w[valid]

def weighted_resample(values: np.ndarray, weights: np.ndarray, n: int, seed: int) -> np.ndarray:
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)

    if values.size == 0:
        return values

    w = np.clip(weights, 0, None)
    s = w.sum()
    rng = np.random.default_rng(seed)

    if not np.isfinite(s) or s <= 0:
        idx = rng.choice(np.arange(values.size), size=min(n, values.size), replace=True)
    else:
        p = w / s
        idx = rng.choice(np.arange(values.size), size=min(n, values.size), replace=True, p=p)

    return values[idx]

def extract_group_values(
    da: xr.DataArray,
    crowns: gpd.GeoDataFrame,
    treatment_value: Optional[int],
    *,
    supersample: int,
    min_weight: float,
    n_sample: int,
    seed: int,
    treat_zero_as_nodata: bool = False,
):
    vals, w = get_pixels_weighted_in_crowns(
        da,
        crowns,
        treatment_value,
        supersample=supersample,
        min_weight=min_weight,
        treat_zero_as_nodata=treat_zero_as_nodata,
    )
    return weighted_resample(vals, w, n=n_sample, seed=seed)

def _draw_violin_set(
    ax,
    arrays: List[np.ndarray],
    positions: np.ndarray,
    *,
    widths: float,
    facecolors: Optional[List[str]] = None,
    alpha: float = 0.5,
):
    nonempty = [(i, np.asarray(a)) for i, a in enumerate(arrays) if np.asarray(a).size > 0]
    if not nonempty:
        return None

    idxs = [i for i, _ in nonempty]
    data = [a for _, a in nonempty]
    positions = np.asarray(positions, dtype=float)
    pos = positions[np.asarray(idxs, dtype=int)]

    parts = ax.violinplot(
        data,
        positions=pos,
        widths=widths,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    bodies = parts.get("bodies", [])
    for k, b in enumerate(bodies):
        b.set_alpha(alpha)
        if facecolors is not None:
            b.set_facecolor(facecolors[idxs[k]])
            b.set_edgecolor("black")
            b.set_linewidth(0.6)

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.0)

    ymin = np.inf
    ymax = -np.inf
    for b in bodies:
        for p in b.get_paths():
            v = p.vertices
            if v.size:
                ymin = min(ymin, float(np.nanmin(v[:, 1])))
                ymax = max(ymax, float(np.nanmax(v[:, 1])))

    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        return (ymin, ymax)
    return None

def _retrieval_greys(retrieval_order: Sequence[str]) -> List[str]:
    return [RETRIEVAL_GREYS.get(r, "#c0c0c0") for r in retrieval_order]

def plot_sif_violin_retrieval_pooled(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    retrieval_order: Sequence[str] = RETRIEVAL_ORDER_DEFAULT,
    title_prefix: str = "SIF pooled",
    ylabel: str = SIF_YLABEL,
    fname: str = "SIF_violin_retrieval_pooled.png",
):
    ensure_dir(out_dir)

    d = df[(df["metric"] == "SIF760") & (df["treatment"].isna())]
    if d.empty:
        return

    ylabel = _normalize_sif_ylabel(ylabel)
    dates_use = _dates_with_2023(dates, include_2023=True)

    fig, axes = plt.subplots(1, len(dates_use), figsize=(6 * len(dates_use), 5), sharey=True)
    if len(dates_use) == 1:
        axes = [axes]

    for i, (ax, date) in enumerate(zip(axes, dates_use)):
        dd = d[d["date"].astype(str) == str(date)]
        arrays = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
        pos = np.arange(1, len(retrieval_order) + 1, dtype=float)

        _draw_violin_set(
            ax,
            arrays,
            pos,
            widths=0.8,
            facecolors=_retrieval_greys(retrieval_order),
            alpha=0.55,
        )

        ax.set_xticks(pos)
        ax.set_xticklabels(list(retrieval_order), rotation=0)
        _date_header(ax, str(date))

        ax.set_ylabel(ylabel if i == 0 else "")
        ax.grid(True, linestyle="--", alpha=0.3)
        _boldify_axes(ax)

    plt.tight_layout()
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_sif_violin_retrieval_by_treatment(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    retrieval_order: Sequence[str] = RETRIEVAL_ORDER_DEFAULT,
    treatment_labels: Sequence[str] = tuple(TREATMENT_LABELS_DEFAULT),
    treatment_colors: Dict[str, str] = TREATMENT_COLORS_DEFAULT,
    title_prefix: str = "SIF by treatment",
    ylabel: str = SIF_YLABEL,
    fname: str = "SIF_violin_retrieval_by_treatment.png",
):
    ensure_dir(out_dir)

    d = df[(df["metric"] == "SIF760") & (df["treatment"].notna())]
    if d.empty:
        return

    ylabel = _normalize_sif_ylabel(ylabel)
    dates_use = _dates_with_2023(dates, include_2023=True)

    fig, axes = plt.subplots(1, len(dates_use), figsize=(7 * len(dates_use), 5), sharey=True)
    if len(dates_use) == 1:
        axes = [axes]

    base = np.arange(1, len(retrieval_order) + 1, dtype=float)
    offsets = np.linspace(-0.30, 0.30, num=len(treatment_labels))
    width = 0.22

    for i, (ax, date) in enumerate(zip(axes, dates_use)):
        dd = d[d["date"].astype(str) == str(date)]

        for k, tlab in enumerate(treatment_labels):
            arrays = [
                dd[(dd["retrieval"] == r) & (dd["treatment"] == tlab)]["value"].to_numpy()
                for r in retrieval_order
            ]
            positions = base + offsets[k]
            colors = [treatment_colors.get(tlab, "#cccccc")] * len(retrieval_order)
            _draw_violin_set(ax, arrays, positions, widths=width, facecolors=colors, alpha=0.55)

        ax.set_xticks(base)
        ax.set_xticklabels(list(retrieval_order), rotation=0)
        _date_header(ax, str(date))

        ax.set_ylabel(ylabel if i == 0 else "")
        ax.grid(True, linestyle="--", alpha=0.3)

        handles = [plt.Line2D([0], [0], color=treatment_colors.get(t, "black"), lw=8) for t in treatment_labels]
        ax.legend(handles, list(treatment_labels), loc="upper right", frameon=False)

        _boldify_axes(ax)

    plt.tight_layout()
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fqe_violin_compact_retrieval_with_treatments(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    downscaling_tags: Sequence[str],
    retrieval_order: Sequence[str] = RETRIEVAL_ORDER_DEFAULT,
    treatment_labels: Sequence[str] = tuple(TREATMENT_LABELS_DEFAULT),
    treatment_colors: Dict[str, str] = TREATMENT_COLORS_DEFAULT,
    title_prefix: str = "FQE",
    ylabel_prefix: str = "FQE",
    fname: str = "FQE_violin_compact_retrieval_with_treatments.png",
):
    ensure_dir(out_dir)

    d = df[df["metric"] == "FQE760"]
    if d.empty:
        return

    ylabel_prefix = _normalize_fqe_ylabel_prefix(ylabel_prefix)
    dates_use = _dates_with_2023(dates, include_2023=True)

    nrows = len(downscaling_tags)
    ncols = len(dates_use)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.6 * ncols, 4.4 * nrows),
        sharey=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.expand_dims(axes, 1)

    base = np.arange(1, len(retrieval_order) + 1, dtype=float)

    hue_order = [POOLED_LABEL] + list(treatment_labels)
    hue_colors = {POOLED_LABEL: POOLED_COLOR, **treatment_colors}
    offsets = np.linspace(-0.33, 0.33, num=len(hue_order))
    width = 0.18

    global_ymin = np.inf
    global_ymax = -np.inf

    for i, tag in enumerate(downscaling_tags):
        for j, date in enumerate(dates_use):
            ax = axes[i, j]
            dd0 = d[(d["date"].astype(str) == str(date)) & (d["downscaling"] == tag)]

            for k, hue in enumerate(hue_order):
                dd = dd0[dd0["treatment"].isna()] if hue == POOLED_LABEL else dd0[dd0["treatment"] == hue]
                arrays = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
                pos = base + offsets[k]
                colors = [hue_colors.get(hue, "#cccccc")] * len(retrieval_order)

                ext = _draw_violin_set(ax, arrays, pos, widths=width, facecolors=colors, alpha=0.60)
                if ext is not None:
                    global_ymin = min(global_ymin, ext[0])
                    global_ymax = max(global_ymax, ext[1])

            ax.set_xticks(base)
            ax.set_xticklabels(list(retrieval_order), rotation=0)
            ax.grid(True, linestyle="--", alpha=0.25)

            if j == 0:
                ax.set_ylabel(f"{ylabel_prefix} ({tag}) [nm$^{{-1}}$]")

            _set_sci_y(ax)
            _boldify_axes(ax)

            if i == 0:
                _date_header(ax, str(date))

    if np.isfinite(global_ymin) and np.isfinite(global_ymax) and (global_ymax > global_ymin):
        pad = 0.06 * (global_ymax - global_ymin)
        y0, y1 = global_ymin - pad, global_ymax + pad
        for ax in axes.ravel():
            ax.set_ylim(y0, y1)

    ax_leg = axes[0, -1]
    handles = [plt.Line2D([0], [0], color=hue_colors[h], lw=8) for h in hue_order]
    ax_leg.legend(
        handles,
        hue_order,
        loc="upper right",
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.6,
        handletextpad=0.5,
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_fqe_violin_grid(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    downscaling_tags: Sequence[str],
    retrieval_order: Sequence[str] = RETRIEVAL_ORDER_DEFAULT,
    treatment_labels: Sequence[str] = tuple(TREATMENT_LABELS_DEFAULT),
    treatment_colors: Dict[str, str] = TREATMENT_COLORS_DEFAULT,
    title_prefix: str = "FQE",
    ylabel_prefix: str = "FQE",
    fname: str = "FQE_violin_grid.png",
):
    ensure_dir(out_dir)

    d = df[df["metric"] == "FQE760"]
    if d.empty:
        return

    ylabel_prefix = _normalize_fqe_ylabel_prefix(ylabel_prefix)
    dates_use = _dates_with_2023(dates, include_2023=True)

    ncols = len(dates_use) * (1 + len(treatment_labels))
    nrows = len(downscaling_tags)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.6 * ncols, 3.6 * nrows),
        sharey="row",
    )
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    base = np.arange(1, len(retrieval_order) + 1, dtype=float)

    for i, tag in enumerate(downscaling_tags):
        for j, date in enumerate(dates_use):
            col0 = j * (1 + len(treatment_labels))

            ax = axes[i, col0]
            dd = d[
                (d["date"].astype(str) == str(date)) &
                (d["downscaling"] == tag) &
                (d["treatment"].isna())
            ]
            arrays = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
            _draw_violin_set(ax, arrays, base, widths=0.8, facecolors=_retrieval_greys(retrieval_order), alpha=0.55)

            ax.set_xticks(base)
            ax.set_xticklabels(list(retrieval_order), rotation=90)
            ax.grid(True, linestyle="--", alpha=0.25)

            if col0 == 0:
                ax.set_ylabel(f"{ylabel_prefix} ({tag}) [nm$^{{-1}}$]")
            _set_sci_y(ax)
            _boldify_axes(ax)

            for k, tlab in enumerate(treatment_labels):
                ax = axes[i, col0 + 1 + k]
                dd = d[
                    (d["date"].astype(str) == str(date)) &
                    (d["downscaling"] == tag) &
                    (d["treatment"] == tlab)
                ]
                arrays = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
                colors = [treatment_colors.get(tlab, "#cccccc")] * len(retrieval_order)
                _draw_violin_set(ax, arrays, base, widths=0.8, facecolors=colors, alpha=0.55)

                ax.set_xticks(base)
                ax.set_xticklabels(list(retrieval_order), rotation=90)
                ax.grid(True, linestyle="--", alpha=0.25)

                _set_sci_y(ax)
                _boldify_axes(ax)

            if i == 0:
                _date_header(axes[i, col0], str(date))

    handles = [
        plt.Line2D([0], [0], color=POOLED_COLOR, lw=8),
        *[plt.Line2D([0], [0], color=treatment_colors.get(t, "black"), lw=8) for t in treatment_labels],
    ]
    labels = [POOLED_LABEL] + list(treatment_labels)
    axes[0, -1].legend(handles, labels, loc="upper right", frameon=False, fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_fqe_violin_compact_retrieval_pooled(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    downscaling_tags: Sequence[str],
    retrieval_order: Sequence[str] = RETRIEVAL_ORDER_DEFAULT,
    title_prefix: str = "FQE pooled",
    ylabel_prefix: str = "FQE",
    fname: str = "FQE_violin_compact_pooled.png",
):
    ensure_dir(out_dir)

    d = df[(df["metric"] == "FQE760") & (df["treatment"].isna())]
    if d.empty:
        return

    ylabel_prefix = _normalize_fqe_ylabel_prefix(ylabel_prefix)
    dates_use = _dates_with_2023(dates, include_2023=True)

    nrows = len(downscaling_tags)
    ncols = len(dates_use)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.6 * ncols, 4.4 * nrows),
        sharey=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.expand_dims(axes, 1)

    xbase = np.arange(1, len(retrieval_order) + 1, dtype=float)

    global_ymin = np.inf
    global_ymax = -np.inf

    for i, tag in enumerate(downscaling_tags):
        for j, date in enumerate(dates_use):
            ax = axes[i, j]
            dd = d[(d["date"].astype(str) == str(date)) & (d["downscaling"] == tag)]

            arrays = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
            ext = _draw_violin_set(
                ax,
                arrays,
                xbase,
                widths=0.75,
                facecolors=_retrieval_greys(retrieval_order),
                alpha=0.60,
            )
            if ext is not None:
                global_ymin = min(global_ymin, ext[0])
                global_ymax = max(global_ymax, ext[1])

            ax.set_xticks(xbase)
            ax.set_xticklabels(list(retrieval_order), rotation=0)
            ax.grid(True, linestyle="--", alpha=0.25)

            if j == 0:
                ax.set_ylabel(f"{ylabel_prefix} ({tag}) [nm$^{{-1}}$]")

            _set_sci_y(ax)
            _boldify_axes(ax)

            if i == 0:
                _date_header(ax, str(date))

    if np.isfinite(global_ymin) and np.isfinite(global_ymax) and (global_ymax > global_ymin):
        pad = 0.06 * (global_ymax - global_ymin)
        y0, y1 = global_ymin - pad, global_ymax + pad
        for ax in axes.ravel():
            ax.set_ylim(y0, y1)

    plt.tight_layout()
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
