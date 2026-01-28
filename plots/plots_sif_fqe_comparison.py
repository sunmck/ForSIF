from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
import xarray as xr

from plots.plots_downscaling import get_pixels_by_treatment_weighted


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def weighted_resample(values: np.ndarray, weights: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Convert weighted pixels into an unweighted sample by resampling ~ weights.
    This makes violin plots visually reflect weights.
    """
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
):
    """
    Returns a (resampled) 1D array of values extracted from crown-masked pixels.

    - treatment_value = 1/2/3 -> crowns filtered by that treatment
    - treatment_value = None  -> pooled across all crowns (still crown-masked)
    """
    if treatment_value is None:
        # pooled: temporarily set all crowns to the same treatment id
        tmp = crowns.copy()
        tmp["__tmp_treat__"] = 1
        px, w, _ = get_pixels_by_treatment_weighted(
            da,
            tmp.rename(columns={"__tmp_treat__": "treatment"}),
            treatment_value=1,
            supersample=supersample,
            min_weight=min_weight,
        )
    else:
        px, w, _ = get_pixels_by_treatment_weighted(
            da,
            crowns,
            treatment_value=treatment_value,
            supersample=supersample,
            min_weight=min_weight,
        )

    return weighted_resample(px, w, n=n_sample, seed=seed)


def violin_by_groups(
    ax,
    data_by_group: List[np.ndarray],
    group_labels: List[str],
    *,
    title: str,
    ylabel: str,
):
    # matplotlib violinplot expects list-like; ensure not empty
    safe = [d if (isinstance(d, np.ndarray) and d.size) else np.array([np.nan]) for d in data_by_group]

    parts = ax.violinplot(
        safe,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for b in parts["bodies"]:
        b.set_alpha(0.5)

    ax.set_xticks(np.arange(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)


def violin_hue_offsets(
    ax,
    data: Dict[str, Dict[str, np.ndarray]],
    x_order: List[str],
    hue_order: List[str],
    *,
    title: str,
    ylabel: str,
    width: float = 0.25,
):
    """
    data[x][hue] -> array
    Draws multiple violins per x position with offsets (hue).
    """
    base = np.arange(1, len(x_order) + 1, dtype=float)
    offsets = np.linspace(-width, width, num=max(1, len(hue_order)))

    for k, hue in enumerate(hue_order):
        arrays = []
        positions = base + offsets[k] if len(hue_order) > 1 else base

        for x in x_order:
            arr = data.get(x, {}).get(hue, np.array([]))
            arrays.append(arr if arr.size else np.array([np.nan]))

        parts = ax.violinplot(
            arrays,
            positions=positions,
            widths=width * 0.9,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for b in parts["bodies"]:
            b.set_alpha(0.45)

    ax.set_xticks(base)
    ax.set_xticklabels(x_order, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles = [plt.Line2D([0], [0], color="black", lw=2) for _ in hue_order]
    ax.legend(handles, hue_order, loc="upper right", frameon=False)


def plot_retrieval_only_pooled(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    downscaling_tags: Sequence[str],
):
    """
    Retrieval compare, pooled across treatments (treatment is NaN in df rows).
    - SIF: (date facet) violin by retrieval
    - FQE: (downscaling x date facet) violin by retrieval
    """
    ensure_dir(out_dir)

    # --- SIF retrieval compare (pooled)
    d = df[(df["metric"] == "SIF760") & (df["treatment"].isna())]
    if not d.empty:
        retrieval_order = sorted(d["retrieval"].unique())

        fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 5), sharey=True)
        if len(dates) == 1:
            axes = [axes]

        for ax, date in zip(axes, dates):
            dd = d[d["date"] == date]
            data_by_group = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
            violin_by_groups(ax, data_by_group, retrieval_order, title=f"SIF760 pooled ({date})", ylabel="SIF760")

        plt.tight_layout()
        fig.savefig(out_dir / "retrieval_only_SIF760_pooled.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- FQE retrieval compare (pooled)
    d = df[(df["metric"] == "FQE760") & (df["treatment"].isna())]
    if not d.empty:
        retrieval_order = sorted(d["retrieval"].unique())
        fig, axes = plt.subplots(
            len(downscaling_tags),
            len(dates),
            figsize=(6 * len(dates), 4 * len(downscaling_tags)),
            sharey="row",
        )
        if len(downscaling_tags) == 1 and len(dates) == 1:
            axes = np.array([[axes]])

        for i, tag in enumerate(downscaling_tags):
            for j, date in enumerate(dates):
                ax = axes[i, j]
                dd = d[(d["date"] == date) & (d["downscaling"] == tag)]
                data_by_group = [dd[dd["retrieval"] == r]["value"].to_numpy() for r in retrieval_order]
                violin_by_groups(ax, data_by_group, retrieval_order, title=f"FQE760 {tag} pooled ({date})", ylabel="FQE760")

        plt.tight_layout()
        fig.savefig(out_dir / "retrieval_only_FQE760_by_downscaling_pooled.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_treatments_only_pooled_methods(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    downscaling_tags: Sequence[str],
    treatment_labels: Sequence[str],
):
    """
    Treatment compare, pooled across retrieval (and for FQE per downscaling tag).
    """
    ensure_dir(out_dir)

    # --- SIF
    d = df[(df["metric"] == "SIF760") & (df["treatment"].notna())]
    if not d.empty:
        fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 5), sharey=True)
        if len(dates) == 1:
            axes = [axes]

        for ax, date in zip(axes, dates):
            dd = d[d["date"] == date]
            data_by_group = [dd[dd["treatment"] == t]["value"].to_numpy() for t in treatment_labels]
            violin_by_groups(ax, data_by_group, list(treatment_labels), title=f"SIF760 treatments (pooled methods) {date}", ylabel="SIF760")

        plt.tight_layout()
        fig.savefig(out_dir / "treatments_only_SIF760_pooled_methods.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- FQE per downscaling tag
    d = df[(df["metric"] == "FQE760") & (df["treatment"].notna())]
    if not d.empty:
        fig, axes = plt.subplots(
            len(downscaling_tags),
            len(dates),
            figsize=(6 * len(dates), 4 * len(downscaling_tags)),
            sharey="row",
        )
        if len(downscaling_tags) == 1 and len(dates) == 1:
            axes = np.array([[axes]])

        for i, tag in enumerate(downscaling_tags):
            for j, date in enumerate(dates):
                ax = axes[i, j]
                dd = d[(d["date"] == date) & (d["downscaling"] == tag)]
                data_by_group = [dd[dd["treatment"] == t]["value"].to_numpy() for t in treatment_labels]
                violin_by_groups(ax, data_by_group, list(treatment_labels), title=f"FQE760 {tag} treatments (pooled methods) {date}", ylabel="FQE760")

        plt.tight_layout()
        fig.savefig(out_dir / "treatments_only_FQE760_pooled_methods.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_treatment_x_retrieval(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dates: Sequence[str],
    downscaling_tags: Sequence[str],
    treatment_labels: Sequence[str],
):
    """
    x = treatment, hue = retrieval, per date.
    For FQE, also facet by downscaling tag.
    """
    ensure_dir(out_dir)

    # --- SIF
    d = df[(df["metric"] == "SIF760") & (df["treatment"].notna())]
    if not d.empty:
        retrieval_order = sorted(d["retrieval"].unique())
        fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 5), sharey=True)
        if len(dates) == 1:
            axes = [axes]

        for ax, date in zip(axes, dates):
            dd = d[d["date"] == date]
            packed = {
                t: {
                    r: dd[(dd["treatment"] == t) & (dd["retrieval"] == r)]["value"].to_numpy()
                    for r in retrieval_order
                }
                for t in treatment_labels
            }
            violin_hue_offsets(
                ax,
                packed,
                x_order=list(treatment_labels),
                hue_order=retrieval_order,
                title=f"SIF760: treatment × retrieval ({date})",
                ylabel="SIF760",
            )

        plt.tight_layout()
        fig.savefig(out_dir / "SIF760_treatment_x_retrieval.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- FQE per downscaling tag
    d = df[(df["metric"] == "FQE760") & (df["treatment"].notna())]
    if not d.empty:
        retrieval_order = sorted(d["retrieval"].unique())
        fig, axes = plt.subplots(
            len(downscaling_tags),
            len(dates),
            figsize=(6 * len(dates), 4 * len(downscaling_tags)),
            sharey="row",
        )
        if len(downscaling_tags) == 1 and len(dates) == 1:
            axes = np.array([[axes]])

        for i, tag in enumerate(downscaling_tags):
            for j, date in enumerate(dates):
                ax = axes[i, j]
                dd = d[(d["date"] == date) & (d["downscaling"] == tag)]
                packed = {
                    t: {
                        r: dd[(dd["treatment"] == t) & (dd["retrieval"] == r)]["value"].to_numpy()
                        for r in retrieval_order
                    }
                    for t in treatment_labels
                }
                violin_hue_offsets(
                    ax,
                    packed,
                    x_order=list(treatment_labels),
                    hue_order=retrieval_order,
                    title=f"FQE760 {tag}: treatment × retrieval ({date})",
                    ylabel="FQE760",
                )

        plt.tight_layout()
        fig.savefig(out_dir / "FQE760_treatment_x_retrieval_by_downscaling.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
