from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ---------- Plot settings ----------

TREATMENT_ORDER = ["control", "irrig.", "irrig. stopped"]
TREATMENT_COLORS = {
    "control": "tab:orange",
    "irrig.": "tab:blue",
    "irrig. stopped": "tab:green",
}
TREATMENT_LABELS = {
    "control": "control",
    "irrig.": "irrig.",
    "irrig. stopped": "irrig. stopped",
}

SIF_BOX_COLOR = "0.68"
METHOD_FACE_COLOR = "0.84"
METHOD_HATCHES = {
    "NIRv": "",
    "FCVI": "///",
    "saR2F": "xxx",
}

_MONTHS = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}

DATE_GAP = 0.55
MEAN_GAP = 0.35
MEAN_BAND_COLOR = "0.955"

# Use these for both SIF figures so they stay directly comparable.
SIF_PANEL_YLIMS = {
    "iFLD": (-0.60, 1.12),
    "SFM": (-0.60, 1.12),
    "SFMNN": (-0.05, 0.45),
}

# Optional test setting for the FQE figure.
FQE_PANEL_YLIMS = {
    "iFLD": None,
    "SFM": None,
    "SFMNN": (0.60e-5, 1.50e-5),
}


# ---------- Helper functions ----------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_publication_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 12.0,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_date(date):
    date = str(date)
    return f"{int(date[6:8])} {_MONTHS[date[4:6]]} {date[:4]}"


def format_time(time):
    time = str(time)
    return f"{time[:2]}:{time[2:]}"


def _with_sample_type(df: pd.DataFrame):
    df = df.copy()
    if "sample_type" not in df.columns:
        df["sample_type"] = np.where(
            df["flight_id"].astype(str).eq("MEAN"),
            "date_mean",
            "flight",
        )
    return df


def _build_sample_axis(df: pd.DataFrame):
    samples = (
        _with_sample_type(df)[
            ["date", "sample_type", "flight_id", "time", "time_min", "direction"]
        ]
        .drop_duplicates()
        .copy()
    )
    samples["date"] = samples["date"].astype(str)

    rows = []
    groups = []
    cursor = 0.0

    for date in sorted(samples["date"].unique()):
        date_samples = samples[samples["date"] == date]
        ff = (
            date_samples[date_samples["sample_type"] == "flight"]
            .sort_values("time_min")
            .reset_index(drop=True)
        )

        mean_rows = (
            date_samples[date_samples["sample_type"] == "date_mean"]
            .drop_duplicates(subset=["date", "sample_type", "flight_id"])
            .reset_index(drop=True)
        )

        date_xs = []
        flight_xs = cursor + np.arange(len(ff), dtype=float)

        for x, row in zip(flight_xs, ff.itertuples()):
            rows.append({
                "date": date,
                "sample_type": "flight",
                "flight_id": row.flight_id,
                "time": row.time,
                "time_min": row.time_min,
                "direction": row.direction,
                "x": x,
            })
            date_xs.append(x)

        if len(mean_rows):
            if len(flight_xs):
                mean_x = flight_xs[-1] + 1.0 + MEAN_GAP
            else:
                mean_x = cursor

            mean_row = mean_rows.iloc[0]
            rows.append({
                "date": date,
                "sample_type": "date_mean",
                "flight_id": mean_row["flight_id"],
                "time": "MEAN",
                "time_min": np.inf,
                "direction": "",
                "x": mean_x,
            })
            date_xs.append(mean_x)

        if not date_xs:
            continue

        date_xs = np.asarray(date_xs, dtype=float)

        groups.append({
            "date": date,
            "left": date_xs[0] - 0.50,
            "right": date_xs[-1] + 0.50,
            "center": float(np.mean(date_xs)),
        })

        cursor = date_xs[-1] + 1.0 + DATE_GAP

    return pd.DataFrame(rows), groups


def _common_ylim(values, pad=0.035):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    span = ymax - ymin

    if span <= 0:
        span = max(abs(ymax), 1.0)

    return ymin - pad * span, ymax + pad * span


def _resolve_panel_ylims(
    retrieval_order: Sequence[str],
    default_ylim,
    retrieval_ylims: Mapping[str, tuple[float, float] | None] | None,
):
    panel_ylims = {}

    for retrieval in retrieval_order:
        ylim = default_ylim
        if retrieval_ylims is not None and retrieval in retrieval_ylims:
            if retrieval_ylims[retrieval] is not None:
                ylim = retrieval_ylims[retrieval]
        panel_ylims[retrieval] = ylim

    return panel_ylims


def _draw_boxplot(
    ax,
    values,
    position,
    *,
    facecolor,
    hatch="",
    width=0.62,
    linewidth=0.95,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return

    bp = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        capwidths=width * 0.72,
        manage_ticks=False,
        boxprops=dict(
            facecolor=facecolor,
            edgecolor="black",
            linewidth=linewidth,
        ),
        medianprops=dict(
            color="black",
            linewidth=1.30,
        ),
        whiskerprops=dict(
            color="black",
            linewidth=0.90,
        ),
        capprops=dict(
            color="black",
            linewidth=0.90,
        ),
    )
    bp["boxes"][0].set_hatch(hatch)


def _shade_mean_positions(ax, samples):
    for row in samples.itertuples():
        if row.sample_type == "date_mean":
            ax.axvspan(
                row.x - 0.48,
                row.x + 0.48,
                color=MEAN_BAND_COLOR,
                zorder=0,
            )



def _style_axis(ax, ylim):
    ax.set_ylim(*ylim)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.65,
        alpha=0.25,
        zorder=0,
    )

    if ylim[0] < 0 < ylim[1]:
        ax.axhline(0, color="0.40", linewidth=0.8, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", width=1.0, length=4, pad=5)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def _add_date_separators(ax, groups):
    for left, right in zip(groups[:-1], groups[1:]):
        boundary = 0.5 * (left["right"] + right["left"])
        ax.axvline(
            boundary,
            color="0.72",
            linestyle=":",
            linewidth=0.95,
            zorder=0,
        )


def _format_bottom_axis(ax, samples, groups):
    ax.set_xticks(samples["x"])
    ax.set_xticklabels(
        [
            "MEAN" if sample_type == "date_mean" else format_time(time)
            for time, sample_type in zip(samples["time"], samples["sample_type"])
        ],
        fontsize=11.0,
        fontweight="bold",
    )
    ax.tick_params(axis="x", pad=7)

    trans = ax.get_xaxis_transform()

    for row in samples.itertuples():
        if row.sample_type == "date_mean":
            lower_label = "mosaic"
        else:
            lower_label = "E  →" if row.direction == "E" else "←  W"
        ax.text(
            row.x,
            -0.165,
            lower_label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="0.38",
        )

    for group in groups:
        ax.text(
            group["center"],
            -0.285,
            format_date(group["date"]),
            transform=trans,
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="bold",
        )

    if len(samples):
        ax.set_xlim(samples["x"].min() - 0.60, samples["x"].max() + 0.60)


def _save_figure(fig, out_dir: Path, fname: str):
    out_dir = ensure_dir(out_dir)
    png_path = out_dir / fname
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# ---------- SIF retrieval comparison ----------

def plot_sif_retrieval_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    retrieval_order: Sequence[str],
    by_treatment=False,
    retrieval_ylims: Mapping[str, tuple[float, float] | None] | None = SIF_PANEL_YLIMS,
    fname="SIF_retrieval_comparison.png",
):
    apply_publication_style()

    data = _with_sample_type(df[df["metric"] == "SIF760"])
    if data.empty:
        return

    samples, groups = _build_sample_axis(data)
    default_ylim = _common_ylim(data["median"].to_numpy())
    panel_ylims = _resolve_panel_ylims(retrieval_order, default_ylim, retrieval_ylims)

    fig_w = max(12.8, 3.4 + 0.68 * len(samples))
    fig, axes = plt.subplots(
        len(retrieval_order),
        1,
        figsize=(fig_w, 13.4),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, retrieval in zip(axes, retrieval_order):
        dd = data[data["retrieval"] == retrieval]
        _shade_mean_positions(ax, samples)

        for sample in samples.itertuples():
            ff = dd[
                (dd["date"].astype(str) == sample.date)
                & (dd["flight_id"] == sample.flight_id)
                & (dd["sample_type"] == sample.sample_type)
            ]

            if ff.empty:
                continue

            if by_treatment:
                offsets = (-0.30, 0.0, 0.30)

                for offset, treatment in zip(offsets, TREATMENT_ORDER):
                    values = ff.loc[
                        ff["treatment"] == treatment,
                        "median",
                    ].to_numpy()

                    _draw_boxplot(
                        ax,
                        values,
                        sample.x + offset,
                        facecolor=TREATMENT_COLORS[treatment],
                        width=0.24,
                        linewidth=1.45 if sample.sample_type == "date_mean" else 0.95,
                    )
            else:
                _draw_boxplot(
                    ax,
                    ff["median"].to_numpy(),
                    sample.x,
                    facecolor=SIF_BOX_COLOR,
                    width=0.66,
                    linewidth=1.55 if sample.sample_type == "date_mean" else 0.95,
                )

        _style_axis(ax, panel_ylims[retrieval])
        _add_date_separators(ax, groups)

        ax.text(
            0.0,
            1.085,
            retrieval,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15.5,
            fontweight="bold",
        )

        if retrieval_ylims is not None and retrieval == "SFMNN":
            ax.text(
                0.99,
                1.015,
                "different y-axis",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="0.35",
            )

    fig.supylabel(
        r"SIF$_{760}$ [mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
        x=0.018,
        fontsize=13.5,
        fontweight="bold",
    )

    _format_bottom_axis(axes[-1], samples, groups)

    if by_treatment:
        handles = [
            Patch(
                facecolor=TREATMENT_COLORS[t],
                edgecolor="black",
                label=TREATMENT_LABELS[t],
            )
            for t in TREATMENT_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            prop={"size": 12.0, "weight": "bold"},
            bbox_to_anchor=(0.5, 0.985),
        )
        top = 0.91
    else:
        top = 0.97

    fig.subplots_adjust(
        left=0.090,
        right=0.995,
        top=top,
        bottom=0.205,
        hspace=0.32,
    )
    fig.supxlabel(
        "flight time / date mean mosaic",
        x=0.54,
        y=0.100,
        fontsize=14.0,
        fontweight="bold",
    )
    _save_figure(fig, out_dir, fname)


# ---------- FQE downscaling-index comparison ----------

def plot_fqe_method_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    retrieval_order: Sequence[str],
    method_order: Sequence[str],
    retrieval_ylims: Mapping[str, tuple[float, float] | None] | None = FQE_PANEL_YLIMS,
    fname="FQE_downscaling_method_comparison.png",
):
    apply_publication_style()

    data = _with_sample_type(df[df["metric"] == "FQE760"])
    if data.empty:
        return

    axis_source = pd.concat(
        [
            _with_sample_type(df[df["metric"] == "SIF760"]),
            data,
        ],
        ignore_index=True,
    )

    samples, groups = _build_sample_axis(axis_source)

    default_ylim = _common_ylim(data["median"].to_numpy())
    panel_ylims = _resolve_panel_ylims(retrieval_order, default_ylim, retrieval_ylims)

    fig_w = max(12.8, 3.4 + 0.68 * len(samples))
    fig, axes = plt.subplots(
        len(retrieval_order),
        1,
        figsize=(fig_w, 13.4),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    offsets = np.linspace(-0.24, 0.24, len(method_order))

    for ax, retrieval in zip(axes, retrieval_order):
        dd = data[data["retrieval"] == retrieval]
        _shade_mean_positions(ax, samples)

        for sample in samples.itertuples():
            ff = dd[
                (dd["date"].astype(str) == sample.date)
                & (dd["flight_id"] == sample.flight_id)
                & (dd["sample_type"] == sample.sample_type)
            ]

            if ff.empty:
                continue

            for offset, method in zip(offsets, method_order):
                values = ff.loc[
                    ff["method"] == method,
                    "median",
                ].to_numpy()

                _draw_boxplot(
                    ax,
                    values,
                    sample.x + offset,
                    facecolor=METHOD_FACE_COLOR,
                    hatch=METHOD_HATCHES[method],
                    width=0.22,
                    linewidth=1.45 if sample.sample_type == "date_mean" else 0.95,
                )

        _style_axis(ax, panel_ylims[retrieval])
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(-2, 2),
            useMathText=False,
        )
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(11)
        offset_text.set_fontweight("bold")
        offset_text.set_x(-0.03)
        offset_text.set_y(1.010)
        offset_text.set_ha("left")
        _add_date_separators(ax, groups)

        ax.text(
            0.0,
            1.085,
            retrieval,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15.5,
            fontweight="bold",
        )

        if retrieval_ylims is not None and retrieval == "SFMNN" and retrieval_ylims.get("SFMNN") is not None:
            ax.text(
                0.99,
                1.015,
                "different y-axis",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="0.35",
            )

    fig.supylabel(
        r"FQE [nm$^{-1}$]",
        x=0.018,
        fontsize=13.5,
        fontweight="bold",
    )

    _format_bottom_axis(axes[-1], samples, groups)

    handles = [
        Patch(
            facecolor=METHOD_FACE_COLOR,
            edgecolor="black",
            hatch=METHOD_HATCHES[m],
            label=m,
        )
        for m in method_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(method_order),
        frameon=False,
        prop={"size": 12.0, "weight": "bold"},
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.subplots_adjust(
        left=0.090,
        right=0.995,
        top=0.91,
        bottom=0.205,
        hspace=0.32,
    )
    fig.supxlabel(
        "flight time / date mean mosaic",
        x=0.54,
        y=0.100,
        fontsize=14.0,
        fontweight="bold",
    )
    _save_figure(fig, out_dir, fname)


# ---------- FQE retrieval comparison by treatment ----------

def plot_fqe_retrieval_comparison_by_treatment(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    retrieval_order: Sequence[str],
    method="saR2F",
    retrieval_ylims: Mapping[str, tuple[float, float] | None] | None = FQE_PANEL_YLIMS,
    fname="FQE_saR2F_retrieval_comparison_by_treatment.png",
):
    """
    Compare one FQE downscaling method between retrievals,
    split by treatment.

    Includes both:
        - individual flight lines
        - date-mean mosaics
    """

    apply_publication_style()

    # Select FQE and one downscaling method only
    data = df[
        (df["metric"] == "FQE760")
        & (df["method"] == method)
    ].copy()

    data = _with_sample_type(data)

    if data.empty:
        print(f"No FQE data found for method '{method}'.")
        return

    # Build the x-axis from all available SIF samples plus the selected FQE data.
    # This keeps dates (e.g. 2026) visible even while FQE is still missing,
    # matching plot_fqe_method_comparison; missing FQE positions stay empty.
    axis_source = pd.concat(
        [
            _with_sample_type(df[df["metric"] == "SIF760"]),
            data,
        ],
        ignore_index=True,
    )

    samples, groups = _build_sample_axis(axis_source)

    default_ylim = _common_ylim(
        data["median"].to_numpy()
    )

    panel_ylims = _resolve_panel_ylims(
        retrieval_order,
        default_ylim,
        retrieval_ylims,
    )

    fig_w = max(
        12.8,
        3.4 + 0.68 * len(samples),
    )

    fig, axes = plt.subplots(
        len(retrieval_order),
        1,
        figsize=(fig_w, 13.4),
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    treatment_offsets = (-0.30, 0.0, 0.30)

    # --------------------------------------------------
    # Retrieval panels
    # --------------------------------------------------

    for ax, retrieval in zip(
        axes,
        retrieval_order,
    ):

        dd = data[
            data["retrieval"] == retrieval
        ]

        # Highlight mean-mosaic positions
        _shade_mean_positions(
            ax,
            samples,
        )

        # ----------------------------------------------
        # Samples
        # ----------------------------------------------

        for sample in samples.itertuples():

            ff = dd[
                (dd["date"].astype(str) == sample.date)
                & (dd["flight_id"] == sample.flight_id)
                & (dd["sample_type"] == sample.sample_type)
            ]

            if ff.empty:
                continue

            # ------------------------------------------
            # Treatments
            # ------------------------------------------

            for offset, treatment in zip(
                treatment_offsets,
                TREATMENT_ORDER,
            ):

                values = ff.loc[
                    ff["treatment"] == treatment,
                    "median",
                ].to_numpy()

                _draw_boxplot(
                    ax,
                    values,
                    sample.x + offset,
                    facecolor=TREATMENT_COLORS[treatment],
                    width=0.24,
                    linewidth=(
                        1.45
                        if sample.sample_type == "date_mean"
                        else 0.95
                    ),
                )

        # ----------------------------------------------
        # Panel formatting
        # ----------------------------------------------

        _style_axis(
            ax,
            panel_ylims[retrieval],
        )

        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(-2, 2),
            useMathText=False,
        )

        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(11)
        offset_text.set_fontweight("bold")
        offset_text.set_x(-0.03)
        offset_text.set_y(1.010)
        offset_text.set_ha("left")

        _add_date_separators(
            ax,
            groups,
        )

        # Retrieval label
        ax.text(
            0.0,
            1.085,
            retrieval,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15.5,
            fontweight="bold",
        )

        # Mark special SFMNN y-axis
        if (
            retrieval_ylims is not None
            and retrieval == "SFMNN"
            and retrieval_ylims.get("SFMNN") is not None
        ):
            ax.text(
                0.99,
                1.015,
                "different y-axis",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="0.35",
            )

    # --------------------------------------------------
    # Shared y-axis label
    # --------------------------------------------------

    fig.supylabel(
        r"FQE [nm$^{-1}$]",
        x=0.018,
        fontsize=13.5,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Shared x-axis
    # --------------------------------------------------

    _format_bottom_axis(
        axes[-1],
        samples,
        groups,
    )

    # --------------------------------------------------
    # Treatment legend
    # --------------------------------------------------

    handles = [
        Patch(
            facecolor=TREATMENT_COLORS[treatment],
            edgecolor="black",
            label=TREATMENT_LABELS[treatment],
        )
        for treatment in TREATMENT_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        prop={
            "size": 12.0,
            "weight": "bold",
        },
        bbox_to_anchor=(0.5, 0.985),
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    fig.subplots_adjust(
        left=0.090,
        right=0.995,
        top=0.91,
        bottom=0.205,
        hspace=0.32,
    )

    fig.supxlabel(
        "flight time",
        x=0.54,
        y=0.100,
        fontsize=14.0,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    _save_figure(
        fig,
        out_dir,
        fname,
    )