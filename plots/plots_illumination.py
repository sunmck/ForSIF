from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from downscaling.illumination.sunlit_fraction import EndmemberLibrary


def plot_endmember_spectra(library: EndmemberLibrary, out_path: Path, title=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wl = np.asarray(library.wavelengths, dtype=float)
    keep = (wl >= 400) & (wl <= 1200)

    fig, ax = plt.subplots(figsize=(10, 6))
    curves = [
        (
            "Vegetation, sunlit",
            library.sunlit,
            library.sunlit_p10,
            library.sunlit_p90,
        ),
        (
            "Vegetation, shaded",
            library.shaded,
            library.shaded_p10,
            library.shaded_p90,
        ),
    ]

    for label, median, p10, p90 in curves:
        line, = ax.plot(wl[keep], median[keep], linewidth=2, label=label)
        ax.fill_between(
            wl[keep],
            p10[keep],
            p90[keep],
            alpha=0.15,
            color=line.get_color(),
        )

    ax.axvline(
        library.red_edge_wavelength_nm,
        linestyle="--",
        linewidth=1,
        label=f"RE = {library.red_edge_wavelength_nm:.1f} nm",
    )
    ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance (-)", title=title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_treatment_spectra(stats, wavelengths, out_path: Path, title=None):
    """Sunlit/shaded candidate-pool spectra for each treatment."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wl = np.asarray(wavelengths, dtype=float)
    keep = (wl >= 400) & (wl <= 1100)

    fig, ax = plt.subplots(figsize=(10, 6))
    for treatment in sorted(stats):
        states = stats[treatment]
        if "sunlit" not in states or "shaded" not in states:
            continue

        sun = states["sunlit"]
        shade = states["shaded"]
        line, = ax.plot(
            wl[keep],
            sun["median"][keep],
            linewidth=2,
            label=f"{treatment} sunlit",
        )
        color = line.get_color()
        ax.fill_between(
            wl[keep],
            sun["p10"][keep],
            sun["p90"][keep],
            alpha=0.10,
            color=color,
        )
        ax.plot(
            wl[keep],
            shade["median"][keep],
            linestyle="--",
            linewidth=2,
            color=color,
            label=f"{treatment} shaded",
        )
        ax.fill_between(
            wl[keep],
            shade["p10"][keep],
            shade["p90"][keep],
            alpha=0.06,
            color=color,
        )

    ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance (-)", title=title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sun_minus_shade(stats, wavelengths, out_path: Path, title=None):
    """Treatment-wise sunlit minus shaded median spectrum."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wl = np.asarray(wavelengths, dtype=float)
    keep = (wl >= 400) & (wl <= 1100)

    fig, ax = plt.subplots(figsize=(10, 5))
    for treatment in sorted(stats):
        states = stats[treatment]
        if "sunlit" in states and "shaded" in states:
            delta = states["sunlit"]["median"] - states["shaded"]["median"]
            ax.plot(wl[keep], delta[keep], linewidth=2, label=treatment)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(
        xlabel="Wavelength (nm)",
        ylabel="Sunlit - shaded reflectance",
        title=title,
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)