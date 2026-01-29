# run_sif_fqe_comparison.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr

from config.config_downscaling import OUT_ROOT
from config.config_downscaling import ProfileConfig, get_profiles

from plots.plots_sif_fqe_comparison import (
    ensure_dir,
    extract_group_values,
    plot_sif_violin_retrieval_pooled,
    plot_sif_violin_retrieval_by_treatment,
    plot_fqe_violin_compact_retrieval_with_treatments,
    plot_fqe_violin_compact_retrieval_pooled,   # <-- add
    plot_fqe_violin_grid,
)

# RUN SETTINGS
DATES = ["20230617", "20240613", "20240823"]

RETRIEVALS = ["iFLD", "SFM", "SFMNN"]
DOWNSCALING_TAGS = ["NIRv", "FCVI", "saR2F"]

TREATMENTS = [1, 2, 3]
TREATMENT_LABELS = ["control", "irrig.", "irrig. stopped"]

DO_SIF = True
DO_FQE = True

SUPERSAMPLE = 10
MIN_WEIGHT = 0.5
SAMPLE_PER_GROUP = 8000
RANDOM_SEED = 42

# Choose which FQE figures to plot
MAKE_FQE_COMPACT = True
MAKE_FQE_COMPACT_POOLED = True
MAKE_FQE_GRID = False

# Addtional helper functions
def open_mean_tif(path: Path):
    da = rxr.open_rasterio(path, masked=True)
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)
    return da


def build_samples_table(
    profiles: Dict[str, ProfileConfig],
    dates: Sequence[str],
):

    rows = []
    seed = RANDOM_SEED

    for _, cfg in profiles.items():
        crowns = gpd.read_file(cfg.crowns_shp)

        for date in dates:
            out_dir = OUT_ROOT / cfg.name / date

            # pick a ref raster that exists so we can project crowns
            ref_path = None
            if DO_SIF:
                p = out_dir / f"{cfg.name}_SIF760_preprocessed_MEAN_{date}.tif"
                if p.exists():
                    ref_path = p
            if ref_path is None and DO_FQE:
                p = out_dir / f"{cfg.name}_FQE760_NIRv_MEAN_{date}.tif"
                if p.exists():
                    ref_path = p

            if ref_path is None:
                continue

            ref = open_mean_tif(ref_path)
            crowns_r = crowns.to_crs(ref.rio.crs)

            # ---- SIF
            if DO_SIF:
                p = out_dir / f"{cfg.name}_SIF760_preprocessed_MEAN_{date}.tif"
                if p.exists():
                    da = open_mean_tif(p)

                    # pooled across crowns
                    vals = extract_group_values(
                        da,
                        crowns_r,
                        None,
                        supersample=SUPERSAMPLE,
                        min_weight=MIN_WEIGHT,
                        n_sample=SAMPLE_PER_GROUP,
                        seed=seed,
                        treat_zero_as_nodata=True, 
                    )
                    seed += 1
                    for v in vals:
                        rows.append(
                            dict(
                                metric="SIF760",
                                date=date,
                                retrieval=cfg.name,
                                downscaling=None,
                                treatment=None,
                                value=float(v),
                            )
                        )

                    # by treatments
                    for t, tlab in zip(TREATMENTS, TREATMENT_LABELS):
                        vals = extract_group_values(
                            da,
                            crowns_r,
                            t,  # IMPORTANT: this must be t
                            supersample=SUPERSAMPLE,
                            min_weight=MIN_WEIGHT,
                            n_sample=SAMPLE_PER_GROUP,
                            seed=seed,
                            treat_zero_as_nodata=True,
                        )
                        seed += 1
                        for v in vals:
                            rows.append(
                                dict(
                                    metric="SIF760",
                                    date=date,
                                    retrieval=cfg.name,
                                    downscaling=None,
                                    treatment=tlab,
                                    value=float(v),
                                )
                            )

            # ---- FQE
            if DO_FQE:
                for tag in DOWNSCALING_TAGS:
                    p = out_dir / f"{cfg.name}_FQE760_{tag}_MEAN_{date}.tif"
                    if not p.exists():
                        continue

                    da = open_mean_tif(p)

                    # pooled across crowns
                    vals = extract_group_values(
                        da,
                        crowns_r,
                        None,
                        supersample=SUPERSAMPLE,
                        min_weight=MIN_WEIGHT,
                        n_sample=SAMPLE_PER_GROUP,
                        seed=seed,
                        treat_zero_as_nodata=False,
                    )
                    seed += 1
                    for v in vals:
                        rows.append(
                            dict(
                                metric="FQE760",
                                date=date,
                                retrieval=cfg.name,
                                downscaling=tag,
                                treatment=None,
                                value=float(v),
                            )
                        )

                    # by treatments
                    for t, tlab in zip(TREATMENTS, TREATMENT_LABELS):
                        vals = extract_group_values(
                            da,
                            crowns_r,
                            t,
                            supersample=SUPERSAMPLE,
                            min_weight=MIN_WEIGHT,
                            n_sample=SAMPLE_PER_GROUP,
                            seed=seed,
                            treat_zero_as_nodata=False,
                        )
                        seed += 1
                        for v in vals:
                            rows.append(
                                dict(
                                    metric="FQE760",
                                    date=date,
                                    retrieval=cfg.name,
                                    downscaling=tag,
                                    treatment=tlab,
                                    value=float(v),
                                )
                            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[np.isfinite(df["value"].to_numpy())]
    return df


def main():
    profiles_all = get_profiles()
    profiles = {k: profiles_all[k] for k in RETRIEVALS if k in profiles_all}
    if not profiles:
        raise RuntimeError(
            f"No profiles found for RETRIEVALS={RETRIEVALS}. Available: {list(profiles_all.keys())}"
        )

    df = build_samples_table(profiles, DATES)
    if df.empty:
        raise RuntimeError("No samples were created. Check that MEAN tif exports exist for the requested dates.")

    out_dir = ensure_dir(OUT_ROOT / "comparisons")

    # 1) SIF pooled across crowns (no treatments)
    plot_sif_violin_retrieval_pooled(
        df,
        out_dir,
        dates=DATES,
        retrieval_order=RETRIEVALS,
        title_prefix="SIF760 pooled",
        ylabel="SIF760",
        fname="SIF_violin_retrieval_pooled.png",
    )

    # 2) SIF grouped by treatments (3 violins per retrieval)
    plot_sif_violin_retrieval_by_treatment(
        df,
        out_dir,
        dates=DATES,
        retrieval_order=RETRIEVALS,
        treatment_labels=TREATMENT_LABELS,
        title_prefix="SIF760 by treatment",
        ylabel="SIF760",
        fname="SIF_violin_retrieval_by_treatment.png",
    )

    # 3a) FQE compact (with treatments)
    if MAKE_FQE_COMPACT:
        plot_fqe_violin_compact_retrieval_with_treatments(
            df,
            out_dir,
            dates=DATES,
            downscaling_tags=DOWNSCALING_TAGS,
            retrieval_order=RETRIEVALS,
            treatment_labels=TREATMENT_LABELS,
            title_prefix="FQE760",
            ylabel_prefix="FQE",
            fname="FQE_violin_compact.png",
        )

    # 3a2) FQE compact pooled only (no treatments)
    if MAKE_FQE_COMPACT_POOLED:
        plot_fqe_violin_compact_retrieval_pooled(
            df,
            out_dir,
            dates=DATES,
            downscaling_tags=DOWNSCALING_TAGS,
            retrieval_order=RETRIEVALS,
            title_prefix="FQE760 pooled",
            ylabel_prefix="FQE",
            fname="FQE_violin_compact_pooled.png",
        )

    # 3b) FQE grid (optional, notebook-style)
    if MAKE_FQE_GRID:
        plot_fqe_violin_grid(
            df,
            out_dir,
            dates=DATES,
            downscaling_tags=DOWNSCALING_TAGS,
            retrieval_order=RETRIEVALS,
            treatment_labels=TREATMENT_LABELS,
            title_prefix="FQE760",
            ylabel_prefix="FQE",
            fname="FQE_violin_grid.png",
        )

    print(f"Saved comparison plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
