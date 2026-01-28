# run_sif_fqe_comparison.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr

from config.config import OUT_ROOT, PLOTS_DIRNAME
from config.config import ProfileConfig, get_profiles

from plots.plots_sif_fqe_comparison import (
    ensure_dir,
    extract_group_values,
    plot_retrieval_only_pooled,
    plot_treatments_only_pooled_methods,
    plot_treatment_x_retrieval,
)

# -----------------------
# Settings
# -----------------------
DATES = ["20240613", "20240823"]
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


def open_mean_tif(path: Path):
    da = rxr.open_rasterio(path, masked=True)
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)
    return da


def build_samples_table(
    profiles: Dict[str, ProfileConfig],
    dates: Sequence[str],
) -> pd.DataFrame:
    """
    Builds a tidy table of sampled crown-masked values from exported MEAN rasters.

    Columns:
      metric: "SIF760" or "FQE760"
      date
      retrieval (profile name)
      downscaling (None for SIF, else tag)
      treatment (None for pooled, else label)
      value
    """
    rows = []
    seed = RANDOM_SEED

    for retrieval_name, cfg in profiles.items():
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

                    # pooled across crown treatments
                    vals = extract_group_values(
                        da,
                        crowns_r,
                        None,
                        supersample=SUPERSAMPLE,
                        min_weight=MIN_WEIGHT,
                        n_sample=SAMPLE_PER_GROUP,
                        seed=seed,
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

                    # split by crown treatments
                    for t, tlab in zip(TREATMENTS, TREATMENT_LABELS):
                        vals = extract_group_values(
                            da,
                            crowns_r,
                            t,
                            supersample=SUPERSAMPLE,
                            min_weight=MIN_WEIGHT,
                            n_sample=SAMPLE_PER_GROUP,
                            seed=seed,
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

                    # pooled across crown treatments
                    vals = extract_group_values(
                        da,
                        crowns_r,
                        None,
                        supersample=SUPERSAMPLE,
                        min_weight=MIN_WEIGHT,
                        n_sample=SAMPLE_PER_GROUP,
                        seed=seed,
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

                    # split by crown treatments
                    for t, tlab in zip(TREATMENTS, TREATMENT_LABELS):
                        vals = extract_group_values(
                            da,
                            crowns_r,
                            t,
                            supersample=SUPERSAMPLE,
                            min_weight=MIN_WEIGHT,
                            n_sample=SAMPLE_PER_GROUP,
                            seed=seed,
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
    df = df[np.isfinite(df["value"])]
    return df


def main():
    profiles_all = get_profiles()
    profiles = {k: profiles_all[k] for k in RETRIEVALS if k in profiles_all}
    if not profiles:
        raise RuntimeError(f"No profiles found for RETRIEVALS={RETRIEVALS}. Available: {list(profiles_all.keys())}")

    df = build_samples_table(profiles, DATES)

    out_root = ensure_dir(OUT_ROOT / "comparisons" / "SIF_FQE")
    plots_dir = ensure_dir(out_root / PLOTS_DIRNAME)

    plot_retrieval_only_pooled(df, plots_dir, dates=DATES, downscaling_tags=DOWNSCALING_TAGS)
    plot_treatments_only_pooled_methods(
        df,
        plots_dir,
        dates=DATES,
        downscaling_tags=DOWNSCALING_TAGS,
        treatment_labels=TREATMENT_LABELS,
    )
    plot_treatment_x_retrieval(
        df,
        plots_dir,
        dates=DATES,
        downscaling_tags=DOWNSCALING_TAGS,
        treatment_labels=TREATMENT_LABELS,
    )

    print(f"Saved comparison plots to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
