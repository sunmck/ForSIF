from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.features import geometry_mask

from config.config_downscaling import OUT_ROOT, get_profiles
from plots.plots_sif_fqe_comparison import (
    ensure_dir,
    plot_fqe_method_comparison,
    plot_fqe_retrieval_comparison_by_treatment,
    plot_sif_retrieval_comparison,
)


# ---------- Run settings ----------

DATES = ["20230617", "20240613", "20240823", "20260529", "20260805"]
RETRIEVALS = ["iFLD", "SFM", "SFMNN"]
DOWNSCALING_TAGS = ["NIRv", "FCVI", "saR2F"]

TREATMENT_FIELD = "treatment"
TREATMENT_LABELS = {
    1: "control",
    2: "irrig.",
    3: "irrig. stopped",
}


# ---------- Helper functions ----------

def open_tif(path: Path):
    da = rxr.open_rasterio(path, masked=True)

    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)

    return da


def parse_flight_id(flight_id):
    time, line, direction = flight_id.split("_")

    hour = int(time[:2])
    minute = int(time[2:])

    return time, hour * 60 + minute, line, direction


def add_polygon_rows(
    rows,
    raster,
    treatment_areas,
    *,
    date,
    retrieval,
    flight_id,
    metric,
    method=None,
    sample_type="flight",
):
    areas = treatment_areas.to_crs(raster.rio.crs).reset_index(drop=True)
    arr = np.asarray(raster.values).squeeze()

    if sample_type == "flight":
        time, time_min, line, direction = parse_flight_id(flight_id)

    elif sample_type == "date_mean":
        time = "MEAN"
        time_min = np.inf
        line = "MEAN"
        direction = ""

    else:
        raise ValueError(f"Unknown sample_type: {sample_type}")

    for i, area in areas.iterrows():

        mask = geometry_mask(
            [area.geometry],
            out_shape=arr.shape,
            transform=raster.rio.transform(),
            invert=True,
        )

        values = arr[mask]
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        treatment_value = area[TREATMENT_FIELD]
        treatment = TREATMENT_LABELS.get(
            treatment_value,
            str(treatment_value),
        )

        rows.append(
            {
                "date": date,
                "retrieval": retrieval,
                "sample_type": sample_type,
                "flight_id": flight_id,
                "time": time,
                "time_min": time_min,
                "line": line,
                "direction": direction,
                "plot_id": i + 1,
                "treatment": treatment,
                "metric": metric,
                "method": method,
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "n_valid": int(values.size),
            }
        )


def build_polygon_table(profiles):
    rows = []

    for retrieval in RETRIEVALS:

        if retrieval not in profiles:
            continue

        cfg = profiles[retrieval]
        treatment_areas = gpd.read_file(cfg.treatment_areas_shp)

        for scene in cfg.scenes:

            if scene.date not in DATES:
                continue

            out_dir = OUT_ROOT / cfg.name / scene.date

            # --------------------------------------------------
            # Individual flight lines
            # --------------------------------------------------

            for flight in scene.flights:

                sif_path = (
                    out_dir
                    / f"{cfg.name}_SIF760_preprocessed_"
                    f"{flight.flight_id}_{scene.date}.tif"
                )

                if sif_path.exists():
                    add_polygon_rows(
                        rows,
                        open_tif(sif_path),
                        treatment_areas,
                        date=scene.date,
                        retrieval=cfg.name,
                        flight_id=flight.flight_id,
                        metric="SIF760",
                    )

                for tag in DOWNSCALING_TAGS:

                    fqe_path = (
                        out_dir
                        / f"{cfg.name}_FQE760_{tag}_"
                        f"{flight.flight_id}_{scene.date}.tif"
                    )

                    if not fqe_path.exists():
                        continue

                    add_polygon_rows(
                        rows,
                        open_tif(fqe_path),
                        treatment_areas,
                        date=scene.date,
                        retrieval=cfg.name,
                        flight_id=flight.flight_id,
                        metric="FQE760",
                        method=tag,
                    )

            # --------------------------------------------------
            # Date-mean SIF mosaic
            # --------------------------------------------------

            mean_sif_path = (
                out_dir
                / f"{cfg.name}_SIF760_preprocessed_MEAN_"
                f"{scene.date}.tif"
            )

            if mean_sif_path.exists():
                add_polygon_rows(
                    rows,
                    open_tif(mean_sif_path),
                    treatment_areas,
                    date=scene.date,
                    retrieval=cfg.name,
                    flight_id="MEAN",
                    metric="SIF760",
                    sample_type="date_mean",
                )

            # --------------------------------------------------
            # Date-mean FQE mosaics
            # --------------------------------------------------

            for tag in DOWNSCALING_TAGS:

                mean_fqe_path = (
                    out_dir
                    / f"{cfg.name}_FQE760_{tag}_MEAN_"
                    f"{scene.date}.tif"
                )

                if not mean_fqe_path.exists():
                    continue

                add_polygon_rows(
                    rows,
                    open_tif(mean_fqe_path),
                    treatment_areas,
                    date=scene.date,
                    retrieval=cfg.name,
                    flight_id="MEAN",
                    metric="FQE760",
                    method=tag,
                    sample_type="date_mean",
                )

    return pd.DataFrame(rows)


# ---------- Main ----------

def main():
    profiles = get_profiles()

    df = build_polygon_table(profiles)

    if df.empty:
        raise RuntimeError("No SIF/FQE rasters found.")

    out_dir = ensure_dir(
        OUT_ROOT / "comparisons"
    )

    df.to_csv(
        out_dir / "polygon_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # SIF comparison
    # --------------------------------------------------

    plot_sif_retrieval_comparison(
        df,
        out_dir,
        retrieval_order=RETRIEVALS,
        by_treatment=False,
        fname="SIF_retrieval_comparison.png",
    )

    # --------------------------------------------------
    # SIF by treatment
    # --------------------------------------------------

    plot_sif_retrieval_comparison(
        df,
        out_dir,
        retrieval_order=RETRIEVALS,
        by_treatment=True,
        fname="SIF_retrieval_comparison_by_treatment.png",
    )

    # --------------------------------------------------
    # FQE downscaling-method comparison
    # --------------------------------------------------

    plot_fqe_method_comparison(
        df,
        out_dir,
        retrieval_order=RETRIEVALS,
        method_order=DOWNSCALING_TAGS,
        fname="FQE_downscaling_method_comparison.png",
    )

    # --------------------------------------------------
    # FQE saR2F by treatment
    # --------------------------------------------------

    plot_fqe_retrieval_comparison_by_treatment(
        df,
        out_dir,
        retrieval_order=RETRIEVALS,
        method="saR2F",
        fname="FQE_saR2F_retrieval_comparison_by_treatment.png",
    )

    print(f"Saved comparison results to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())