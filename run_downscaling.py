# downscaling/pipeline.py
from __future__ import annotations

from typing import Dict

import geopandas as gpd

from config.config import NODATA_OUT, OUT_ROOT, DEFAULT_RASTER_CRS
from config.config import ProfileConfig, get_profiles
from downscaling.io import load_wavelengths, save_tif
from downscaling.compute_downscaling_indices import process_month_indices
from downscaling.sif_preprocessing import load_sif_stack
from downscaling.compute_fqe import compute_fqe_stack, compute_sifleaf_stack
from plots.plots import PlotOptions, make_scene_plots, mean_mosaic

# RUN SETTINGS
# Can be:
#   - "SFMNN" / "SFM" / "iFLD" / "all"
#   - ["SFMNN", "iFLD"]  (any 1..N profiles)
PROFILE_TO_RUN = ["iFLD"]

EXPORT_PREPROCESSED_SIF = True
EXPORT_FQE = True
EXPORT_SIFLEAF = False
MAKE_PLOTS = True
DRY_RUN = False


def run_profile(
    cfg: ProfileConfig,
    export_preprocessed_sif: bool = False,
    export_fqe: bool = True,
    export_sifleaf: bool = False,
    dry_run: bool = False,
    make_plots: bool = True,
):
    wavelengths = load_wavelengths(cfg.hdr_path_for_wavelengths)

    # Load vectors once per profile
    crowns = gpd.read_file(cfg.crowns_shp)
    treatment_areas = gpd.read_file(cfg.treatment_areas_shp)

    # treatment config
    treatments = [1, 2, 3]
    treatment_labels = ["control", "irrig.", "irrig. stopped"]
    treatment_color_map = {1: "tab:orange", 2: "tab:blue", 3: "tab:green"}

    # plot options
    plot_opts = PlotOptions(
        save=True,
        make_overview_maps=True, 
        make_monthly_comparisons=True, 
        make_boxplots=True,
        make_weighted_stats=True,
    )

    for scene in cfg.scenes:
        out_dir = OUT_ROOT / cfg.name / scene.date
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {cfg.name} | {scene.date} ===")
        print(f"SIF rasters: {len(scene.sif_files)} | TOC refl rasters: {len(scene.toc_refl_files)}")
        print(f"Band={cfg.sif_o2a_band} | factor={cfg.sif_to_sif760_factor} | PAR={scene.par_mW_m2:.2f} mW/m²")
        print(f"Output: {out_dir}")

        if dry_run:
            continue

        # 1) Load SIF stack (per flightline) + reference grid
        sif_stack, sif_names = load_sif_stack(
            scene.sif_files,
            sif_o2a_band=cfg.sif_o2a_band,
            sif_to_sif760_factor=cfg.sif_to_sif760_factor,
        )
        ref_raster = sif_stack[0]

        # Reproject vectors to raster CRS
        crowns_r = crowns.to_crs(ref_raster.rio.crs)
        treatment_areas_r = treatment_areas.to_crs(ref_raster.rio.crs)

        # 2) Compute indices & fesc from TOC reflectance
        refl = process_month_indices(
            scene.toc_refl_files,
            wavelengths,
            ndvi_threshold=cfg.ndvi_threshold,
            fcvi_threshold=cfg.fcvi_threshold,
            default_crs= DEFAULT_RASTER_CRS,
        )

        # align required fields to SIF grid
        nirv = refl["NIRv"].rio.reproject_match(ref_raster)
        fcvi = refl["FCVI"].rio.reproject_match(ref_raster)
        sar2f = refl["saR2F"].rio.reproject_match(ref_raster)

        fesc_nirv = refl["fesc_SIF760_NIRv"].rio.reproject_match(ref_raster)
        fesc_fcvi = refl["fesc_SIF760_FCVI"].rio.reproject_match(ref_raster)
        fesc_sar2f = refl["fesc_SIF760_saR2F"].rio.reproject_match(ref_raster)

        # Empty dict to collect mean mosaics for plots
        means: Dict[str, object] = {}

        # 3) Export preprocessed SIF760 (per flightline) + mean mosaic
        if export_preprocessed_sif:
            for da, name in zip(sif_stack, sif_names):
                out = out_dir / f"{cfg.name}_SIF760_preprocessed_{name}_{scene.date}.tif"
                save_tif(da, out, nodata_out=NODATA_OUT)

            sif_mean = mean_mosaic(sif_stack)
            save_tif(
                sif_mean,
                out_dir / f"{cfg.name}_SIF760_preprocessed_MEAN_{scene.date}.tif",
                nodata_out=NODATA_OUT,
            )
            means["SIF760_preprocessed_mean"] = sif_mean
            print("Exported preprocessed SIF760 (flights + mean)")

        # 4) For each selected fesc method: export stacks + mean mosaics
        method_map: Dict[str, Dict[str, object]] = {
            "nirv": {"index": nirv, "fesc": fesc_nirv, "tag": "NIRv"},
            "fcvi": {"index": fcvi, "fesc": fesc_fcvi, "tag": "FCVI"},
            "sar2f": {"index": sar2f, "fesc": fesc_sar2f, "tag": "saR2F"},
        }

        for m in cfg.fesc_methods:
            if m not in method_map:
                raise ValueError(f"Unknown method '{m}'. Use one of: {list(method_map.keys())}")

            idx = method_map[m]["index"]
            fesc = method_map[m]["fesc"]
            tag = method_map[m]["tag"]

            if export_sifleaf:
                sifleaf_stack = compute_sifleaf_stack(sif_stack, fesc)

                for da, name in zip(sifleaf_stack, sif_names):
                    out = out_dir / f"{cfg.name}_SIFleaf760_{tag}_{name}_{scene.date}.tif"
                    save_tif(da, out, nodata_out=NODATA_OUT)

                sifleaf_mean = mean_mosaic(sifleaf_stack)
                save_tif(
                    sifleaf_mean,
                    out_dir / f"{cfg.name}_SIFleaf760_{tag}_MEAN_{scene.date}.tif",
                    nodata_out=NODATA_OUT,
                )
                means[f"SIFleaf760_{tag}_mean"] = sifleaf_mean
                print(f"Exported SIFleaf ({tag}) (flights + mean)")

            if export_fqe:
                fqe_stack = compute_fqe_stack(sif_stack, idx, scene.par_mW_m2)

                for da, name in zip(fqe_stack, sif_names):
                    out = out_dir / f"{cfg.name}_FQE760_{tag}_{name}_{scene.date}.tif"
                    save_tif(da, out, nodata_out=NODATA_OUT)

                fqe_mean = mean_mosaic(fqe_stack)
                save_tif(
                    fqe_mean,
                    out_dir / f"{cfg.name}_FQE760_{tag}_MEAN_{scene.date}.tif",
                    nodata_out=NODATA_OUT,
                )
                means[f"FQE760_{tag}_mean"] = fqe_mean
                print(f"Exported FQE ({tag}) (flights + mean)")

        # 5) Plots (saved inside out_dir/plots)
        if make_plots:
            layer_names = list(sif_names)

            make_scene_plots(
                out_dir=out_dir,
                opts=plot_opts,
                ref_raster=ref_raster,
                crowns=crowns_r,
                treatment_areas=treatment_areas_r,
                treatments=treatments,
                treatment_labels=treatment_labels,
                treatment_color_map=treatment_color_map,
                means=means,
                stacks=None,
                layer_names=layer_names,
            )
            print("All plots saved")

        print("Done.")


def main():
    profiles = get_profiles()
    keymap = {k.lower(): k for k in profiles.keys()}  # allow lowercase input

    preferred_order = ["SFMNN", "SFM", "iFLD"]

    def normalize_one(x: str) -> str:
        s = str(x).strip().lower()
        if s == "all":
            return "all"
        if s not in keymap:
            raise ValueError(
                f"Unknown profile '{x}'. Choose from {list(profiles.keys())} or 'all'."
            )
        return keymap[s]

    # ----- Decide selection -----
    if isinstance(PROFILE_TO_RUN, (list, tuple, set)):
        normalized = [normalize_one(x) for x in PROFILE_TO_RUN]
        if "all" in normalized:
            selected = [p for p in preferred_order if p in profiles]
        else:
            # keep user's order, remove duplicates while preserving order
            seen = set()
            selected = []
            for p in normalized:
                if p not in seen:
                    selected.append(p)
                    seen.add(p)
    else:
        one = normalize_one(PROFILE_TO_RUN)
        if one == "all":
            selected = [p for p in preferred_order if p in profiles]
        else:
            selected = [one]

    print(f"\nRunning profiles: {selected}\n")

    # ----- Run -----
    for name in selected:
        cfg = profiles[name]
        run_profile(
            cfg,
            export_preprocessed_sif=EXPORT_PREPROCESSED_SIF,
            export_fqe=EXPORT_FQE,
            export_sifleaf=EXPORT_SIFLEAF,
            dry_run=DRY_RUN,
            make_plots=MAKE_PLOTS,
        )

    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main())
