from __future__ import annotations

import rasterio
import xarray as xr

from config.config_downscaling import (
    NODATA_OUT,
    OUT_ROOT,
    DEFAULT_RASTER_CRS,
    PAR_UMOL_TO_W,
    ProfileConfig,
    get_profiles,
)

from downscaling.io import load_wavelengths, save_tif
from downscaling.compute_downscaling_indices import compute_indices
from downscaling.sif_preprocessing import load_sif
from downscaling.compute_fqe import compute_fqe, compute_sifleaf


# ---------- Run settings ----------

PROFILE_TO_RUN = "all"  # options: "all", "SFMNN", "SFM", "iFLD"

EXPORT_CUSTOM_VIS = True
EXPORT_PREPROCESSED_SIF = True
EXPORT_FQE = True
EXPORT_SIFLEAF = False

VI_BAND_NAMES = ["pri_inv", "wbi_inv", "nirv", "fcvi", "sar2f", "fAPARchl"]

# ---------- Helper functions ----------

def mean_stack(stack):
    return xr.concat(stack, dim="flight").mean(dim="flight", skipna=True)


def save_vi_cube(vi, out_path):
    layers = [
        vi["PRI"].astype("float32").expand_dims(band=[1]),
        vi["WBI"].astype("float32").expand_dims(band=[2]),
        vi["NIRv"].astype("float32").expand_dims(band=[3]),
        vi["FCVI"].astype("float32").expand_dims(band=[4]),
        vi["saR2F"].astype("float32").expand_dims(band=[5]),
        vi["fAPARchl"].astype("float32").expand_dims(band=[6]),
    ]

    cube = xr.concat(layers, dim="band")
    save_tif(cube, out_path, nodata_out=NODATA_OUT)

    with rasterio.open(out_path, "r+") as dst:
        dst.descriptions = tuple(VI_BAND_NAMES)


# ---------- Run profile ----------

def run_profile(
    cfg: ProfileConfig,
    export_preprocessed_sif=False,
    export_fqe=True,
    export_sifleaf=False,
    export_custom_vis=True,
):
    wavelengths = load_wavelengths(cfg.hdr_path_for_wavelengths)

    method_map = {
        "nirv": ("NIRv", "fesc_SIF760_NIRv", "NIRv"),
        "fcvi": ("FCVI_valid", "fesc_SIF760_FCVI", "FCVI"),
        "sar2f": ("saR2F", "fesc_SIF760_saR2F", "saR2F"),
    }

    for scene in cfg.scenes:
        out_dir = OUT_ROOT / cfg.name / scene.date
        out_dir.mkdir(parents=True, exist_ok=True)

        vi_out_dir = OUT_ROOT / "VIs" / cfg.name / scene.date
        vi_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {cfg.name} | {scene.date} ===")

        ref_grid = None
        sif_stack = []

        vi_stacks = {
            "NDVI": [],
            "PRI": [],
            "WBI": [],
            "NIRv": [],
            "FCVI": [],
            "saR2F": [],
            "fAPARchl": [],
        }

        fqe_stacks = {
            method_map[m][2]: []
            for m in cfg.fesc_methods
        }

        sifleaf_stacks = {
            method_map[m][2]: []
            for m in cfg.fesc_methods
        }

        for flight in scene.flights:
            print(f"  {flight.flight_id}")

            sif = load_sif(
                flight.sif_file,
                sif_o2a_band=cfg.sif_o2a_band,
                sif_to_sif760_factor=cfg.sif_to_sif760_factor,
                sif_scale_factor=cfg.sif_scale_factor,
                nodata_in=NODATA_OUT,
            )

            vi = compute_indices(
                flight.toc_refl_file,
                wavelengths,
                ndvi_threshold=cfg.ndvi_threshold,
                fcvi_threshold=cfg.fcvi_threshold,
                default_crs=DEFAULT_RASTER_CRS,
            )

            sif_grid = (
                sif.rio.crs,
                sif.rio.shape,
                sif.rio.transform(),
            )
            toc_grid = (
                vi["NDVI"].rio.crs,
                vi["NDVI"].rio.shape,
                vi["NDVI"].rio.transform(),
            )

            if sif_grid != toc_grid:
                raise ValueError(
                    f"{cfg.name} {scene.date} {flight.flight_id}: "
                    "SIF and TOC grids do not match"
                )

            if ref_grid is None:
                ref_grid = sif_grid
            elif sif_grid != ref_grid:
                raise ValueError(
                    f"{cfg.name} {scene.date} {flight.flight_id}: "
                    "flight does not match shared date grid"
                )

            sif_stack.append(sif)
            for key in vi_stacks:
                vi_stacks[key].append(vi[key])

            if export_custom_vis:
                save_vi_cube(
                    vi,
                    vi_out_dir / f"custom_vi_{flight.flight_id}_{scene.date}.tif",
                )

            if export_preprocessed_sif:
                save_tif(
                    sif,
                    out_dir /
                    f"{cfg.name}_SIF760_preprocessed_{flight.flight_id}_{scene.date}.tif",
                    nodata_out=NODATA_OUT,
                )

            par_umol = (
                flight.par_umol_m2_s
                if flight.par_umol_m2_s is not None
                else scene.par_umol_m2_s
            )
            par_mW_m2 = (
                par_umol * PAR_UMOL_TO_W * 1000.0
                if par_umol is not None
                else None
            )

            for method in cfg.fesc_methods:
                index_name, fesc_name, tag = method_map[method]

                if export_sifleaf:
                    sifleaf = compute_sifleaf(sif, vi[fesc_name])
                    sifleaf_stacks[tag].append(sifleaf)
                    save_tif(
                        sifleaf,
                        out_dir /
                        f"{cfg.name}_SIFleaf760_{tag}_{flight.flight_id}_{scene.date}.tif",
                        nodata_out=NODATA_OUT,
                    )

                if export_fqe and par_mW_m2 is not None:
                    fqe = compute_fqe(sif, vi[index_name], par_mW_m2)
                    fqe_stacks[tag].append(fqe)
                    save_tif(
                        fqe,
                        out_dir /
                        f"{cfg.name}_FQE760_{tag}_{flight.flight_id}_{scene.date}.tif",
                        nodata_out=NODATA_OUT,
                    )

        if export_preprocessed_sif:
            save_tif(
                mean_stack(sif_stack),
                out_dir / f"{cfg.name}_SIF760_preprocessed_MEAN_{scene.date}.tif",
                nodata_out=NODATA_OUT,
            )

        if export_custom_vis:
            vi_means = {
                key: mean_stack(stack)
                for key, stack in vi_stacks.items()
            }
            save_vi_cube(
                vi_means,
                vi_out_dir / f"custom_vi_MEAN_{scene.date}.tif",
            )

        if export_sifleaf:
            for tag, stack in sifleaf_stacks.items():
                if stack:
                    save_tif(
                        mean_stack(stack),
                        out_dir / f"{cfg.name}_SIFleaf760_{tag}_MEAN_{scene.date}.tif",
                        nodata_out=NODATA_OUT,
                    )

        if export_fqe:
            for tag, stack in fqe_stacks.items():
                if stack:
                    save_tif(
                        mean_stack(stack),
                        out_dir / f"{cfg.name}_FQE760_{tag}_MEAN_{scene.date}.tif",
                        nodata_out=NODATA_OUT,
                    )
                else:
                    print(f"  FQE {tag}: skipped, PAR missing")

        print("Done.")


# ---------- Main ----------

def main():
    profiles = get_profiles()

    if PROFILE_TO_RUN == "all":
        selected = [
            name
            for name in ("SFMNN", "SFM", "iFLD")
            if name in profiles
        ]
    else:
        if PROFILE_TO_RUN not in profiles:
            raise ValueError(
                f"Unknown profile '{PROFILE_TO_RUN}'. "
                f"Choose from {list(profiles)} or 'all'."
            )
        selected = [PROFILE_TO_RUN]

    print(f"\nRunning profiles: {selected}\n")

    for name in selected:
        run_profile(
            profiles[name],
            export_preprocessed_sif=EXPORT_PREPROCESSED_SIF,
            export_fqe=EXPORT_FQE,
            export_sifleaf=EXPORT_SIFLEAF,
            export_custom_vis=EXPORT_CUSTOM_VIS,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())