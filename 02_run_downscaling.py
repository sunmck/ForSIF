from __future__ import annotations

from dataclasses import asdict

import rasterio
import rioxarray
import xarray as xr

from config.config_downscaling import (
    NODATA_OUT,
    OUT_ROOT,
    PLOTS_DIRNAME,
    DEFAULT_RASTER_CRS,
    PAR_UMOL_TO_W,
    ProfileConfig,
    get_profiles,
)

from downscaling.io import load_wavelengths, save_tif
from downscaling.compute_downscaling_indices import compute_indices
from config.config_illumination import (
    FQE_ILLUMINATION_QC,
    ILLUMINATION_CONFIG,
    ILLUMINATION_PRODUCTS_TO_SAVE,
)
from config.config_formask import forest_fraction_path, scaffold_fraction_path
from downscaling.illumination.sunlit_fraction import (
    compute_sunlit_fraction,
    save_endmember_library,
    save_endmember_points,
)
from plots.plots_illumination import (
    plot_endmember_spectra,
    plot_treatment_spectra,
    plot_sun_minus_shade,
)
from downscaling.sif_preprocessing import load_sif
from downscaling.compute_fqe import compute_fqe, compute_sifleaf


# ---------- Run settings ----------

PROFILE_TO_RUN = "all"  # options: "all", "SFMNN", "SFM", "iFLD"

EXPORT_CUSTOM_VIS = True
EXPORT_PREPROCESSED_SIF = True
EXPORT_FQE = True
EXPORT_SIFLEAF = False

# Sunlit/shaded retrieval
EXPORT_ILLUMINATION = True
EXPORT_ENDMEMBER_PLOTS = True

# The methodological settings, FQE illumination-QC thresholds, and list of
# illumination products to save are defined in config/config_illumination.py.

# Keep all physically valid NDVI values during preprocessing.
# Forest/vegetation masking is applied later during analysis.
NDVI_PROCESSING_MIN = -1.0

VI_BAND_NAMES = ["pri_inv", "wbi_inv", "nirv", "fcvi_valid", "sar2f", "fAPARchl"]


# ---------- Helper functions ----------

def mean_stack(stack):
    return xr.concat(stack, dim="flight").mean(dim="flight", skipna=True)


def save_vi_cube(vi, out_path):
    layers = [
        vi["PRI"].astype("float32").expand_dims(band=[1]),
        vi["WBI"].astype("float32").expand_dims(band=[2]),
        vi["NIRv"].astype("float32").expand_dims(band=[3]),
        vi["FCVI_valid"].astype("float32").expand_dims(band=[4]),
        vi["saR2F"].astype("float32").expand_dims(band=[5]),
        vi["fAPARchl"].astype("float32").expand_dims(band=[6]),
    ]

    cube = xr.concat(layers, dim="band")
    save_tif(cube, out_path, nodata_out=NODATA_OUT)

    with rasterio.open(out_path, "r+") as dst:
        dst.descriptions = tuple(VI_BAND_NAMES)


def illumination_qc_enabled():
    return (
        FQE_ILLUMINATION_QC.sunlit_veg_min is not None
        or FQE_ILLUMINATION_QC.unmix_rmse_max is not None
    )


def required_illumination_qc_products():
    products = []
    if FQE_ILLUMINATION_QC.sunlit_veg_min is not None:
        products.append("f_sun_veg")
    if FQE_ILLUMINATION_QC.unmix_rmse_max is not None:
        products.append("rmse")
    return products


def save_illumination_products(illum, out_dir, flight_id, product_names, reference_raster):
    """Save products and force the exact source CRS/transform onto each GeoTIFF."""
    with rasterio.open(reference_raster) as src:
        crs = src.crs or DEFAULT_RASTER_CRS
        transform = src.transform
        shape = (src.height, src.width)

    for key in product_names:
        if key not in illum:
            raise KeyError(f"Illumination result does not contain '{key}'")

        path = out_dir / f"{key}_{flight_id}.tif"
        save_tif(illum[key], path, nodata_out=NODATA_OUT)

        # This makes CRS handling independent of save_tif internals.
        with rasterio.open(path, "r+") as dst:
            if (dst.height, dst.width) != shape:
                raise ValueError(f"Grid mismatch for {path}")
            dst.crs = crs
            dst.transform = transform


def load_illumination_for_qc(date, flight_id):
    """Load only the SFMNN-derived illumination layers needed for FQE QC."""
    if not illumination_qc_enabled():
        return None

    illumination_out_dir = OUT_ROOT / "Illumination" / date
    illum = {}

    for key in required_illumination_qc_products():
        path = illumination_out_dir / f"{key}_{flight_id}.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing SFMNN-derived illumination product required for FQE QC: {path}"
            )

        da = rioxarray.open_rasterio(path, masked=True).astype("float32")
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        illum[key] = da

    return illum


def derive_illumination_from_sfmnn(
    sfmnn_cfg: ProfileConfig,
    required_flights,
    export_illumination=True,
    export_endmember_plots=True,
):
    """Derive illumination once, always using the SFMNN flight/DUAL mapping."""
    if sfmnn_cfg.name != "SFMNN":
        raise ValueError("Illumination source profile must be SFMNN")

    illumination_needed = (
        export_illumination
        or export_endmember_plots
        or illumination_qc_enabled()
    )
    if not illumination_needed:
        return

    wavelengths = load_wavelengths(sfmnn_cfg.hdr_path_for_wavelengths)
    required_flights = set(required_flights)
    forest_raster = forest_fraction_path()
    scaffold_raster = scaffold_fraction_path()

    for label, path in (
        ("forest fraction", forest_raster),
        ("scaffold fraction", scaffold_raster),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}. Run 01_run_formask.py first."
            )

    print("\n=== Deriving illumination from SFMNN DUAL flights ===")

    for scene in sfmnn_cfg.scenes:
        scene_required = [
            flight
            for flight in scene.flights
            if (scene.date, flight.flight_id) in required_flights
        ]
        if not scene_required:
            continue

        illumination_out_dir = OUT_ROOT / "illumination" / scene.date
        illumination_plot_dir = (
            OUT_ROOT / PLOTS_DIRNAME / "illumination" / scene.date
        )

        # The endmember library is always saved when illumination is derived.
        illumination_out_dir.mkdir(parents=True, exist_ok=True)
        if export_endmember_plots:
            illumination_plot_dir.mkdir(parents=True, exist_ok=True)

        for flight in scene_required:
            print(f"  {scene.date} | {flight.flight_id}")

            illum, endmembers, candidates = compute_sunlit_fraction(
                flight.toc_refl_file,
                wavelengths,
                default_crs=DEFAULT_RASTER_CRS,
                config=ILLUMINATION_CONFIG,
                treatment_areas_shp=sfmnn_cfg.treatment_areas_shp,
                forest_fraction_raster=forest_raster,
                scaffold_fraction_raster=scaffold_raster,
            )

            print(
                f"    red edge = {endmembers.red_edge_wavelength_nm:.1f} nm | "
                f"support spectra: sun={endmembers.candidate_counts['sunlit']}, "
                f"shade={endmembers.candidate_counts['shaded']}"
            )
            print(
                f"    local NDCSI thresholds from "
                f"{len(candidates['per_plot_thresholds'])} plots | "
                f"candidate pools: sun={candidates['candidate_pool_counts']['sunlit']}, "
                f"shade={candidates['candidate_pool_counts']['shaded']}"
            )

            if export_illumination:
                save_illumination_products(
                    illum,
                    illumination_out_dir,
                    flight.flight_id,
                    ILLUMINATION_PRODUCTS_TO_SAVE,
                    flight.toc_refl_file,
                )
            elif illumination_qc_enabled():
                # Save only the layers needed by downstream FQE QC.
                save_illumination_products(
                    illum,
                    illumination_out_dir,
                    flight.flight_id,
                    required_illumination_qc_products(),
                    flight.toc_refl_file,
                )

            library_stem = illumination_out_dir / f"endmembers_{flight.flight_id}"
            save_endmember_library(
                endmembers,
                library_stem.with_suffix(".npz"),
                library_stem.with_suffix(".json"),
                extra_metadata={
                    "source_profile": "SFMNN",
                    "date": scene.date,
                    "flight_id": flight.flight_id,
                    "per_plot_thresholds": candidates.get("per_plot_thresholds"),
                    "forest_fraction_raster": str(forest_raster),
                    "scaffold_fraction_raster": str(scaffold_raster),
                    "sunlit_config": asdict(ILLUMINATION_CONFIG),
                },
            )

            save_endmember_points(
                candidates["selected_points"],
                flight.toc_refl_file,
                illumination_out_dir / f"endmember_pixels_{flight.flight_id}.gpkg",
            )

            if export_endmember_plots:
                title = f"{scene.date} | {flight.flight_id}"
                plot_endmember_spectra(
                    endmembers,
                    illumination_plot_dir / f"endmembers_{flight.flight_id}.png",
                    title=title,
                )

                stats = candidates.get("spectra_by_treatment")
                if stats:
                    plot_treatment_spectra(
                        stats, wavelengths,
                        illumination_plot_dir / f"treatment_sun_shade_{flight.flight_id}.png",
                        title=title,
                    )
                    plot_sun_minus_shade(
                        stats, wavelengths,
                        illumination_plot_dir / f"treatment_sun_minus_shade_{flight.flight_id}.png",
                        title=title,
                    )

    print("Illumination derivation done.\n")


def apply_fqe_illumination_qc(fqe, illum):
    """Optionally mask FQE using validated illumination/unmixing thresholds."""
    if illum is None:
        return fqe

    valid = xr.ones_like(fqe, dtype=bool)

    if FQE_ILLUMINATION_QC.sunlit_veg_min is not None:
        valid = valid & (
            illum["f_sun_veg"] >= FQE_ILLUMINATION_QC.sunlit_veg_min
        )

    if FQE_ILLUMINATION_QC.unmix_rmse_max is not None:
        valid = valid & (
            illum["rmse"] <= FQE_ILLUMINATION_QC.unmix_rmse_max
        )

    return fqe.where(valid)


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
            "FCVI_valid": [],
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
                # No NDVI > 0.5 vegetation mask here. Retain all physically
                # valid NDVI values and apply the forest mask later in analysis.
                ndvi_threshold=NDVI_PROCESSING_MIN,
                fcvi_threshold=cfg.fcvi_threshold,
                default_crs=DEFAULT_RASTER_CRS,
            )

            # Illumination is derived once from the SFMNN flight/DUAL mapping
            # before any SIF profile is processed. For FQE QC, load only the
            # required SFMNN-derived layers here.
            illum = load_illumination_for_qc(
                scene.date,
                flight.flight_id,
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

            if illum is not None:
                illum_sample = next(iter(illum.values()))
                illum_grid = (
                    illum_sample.rio.crs,
                    illum_sample.rio.shape,
                    illum_sample.rio.transform(),
                )
                if illum_grid != toc_grid:
                    raise ValueError(
                        f"{cfg.name} {scene.date} {flight.flight_id}: "
                        "illumination and TOC grids do not match"
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

            par_umol = flight.par_umol_m2_s

            if export_fqe and par_umol is None:
                raise ValueError(
                    f"{cfg.name} {scene.date} {flight.flight_id}: "
                    "PAR is unavailable for this flight. "
                    "Check PAR_MEASUREMENTS and the flight time."
                )

            par_mW_m2 = (
                par_umol * PAR_UMOL_TO_W * 1000.0
                if par_umol is not None
                else None
            )

            if par_umol is not None:
                print(
                    f"    PAR = {par_umol:.1f} µmol m-2 s-1 "
                    f"({par_mW_m2:.1f} mW m-2)"
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
                    fqe = apply_fqe_illumination_qc(fqe, illum)
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

    illumination_needed = (
        EXPORT_ILLUMINATION
        or EXPORT_ENDMEMBER_PLOTS
        or illumination_qc_enabled()
    )

    if illumination_needed:
        if "SFMNN" not in profiles:
            raise KeyError(
                "SFMNN profile is required because illumination is always "
                "derived from the SFMNN flight/DUAL mapping."
            )

        required_flights = {
            (scene.date, flight.flight_id)
            for name in selected
            for scene in profiles[name].scenes
            for flight in scene.flights
        }
        sfmnn_flights = {
            (scene.date, flight.flight_id)
            for scene in profiles["SFMNN"].scenes
            for flight in scene.flights
        }
        missing = required_flights - sfmnn_flights
        if missing:
            missing_text = ", ".join(
                f"{date}/{flight_id}"
                for date, flight_id in sorted(missing)
            )
            raise ValueError(
                "The selected SIF profiles contain flights that are not "
                "available in the SFMNN profile, so SFMNN-based illumination "
                f"cannot be derived for: {missing_text}"
            )

        derive_illumination_from_sfmnn(
            profiles["SFMNN"],
            required_flights=required_flights,
            export_illumination=EXPORT_ILLUMINATION,
            export_endmember_plots=EXPORT_ENDMEMBER_PLOTS,
        )

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