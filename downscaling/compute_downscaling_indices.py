from __future__ import annotations

from typing import Dict, Tuple
from weakref import ref

import numpy as np
import xarray as xr
import rioxarray

from config.config_downscaling import (
    REFL_SCALE, FILL_VALUE,
    RED_RANGE, NIR_RANGE, VIS_RANGE,
    NIR_FCVI_RANGE,
    MTCI_NIR_RANGE, MTCI_REDEDGE_RANGE, MTCI_RED_RANGE,
    NIR_saR2F_RANGE, RED_saR2F_RANGE, BLUE_saR2F_RANGE,
    PRI_531_RANGE, PRI_570_RANGE,
    WBI_NIR1_RANGE, WBI_NIR2_RANGE,
)


def open_and_scale(filepath):
    cube = rioxarray.open_rasterio(filepath).astype("float32")
    cube = cube.where((cube >= 0) & (cube <= REFL_SCALE) & (cube != FILL_VALUE))
    cube = cube / REFL_SCALE
    return cube


def compute_mean_reflectance(cube, wavelengths, ranges: Dict[str, Tuple[float, float]]):
    results = {}
    for name, (low, high) in ranges.items():
        idx = [i for i, wl in enumerate(wavelengths) if low <= wl <= high]
        if len(idx) == 0:
            center = 0.5 * (low + high)
            nearest = int(np.argmin(np.abs(np.array(wavelengths) - center)))
            idx = [nearest]
        da = cube.isel(band=idx).mean(dim="band")
        results[name] = da
    return results


def process_month_indices(
    raster_files,
    wavelengths,
    ndvi_threshold=0.5,
    fcvi_threshold=0.18,
    default_crs=None,
):
    """
    - compute Rred, Rnir, Rvis, FCVI, NIRv, saR2F, etc.
    - compute fesc (NIRv / FCVI / saR2F) / fAPARchl
    - align scenes via reproject_match to the first scene
    - apply NDVI mask
    - average across scenes
    """
    red_list, nir_list, vis_list, nir_fcvi_list = [], [], [], []
    ndvi_list, nirv_list, fcvi_list, saR2F_list, wdrvi_list = [], [], [], [], []
    pri_list, wbi_list = [], []
    fapar_green_list, fapar_chl_list = [], []
    fesc_sif760_fcvi_list, fesc_sif760_nirv_list, fesc_saR2F_list = [], [], []

    ref = None
    eps = 1e-6 # make sure not to divide by 0

    for i, (label, filepath) in enumerate(raster_files.items()):
        cube = open_and_scale(filepath)

        # Ensure CRS exists on the source cube
        if cube.rio.crs is None:
            if default_crs is None:
                raise ValueError(
                    f"Raster has no CRS: {filepath}. "
                    f"Set DEFAULT_RASTER_CRS in config and pass it as default_crs."
                )
            cube = cube.rio.write_crs(default_crs)

        # --- Compute mean reflectances from wavelength ranges ---
        refl = compute_mean_reflectance(
            cube,
            wavelengths,
            {
                "red": RED_RANGE,
                "nir": NIR_RANGE,
                "vis": VIS_RANGE,
                "nir_fcvi": NIR_FCVI_RANGE,
                "mtci_nir": MTCI_NIR_RANGE,
                "mtci_re": MTCI_REDEDGE_RANGE,
                "mtci_red": MTCI_RED_RANGE,
                "nir_saR2F": NIR_saR2F_RANGE,
                "red_saR2F": RED_saR2F_RANGE,
                "blue_saR2F": BLUE_saR2F_RANGE,
                "pri_531": PRI_531_RANGE,
                "pri_570": PRI_570_RANGE,
                "wbi_890_905": WBI_NIR1_RANGE,
                "wbi_955_970": WBI_NIR2_RANGE,
            },
        )

        # --- Replace invalid/fill values with NaN ---
        #for key in refl:
        #    refl[key] = refl[key].where(refl[key] > 0)

        # --- Indices ---
        ndvi = (refl["nir"] - refl["red"]) / (refl["nir"] + refl["red"])
        nirv = ndvi * refl["nir"]
        fcvi = refl["nir_fcvi"] - refl["vis"]
        saR2F = refl["nir_saR2F"] - 1.4 * refl["red_saR2F"] + 0.4 * refl["blue_saR2F"]
        wdrvi = (0.1 * refl["nir"] - refl["red"]) / (0.1 * refl["nir"] + refl["red"])

        # flipped PRI direction
        pri_den = refl["pri_531"] + refl["pri_570"]
        pri = xr.where(np.abs(pri_den) > eps,
                    (refl["pri_531"] - refl["pri_570"]) / pri_den,
                    np.nan)

        # flipped WBI direction
        wbi = xr.where(np.abs(refl["wbi_955_970"]) > eps,
                       refl["wbi_890_905"] / refl["wbi_955_970"],
                       np.nan)

        # --- fAPAR ---
        fapar_green = 0.516 * wdrvi + 0.726
        fapar_chl = 0.79 * fapar_green

        # --- fesc ---
        fesc_sif760_nirv = xr.where(ndvi >= ndvi_threshold, nirv / fapar_chl, np.nan)
        fesc_sif760_fcvi = xr.where(ndvi >= ndvi_threshold, fcvi / fapar_chl, np.nan)
        fesc_sif760_saR2F = xr.where(ndvi >= ndvi_threshold, saR2F / fapar_chl, np.nan)

        # Re-attach CRS so rioxarray can reproject.
        crs = cube.rio.crs
        for arr in [
            refl["red"], refl["nir"], refl["vis"], refl["nir_fcvi"],
            refl["mtci_nir"], refl["mtci_re"], refl["mtci_red"],
            refl["nir_saR2F"], refl["red_saR2F"], refl["blue_saR2F"],
            refl["pri_531"], refl["pri_570"],
            refl["wbi_890_905"], refl["wbi_955_970"],
            ndvi, nirv, fcvi, saR2F, wdrvi, pri, wbi,
            fapar_green, fapar_chl,
            fesc_sif760_nirv, fesc_sif760_fcvi, fesc_sif760_saR2F,
        ]:
            if arr.rio.crs is None:
                arr.rio.write_crs(crs, inplace=True)

        # --- Align scenes ---
        if i == 0:
            ref = refl["red"]
            if ref.rio.crs is None:
                ref = ref.rio.write_crs(crs)
        else:
            for key in refl:
                refl[key] = refl[key].rio.reproject_match(ref)

            ndvi = ndvi.rio.reproject_match(ref)
            pri = pri.rio.reproject_match(ref)
            wbi = wbi.rio.reproject_match(ref)
            nirv = nirv.rio.reproject_match(ref)
            fcvi = fcvi.rio.reproject_match(ref)
            saR2F = saR2F.rio.reproject_match(ref)
            wdrvi = wdrvi.rio.reproject_match(ref)
            fapar_green = fapar_green.rio.reproject_match(ref)
            fapar_chl = fapar_chl.rio.reproject_match(ref)
            fesc_sif760_nirv = fesc_sif760_nirv.rio.reproject_match(ref)
            fesc_sif760_fcvi = fesc_sif760_fcvi.rio.reproject_match(ref)
            fesc_sif760_saR2F = fesc_sif760_saR2F.rio.reproject_match(ref)

        # --- Apply NDVI mask ---
        mask = ndvi >= ndvi_threshold
        for key in refl:
            refl[key] = refl[key].where(mask)

        ndvi = ndvi.where(mask)
        pri = pri.where(mask)
        wbi = wbi.where(mask)
        nirv = nirv.where(mask)
        fcvi = fcvi.where(mask)
        saR2F = saR2F.where(mask)
        wdrvi = wdrvi.where(mask)
        fapar_green = fapar_green.where(mask)
        fapar_chl = fapar_chl.where(mask)
        fesc_sif760_nirv = fesc_sif760_nirv.where(mask)
        fesc_sif760_fcvi = fesc_sif760_fcvi.where(mask)
        fesc_sif760_saR2F = fesc_sif760_saR2F.where(mask)

        # --- Append ---
        red_list.append(refl["red"])
        nir_list.append(refl["nir"])
        vis_list.append(refl["vis"])
        pri_list.append(pri)
        wbi_list.append(wbi)
        nir_fcvi_list.append(refl["nir_fcvi"])
        ndvi_list.append(ndvi)
        nirv_list.append(nirv)
        fcvi_list.append(fcvi)
        saR2F_list.append(saR2F)
        wdrvi_list.append(wdrvi)
        fapar_green_list.append(fapar_green)
        fapar_chl_list.append(fapar_chl)
        fesc_sif760_nirv_list.append(fesc_sif760_nirv)
        fesc_sif760_fcvi_list.append(fesc_sif760_fcvi)
        fesc_saR2F_list.append(fesc_sif760_saR2F)

    # --- Mean across scenes ---
    return {
        "Rred": xr.concat(red_list, dim="scene").mean(dim="scene"),
        "Rnir": xr.concat(nir_list, dim="scene").mean(dim="scene"),
        "Rvis": xr.concat(vis_list, dim="scene").mean(dim="scene"),
        "PRI": xr.concat(pri_list, dim="scene").mean(dim="scene"),
        "WBI": xr.concat(wbi_list, dim="scene").mean(dim="scene"),
        "Rnir_fcvi": xr.concat(nir_fcvi_list, dim="scene").mean(dim="scene"),
        "NDVI": xr.concat(ndvi_list, dim="scene").mean(dim="scene"),
        "NIRv": xr.concat(nirv_list, dim="scene").mean(dim="scene"),
        "FCVI": xr.concat(fcvi_list, dim="scene").mean(dim="scene"),
        "saR2F": xr.concat(saR2F_list, dim="scene").mean(dim="scene"),
        "WDRVI": xr.concat(wdrvi_list, dim="scene").mean(dim="scene"),
        "fAPARgreen": xr.concat(fapar_green_list, dim="scene").mean(dim="scene"),
        "fAPARchl": xr.concat(fapar_chl_list, dim="scene").mean(dim="scene"),
        "fesc_SIF760_NIRv": xr.concat(fesc_sif760_nirv_list, dim="scene").mean(dim="scene"),
        "fesc_SIF760_FCVI": xr.concat(fesc_sif760_fcvi_list, dim="scene").mean(dim="scene"),
        "fesc_SIF760_saR2F": xr.concat(fesc_saR2F_list, dim="scene").mean(dim="scene"),
    }
