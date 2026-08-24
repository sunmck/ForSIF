from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr
import rioxarray

from config.config_downscaling import (
    REFL_SCALE,
    FILL_VALUE,
    RED_RANGE,
    NIR_RANGE,
    VIS_RANGE,
    NIR_FCVI_RANGE,
    NIR_saR2F_RANGE,
    RED_saR2F_RANGE,
    BLUE_saR2F_RANGE,
    PRI_531_RANGE,
    PRI_570_RANGE,
    WBI_NIR1_RANGE,
    WBI_NIR2_RANGE,
)


# ---------- Helper functions ----------

def open_and_scale(filepath):
    cube = rioxarray.open_rasterio(filepath).astype("float32")
    cube = cube.where(
        (cube >= 0)
        & (cube <= REFL_SCALE)
        & (cube != FILL_VALUE)
    )
    return cube / REFL_SCALE


def compute_mean_reflectance(
    cube,
    wavelengths,
    ranges: Dict[str, Tuple[float, float]],
):
    results = {}

    for name, (low, high) in ranges.items():

        idx = [
            i
            for i, wl in enumerate(wavelengths)
            if low <= wl <= high
        ]

        if not idx:
            center = 0.5 * (low + high)
            idx = [
                int(
                    np.argmin(
                        np.abs(np.asarray(wavelengths) - center)
                    )
                )
            ]

        results[name] = cube.isel(band=idx).mean(dim="band")

    return results


# ---------- Calculate indices ----------

def compute_indices(
    filepath,
    wavelengths,
    ndvi_threshold=0.5,
    fcvi_threshold=0.18,
    default_crs=None,
):

    cube = open_and_scale(filepath)

    if cube.rio.crs is None:
        if default_crs is None:
            raise ValueError(f"Raster has no CRS: {filepath}")

        cube = cube.rio.write_crs(default_crs)

    refl = compute_mean_reflectance(
        cube,
        wavelengths,
        {
            "red": RED_RANGE,
            "nir": NIR_RANGE,
            "vis": VIS_RANGE,
            "nir_fcvi": NIR_FCVI_RANGE,
            "nir_saR2F": NIR_saR2F_RANGE,
            "red_saR2F": RED_saR2F_RANGE,
            "blue_saR2F": BLUE_saR2F_RANGE,
            "pri_531": PRI_531_RANGE,
            "pri_570": PRI_570_RANGE,
            "wbi_890_905": WBI_NIR1_RANGE,
            "wbi_955_970": WBI_NIR2_RANGE,
        },
    )

    eps = 1e-6

    # vegetation indices
    ndvi = (
        (refl["nir"] - refl["red"])
        / (refl["nir"] + refl["red"])
    )

    nirv = ndvi * refl["nir"]
    fcvi = refl["nir_fcvi"] - refl["vis"]

    sar2f = (
        refl["nir_saR2F"]
        - 1.4 * refl["red_saR2F"]
        + 0.4 * refl["blue_saR2F"]
    )

    wdrvi = (
        (0.1 * refl["nir"] - refl["red"])
        / (0.1 * refl["nir"] + refl["red"])
    )

    pri_den = refl["pri_531"] + refl["pri_570"]
    pri = xr.where(
        np.abs(pri_den) > eps,
        (refl["pri_531"] - refl["pri_570"]) / pri_den,
        np.nan,
    )

    wbi = xr.where(
        np.abs(refl["wbi_955_970"]) > eps,
        refl["wbi_890_905"] / refl["wbi_955_970"],
        np.nan,
    )

    # fAPAR
    fapar_green = 0.516 * wdrvi + 0.726
    fapar_chl = 0.79 * fapar_green

    # NDVI mask
    mask = ndvi >= ndvi_threshold

    ndvi = ndvi.where(mask)
    pri = pri.where(mask)
    wbi = wbi.where(mask)
    nirv = nirv.where(mask)
    fcvi = fcvi.where(mask)
    sar2f = sar2f.where(mask)
    wdrvi = wdrvi.where(mask)
    fapar_green = fapar_green.where(mask)
    fapar_chl = fapar_chl.where(mask)

    # FCVI validity mask for FCVI-based SIF correction
    fcvi_valid = fcvi.where(fcvi >= fcvi_threshold)

    # escape fraction
    fesc_nirv = nirv / fapar_chl
    fesc_fcvi = fcvi / fapar_chl
    fesc_sar2f = sar2f / fapar_chl

    return {
        "NDVI": ndvi,
        "PRI": pri,
        "WBI": wbi,
        "NIRv": nirv,
        "FCVI": fcvi,
        "FCVI_valid": fcvi_valid,
        "saR2F": sar2f,
        "WDRVI": wdrvi,
        "fAPARgreen": fapar_green,
        "fAPARchl": fapar_chl,
        "fesc_SIF760_NIRv": fesc_nirv,
        "fesc_SIF760_FCVI": fesc_fcvi,
        "fesc_SIF760_saR2F": fesc_sar2f,
    }