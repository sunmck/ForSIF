from __future__ import annotations

from typing import Dict, List, Tuple

import rioxarray
import numpy as np


def load_sif_stack(
    sif_files: Dict[str, "str"],
    sif_o2a_band: int,
    sif_to_sif760_factor: float,
    *,
    sif_scale_factor: float = 1.0,
    nodata_in: float = -999.0,
):
    rasters = []
    names = list(sif_files.keys())

    ref = None
    for i, name in enumerate(names):
        path = sif_files[name]

        da = (
            rioxarray.open_rasterio(path)
            .sel(band=[sif_o2a_band])
            .squeeze("band", drop=True)
            .astype("float32")
        )

        # --- NODATA handling ---
        nd = da.rio.nodata
        if nd is None:
            nd = float(nodata_in)

        # Turn nodata sentinel into NaN (keeps real zeros!)
        da = da.where(da != nd)

        # Undo integer encoding
        if sif_scale_factor is not None:
            sf = float(sif_scale_factor)
            if np.isfinite(sf) and sf not in (0.0, 1.0):
                da = da / sf

        # Convert to SIF760
        if sif_to_sif760_factor != 1.0:
            da = da * float(sif_to_sif760_factor)

        if i == 0:
            ref = da
        else:
            da = da.rio.reproject_match(ref)

        rasters.append(da)

    return rasters, names