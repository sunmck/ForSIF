from __future__ import annotations

import numpy as np
import rioxarray


# ---------- Load SIF ----------

def load_sif(
    filepath,
    sif_o2a_band,
    sif_to_sif760_factor,
    sif_scale_factor=1.0,
    nodata_in=-999.0,
):

    sif = (
        rioxarray.open_rasterio(filepath)
        .sel(band=sif_o2a_band, drop=True)
        .astype("float32")
)

    # nodata
    nodata = sif.rio.nodata

    if nodata is None:
        nodata = nodata_in

    sif = sif.where(sif != nodata)

    # undo integer encoding
    scale = float(sif_scale_factor)

    if np.isfinite(scale) and scale not in (0.0, 1.0):
        sif = sif / scale

    # convert to SIF760
    if sif_to_sif760_factor != 1.0:
        sif = sif * float(sif_to_sif760_factor)

    return sif