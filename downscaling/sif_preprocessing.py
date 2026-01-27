from __future__ import annotations

from typing import Dict, List, Tuple

import rioxarray


def load_sif_stack(
    sif_files: Dict[str, "str"],
    sif_o2a_band: int,
    sif_to_sif760_factor: float,
):
    """
    Load SIF rasters:
      - selects band (O2-A)
      - squeezes band dim
      - scales to SIF760 (if factor != 1)
      - reproject_match all to first raster
    Returns: (list_of_DataArrays, list_of_names)
    """
    rasters = []
    names = list(sif_files.keys())

    ref = None
    for i, name in enumerate(names):
        path = sif_files[name]
        da = rioxarray.open_rasterio(path).sel(band=[sif_o2a_band]).squeeze("band", drop=True)
        if sif_to_sif760_factor != 1.0:
            da = da * sif_to_sif760_factor

        if i == 0:
            ref = da
        else:
            da = da.rio.reproject_match(ref)

        rasters.append(da)

    return rasters, names