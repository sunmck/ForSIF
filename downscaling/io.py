from pathlib import Path
from typing import List

import numpy as np
import rioxarray
from spectral.io import envi


def load_wavelengths(hdr_path: Path) -> List[float]:
    hdr = envi.read_envi_header(str(hdr_path))
    return [float(w) for w in hdr["wavelength"]]


def open_raster(path: Path):
    return rioxarray.open_rasterio(path)


def save_tif(da, outpath: Path, nodata_out: float = -9999.0, compress: str = "DEFLATE"):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    da2 = da.copy()

    # --- FIX: make rioxarray band descriptions consistent ---
    ln = da2.attrs.get("long_name", None)

    # how many bands are we writing?
    nbands = da2.sizes["band"] if "band" in da2.dims else 1

    if ln is not None:
        # rioxarray expects either a string OR a list with length == nbands
        if isinstance(ln, (list, tuple)):
            if len(ln) != nbands:
                # simplest: drop it
                da2.attrs.pop("long_name", None)
            elif nbands == 1:
                # normalize [name] -> "name"
                da2.attrs["long_name"] = str(ln[0])
        else:
            # scalar long_name is fine for single band; for multiband, expand if you want
            if nbands > 1:
                da2.attrs["long_name"] = [str(ln)] * nbands

    da2 = da2.where(np.isfinite(da2), nodata_out)
    da2 = da2.rio.write_nodata(nodata_out)

    da2.rio.to_raster(str(outpath), compress=compress)
