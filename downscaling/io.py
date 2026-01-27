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
    """
    Save DataArray as GeoTIFF with a numeric nodata (more compatible than NaN).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    da2 = da.copy()
    da2 = da2.where(np.isfinite(da2), nodata_out)
    da2 = da2.rio.write_nodata(nodata_out)

    da2.rio.to_raster(str(outpath), compress=compress)
