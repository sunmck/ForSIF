from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr


def compute_sifleaf_stack(sif_stack: List, fesc_da):
    """
    SIFleaf = (pi * SIF) / fesc
    """
    out = []
    for sif in sif_stack:
        out.append((np.pi * sif) / fesc_da)
    return out


def compute_fqe_stack(sif_stack: List, index_da, par_mW_m2: float):
    """
    FQE = (pi * SIF) / (index * PAR)
    where index is NIRv, FCVI or saR2F
    """
    out = []
    for sif in sif_stack:
        out.append((np.pi * sif) / (index_da * par_mW_m2))
    return out


def mean_stack(stack: List):
    return xr.concat(stack, dim="scene").mean(dim="scene")