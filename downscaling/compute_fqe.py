from __future__ import annotations

import numpy as np


# ---------- SIFleaf ----------

def compute_sifleaf(sif, fesc):
    return (np.pi * sif) / fesc


# ---------- FQE ----------

def compute_fqe(sif, index, par_mW_m2):
    return (np.pi * sif) / (index * par_mW_m2)