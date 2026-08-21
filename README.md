# ForSIF

**Downscaling Solar-Induced Chlorophyll Fluorescence in Forests**

ForSIF is a Python workflow for processing airborne HyPlant SIF and top-of-canopy reflectance data, calculating fluorescence quantum efficiency (FQE), and comparing different SIF retrieval methods and downscaling indices.

The workflow processes matched SIF and reflectance flight lines individually before calculating date-level averages. This allows flight-to-flight variability to be retained and compared across iFLD, SFM and SFMNN, while FQE can be evaluated using NIRv, FCVI and saR2F.

## Repository structure

```text
ForSIF/
├── config/
│   └── config_downscaling.py
├── downscaling/
│   ├── compute_downscaling_indices.py
│   ├── compute_fqe.py
│   ├── io.py
│   └── sif_preprocessing.py
├── plots/
│   └── plots_sif_fqe_comparison.py
├── run_downscaling.py
├── run_sif_fqe_comparison.py
└── README.md
```

## Scripts

### `config/config_downscaling.py`

Defines the data paths, spectral ranges, PAR values, retrieval-specific settings, treatment polygons, and matched SIF–TOC flight pairs used by the workflow.

### `downscaling/sif_preprocessing.py`

Loads and standardizes individual SIF rasters, including band selection, nodata handling, scaling, and conversion to SIF760.

### `downscaling/compute_downscaling_indices.py`

Calculates reflectance-based vegetation indices from individual TOC reflectance flights, including NDVI, NIRv, FCVI, saR2F, PRI, WBI and fAPAR-related quantities.

### `downscaling/compute_fqe.py`

Contains the calculations for leaf-level SIF proxies and FQE:

```text
SIFleaf = π × SIF / fesc
FQE     = π × SIF / (index × PAR)
```

FQE is calculated separately using NIRv, FCVI and saR2F.

### `downscaling/io.py`

Contains small utilities for reading wavelength information and writing raster outputs.

### `run_downscaling.py`

Main processing script. For each matched SIF–TOC flight pair it:

- loads and preprocesses SIF,
- calculates vegetation indices,
- checks that SIF and TOC are on the same shared grid,
- calculates FQE when PAR is available,
- exports individual-flight products,
- creates date-level mean products.

### `plots/plots_sif_fqe_comparison.py`

Creates the publication-style comparison figures for:

- between-flight variability of iFLD, SFM and SFMNN,
- treatment-specific SIF variability,
- sensitivity of FQE to NIRv, FCVI and saR2F.

### `run_sif_fqe_comparison.py`

Reads the individual-flight outputs, summarizes values within the treatment polygons, saves the polygon-level summary table, and creates the comparison figures.

## Running

First run the downscaling:

```bash
python run_downscaling.py
```

Then run the comparison analysis:

```bash
python run_sif_fqe_comparison.py
```

Output rasters and comparison figures are saved below the `OUT_ROOT` directory defined in `config/config_downscaling.py`.

## Notes

The input `*_coreg_shared.tif` rasters are expected to already share the same spatial grid. The workflow checks this and raises an error if SIF and reflectance grids do not match.

For the 2026 SFMNN flights, SIF and vegetation indices can be processed, but FQE is skipped until PAR data are available.
