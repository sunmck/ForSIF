# ForSIF

**Downscaling Solar-Induced Chlorophyll Fluorescence in Forests**

ForSIF is a Python workflow for processing airborne HyPlant SIF and top-of-canopy (TOC) hyperspectral reflectance data, from the FLUO and DUAL sensors, respectively. The workflow includes cropping the data to an aoi, improving the co-registration based on a high-resolution UAV reference image, deriving vegetation sunlit/shaded fractions and calculating the fluorescence quantum efficiency (FQE) using different vegetation indices (NIRv, FCVI and saR2F). A comparison of different SIF retrieval methods and downscaling indices can also be run.


## Repository structure

```text
ForSIF/
├── config/
│   ├── config_downscaling.py
│   ├── config_formask.py
│   └── config_illumination.py
├── downscaling/
│   ├── illumination/
│   │   └── sunlit_fraction.py
│   ├── compute_downscaling_indices.py
│   ├── compute_fqe.py
│   ├── io.py
│   └── sif_preprocessing.py
├── plots/
│   ├── plots_illumination.py
│   └── plots_sif_fqe_comparison.py
├── 01_run_formask.py
├── 02_run_downscaling.py
├── 03_run_sif_fqe_comparison.py
└── README.md
```

## Scripts

### `config/config_downscaling.py`

Defines the data paths, spectral ranges, PAR values, retrieval-specific settings, treatment polygons and matched SIF–TOC_REFL flight pairs used by the workflow.

### `config/config_formask.py`

Defines the inputs and thresholds used to create the forest-fraction and scaffold-fraction rasters required by the illumination retrieval.

### `config/config_illumination.py`

Defines the sunlit-fraction retrieval settings, including spectral ranges, NDCSI quantiles, minimum forest fraction, endmember selection and the spectral range used for two-endmember unmixing.

It also defines optional illumination-based FQE quality-control thresholds:

- `sunlit_veg_min`: minimum accepted vegetation sunlit fraction,
- `unmix_rmse_max`: maximum accepted spectral-unmixing RMSE.

Both thresholds are `None` by default, so illumination-based FQE masking is disabled until the retrieval has been validated.

### `downscaling/sif_preprocessing.py`

Loads and standardizes individual SIF rasters, including band selection, nodata handling, scaling and conversion to SIF760.

### `downscaling/compute_downscaling_indices.py`

Calculates reflectance-based VIs from individual TOC reflectance flights, including NDVI, NIRv, FCVI, saR2F, PRI, WBI and fAPAR-related quantities.

### `downscaling/compute_fqe.py`

Contains the calculations for leaf-level SIF proxies and FQE:

```text
SIFleaf = π × SIF / fesc with fesc = index/fAPAR
FQE     = π × SIF / (index × PAR)
```

FQE is calculated separately using NIRv, FCVI and saR2F.

### `downscaling/io.py`

Contains small utilities for reading wavelength information and writing raster outputs.

### `01_run_formask.py`

Builds the ancillary fraction rasters used by the illumination retrieval on the shared hyperspectral reference grid:

- `forest_fraction`: fraction of each target pixel occupied by vegetation above the configured nDSM height threshold,
- `scaffold_fraction`: exact polygon–pixel area fraction covered by known scaffold structures.

Run this script before deriving illumination products.

### `downscaling/illumination/sunlit_fraction.py`

Derives a vegetation sunlit fraction from TOC reflectance using scene-specific sunlit and shaded forest endmembers.

The retrieval:

1. uses the forest-fraction raster to define the forest application domain;
2. excludes scaffold-affected pixels from endmember learning and NDCSI scaling;
3. detects a scene-specific red-edge wavelength and calculates the normalized difference canopy shadow index (NDCSI);
4. identifies local sunlit and shaded candidate tails within each treatment plot;
5. builds robust sunlit and shaded spectral endmembers from a spatially distributed, plot-balanced support set;
6. performs constrained two-endmember spectral unmixing,

```text
y = shade + f_sun × (sun - shade),    0 ≤ f_sun ≤ 1
```

and returns the vegetation sunlit fraction together with the spectral fit error.

The main raster products are:

- `f_sun_veg`: vegetation sunlit fraction,
- `rmse`: two-endmember spectral-unmixing RMSE,
- `NDCSI`: normalized difference canopy shadow index.

Treatment polygons are used for endmember learning only. The learned endmembers are then applied to the full forest mask. Scaffold-affected forest pixels remain in the application domain so their RMSE can be inspected as a diagnostic.

### `02_run_downscaling.py`

Main processing script. Illumination, when requested, is derived first and always from the SFMNN DUAL flight mapping. The resulting illumination layers can then be used consistently with any selected SIF retrieval profile for the same date/flight.

For each matched SIF–TOC flight pair the script:

- loads and preprocesses SIF,
- calculates vegetation indices,
- checks that SIF, TOC and any illumination layers are on the same shared grid,
- calculates FQE when PAR is available,
- optionally masks FQE using configured illumination/unmixing QC thresholds,
- exports individual-flight products,
- creates date-level mean products.


### `03_run_sif_fqe_comparison.py`

Reads the individual-flight outputs, summarizes values within the treatment polygons, saves the polygon-level summary table, and creates the comparison figures.

## Running

For the full workflow including illumination retrieval, first create the forest/scaffold fraction rasters:

```bash
python 01_run_formask.py
```

Then configure the desired illumination outputs/QC settings and run the downscaling:

```bash
python 02_run_downscaling.py
```

Finally run the comparison analysis:

```bash
python 03_run_sif_fqe_comparison.py
```

If illumination export, illumination plots and illumination-based FQE QC are all disabled, the forest/scaffold preprocessing step is not required for the standard SIF/FQE workflow.

Output rasters and comparison figures are saved below the `OUT_ROOT` directory defined in `config/config_downscaling.py`.

## Notes

The input `*_coreg_shared.tif` rasters are expected to already share the same spatial grid. The workflow checks this and raises an error if SIF and reflectance grids do not match; illumination layers are checked against the same grid before they are applied to FQE.

Illumination products are derived from SFMNN TOC reflectance only. Therefore, every date/flight processed with illumination enabled must also exist in the SFMNN profile mapping.

The default illumination configuration uses a minimum `forest_fraction` of 0.5, local 85th/15th percentile NDCSI tails for sunlit/shaded candidates, and constrained spectral unmixing over 400–1100 nm. These values can be changed in `config/config_illumination.py`.