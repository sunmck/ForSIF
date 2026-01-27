# downscaling/geo.py
from __future__ import annotations

import geopandas as gpd


def load_vectors(crowns_shp, treatment_areas_shp, ref_raster):
    crowns = gpd.read_file(crowns_shp).to_crs(ref_raster.rio.crs)
    treatment_areas = gpd.read_file(treatment_areas_shp).to_crs(ref_raster.rio.crs)
    return crowns, treatment_areas
