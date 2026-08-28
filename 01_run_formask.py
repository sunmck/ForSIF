from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from config.config_downscaling import DEFAULT_RASTER_CRS, NODATA_OUT, get_profiles
from config.config_formask import (
    FOREST_MASK_CONFIG,
    FOREST_MASK_PROFILE,
    FOREST_NDSM_RASTER,
    SCAFFOLD_VECTOR,
    forest_fraction_path,
    scaffold_fraction_path,
    forest_mask_path,
    scaffold_mask_path,
)


def _reference_grid(reference_raster):
    with rasterio.open(reference_raster) as ref:
        crs = ref.crs or DEFAULT_RASTER_CRS
        transform = ref.transform
        shape = (ref.height, ref.width)
        profile = ref.profile.copy()
    return crs, transform, shape, profile


def _write_fraction(out_path, fraction, profile, crs, transform, description, tags):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA_OUT,
        crs=crs,
        transform=transform,
        compress="deflate",
    )

    data = np.where(np.isfinite(fraction), fraction, NODATA_OUT).astype("float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)
        dst.update_tags(**tags)

def build_binary_mask_from_fraction(
    fraction_path,
    out_path,
    threshold,
    *,
    inclusive=True,
    description="binary_mask",
):
    """
    Convert a fractional raster to a uint8 binary mask.

    Values:
        0   = False
        1   = True
        255 = nodata
    """
    with rasterio.open(fraction_path) as src:
        fraction = src.read(1).astype("float32")
        profile = src.profile.copy()

        valid = np.isfinite(fraction)
        if src.nodata is not None:
            valid &= fraction != src.nodata

        data = np.full(fraction.shape, 255, dtype="uint8")

        if inclusive:
            data[valid] = (fraction[valid] >= threshold).astype("uint8")
        else:
            data[valid] = (fraction[valid] > threshold).astype("uint8")

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, description)
            dst.update_tags(
                source_fraction=str(fraction_path),
                binary_threshold=threshold,
                threshold_operator=">=" if inclusive else ">",
            )

def build_forest_fraction(ndsm_path, reference_raster, out_path):
    cfg = FOREST_MASK_CONFIG
    dst_crs, dst_transform, dst_shape, profile = _reference_grid(reference_raster)

    with rasterio.open(ndsm_path) as src:
        if src.crs is None:
            raise ValueError(f"nDSM has no CRS: {ndsm_path}")

        z = src.read(1).astype("float32")
        valid = np.isfinite(z)
        if src.nodata is not None:
            valid &= z != src.nodata

        high = (valid & (z >= cfg.height_threshold_m)).astype("float32")
        valid = valid.astype("float32")

        high_avg = np.full(dst_shape, np.nan, dtype="float32")
        valid_avg = np.full(dst_shape, np.nan, dtype="float32")

        common = dict(
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            dst_nodata=np.nan,
            init_dest_nodata=True,
        )
        reproject(high, high_avg, **common)
        reproject(valid, valid_avg, **common)

    fraction = np.full(dst_shape, np.nan, dtype="float32")
    ok = np.isfinite(valid_avg) & (valid_avg >= cfg.valid_fraction_min)
    fraction[ok] = high_avg[ok] / np.maximum(valid_avg[ok], 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)

    _write_fraction(
        out_path,
        fraction,
        profile,
        dst_crs,
        dst_transform,
        "forest_fraction",
        {
            "source_ndsm": str(ndsm_path),
            "height_threshold_m": cfg.height_threshold_m,
            "valid_fraction_min": cfg.valid_fraction_min,
            "reference_raster": str(reference_raster),
        },
    )


def build_scaffold_fraction(scaffold_vector, reference_raster, out_path):
    """Exact scaffold-polygon area fraction for every 2 m reference pixel."""
    dst_crs, transform, shape, profile = _reference_grid(reference_raster)
    height, width = shape

    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError("Rotated reference grids are not supported for exact pixel-area intersection.")

    gdf = gpd.read_file(scaffold_vector)
    if gdf.crs is None:
        raise ValueError(f"Scaffold vector has no CRS: {scaffold_vector}")

    gdf = gdf.to_crs(dst_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    fraction = np.zeros(shape, dtype="float32")

    if not gdf.empty:
        scaffold = gdf.geometry.union_all()
        if scaffold.is_empty or scaffold.area <= 0:
            raise ValueError("Scaffold vector contains no polygon area.")

        raster_left = transform.c
        raster_top = transform.f
        raster_right = transform.c + width * transform.a
        raster_bottom = transform.f + height * transform.e
        raster_bounds = (
            min(raster_left, raster_right),
            min(raster_bottom, raster_top),
            max(raster_left, raster_right),
            max(raster_bottom, raster_top),
        )

        overlap = scaffold.intersection(box(*raster_bounds))
        if not overlap.is_empty and overlap.area > 0:
            raw_window = from_bounds(*overlap.bounds, transform=transform)
            c0 = max(0, int(np.floor(raw_window.col_off)))
            r0 = max(0, int(np.floor(raw_window.row_off)))
            c1 = min(width, int(np.ceil(raw_window.col_off + raw_window.width)))
            r1 = min(height, int(np.ceil(raw_window.row_off + raw_window.height)))
            window = Window(c0, r0, c1 - c0, r1 - r0)

            pixel_area = abs(transform.a * transform.e - transform.b * transform.d)

            for row in range(int(window.row_off), int(window.row_off + window.height)):
                y0 = transform.f + row * transform.e
                y1 = transform.f + (row + 1) * transform.e
                bottom, top = sorted((y0, y1))

                for col in range(int(window.col_off), int(window.col_off + window.width)):
                    x0 = transform.c + col * transform.a
                    x1 = transform.c + (col + 1) * transform.a
                    left, right = sorted((x0, x1))

                    pixel = box(left, bottom, right, top)
                    if not overlap.intersects(pixel):
                        continue

                    area = overlap.intersection(pixel).area
                    if area > 0:
                        fraction[row, col] = min(1.0, area / pixel_area)

    _write_fraction(
        out_path,
        fraction,
        profile,
        dst_crs,
        transform,
        "scaffold_fraction",
        {
            "source_scaffold_vector": str(scaffold_vector),
            "method": "exact polygon-pixel area intersection",
            "reference_raster": str(reference_raster),
        },
    )


def main():
    profile = get_profiles()[FOREST_MASK_PROFILE]
    reference = profile.scenes[0].flights[0].toc_refl_file

    forest_out = forest_fraction_path()
    scaffold_out = scaffold_fraction_path()
    forest_mask_out = forest_mask_path()
    scaffold_mask_out = scaffold_mask_path()

    # ------------------------------------------------------------------
    # Fractional forest raster
    # ------------------------------------------------------------------
    if FOREST_MASK_CONFIG.overwrite or not forest_out.exists():
        print(f"\n=== Forest fraction from nDSM | reference: {reference} ===")
        build_forest_fraction(FOREST_NDSM_RASTER, reference, forest_out)
        print(f"Saved: {forest_out}")
    else:
        print(f"Forest fraction already exists: {forest_out}")

    # ------------------------------------------------------------------
    # Fractional scaffold raster
    # ------------------------------------------------------------------
    if FOREST_MASK_CONFIG.overwrite or not scaffold_out.exists():
        print(f"\n=== Scaffold fraction from polygons | reference: {reference} ===")
        build_scaffold_fraction(SCAFFOLD_VECTOR, reference, scaffold_out)
        print(f"Saved: {scaffold_out}")
    else:
        print(f"Scaffold fraction already exists: {scaffold_out}")

    # ------------------------------------------------------------------
    # Binary forest mask: forest_fraction >= forest_binary_fraction_threshold
    # ------------------------------------------------------------------
    if FOREST_MASK_CONFIG.overwrite or not forest_mask_out.exists():
        print(
            "\n=== Binary forest mask "
            f"| fraction >= {FOREST_MASK_CONFIG.forest_binary_fraction_threshold} ==="
        )

        build_binary_mask_from_fraction(
            forest_out,
            forest_mask_out,
            threshold=FOREST_MASK_CONFIG.forest_binary_fraction_threshold,
            inclusive=True,
            description="forest_mask",
        )

        print(f"Saved: {forest_mask_out}")
    else:
        print(f"Forest mask already exists: {forest_mask_out}")

    # ------------------------------------------------------------------
    # Binary scaffold mask: scaffold_fraction > scaffold_binary_fraction_threshold
    # ------------------------------------------------------------------
    if FOREST_MASK_CONFIG.overwrite or not scaffold_mask_out.exists():
        print(
            "\n=== Binary scaffold mask "
            f"| fraction > {FOREST_MASK_CONFIG.scaffold_binary_fraction_threshold} ==="
        )

        build_binary_mask_from_fraction(
            scaffold_out,
            scaffold_mask_out,
            threshold=FOREST_MASK_CONFIG.scaffold_binary_fraction_threshold,
            inclusive=False,
            description="scaffold_mask",
        )

        print(f"Saved: {scaffold_mask_out}")
    else:
        print(f"Scaffold mask already exists: {scaffold_mask_out}")

    print()
    return 0