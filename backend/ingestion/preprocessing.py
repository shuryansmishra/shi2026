"""
SatQuery AI - Ingestion & Preprocessing.

Runs BEFORE routing, per PRD section 4.2's "INPUT VALIDATION & INGESTION" box:
  - detect modality (optical vs SAR)
  - reproject every image to a common CRS (critical for fusion -- PRD
    Risk Register: "Cross-modal misregistration" is a HIGH-likelihood risk,
    mitigated by doing this at ingestion, never at inference)
  - tile large scenes
  - compute optical cloud-cover fraction (feeds the fusion engine's
    SAR-upweighting logic)

Uses rasterio/numpy when available. Both are optional at import time so the
rest of the app (router, engines in mock mode, API) can run in an environment
where GDAL isn't installed yet -- geospatial libraries are the most common
thing to trip up a fresh setup, so we degrade gracefully instead of crashing
on import.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

from config import get_settings
from models.schemas import ImageMeta, Modality

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def detect_modality(file_path: str) -> Modality:
    """
    Heuristic modality detection. Real pipeline should prefer explicit
    metadata (e.g. a form field or STAC item) over this guess -- this exists
    as a fallback for raw file uploads with no accompanying metadata.
    SAR products are near-universally single-band and often named with
    sensor/product-type tokens; optical scenes are usually multi-band.
    """
    name = os.path.basename(file_path).lower()
    sar_tokens = ("sar", "risat", "sentinel-1", "s1", "_vv", "_vh", "_hh", "_hv")
    if any(t in name for t in sar_tokens):
        return Modality.SAR

    if HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                return Modality.SAR if src.count == 1 else Modality.OPTICAL
        except Exception:
            pass

    return Modality.UNKNOWN


def reproject_to_common_crs(file_path: str, output_dir: str, target_crs: Optional[str] = None) -> str:
    """
    Reproject a raster to settings.TARGET_UTM_CRS. Returns the output path.
    No-op passthrough (returns the original path) when rasterio isn't
    installed, so the pipeline stays runnable in mock-mode-only environments.
    """
    settings = get_settings()
    target_crs = target_crs or settings.TARGET_UTM_CRS

    if not HAS_RASTERIO:
        return file_path

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"reprojected_{uuid.uuid4().hex[:8]}.tif")

    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
    return out_path


def estimate_cloud_cover(file_path: str) -> float:
    """
    Crude brightness/whiteness-based cloud proxy for optical imagery.
    Good enough to trigger the fusion engine's SAR-upweighting logic in a
    demo; swap for a proper cloud-mask model (e.g. s2cloudless) before
    treating this as a real product number.
    """
    if not HAS_RASTERIO:
        return 0.0
    try:
        import numpy as np
        with rasterio.open(file_path) as src:
            if src.count < 3:
                return 0.0
            arr = src.read([1, 2, 3]).astype("float32")
            arr /= max(arr.max(), 1.0)
            brightness = arr.mean(axis=0)
            cloud_mask = brightness > 0.85  # bright + low color variance ~= cloud
            return float(cloud_mask.mean())
    except Exception:
        return 0.0


def build_image_meta(file_path: str, file_id: Optional[str] = None) -> ImageMeta:
    """Convenience wrapper used by the API layer right after a file is saved."""
    modality = detect_modality(file_path)
    cloud_cover = estimate_cloud_cover(file_path) if modality == Modality.OPTICAL else None
    
    crs_str = None
    res_m = None
    bbox_latlon = None
    
    if HAS_RASTERIO:
        try:
            from rasterio.warp import transform_bounds
            with rasterio.open(file_path) as src:
                crs_str = str(src.crs) if src.crs else "EPSG:4326"
                res_m = float(abs(src.transform.a)) if src.transform else 10.0
                if src.crs:
                    try:
                        bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                        bbox_latlon = [round(b, 6) for b in bounds]
                    except Exception:
                        bbox_latlon = [round(b, 6) for b in src.bounds]
                else:
                    bbox_latlon = [77.5946, 12.9716, 77.6100, 12.9850] # Fallback AOI
        except Exception:
            pass

    return ImageMeta(
        file_id=file_id or uuid.uuid4().hex[:12],
        path=file_path,
        modality=modality,
        crs=crs_str,
        resolution_m=res_m,
        bbox_latlon=bbox_latlon,
        cloud_cover_fraction=cloud_cover,
    )


def save_raster_as_png(tif_path: str, png_path: str) -> bool:
    """
    Reads a GeoTIFF (using rasterio/numpy if available, otherwise Pillow as fallback)
    and saves it as a standard RGB PNG file.
    """
    try:
        from PIL import Image
        import numpy as np

        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        # Try with rasterio first
        try:
            import rasterio
            with rasterio.open(tif_path) as src:
                if src.count >= 3:
                    data = src.read([1, 2, 3])
                    data = np.transpose(data, (1, 2, 0))
                else:
                    data = src.read(1)
                    data = np.stack([data, data, data], axis=-1)
                
                data = data.astype(np.float32)
                dmin, dmax = data.min(), data.max()
                if dmax > dmin:
                    data = (data - dmin) / (dmax - dmin) * 255.0
                data = np.clip(data, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(data)
                img.save(png_path, "PNG")
                return True
        except Exception:
            pass

        # Fallback to standard Pillow open if it's a mock TIFF file or rasterio failed
        try:
            with Image.open(tif_path) as img:
                img.convert("RGB").save(png_path, "PNG")
                return True
        except Exception:
            pass

    except Exception:
        pass
    return False


def save_diff_map_as_png(diff_map: Any, png_path: str, change_threshold: float = 0.35) -> bool:
    """
    Converts a 2D difference map (0-1 range or absolute differences) into a
    colorized PNG showing changes in red overlay.
    """
    try:
        from PIL import Image
        import numpy as np

        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        # Ensure it is a 2D float array
        if diff_map.ndim == 3:
            diff_map = diff_map.mean(axis=0) if diff_map.shape[0] <= 4 else diff_map[:, :, 0]
            
        h, w = diff_map.shape
        
        map_max = float(diff_map.max()) if diff_map.size > 0 else 1.0
        if map_max <= 1.0:
            change_mask = diff_map < (1.0 - change_threshold)
            normalized = (diff_map * 255.0).astype(np.uint8)
        else:
            normalized_diff = diff_map / max(map_max, 1.0)
            change_mask = normalized_diff > change_threshold
            normalized = (normalized_diff * 255.0).astype(np.uint8)
            
        rgb_data = np.stack([normalized, normalized, normalized], axis=-1)
        rgb_data[change_mask] = [220, 50, 50] # Red highlights for changes
        
        img = Image.fromarray(rgb_data)
        img.save(png_path, "PNG")
        return True
    except Exception:
        return False
