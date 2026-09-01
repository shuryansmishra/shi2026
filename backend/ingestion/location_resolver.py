"""
SatQuery AI - Location Resolver.

Resolves natural language location names to geographical coordinates,
determines the appropriate satellite analysis task from query text, and
retrieves or generates mock satellite imagery for the target area.
"""
from __future__ import annotations

import os
import json
import hashlib
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from models.schemas import ImageMeta, Modality, TaskType

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO_LIBS = True
except ImportError:
    HAS_RASTERIO_LIBS = False


# Predefined high-fidelity fallbacks for popular locations in India
# Useful for offline mode, testing, and sandboxed networks
OFFLINE_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "hardoi": {
        "lat": 27.3828,
        "lon": 80.1287,
        "bbox": [80.1087, 27.3628, 80.1487, 27.4028],
        "name": "Hardoi, Uttar Pradesh, India"
    },
    "bangalore": {
        "lat": 12.9716,
        "lon": 77.5946,
        "bbox": [77.5746, 12.9516, 77.6146, 12.9916],
        "name": "Bengaluru, Karnataka, India"
    },
    "bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "bbox": [77.5746, 12.9516, 77.6146, 12.9916],
        "name": "Bengaluru, Karnataka, India"
    },
    "delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "bbox": [77.1890, 28.5939, 77.2290, 28.6339],
        "name": "Delhi, India"
    },
    "hyderabad": {
        "lat": 17.3850,
        "lon": 78.4867,
        "bbox": [78.4667, 17.3650, 78.5067, 17.4050],
        "name": "Hyderabad, Telangana, India"
    },
    "uttar pradesh": {
        "lat": 26.8467,
        "lon": 80.9462,
        "bbox": [80.9262, 26.8267, 80.9662, 26.8667],
        "name": "Lucknow, Uttar Pradesh, India"
    },
    "up": {
        "lat": 26.8467,
        "lon": 80.9462,
        "bbox": [80.9262, 26.8267, 80.9662, 26.8667],
        "name": "Lucknow, Uttar Pradesh, India"
    }
}


def geocode(place_name: str) -> Tuple[float, float, List[float], str]:
    """
    Geocodes a place name to (latitude, longitude, bbox_latlon, resolved_name).
    Uses OpenStreetMap's Nominatim with an offline fallback.
    """
    clean_name = place_name.lower().strip()
    
    # Try Nominatim API
    try:
        encoded_query = urllib.parse.quote(place_name)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SatQuery-AI-SIH26167-Agent/1.0 (Google Antigravity)"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                # Nominatim returns bbox as string list: [lat_min, lat_max, lon_min, lon_max]
                nb = [float(x) for x in data[0]["boundingbox"]]
                bbox = [nb[2], nb[0], nb[3], nb[1]] # convert to [min_lon, min_lat, max_lon, max_lat]
                display_name = data[0]["display_name"]
                return lat, lon, bbox, display_name
    except Exception:
        # Fall through to offline cache/lookup
        pass

    # Offline / Predefined fallback logic
    for key, data in OFFLINE_LOCATIONS.items():
        if key in clean_name:
            return data["lat"], data["lon"], data["bbox"], data["name"]

    # Global default: Hardoi, UP
    default_data = OFFLINE_LOCATIONS["hardoi"]
    return (
        default_data["lat"],
        default_data["lon"],
        default_data["bbox"],
        f"{place_name} (Resolved to fallback Hardoi, UP)"
    )


def infer_task_intent(query_text: str) -> TaskType:
    """
    Infers the correct TaskType category from the query text.
    """
    q = query_text.lower()
    
    change_keywords = [
        "change", "compare", "history", "before", "after", "since",
        "last month", "last year", "temporal", "shrink", "grow", "past", "difference"
    ]
    fusion_keywords = [
        "cloud", "monsoon", "rain", "fusion", "fuse", "sar", "radar", "optical", "mixed"
    ]

    if any(k in q for k in change_keywords):
        return TaskType.BI_TEMPORAL_CHANGE
    if any(k in q for k in fusion_keywords):
        return TaskType.CROSS_MODAL_FUSION
        
    return TaskType.SINGLE_IMAGE


def water_level_caveat(query_text: str) -> Optional[str]:
    """
    Checks if a query relates to water level / depth and appends a scientific caveat.
    """
    q = query_text.lower()
    if "level" in q or "depth" in q or "volume" in q:
        return (
            "Note: Passive satellite sensors analyze surface water area (extent). "
            "An accurate estimation of depth, level, or volume additionally requires "
            "bathymetric survey data of the water basin."
        )
    return None


def _build_meta_from_raster(
    file_id: str,
    path: str,
    modality: Modality,
    fallback_bbox: List[float],
    capture_date: str,
) -> ImageMeta:
    """Helper to construct ImageMeta from downloaded GeoTIFF file properties."""
    crs = "EPSG:4326"
    res = 10.0
    bbox = fallback_bbox
    
    if HAS_RASTERIO_LIBS:
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(path) as src:
                crs = str(src.crs) if src.crs else crs
                res = abs(src.transform.a) if src.transform else res
                bounds = src.bounds
                if src.crs and str(src.crs) != "EPSG:4326":
                    bbox = list(transform_bounds(src.crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top))
                else:
                    bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        except Exception:
            pass
            
    return ImageMeta(
        file_id=file_id,
        path=path,
        modality=modality,
        crs=crs,
        resolution_m=res,
        bbox_latlon=bbox,
        capture_date=capture_date,
    )


def fetch_scenes_for_location(
    lat: float,
    lon: float,
    bbox: List[float],
    task_type: TaskType
) -> List[ImageMeta]:
    """
    Retrieves real satellite scenes from Bhoonidhi NRSC if credentials are set,
    otherwise generates mock satellite scenes centered on coordinates.
    Returns a list of ImageMeta ready for routing.
    """
    settings = get_settings()
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
    
    images: List[ImageMeta] = []
    
    # Try downloading real Bhoonidhi scenes if credentials are set
    if settings.BHOONIDHI_USER and settings.BHOONIDHI_PASSWORD:
        try:
            from data_access.bhoonidhi_client import BhoonidhiClient
            client = BhoonidhiClient()
            print(f"[Bhoonidhi] Querying real scenes near lat {lat:.4f}, lon {lon:.4f}...")
            
            if task_type == TaskType.SINGLE_IMAGE:
                q = f"Cartosat optical scene near latitude {lat:.4f} longitude {lon:.4f}"
                results = client.smart_search(q)
                if results:
                    scene_id = results[0]["scene_id"]
                    dest = client.download_scene(scene_id, settings.PROCESSED_DIR)
                    if dest:
                        if os.path.isdir(dest):
                            from pathlib import Path
                            tifs = sorted(Path(dest).glob("*.tif"))
                            if tifs:
                                dest = str(tifs[0])
                        images.append(_build_meta_from_raster(scene_id, dest, Modality.OPTICAL, bbox, "2026-08-20"))
                        print(f"[Bhoonidhi] Successfully downloaded Cartosat optical scene: {scene_id}")
                        
            elif task_type == TaskType.BI_TEMPORAL_CHANGE:
                q = f"Cartosat optical scene near latitude {lat:.4f} longitude {lon:.4f}"
                results = client.smart_search(q)
                if len(results) >= 2:
                    for i, res_item in enumerate(results[:2]):
                        scene_id = res_item["scene_id"]
                        dest = client.download_scene(scene_id, settings.PROCESSED_DIR)
                        if dest:
                            if os.path.isdir(dest):
                                from pathlib import Path
                                tifs = sorted(Path(dest).glob("*.tif"))
                                if tifs:
                                    dest = str(tifs[0])
                            date = "2026-07-20" if i == 0 else "2026-08-20"
                            images.append(_build_meta_from_raster(scene_id, dest, Modality.OPTICAL, bbox, date))
                            print(f"[Bhoonidhi] Successfully downloaded bi-temporal scene {i+1}: {scene_id}")
                            
            else:  # TaskType.CROSS_MODAL_FUSION
                q_opt = f"Cartosat optical scene near latitude {lat:.4f} longitude {lon:.4f}"
                results_opt = client.smart_search(q_opt)
                q_sar = f"RISAT SAR scene near latitude {lat:.4f} longitude {lon:.4f}"
                results_sar = client.smart_search(q_sar)
                
                if results_opt and results_sar:
                    scene_opt = results_opt[0]["scene_id"]
                    dest_opt = client.download_scene(scene_opt, settings.PROCESSED_DIR)
                    if dest_opt:
                        if os.path.isdir(dest_opt):
                            from pathlib import Path
                            tifs = sorted(Path(dest_opt).glob("*.tif"))
                            if tifs:
                                dest_opt = str(tifs[0])
                        images.append(_build_meta_from_raster(scene_opt, dest_opt, Modality.OPTICAL, bbox, "2026-08-20"))
                        
                    scene_sar = results_sar[0]["scene_id"]
                    dest_sar = client.download_scene(scene_sar, settings.PROCESSED_DIR)
                    if dest_sar:
                        if os.path.isdir(dest_sar):
                            from pathlib import Path
                            tifs = sorted(Path(dest_sar).glob("*.tif"))
                            if tifs:
                                dest_sar = str(tifs[0])
                        images.append(_build_meta_from_raster(scene_sar, dest_sar, Modality.SAR, bbox, "2026-08-20"))
                        
                    print(f"[Bhoonidhi] Successfully downloaded fusion pair: {scene_opt} and {scene_sar}")
        except Exception as e:
            print(f"[!] Bhoonidhi live download failed: {e}. Falling back to mock generator...")
            images = []

    # Fallback to generating mock rasters if downloads failed or credentials aren't set
    if not images:
        if task_type == TaskType.SINGLE_IMAGE:
            # Create single optical scene
            file_id = _stable_scene_id("loc_opt", lat, lon, bbox, task_type)
            path = os.path.join(settings.PROCESSED_DIR, f"{file_id}.tif")
            _create_mock_geotiff(path, bands=3, lat=lat, lon=lon, is_sar=False)
            img = ImageMeta(
                file_id=file_id,
                path=path,
                modality=Modality.OPTICAL,
                crs="EPSG:4326",
                resolution_m=10.0,
                bbox_latlon=bbox,
                capture_date="2026-08-20",
            )
            images.append(img)
            
        elif task_type == TaskType.BI_TEMPORAL_CHANGE:
            # Create two optical scenes (T1 and T2)
            file_id_t1 = _stable_scene_id("loc_opt_t1", lat, lon, bbox, task_type)
            path_t1 = os.path.join(settings.PROCESSED_DIR, f"{file_id_t1}.tif")
            _create_mock_geotiff(path_t1, bands=3, lat=lat, lon=lon, is_sar=False, variant="t1")
            
            file_id_t2 = _stable_scene_id("loc_opt_t2", lat, lon, bbox, task_type)
            path_t2 = os.path.join(settings.PROCESSED_DIR, f"{file_id_t2}.tif")
            _create_mock_geotiff(path_t2, bands=3, lat=lat, lon=lon, is_sar=False, variant="t2")
            
            images.append(ImageMeta(
                file_id=file_id_t1,
                path=path_t1,
                modality=Modality.OPTICAL,
                crs="EPSG:4326",
                resolution_m=10.0,
                bbox_latlon=bbox,
                capture_date="2026-07-20",
            ))
            images.append(ImageMeta(
                file_id=file_id_t2,
                path=path_t2,
                modality=Modality.OPTICAL,
                crs="EPSG:4326",
                resolution_m=10.0,
                bbox_latlon=bbox,
                capture_date="2026-08-20",
            ))
            
        else:
            # TaskType.CROSS_MODAL_FUSION (Optical + SAR)
            file_id_opt = _stable_scene_id("loc_fuse_opt", lat, lon, bbox, task_type)
            path_opt = os.path.join(settings.PROCESSED_DIR, f"{file_id_opt}.tif")
            _create_mock_geotiff(path_opt, bands=3, lat=lat, lon=lon, is_sar=False)
            
            file_id_sar = _stable_scene_id("loc_fuse_sar", lat, lon, bbox, task_type)
            path_sar = os.path.join(settings.PROCESSED_DIR, f"{file_id_sar}.tif")
            _create_mock_geotiff(path_sar, bands=1, lat=lat, lon=lon, is_sar=True)
            
            images.append(ImageMeta(
                file_id=file_id_opt,
                path=path_opt,
                modality=Modality.OPTICAL,
                crs="EPSG:4326",
                resolution_m=10.0,
                bbox_latlon=bbox,
                capture_date="2026-08-20",
                cloud_cover_fraction=0.65, # high cloud cover to trigger SAR up-weighting
            ))
            images.append(ImageMeta(
                file_id=file_id_sar,
                path=path_sar,
                modality=Modality.SAR,
                crs="EPSG:4326",
                resolution_m=10.0,
                bbox_latlon=bbox,
                capture_date="2026-08-20",
            ))
        
    return images


def _stable_scene_id(
    prefix: str,
    lat: float,
    lon: float,
    bbox: List[float],
    task_type: TaskType,
) -> str:
    """
    Builds a repeatable mock scene id for location-mode requests.

    The mock vision engines seed their synthetic answers from query text and
    image file ids. Using random ids here made identical location requests
    produce different demo answers, so the id must represent the requested
    scene instead of the request attempt.
    """
    bbox_key = ",".join(f"{value:.6f}" for value in bbox)
    stable_key = f"{task_type.value}|{lat:.6f}|{lon:.6f}|{bbox_key}|{prefix}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def _create_mock_geotiff(
    filename: str,
    bands: int,
    lat: float,
    lon: float,
    is_sar: bool = False,
    variant: str = "default",
) -> None:
    """
    Generates a localized mock GeoTIFF raster using rasterio if available,
    otherwise writes an empty dummy file.
    """
    width, height = 256, 256
    
    if not HAS_RASTERIO_LIBS:
        # Minimal file write fallback
        with open(filename, "wb") as f:
            f.write(b"MOCK_TIFF_DATA")
        return
        
    # Geotransform centered on target coordinates
    transform = from_origin(lon - 0.01, lat + 0.01, 0.0001, 0.0001)
    
    seed_key = f"{lat:.6f}|{lon:.6f}|{bands}|{is_sar}|{variant}"
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)

    data = []
    for _ in range(bands):
        band_data = rng.integers(0, 255, (height, width), dtype=np.uint8)
        # Add basic structure features (e.g. blocks/lines representing land features)
        for _ in range(5):
            x = rng.integers(0, width - 20)
            y = rng.integers(0, height - 20)
            w = rng.integers(10, 40)
            h = rng.integers(10, 40)
            val = rng.integers(0, 255)
            band_data[y:y+h, x:x+w] = val
            
        if is_sar:
            speckle = rng.normal(1.0, 0.15, (height, width))
            band_data = np.clip(band_data * speckle, 0, 255).astype(np.uint8)
            
        data.append(band_data)
        
    data_arr = np.array(data)
    
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(data_arr[i], i + 1)
