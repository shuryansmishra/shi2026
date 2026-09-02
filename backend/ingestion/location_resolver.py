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
    import rasterio  # type: ignore[import-not-found,import-untyped]
    from rasterio.transform import from_origin  # type: ignore[import-not-found,import-untyped]
    HAS_RASTERIO_LIBS = True
except ImportError:
    HAS_RASTERIO_LIBS = False



# Predefined high-fidelity fallbacks for popular locations in India
# Useful for offline mode, testing, and sandboxed networks
OFFLINE_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "bbox": [72.8200, 18.9000, 72.9800, 19.2500],
        "name": "Mumbai Metropolitan, Maharashtra, India"
    },
    "delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "bbox": [77.1890, 28.5939, 77.2290, 28.6339],
        "name": "Delhi NCR Region, India"
    },
    "chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "bbox": [80.2000, 12.9800, 80.3200, 13.1500],
        "name": "Chennai Coastal Sector, Tamil Nadu, India"
    },
    "bangalore": {
        "lat": 12.9716,
        "lon": 77.5946,
        "bbox": [77.5746, 12.9516, 77.6146, 12.9916],
        "name": "Bengaluru Urban Corridor, Karnataka, India"
    },
    "bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "bbox": [77.5746, 12.9516, 77.6146, 12.9916],
        "name": "Bengaluru Urban Corridor, Karnataka, India"
    },
    "hardoi": {
        "lat": 27.3828,
        "lon": 80.1287,
        "bbox": [80.1087, 27.3628, 80.1487, 27.4028],
        "name": "Hardoi Farmland District, Uttar Pradesh, India"
    },
    "punjab": {
        "lat": 30.9010,
        "lon": 75.8573,
        "bbox": [75.7500, 30.8000, 75.9500, 31.0000],
        "name": "Punjab Agricultural Belt, India"
    },
    "chilika": {
        "lat": 19.7165,
        "lon": 85.3218,
        "bbox": [85.1500, 19.5500, 85.4500, 19.8500],
        "name": "Chilika Lake Wetland Basin, Odisha, India"
    },
    "hyderabad": {
        "lat": 17.3850,
        "lon": 78.4867,
        "bbox": [78.4667, 17.3650, 78.5067, 17.4050],
        "name": "Hyderabad Tech Hub, Telangana, India"
    },
    "kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "bbox": [88.3000, 22.5000, 88.4200, 22.6500],
        "name": "Kolkata Estuary Zone, West Bengal, India"
    },
    "lucknow": {
        "lat": 26.8467,
        "lon": 80.9462,
        "bbox": [80.9262, 26.8267, 80.9662, 26.8667],
        "name": "Lucknow Capital Zone, Uttar Pradesh, India"
    },
    "uttar pradesh": {
        "lat": 26.8467,
        "lon": 80.9462,
        "bbox": [80.9262, 26.8267, 80.9662, 26.8667],
        "name": "Uttar Pradesh Gangetic Plain, India"
    },
    "up": {
        "lat": 26.8467,
        "lon": 80.9462,
        "bbox": [80.9262, 26.8267, 80.9662, 26.8667],
        "name": "Uttar Pradesh Gangetic Plain, India"
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
            import rasterio  # type: ignore[import-not-found,import-untyped]
            from rasterio.warp import transform_bounds  # type: ignore[import-not-found,import-untyped]
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
    
    # Base terrain seed tied to coordinates
    base_seed_key = f"{lat:.4f}|{lon:.4f}|{bands}|{is_sar}"
    base_seed = int(hashlib.sha256(base_seed_key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(base_seed)

    # Contextual land classification base
    is_coastal = any(k in filename.lower() for k in ["mumbai", "chennai", "chilika", "kolkata"]) or (lon > 80.0 and lat < 20.0)
    is_urban = any(k in filename.lower() for k in ["delhi", "bangalore", "bengaluru", "hyderabad"])

    data = []
    for b_idx in range(bands):
        # Base background texture
        if is_coastal and b_idx == 0:
            band_data = rng.integers(20, 90, (height, width), dtype=np.uint8) # Dark water absorption
        elif is_urban:
            band_data = rng.integers(120, 210, (height, width), dtype=np.uint8) # High urban albedo
        else:
            band_data = rng.integers(60, 160, (height, width), dtype=np.uint8) # Mixed agriculture/canopy

        # Add structured terrain blocks (building grids, crop parcels, waterways)
        for _ in range(8):
            x = rng.integers(10, width - 60)
            y = rng.integers(10, height - 60)
            w = rng.integers(20, 60)
            h = rng.integers(20, 60)
            val = rng.integers(40, 240)
            band_data[y:y+h, x:x+w] = val

        # If T2 (Target observation), inject distinct morphological changes (new built-up or flood extent)
        if variant == "t2":
            t2_rng = np.random.default_rng(base_seed + 999)
            for _ in range(5):
                cx = t2_rng.integers(20, width - 80)
                cy = t2_rng.integers(20, height - 80)
                cw = t2_rng.integers(30, 70)
                ch = t2_rng.integers(30, 70)
                # Significant spectral shift representing new construction / land use shift
                shift_val = t2_rng.integers(180, 255) if b_idx == 0 else t2_rng.integers(10, 80)
                band_data[cy:cy+ch, cx:cx+cw] = shift_val

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
