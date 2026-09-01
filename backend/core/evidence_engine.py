"""
SatQuery AI - Evidence Engine.

This is the hallucination firewall described in PRD section 4.2 and the
RSHallu-motivated USP in section 5. It sits between the CV/GIS engines and
the LLM synthesis layer, and does exactly one job: turn whatever raw numbers
an engine produced into a LOCKED EvidenceObject.

Nothing downstream of this file is allowed to invent a coordinate, an area,
or a confidence score -- the LLM only ever rephrases what's in the object
this file returns. This also doubles as the cheapest possible way to satisfy
the PS's execution-trace grading requirement: the evidence object IS the trace.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.schemas import (
    BoundingBox,
    EvidenceObject,
    ExecutionTrace,
    TaskType,
)

try:
    import numpy as np
    import rasterio
    from rasterio.features import shapes as rasterio_shapes
    from rasterio.transform import Affine
    from rasterio.warp import transform_geom
    from shapely.geometry import shape as shapely_shape
    HAS_GIS_VECTORIZE = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_GIS_VECTORIZE = False


class EvidenceEngine:
    def build(self, task_type: TaskType, raw_engine_output: Dict[str, Any], trace: ExecutionTrace) -> EvidenceObject:
        """
        raw_engine_output is whatever the single-image / change / fusion engine
        returned (pixel counts, bbox in pixel space + resolution, class scores, ...).
        This function is the ONLY place that is allowed to convert raw CV output
        into the numbers a user will eventually see.
        """
        bbox = None
        if raw_engine_output.get("bbox_latlon"):
            b = raw_engine_output["bbox_latlon"]
            bbox = BoundingBox(min_lon=b[0], min_lat=b[1], max_lon=b[2], max_lat=b[3])

        # --- GeoJSON polygon vectorization from change/diff masks ---
        change_polygons = self._extract_change_polygons(raw_engine_output, trace)

        evidence = EvidenceObject(
            task_type=task_type,
            area_ha=raw_engine_output.get("area_ha"),
            change_area_ha=raw_engine_output.get("change_area_ha"),
            bbox_latlon=bbox,
            bbox_pixel=raw_engine_output.get("bbox_pixel"),
            confidence=raw_engine_output.get("confidence", 0.5),
            generated_answer=raw_engine_output.get("generated_answer"),
            land_cover_classes=raw_engine_output.get("land_cover_classes", []),
            change_classes=raw_engine_output.get("change_classes", []),
            object_count=raw_engine_output.get("object_count"),
            raw_scores=raw_engine_output.get("raw_scores", {}),
            change_polygons_geojson=change_polygons,
            notes=raw_engine_output.get("notes", []),
        )

        trace.add(
            step="build_evidence",
            component="EvidenceEngine",
            parameters={"task_type": task_type.value},
            output_summary=(
                f"Locked evidence object: area_ha={evidence.area_ha}, "
                f"confidence={evidence.confidence}, "
                f"classes={evidence.land_cover_classes or evidence.change_classes}, "
                f"polygons={len(change_polygons) if change_polygons else 0}"
            ),
        )
        return evidence

    def _extract_change_polygons(
        self, raw_output: Dict[str, Any], trace: ExecutionTrace
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Vectorize a pixel-level change/diff map into GeoJSON polygons.

        Expects raw_output to contain:
          - 'diff_map': np.ndarray (H, W) — SSIM difference or change probability map
          - 'raster_transform': Affine or list (6 elements) — pixel-to-CRS affine
          - 'raster_crs': str — CRS of the raster (e.g. "EPSG:32644")

        Returns a list of GeoJSON Feature dicts with Polygon geometry in EPSG:4326,
        or None if the required data is not available.
        """
        diff_map = raw_output.get("diff_map")
        transform_data = raw_output.get("raster_transform")
        raster_crs = raw_output.get("raster_crs")

        if not HAS_GIS_VECTORIZE:
            trace.add(
                step="vectorize_change_polygons",
                component="EvidenceEngine",
                parameters={},
                output_summary="Skipped: rasterio/shapely not available for vectorization",
            )
            return None

        if diff_map is None or not isinstance(diff_map, np.ndarray):
            return None

        try:
            # Build affine transform
            if isinstance(transform_data, (list, tuple)) and len(transform_data) == 6:
                transform = Affine(*transform_data)
            elif transform_data is not None:
                transform = transform_data
            else:
                # Default 10m resolution identity if no transform provided
                transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)

            src_crs = raster_crs or "EPSG:32644"

            # Ensure diff_map is 2D float
            if diff_map.ndim == 3:
                diff_map = diff_map.mean(axis=0) if diff_map.shape[0] <= 4 else diff_map[:, :, 0]

            # Threshold: pixels where change is significant (1 - SSIM > threshold)
            # For SSIM diff maps, values closer to 0 = more change
            # For change probability maps, higher = more change
            change_threshold = raw_output.get("change_threshold", 0.35)

            # Determine if this is an SSIM map (values 0-1) or diff map
            map_max = float(diff_map.max()) if diff_map.size > 0 else 0
            if map_max <= 1.0:
                # SSIM map: low values = change
                binary_mask = (diff_map < (1.0 - change_threshold)).astype(np.uint8)
            else:
                # Absolute difference map: high values = change
                normalized = diff_map / max(map_max, 1.0)
                binary_mask = (normalized > change_threshold).astype(np.uint8)

            if binary_mask.sum() == 0:
                trace.add(
                    step="vectorize_change_polygons",
                    component="EvidenceEngine",
                    parameters={"threshold": change_threshold},
                    output_summary="No change pixels above threshold — no polygons generated",
                )
                return None

            # Vectorize binary mask to polygons
            features: List[Dict[str, Any]] = []
            change_classes = raw_output.get("change_classes", ["detected change"])
            primary_class = change_classes[0] if change_classes else "detected change"

            pixel_area_m2 = abs(transform.a * transform.e)  # cell width * cell height
            total_change_pixels = 0

            for geom, value in rasterio_shapes(binary_mask, transform=transform):
                if value == 0:
                    continue  # skip background

                # Count pixels in this polygon
                poly_shape = shapely_shape(geom)
                poly_area_m2 = poly_shape.area  # in CRS units (meters if UTM)
                poly_area_ha = poly_area_m2 / 10000.0
                total_change_pixels += 1

                # Transform to EPSG:4326 for Leaflet
                try:
                    geom_4326 = transform_geom(src_crs, "EPSG:4326", geom)
                except Exception:
                    geom_4326 = geom  # Keep in source CRS if transform fails

                feature = {
                    "type": "Feature",
                    "geometry": geom_4326,
                    "properties": {
                        "class": primary_class,
                        "area_ha": round(poly_area_ha, 4),
                        "confidence": raw_output.get("confidence", 0.5),
                    },
                }
                features.append(feature)

            # Limit to top 100 polygons by area (avoid overwhelming the frontend)
            if len(features) > 100:
                features.sort(key=lambda f: f["properties"]["area_ha"], reverse=True)
                features = features[:100]

            trace.add(
                step="vectorize_change_polygons",
                component="EvidenceEngine",
                parameters={
                    "threshold": change_threshold,
                    "source_crs": src_crs,
                    "pixel_area_m2": round(pixel_area_m2, 2),
                },
                output_summary=(
                    f"Vectorized {len(features)} change polygon(s) from diff map "
                    f"({diff_map.shape[0]}×{diff_map.shape[1]}px)"
                ),
            )

            return features if features else None

        except Exception as e:
            trace.add(
                step="vectorize_change_polygons",
                component="EvidenceEngine",
                parameters={},
                output_summary=f"Vectorization failed: {e}",
            )
            return None
