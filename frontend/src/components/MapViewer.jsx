import React from "react";
import { MapContainer, TileLayer, Rectangle, Popup, GeoJSON } from "react-leaflet";

/**
 * Renders the evidence object's bounding box (backend/models/schemas.py
 * EvidenceObject.bbox_latlon) and vectorized change polygons on an OSM basemap.
 */
export default function MapViewer({ evidence }) {
  const bbox = evidence?.bbox_latlon;
  const polygons = evidence?.change_polygons_geojson;
  
  // Calculate map center based on bbox or polygons
  let center = [20.5937, 78.9629]; // fallback: center of India
  let zoom = 4;

  if (bbox) {
    center = [(bbox.min_lat + bbox.max_lat) / 2, (bbox.min_lon + bbox.max_lon) / 2];
    zoom = 12;
  } else if (polygons && polygons.length > 0) {
    // If no bbox but polygons exist, center on first polygon coords
    const coords = polygons[0].geometry.coordinates[0][0];
    if (coords && coords.length >= 2) {
      center = [coords[1], coords[0]]; // GeoJSON is [lng, lat], Leaflet is [lat, lng]
      zoom = 13;
    }
  }

  const rectangle = bbox
    ? [
        [bbox.min_lat, bbox.min_lon],
        [bbox.max_lat, bbox.max_lon],
      ]
    : null;

  // Use dynamic key to force map reconstruction on coordinates change
  const mapKey = bbox 
    ? `${bbox.min_lat}_${bbox.min_lon}` 
    : (polygons && polygons.length > 0 ? JSON.stringify(polygons[0].geometry.coordinates[0][0]) : "default");

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>Result overlay</h3>
      <div style={styles.mapWrap}>
        <MapContainer key={mapKey} center={center} zoom={zoom} style={styles.map}>
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {rectangle && (
            <Rectangle bounds={rectangle} pathOptions={{ color: "blue", weight: 2, fillOpacity: 0.05 }}>
              <Popup>
                Bounding Box Area
                {evidence.area_ha != null && <><br />Area: {evidence.area_ha} ha</>}
                {evidence.change_area_ha != null && <><br />Change: {evidence.change_area_ha} ha</>}
              </Popup>
            </Rectangle>
          )}
          {polygons && polygons.length > 0 && (
            <GeoJSON
              data={{
                type: "FeatureCollection",
                features: polygons
              }}
              style={{ color: "#d32f2f", weight: 3, fillColor: "#ffcdd2", fillOpacity: 0.5 }}
              onEachFeature={(feature, layer) => {
                if (feature.properties) {
                  layer.bindPopup(
                    `<strong>Detected Feature:</strong> ${feature.properties.class}<br/>` +
                    `<strong>Area:</strong> ${feature.properties.area_ha} ha<br/>` +
                    `<strong>Confidence:</strong> ${(feature.properties.confidence * 100).toFixed(0)}%`
                  );
                }
              }}
            />
          )}
        </MapContainer>
      </div>
      {!bbox && (!polygons || polygons.length === 0) && (
        <p style={styles.hint}>No bounding box or polygons in this result -- map shown at default view.</p>
      )}
    </div>
  );
}

const styles = {
  panel: { border: "1px solid #ddd", borderRadius: 8, padding: 16, marginBottom: 16 },
  heading: { margin: "0 0 8px 0" },
  mapWrap: { height: 320, borderRadius: 6, overflow: "hidden" },
  map: { height: "100%", width: "100%" },
  hint: { color: "#666", fontSize: 13 },
};
