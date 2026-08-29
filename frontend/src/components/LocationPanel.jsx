import React from "react";

/**
 * Panel to select location-based queries instead of uploading imagery.
 * Wires place name and predefined presets to the client.
 */
export default function LocationPanel({ placeName, setPlaceName }) {
  const presets = [
    "Hardoi, Uttar Pradesh",
    "Bengaluru, Karnataka",
    "Delhi, India",
    "Hyderabad, Telangana"
  ];

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>1. Specify Location</h3>
      <p style={styles.hint}>
        Type a place name or GPS coordinates to automatically fetch and analyze
        satellite scenes.
      </p>
      
      <input
        type="text"
        placeholder="Enter village, district, city, or coordinates..."
        value={placeName}
        onChange={(e) => setPlaceName(e.target.value)}
        style={styles.input}
      />

      <div style={styles.presetsContainer}>
        <span style={styles.presetLabel}>Presets:</span>
        {presets.map((preset) => (
          <button
            key={preset}
            onClick={() => setPlaceName(preset)}
            style={{
              ...styles.presetBtn,
              background: placeName === preset ? "#e6f0fa" : "#f5f5f5",
              borderColor: placeName === preset ? "#0066cc" : "#ddd",
            }}
          >
            {preset.split(",")[0]}
          </button>
        ))}
      </div>
    </div>
  );
}

const styles = {
  panel: { border: "1px solid #ddd", borderRadius: 8, padding: 16, marginBottom: 16 },
  heading: { margin: "0 0 8px 0" },
  hint: { color: "#666", fontSize: 13, marginTop: 0 },
  input: {
    width: "100%",
    padding: "8px 12px",
    borderRadius: 4,
    border: "1px solid #ccc",
    boxSizing: "border-box",
    fontSize: 14,
    marginBottom: 12,
  },
  presetsContainer: { display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 },
  presetLabel: { fontSize: 12, color: "#666", fontWeight: "bold" },
  presetBtn: {
    padding: "4px 8px",
    fontSize: 12,
    border: "1px solid #ddd",
    borderRadius: 16,
    cursor: "pointer",
    transition: "all 0.2s",
  }
};
