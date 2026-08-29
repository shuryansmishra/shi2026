import React from "react";

/**
 * Accepts 1 or 2 image files. The router (backend/core/router.py) decides
 * the task purely from how many images arrive and their detected modality --
 * this panel doesn't need to know the routing rules, just collect files.
 */
export default function UploadPanel({ files, setFiles, dates, setDates }) {
  function onFilesSelected(e) {
    const selected = Array.from(e.target.files).slice(0, 2);
    setFiles(selected);
    setDates(selected.map(() => ""));
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>1. Upload imagery</h3>
      <p style={styles.hint}>
        Upload 1 image for single-scene VQA, or 2 images for change detection
        (same sensor, two dates) or optical+SAR fusion.
      </p>
      <input type="file" accept="image/*,.tif,.tiff" multiple onChange={onFilesSelected} />

      {files.length > 0 && (
        <ul style={styles.fileList}>
          {files.map((f, i) => (
            <li key={f.name + i} style={styles.fileRow}>
              <span>{f.name}</span>
              <input
                type="date"
                value={dates[i] || ""}
                onChange={(e) => {
                  const next = [...dates];
                  next[i] = e.target.value;
                  setDates(next);
                }}
                title="Capture date (optional, required for meaningful change detection)"
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const styles = {
  panel: { border: "1px solid #ddd", borderRadius: 8, padding: 16, marginBottom: 16 },
  heading: { margin: "0 0 8px 0" },
  hint: { color: "#666", fontSize: 13, marginTop: 0 },
  fileList: { listStyle: "none", padding: 0, marginTop: 12 },
  fileRow: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 0" },
};
