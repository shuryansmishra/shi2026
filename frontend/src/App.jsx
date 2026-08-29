import React, { useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import LocationPanel from "./components/LocationPanel.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import MapViewer from "./components/MapViewer.jsx";
import TraceViewer from "./components/TraceViewer.jsx";
import { runQuery, runLocationQuery } from "./api.js";

export default function App() {
  const [mode, setMode] = useState("upload"); // "upload" or "location"
  const [files, setFiles] = useState([]);
  const [dates, setDates] = useState([]);
  const [placeName, setPlaceName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleAsk(queryText) {
    if (mode === "upload" && files.length === 0) {
      setError("Upload at least one image first.");
      return;
    }
    if (mode === "location" && !placeName.trim()) {
      setError("Please specify a location name first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      let data;
      if (mode === "upload") {
        data = await runQuery(queryText, files, dates);
      } else {
        data = await runLocationQuery(queryText, placeName);
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>SatQuery AI</h1>
        <p style={styles.subtitle}>
          Ask natural-language questions of satellite imagery — single-scene VQA,
          bi-temporal change detection, and optical-SAR fusion, all under one agentic router.
        </p>
      </header>

      <main style={styles.main}>
        <div style={styles.left}>
          {/* Mode Toggle Tabs */}
          <div style={styles.tabs}>
            <button
              onClick={() => {
                setMode("upload");
                setError(null);
              }}
              style={{
                ...styles.tab,
                borderBottomColor: mode === "upload" ? "#0066cc" : "transparent",
                fontWeight: mode === "upload" ? "bold" : "normal",
                color: mode === "upload" ? "#0066cc" : "#666",
              }}
            >
              Upload Imagery
            </button>
            <button
              onClick={() => {
                setMode("location");
                setError(null);
              }}
              style={{
                ...styles.tab,
                borderBottomColor: mode === "location" ? "#0066cc" : "transparent",
                fontWeight: mode === "location" ? "bold" : "normal",
                color: mode === "location" ? "#0066cc" : "#666",
              }}
            >
              Use My Location
            </button>
          </div>

          {mode === "upload" ? (
            <UploadPanel files={files} setFiles={setFiles} dates={dates} setDates={setDates} />
          ) : (
            <LocationPanel placeName={placeName} setPlaceName={setPlaceName} />
          )}

          <ChatPanel onSubmit={handleAsk} loading={loading} />
          {error && <p style={styles.error}>{error}</p>}
          
          {result && (
            <div style={styles.resultContainer}>
              <p style={styles.answer}>{result.answer}</p>
              
              {(result.input_image_urls?.length > 0 || result.result_image_url) && (
                <div style={styles.visuals}>
                  <h4 style={styles.visualsTitle}>Visual Imagery Analysis</h4>
                  
                  {/* Inputs */}
                  {result.input_image_urls?.length > 0 && (
                    <div style={styles.imageGrid}>
                      {result.input_image_urls.map((url, idx) => {
                        let label = `Input Scene ${idx + 1}`;
                        if (result.route?.task_type === "bi_temporal_change") {
                          label = idx === 0 ? "Before Scene (T1)" : "After Scene (T2)";
                        } else if (result.route?.task_type === "cross_modal_fusion") {
                          label = idx === 0 ? "Optical Band" : "SAR Backscatter Band";
                        }
                        return (
                          <div key={url} style={styles.imageCard}>
                            <span style={styles.imageLabel}>{label}</span>
                            <img src={url} alt={label} style={styles.satImage} />
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Difference Map */}
                  {result.result_image_url && (
                    <div style={styles.resultCard}>
                      <span style={styles.imageLabel}>Analysis Overlay (Red Highlights = Detected Changes)</span>
                      <img src={result.result_image_url} alt="Analysis Overlay" style={styles.resultImage} />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div style={styles.right}>
          <MapViewer evidence={result?.evidence} />
          <TraceViewer trace={result?.trace} route={result?.route} />
        </div>
      </main>
    </div>
  );
}

const styles = {
  app: { fontFamily: "system-ui, sans-serif", maxWidth: 1100, margin: "0 auto", padding: 24 },
  header: { marginBottom: 24 },
  title: { margin: 0 },
  subtitle: { color: "#555" },
  main: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 },
  left: {},
  right: {},
  tabs: { display: "flex", borderBottom: "1px solid #ddd", marginBottom: 16 },
  tab: {
    padding: "8px 16px",
    background: "none",
    border: "none",
    borderBottom: "2px solid transparent",
    cursor: "pointer",
    fontSize: 14,
    outline: "none",
  },
  error: { color: "#b00020" },
  answer: { background: "#f4f8ff", padding: 12, borderRadius: 8, fontSize: 15, whiteSpace: "pre-line", marginBottom: 16 },
  resultContainer: { display: "flex", flexDirection: "column", gap: 12 },
  visuals: { border: "1px solid #eee", borderRadius: 8, padding: 16, background: "#fafafa" },
  visualsTitle: { margin: "0 0 12px 0", borderBottom: "1px solid #ddd", paddingBottom: 6 },
  imageGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 },
  imageCard: { display: "flex", flexDirection: "column", gap: 6, background: "#fff", padding: 8, borderRadius: 4, border: "1px solid #e0e0e0" },
  imageLabel: { fontSize: 11, fontWeight: "bold", color: "#666", textTransform: "uppercase" },
  satImage: { width: "100%", height: "auto", borderRadius: 4, objectFit: "cover" },
  resultCard: { display: "flex", flexDirection: "column", gap: 6, background: "#fff", padding: 8, borderRadius: 4, border: "1px solid #e0e0e0" },
  resultImage: { width: "100%", height: "auto", borderRadius: 4, objectFit: "cover" },
};
