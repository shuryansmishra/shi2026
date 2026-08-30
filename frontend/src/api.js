/**
 * SatQuery AI API Client
 * Interfaces with the FastAPI backend endpoints:
 *   - POST /api/query
 *   - POST /api/query_by_location
 *   - GET  /health
 */

const API_BASE = import.meta.env.VITE_BACKEND_URL ? import.meta.env.VITE_BACKEND_URL.replace(/\/$/, "") : "";

export async function runQuery(queryText, imageFiles, captureDates = []) {
  const form = new FormData();
  form.append("query_text", queryText);
  
  imageFiles.forEach((file) => {
    form.append("files", file);
  });
  
  captureDates.forEach((date) => {
    if (date) form.append("capture_dates", date);
  });

  const res = await fetch(`${API_BASE}/api/query`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Query failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return { status: "error" };
    return res.json();
  } catch (e) {
    return { status: "offline", error: e.message };
  }
}

export async function runLocationQuery(queryText, placeName) {
  const form = new FormData();
  form.append("query_text", queryText);
  form.append("place_name", placeName);

  const res = await fetch(`${API_BASE}/api/query_by_location`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Location Query failed (${res.status}): ${detail}`);
  }
  return res.json();
}

/**
 * Creates a synthetic demo File object (PNG dummy) if the user clicks quick demo without manual upload.
 */
export function createDemoFile(name = "sample_satellite.png") {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  
  // draw satellite-like texture
  ctx.fillStyle = "#1e3a5f";
  ctx.fillRect(0, 0, 512, 512);
  ctx.fillStyle = "#2d5a27";
  ctx.beginPath();
  ctx.arc(200, 200, 120, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#8a795d";
  ctx.fillRect(250, 100, 180, 260);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(new File([blob], name, { type: "image/png" }));
    }, "image/png");
  });
}
