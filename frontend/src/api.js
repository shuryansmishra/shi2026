/**
 * SatQuery AI API Client
 * Interfaces with the FastAPI backend endpoints:
 *   - POST /api/query
 *   - POST /api/query_by_location
 *   - GET  /health
 *
 * ROUTING STRATEGY:
 * - In development (localhost): always use RELATIVE paths so Vite's proxy
 *   forwards them to the backend (avoids CORS preflight entirely).
 * - In production (Vercel/etc): use VITE_BACKEND_URL if it points to a
 *   different host, otherwise fall back to relative paths.
 */

const rawEnvUrl = import.meta.env.VITE_BACKEND_URL
  ? import.meta.env.VITE_BACKEND_URL.replace(/\/$/, "")
  : "";

// Check if running on local dev machine
const IS_LOCAL =
  typeof window !== "undefined" &&
  (window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1");

// Determine API_BASE:
// 1. On localhost: always use relative path "" (Vite proxy forwards /api, /health to backend).
// 2. On production (e.g. deployed on Vercel):
//    - If VITE_BACKEND_URL points to localhost/127.0.0.1, ignore it (cannot reach dev machine).
//      Instead, use same-origin relative path "" (Vercel serverless functions handle /api and /health).
//    - If VITE_BACKEND_URL is a valid remote cloud URL (e.g. https://...onrender.com), use it.
//    - Otherwise, use same-origin relative path "".
let API_BASE = "";
if (IS_LOCAL) {
  API_BASE = "";
} else if (rawEnvUrl && !rawEnvUrl.includes("localhost") && !rawEnvUrl.includes("127.0.0.1")) {
  API_BASE = rawEnvUrl;
} else {
  API_BASE = "";
}

async function fetchWithFallback(endpoint, options) {
  const url = API_BASE + endpoint;  // e.g. "/api/query" (proxied) or "https://xxx/api/query"
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = "";
    try { detail = await res.text(); } catch (_) {}
    throw new Error(`Request to ${endpoint} failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function runQuery(queryText, imageFiles, captureDates = []) {
  const form = new FormData();
  form.append("query_text", queryText);
  
  imageFiles.forEach((file) => {
    form.append("files", file);
  });
  
  captureDates.forEach((date) => {
    if (date) form.append("capture_dates", date);
  });

  return fetchWithFallback("/api/query", { method: "POST", body: form });
}

export async function checkHealth() {
  try {
    return await fetchWithFallback("/health", { method: "GET" });
  } catch (e) {
    return { status: "offline", error: e.message };
  }
}

export async function runLocationQuery(queryText, placeName) {
  const form = new FormData();
  form.append("query_text", queryText);
  form.append("place_name", placeName);

  return fetchWithFallback("/api/query_by_location", { method: "POST", body: form });
}

/**
 * Creates a synthetic demo File object with realistic satellite terrain textures.
 * Generates distinct signatures for T1 Baseline vs T2 Target so bi-temporal
 * change detection and SSIM algorithms detect real, measurable deltas.
 */
export function createDemoFile(name = "sample_satellite.png", variant = "default") {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  const isT2 = variant === "t2" || name.includes("t2") || name.includes("target");
  const isSar = variant === "sar" || name.includes("sar") || name.includes("radar");

  if (isSar) {
    // SAR radar backscatter texture with speckle
    ctx.fillStyle = "#2b2b2b";
    ctx.fillRect(0, 0, 512, 512);
    ctx.fillStyle = "#666666";
    ctx.fillRect(150, 100, 200, 150);
    // Radar scatter noise
    for (let i = 0; i < 600; i++) {
      ctx.fillStyle = Math.random() > 0.4 ? "#ffffff" : "#111111";
      ctx.fillRect(Math.random() * 512, Math.random() * 512, 2, 2);
    }
  } else if (isT2) {
    // T2 Target Observation: Shows significant new urban built-up expansion & road connectivity
    ctx.fillStyle = "#1e3a5f";
    ctx.fillRect(0, 0, 512, 512);
    // Shrunk water / altered wetland boundary
    ctx.fillStyle = "#2d5a27";
    ctx.beginPath();
    ctx.arc(220, 180, 85, 0, Math.PI * 2);
    ctx.fill();
    // Cleared land
    ctx.fillStyle = "#b09572";
    ctx.fillRect(220, 80, 220, 290);
    // NEW urban construction clusters (High-contrast red/orange footprints)
    ctx.fillStyle = "#d35400";
    ctx.fillRect(80, 300, 140, 120);
    ctx.fillStyle = "#c0392b";
    ctx.fillRect(260, 120, 90, 90);
    ctx.fillStyle = "#e67e22";
    ctx.fillRect(360, 220, 100, 80);
    // New connecting arterial transit corridor
    ctx.strokeStyle = "#f39c12";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(0, 360);
    ctx.lineTo(512, 360);
    ctx.stroke();
  } else {
    // T1 Baseline Observation: Dense vegetation & original agricultural zoning
    ctx.fillStyle = "#1e3a5f";
    ctx.fillRect(0, 0, 512, 512);
    // Natural water reservoir
    ctx.fillStyle = "#27ae60";
    ctx.beginPath();
    ctx.arc(200, 200, 120, 0, Math.PI * 2);
    ctx.fill();
    // Agricultural fields / open soil
    ctx.fillStyle = "#8a795d";
    ctx.fillRect(250, 100, 180, 260);
    ctx.fillStyle = "#2ecc71";
    ctx.fillRect(60, 280, 160, 150);
  }

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(new File([blob], name, { type: "image/png" }));
    }, "image/png");
  });
}

