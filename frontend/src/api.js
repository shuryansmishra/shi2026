/**
 * Thin wrapper around the SatQuery AI backend's single endpoint.
 * Sends 1-2 images + a natural-language query, gets back
 * { answer, evidence, trace, route } -- see backend/models/schemas.py.
 */
export async function runQuery(queryText, imageFiles, captureDates = []) {
  const form = new FormData();
  form.append("query_text", queryText);
  imageFiles.forEach((file) => form.append("files", file));
  captureDates.forEach((date) => form.append("capture_dates", date));

  const res = await fetch("/api/query", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Query failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch("/health");
  return res.json();
}

export async function runLocationQuery(queryText, placeName) {
  const form = new FormData();
  form.append("query_text", queryText);
  form.append("place_name", placeName);

  const res = await fetch("/api/query_by_location", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Location Query failed (${res.status}): ${detail}`);
  }
  return res.json();
}
