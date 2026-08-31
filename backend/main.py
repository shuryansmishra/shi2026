"""
SatQuery AI - FastAPI entrypoint.

Wires the full pipeline together end to end:
  upload -> ingestion -> agentic router -> vision engine -> evidence engine
  -> LLM synthesis -> response (answer + evidence + execution trace)

Run:
    uvicorn main:app --reload --port 8000

With VQA_MOCK_MODE=True (the default), this is fully functional right now --
no models, no GPU, no API keys required. See README.md at the project root
for what to add for each layer of realism.
"""
from __future__ import annotations

import os
import shutil
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_settings
from core.evidence_engine import EvidenceEngine
from core.router import AgenticRouter, RoutingError
from engines.change_engine import ChangeEngine
from engines.fusion_engine import FusionEngine
from engines.single_image_engine import SingleImageEngine
from ingestion.preprocessing import build_image_meta, save_raster_as_png, save_diff_map_as_png
from ingestion.location_resolver import geocode, infer_task_intent, fetch_scenes_for_location, water_level_caveat
from llm.synthesis import LLMSynthesis
from models.schemas import ExecutionTrace, QueryResponse, TaskType

settings = get_settings()

app = FastAPI(title=settings.APP_NAME, version="0.1.0")

# Parse allowed origins with Vercel regex support
raw_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
has_vercel_wildcard = any("vercel.app" in o for o in raw_origins)
clean_origins = [o for o in raw_origins if "*" not in o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=clean_origins if clean_origins else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app" if has_vercel_wildcard else None,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Isolate public static storage (internal DB/rasters cannot be read via /static)
os.makedirs(os.path.join(settings.PUBLIC_STORAGE_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(settings.PUBLIC_STORAGE_DIR, "processed"), exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.PUBLIC_STORAGE_DIR), name="static")

router_engine = AgenticRouter()
evidence_engine = EvidenceEngine()
single_image_engine = SingleImageEngine()
change_engine = ChangeEngine()
fusion_engine = FusionEngine()
llm_synthesis = LLMSynthesis()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "mock_mode": settings.VQA_MOCK_MODE,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(
    query_text: str = Form(...),
    files: List[UploadFile] = File(...),
    capture_dates: Optional[List[str]] = Form(None),
) -> QueryResponse:
    """
    Accepts 1 or 2 images plus a natural-language question.
    capture_dates (optional) should line up 1:1 with files, e.g.
    ["2024-01-15", "2024-06-20"] for a bi-temporal pair.
    """
    if len(files) not in (1, 2):
        raise HTTPException(400, "Provide exactly 1 or 2 images.")

    trace = ExecutionTrace()
    images = []

    for i, upload in enumerate(files):
        if upload.size and upload.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(400, f"{upload.filename} exceeds MAX_UPLOAD_MB={settings.MAX_UPLOAD_MB}")

        # Sanitize filename and validate against allowed extensions
        safe_filename = os.path.basename(upload.filename or "image.tif")
        ext = os.path.splitext(safe_filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            )

        file_id = uuid.uuid4().hex[:12]
        dest_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{ext}")
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        img_meta = build_image_meta(dest_path, file_id=file_id)
        if capture_dates and i < len(capture_dates):
            img_meta.capture_date = capture_dates[i]
        images.append(img_meta)

    trace.add(
        step="ingest",
        component="ingestion.preprocessing",
        parameters={"file_count": len(images)},
        output_summary=f"Ingested {len(images)} image(s): "
        + ", ".join(f"{img.file_id}({img.modality.value})" for img in images),
    )

    try:
        route = router_engine.route(query_text, images, trace)
    except RoutingError as exc:
        raise HTTPException(400, str(exc)) from exc

    if route.task_type == TaskType.SINGLE_IMAGE:
        raw_output = single_image_engine.run(query_text, images[0], route.sub_tasks, trace)
    elif route.task_type == TaskType.BI_TEMPORAL_CHANGE:
        raw_output = change_engine.run(query_text, images, trace)
    else:
        raw_output = fusion_engine.run(query_text, images, trace)

    evidence = evidence_engine.build(route.task_type, raw_output, trace)
    answer = llm_synthesis.synthesize(query_text, route, evidence, trace)

    input_image_urls = []
    for img in images:
        png_filename = f"{img.file_id}.png"
        png_path = os.path.join(settings.PUBLIC_STORAGE_DIR, "uploads", png_filename)
        if save_raster_as_png(img.path, png_path):
            input_image_urls.append(f"/static/uploads/{png_filename}")

    result_image_url = None
    if "diff_map" in raw_output:
        diff_filename = f"diff_{uuid.uuid4().hex[:12]}.png"
        diff_path = os.path.join(settings.PUBLIC_STORAGE_DIR, "processed", diff_filename)
        change_thresh = raw_output.get("change_threshold", settings.CHANGE_THRESHOLD)
        if save_diff_map_as_png(raw_output["diff_map"], diff_path, change_thresh):
            result_image_url = f"/static/processed/{diff_filename}"

    return QueryResponse(
        answer=answer,
        evidence=evidence,
        trace=trace,
        route=route,
        input_image_urls=input_image_urls,
        result_image_url=result_image_url
    )


@app.post("/api/query_by_location", response_model=QueryResponse)
async def query_by_location(
    query_text: str = Form(...),
    place_name: str = Form(...),
) -> QueryResponse:
    """
    Resolves place_name to lat/lon, fetches/mocks images covering it based on inferred intent,
    and runs the standard query pipeline. Appends caveats if querying water levels.
    """
    trace = ExecutionTrace()

    # 1. Geocode location
    lat, lon, bbox, display_name = geocode(place_name)
    trace.add(
        step="geocode",
        component="ingestion.location_resolver",
        parameters={"place_name": place_name},
        output_summary=f"Resolved '{place_name}' to (lat={lat:.4f}, lon={lon:.4f}). BBox: {bbox}"
    )

    # 2. Infer task intent from query text
    task_type = infer_task_intent(query_text)
    trace.add(
        step="infer_intent",
        component="ingestion.location_resolver",
        parameters={"query_text": query_text},
        output_summary=f"Inferred task type from query: {task_type.value}"
    )

    # 3. Fetch/Mock matching scenes
    images = fetch_scenes_for_location(lat, lon, bbox, task_type)
    trace.add(
        step="ingest",
        component="ingestion.preprocessing",
        parameters={"location": display_name, "task_type": task_type.value},
        output_summary=f"Mock-fetched {len(images)} scene(s) for resolved location: "
        + ", ".join(f"{img.file_id}({img.modality.value})" for img in images)
    )

    # 4. Route decision
    try:
        route = router_engine.route(query_text, images, trace)
    except RoutingError as exc:
        raise HTTPException(400, str(exc)) from exc

    # 5. Execute engine analysis
    if route.task_type == TaskType.SINGLE_IMAGE:
        raw_output = single_image_engine.run(query_text, images[0], route.sub_tasks, trace)
    elif route.task_type == TaskType.BI_TEMPORAL_CHANGE:
        raw_output = change_engine.run(query_text, images, trace)
    else:
        raw_output = fusion_engine.run(query_text, images, trace)

    # 6. Build locked evidence object
    evidence = evidence_engine.build(route.task_type, raw_output, trace)
    
    # 7. Synthesize grounded answer
    answer = llm_synthesis.synthesize(query_text, route, evidence, trace)

    # 8. Append scientific water level caveats if relevant
    caveat = water_level_caveat(query_text)
    if caveat:
        answer = f"{answer}\n\n{caveat}"
        trace.add(
            step="water_level_caveat",
            component="ingestion.location_resolver",
            parameters={},
            output_summary="Appended water level surface area/depth caveat to final answer."
        )

    input_image_urls = []
    for img in images:
        png_filename = f"{img.file_id}.png"
        png_path = os.path.join(settings.PUBLIC_STORAGE_DIR, "uploads", png_filename)
        if save_raster_as_png(img.path, png_path):
            input_image_urls.append(f"/static/uploads/{png_filename}")

    result_image_url = None
    if "diff_map" in raw_output:
        diff_filename = f"diff_{uuid.uuid4().hex[:12]}.png"
        diff_path = os.path.join(settings.PUBLIC_STORAGE_DIR, "processed", diff_filename)
        change_thresh = raw_output.get("change_threshold", settings.CHANGE_THRESHOLD)
        if save_diff_map_as_png(raw_output["diff_map"], diff_path, change_thresh):
            result_image_url = f"/static/processed/{diff_filename}"

    return QueryResponse(
        answer=answer,
        evidence=evidence,
        trace=trace,
        route=route,
        input_image_urls=input_image_urls,
        result_image_url=result_image_url
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
