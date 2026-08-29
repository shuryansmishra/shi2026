"""
Unit and integration tests for the Location Resolver and query_by_location API.
"""
import pytest
from fastapi.testclient import TestClient

from ingestion.location_resolver import (
    geocode,
    infer_task_intent,
    water_level_caveat,
    fetch_scenes_for_location
)
from models.schemas import TaskType
from main import app

client = TestClient(app)


def test_geocode_offline_fallbacks():
    # Test Hardoi fallback or real API
    lat, lon, bbox, name = geocode("Hardoi, Uttar Pradesh")
    assert 27.0 < lat < 28.0
    assert 80.0 < lon < 81.0
    assert "Hardoi" in name

    # Test Bengaluru fallback or real API
    lat, lon, bbox, name = geocode("Bengaluru")
    assert 12.0 < lat < 13.5
    assert 77.0 < lon < 78.5

    # Test default fallback
    lat, lon, bbox, name = geocode("Some unknown random place")
    assert 27.0 < lat < 28.0
    assert "fallback" in name.lower() or "hardoi" in name.lower()


def test_infer_task_intent():
    # Test change detection intent
    assert infer_task_intent("has the water body changed since last month?") == TaskType.BI_TEMPORAL_CHANGE
    assert infer_task_intent("compare before and after scenes") == TaskType.BI_TEMPORAL_CHANGE

    # Test fusion intent
    assert infer_task_intent("what does it look like in monsoon with clouds?") == TaskType.CROSS_MODAL_FUSION
    assert infer_task_intent("combine optical and SAR views") == TaskType.CROSS_MODAL_FUSION

    # Test default single image VQA
    assert infer_task_intent("identify the dominant land cover class") == TaskType.SINGLE_IMAGE


def test_water_level_caveat():
    # Test triggered caveat
    caveat_text = water_level_caveat("What is the water level of this pond?")
    assert caveat_text is not None
    assert "Passive satellite sensors" in caveat_text

    # Test non-triggered caveat
    assert water_level_caveat("What land cover is visible?") is None


def test_fetch_scenes_for_location():
    lat, lon, bbox = 27.3828, 80.1287, [80.10, 27.36, 80.15, 27.40]

    # Test Single Image scene fetching
    scenes = fetch_scenes_for_location(lat, lon, bbox, TaskType.SINGLE_IMAGE)
    assert len(scenes) == 1
    assert scenes[0].modality.value == "optical"

    # Test Change Detection scene fetching
    scenes_change = fetch_scenes_for_location(lat, lon, bbox, TaskType.BI_TEMPORAL_CHANGE)
    assert len(scenes_change) == 2
    assert scenes_change[0].modality.value == "optical"
    assert scenes_change[1].modality.value == "optical"

    # Test Fusion scene fetching
    scenes_fusion = fetch_scenes_for_location(lat, lon, bbox, TaskType.CROSS_MODAL_FUSION)
    assert len(scenes_fusion) == 2
    modalities = [s.modality.value for s in scenes_fusion]
    assert "optical" in modalities
    assert "sar" in modalities


def test_fetch_scenes_for_location_is_stable_for_same_input():
    lat, lon, bbox = 27.3828, 80.1287, [80.10, 27.36, 80.15, 27.40]

    first = fetch_scenes_for_location(lat, lon, bbox, TaskType.SINGLE_IMAGE)
    second = fetch_scenes_for_location(lat, lon, bbox, TaskType.SINGLE_IMAGE)

    assert [scene.file_id for scene in first] == [scene.file_id for scene in second]
    assert [scene.path for scene in first] == [scene.path for scene in second]


def test_api_query_by_location_endpoint():
    response = client.post(
        "/api/query_by_location",
        data={
            "query_text": "What is the water level of my farm area?",
            "place_name": "Hardoi, Uttar Pradesh"
        }
      )
    assert response.status_code == 200
    res_data = response.json()
    assert "answer" in res_data
    assert "evidence" in res_data
    assert "trace" in res_data
    assert "route" in res_data

    # Verify that geocode and water level caveats are included in the trace
    trace_steps = [s["step"] for s in res_data["trace"]["steps"]]
    assert "geocode" in trace_steps
    assert "water_level_caveat" in trace_steps
    assert "Note: Passive satellite sensors" in res_data["answer"]
