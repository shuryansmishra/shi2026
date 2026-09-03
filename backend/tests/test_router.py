"""
Unit tests for the rule-based agentic router. Run with: pytest tests/
"""
import pytest

from core.router import AgenticRouter, RoutingError
from models.schemas import ExecutionTrace, ImageMeta, Modality, TaskType


def make_image(file_id: str, modality: Modality) -> ImageMeta:
    return ImageMeta(file_id=file_id, path=f"/tmp/{file_id}.tif", modality=modality)


def test_single_image_routes_to_single_image_task():
    router = AgenticRouter()
    trace = ExecutionTrace()
    images = [make_image("a", Modality.OPTICAL)]

    decision = router.route("what land cover is this?", images, trace)

    assert decision.task_type == TaskType.SINGLE_IMAGE
    assert len(trace.steps) >= 1  # LangGraph produces 3 steps; rule-based produces 1


def test_two_optical_images_route_to_change_detection():
    router = AgenticRouter()
    trace = ExecutionTrace()
    images = [make_image("a", Modality.OPTICAL), make_image("b", Modality.OPTICAL)]

    decision = router.route("what changed between these two images?", images, trace)

    assert decision.task_type == TaskType.BI_TEMPORAL_CHANGE


def test_optical_plus_sar_routes_to_fusion():
    router = AgenticRouter()
    trace = ExecutionTrace()
    images = [make_image("a", Modality.OPTICAL), make_image("b", Modality.SAR)]

    decision = router.route("combine these two views", images, trace)

    assert decision.task_type == TaskType.CROSS_MODAL_FUSION


def test_two_sar_images_with_unknown_modality_pair_raises():
    router = AgenticRouter()
    trace = ExecutionTrace()
    images = [
        make_image("a", Modality.UNKNOWN),
        make_image("b", Modality.UNKNOWN),
    ]

    with pytest.raises(RoutingError):
        router.route("what is this?", images, trace)


def test_grounding_subtask_selected_for_localisation_query():
    router = AgenticRouter()
    trace = ExecutionTrace()
    images = [make_image("a", Modality.OPTICAL)]

    decision = router.route("where is the airstrip located?", images, trace)

    from models.schemas import SubTask
    assert SubTask.GROUNDING in decision.sub_tasks


def test_query_endpoint_handles_single_capture_date():
    from fastapi.testclient import TestClient
    from main import app
    import io
    from PIL import Image

    client = TestClient(app)
    img = Image.new('RGB', (10, 10), color='green')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    res = client.post(
        '/api/query',
        data={'query_text': 'What land cover is visible?', 'capture_dates': '2024-08-20'},
        files={'files': ('scene.png', buf, 'image/png')}
    )
    assert res.status_code == 200
    assert "answer" in res.json()


def test_query_endpoint_handles_bitemporal_capture_dates():
    from fastapi.testclient import TestClient
    from main import app
    import io
    from PIL import Image

    client = TestClient(app)
    img = Image.new('RGB', (10, 10), color='green')
    buf1 = io.BytesIO()
    img.save(buf1, format='PNG')
    buf1.seek(0)
    buf2 = io.BytesIO()
    img.save(buf2, format='PNG')
    buf2.seek(0)

    res = client.post(
        '/api/query',
        data={'query_text': 'Detect changes between 2020 and 2024'},
        files=[
            ('files', ('t1.png', buf1, 'image/png')),
            ('files', ('t2.png', buf2, 'image/png')),
            ('capture_dates', (None, '2020-01-15')),
            ('capture_dates', (None, '2024-08-20')),
        ]
    )
    assert res.status_code == 200
    assert "answer" in res.json()
