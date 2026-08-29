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
