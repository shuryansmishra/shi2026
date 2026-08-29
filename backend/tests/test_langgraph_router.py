"""
Unit tests for the LangGraph agentic router.
Tests the full StateGraph pipeline: intent classification -> input validation -> specialist dispatch.
"""
import pytest

from core.langgraph_router import (
    LangGraphRouter,
    RouterState,
    _score_intents,
    intent_classifier_node,
    input_validation_node,
    specialist_dispatch_node,
)
from core.router import AgenticRouter, RoutingError
from models.schemas import ExecutionTrace, ImageMeta, Modality, SubTask, TaskType


def make_image(file_id: str, modality: Modality) -> ImageMeta:
    return ImageMeta(file_id=file_id, path=f"/tmp/{file_id}.tif", modality=modality)


# ---------------------------------------------------------------------------
# Unit tests for individual node functions
# ---------------------------------------------------------------------------

class TestIntentScoring:
    def test_change_keywords_score_high(self):
        scores = _score_intents("what changed between these two images before and after?")
        assert scores["change"] > scores["vqa"]

    def test_grounding_keywords_score_high(self):
        scores = _score_intents("where is the airstrip located? find the bounding box")
        assert scores["grounding"] > scores["change"]

    def test_fusion_keywords_score_high(self):
        scores = _score_intents("combine optical and SAR data for multi-modal analysis")
        assert scores["fusion"] > scores["change"]

    def test_vqa_keywords_score_high(self):
        scores = _score_intents("what type of land cover and vegetation is visible?")
        assert scores["vqa"] >= scores["change"]


class TestIntentClassifierNode:
    def test_returns_intent_scores_and_primary(self):
        state: RouterState = {
            "query_text": "what has changed between these two dates?",
            "images": [],
            "image_count": 2,
            "modalities": ["optical", "optical"],
        }
        result = intent_classifier_node(state)
        assert "intent_scores" in result
        assert "primary_intent" in result
        assert result["primary_intent"] == "change"


class TestInputValidationNode:
    def test_single_image_is_valid(self):
        state: RouterState = {
            "query_text": "what is this?",
            "images": [],
            "image_count": 1,
            "modalities": ["optical"],
        }
        result = input_validation_node(state)
        assert result["is_valid"] is True
        assert result["candidate_task"] == "single_image"

    def test_two_optical_is_change_detection(self):
        state: RouterState = {
            "query_text": "compare",
            "images": [],
            "image_count": 2,
            "modalities": ["optical", "optical"],
        }
        result = input_validation_node(state)
        assert result["is_valid"] is True
        assert result["candidate_task"] == "bi_temporal_change"

    def test_optical_sar_is_fusion(self):
        state: RouterState = {
            "query_text": "fuse",
            "images": [],
            "image_count": 2,
            "modalities": ["optical", "sar"],
        }
        result = input_validation_node(state)
        assert result["is_valid"] is True
        assert result["candidate_task"] == "cross_modal_fusion"

    def test_three_images_is_invalid(self):
        state: RouterState = {
            "query_text": "test",
            "images": [],
            "image_count": 3,
            "modalities": ["optical", "optical", "sar"],
        }
        result = input_validation_node(state)
        assert result["is_valid"] is False
        assert result["validation_error"] is not None

    def test_two_unknown_is_invalid(self):
        state: RouterState = {
            "query_text": "test",
            "images": [],
            "image_count": 2,
            "modalities": ["unknown", "unknown"],
        }
        result = input_validation_node(state)
        assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# Integration tests for the full LangGraph pipeline
# ---------------------------------------------------------------------------

class TestLangGraphRouter:
    def setup_method(self):
        self.router = LangGraphRouter()

    def test_langgraph_is_available(self):
        assert self.router.is_available, "langgraph should be installed"

    def test_single_image_routes_correctly(self):
        trace = ExecutionTrace()
        images = [make_image("a", Modality.OPTICAL)]
        decision = self.router.route("what land cover is this?", images, trace)

        assert decision.task_type == TaskType.SINGLE_IMAGE
        # LangGraph produces 3 trace steps (intent + validation + dispatch)
        assert len(trace.steps) >= 3

    def test_bitemporal_routes_correctly(self):
        trace = ExecutionTrace()
        images = [make_image("a", Modality.OPTICAL), make_image("b", Modality.OPTICAL)]
        decision = self.router.route("what changed between these two?", images, trace)

        assert decision.task_type == TaskType.BI_TEMPORAL_CHANGE

    def test_fusion_routes_correctly(self):
        trace = ExecutionTrace()
        images = [make_image("a", Modality.OPTICAL), make_image("b", Modality.SAR)]
        decision = self.router.route("combine optical and SAR", images, trace)

        assert decision.task_type == TaskType.CROSS_MODAL_FUSION

    def test_invalid_input_raises_routing_error(self):
        trace = ExecutionTrace()
        images = [make_image("a", Modality.UNKNOWN), make_image("b", Modality.UNKNOWN)]

        with pytest.raises(RoutingError):
            self.router.route("test", images, trace)

    def test_grounding_subtask_for_localisation(self):
        trace = ExecutionTrace()
        images = [make_image("a", Modality.OPTICAL)]
        decision = self.router.route("where is the airstrip located?", images, trace)

        assert decision.task_type == TaskType.SINGLE_IMAGE
        assert SubTask.GROUNDING in decision.sub_tasks

    def test_caption_subtask_for_description(self):
        trace = ExecutionTrace()
        images = [make_image("a", Modality.OPTICAL)]
        decision = self.router.route("describe the land cover", images, trace)

        assert decision.task_type == TaskType.SINGLE_IMAGE
        assert SubTask.CAPTION in decision.sub_tasks


# ---------------------------------------------------------------------------
# Test that AgenticRouter uses LangGraph as primary
# ---------------------------------------------------------------------------

class TestAgenticRouterLangGraphIntegration:
    def test_agentic_router_uses_langgraph(self):
        """AgenticRouter should produce LangGraph trace entries when langgraph is available."""
        router = AgenticRouter()
        trace = ExecutionTrace()
        images = [make_image("a", Modality.OPTICAL)]
        decision = router.route("what land cover is this?", images, trace)

        assert decision.task_type == TaskType.SINGLE_IMAGE
        # Should have LangGraph-prefixed trace steps
        step_names = [s.step for s in trace.steps]
        assert any("langgraph" in name for name in step_names), (
            f"Expected LangGraph trace steps, got: {step_names}"
        )
