"""
Unit tests for the Evidence Engine -- the hallucination firewall.
"""
from core.evidence_engine import EvidenceEngine
from models.schemas import ExecutionTrace, TaskType


def test_evidence_engine_locks_only_known_fields():
    engine = EvidenceEngine()
    trace = ExecutionTrace()
    raw = {
        "area_ha": 12.8,
        "confidence": 0.87,
        "land_cover_classes": ["forest cover"],
        "bbox_latlon": [77.1, 28.5, 77.2, 28.6],
        "some_untrusted_field": "should be dropped silently",
    }

    evidence = engine.build(TaskType.SINGLE_IMAGE, raw, trace)

    assert evidence.area_ha == 12.8
    assert evidence.confidence == 0.87
    assert evidence.bbox_latlon.min_lon == 77.1
    assert not hasattr(evidence, "some_untrusted_field")
    assert len(trace.steps) == 1


def test_evidence_engine_defaults_confidence_when_missing():
    engine = EvidenceEngine()
    trace = ExecutionTrace()
    evidence = engine.build(TaskType.BI_TEMPORAL_CHANGE, {}, trace)

    assert evidence.confidence == 0.5
    assert evidence.change_classes == []
