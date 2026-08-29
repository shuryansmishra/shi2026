"""
SatQuery AI - LangGraph Agentic Router.

Implements a proper LangGraph StateGraph with three sequential nodes:
  1. IntentClassifier — scores query intent tokens (change, grounding, fusion, VQA)
  2. InputValidator — validates image count × modality combinations
  3. SpecialistDispatcher — selects TaskType using intent + validation + RL refinement

The typed RouterState flows through each node and is fully serializable for
the graded execution trace (SIH26167 requirement).

Falls back gracefully to the existing rule-based router if langgraph is
not installed (import-time check).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from models.schemas import (
    ImageMeta,
    Modality,
    RouteDecision,
    SubTask,
    TaskType,
    ExecutionTrace,
)


# ---------------------------------------------------------------------------
# Typed State that flows through the LangGraph
# ---------------------------------------------------------------------------

class RouterState(TypedDict, total=False):
    """Typed state dict flowing through the LangGraph StateGraph."""
    # Inputs
    query_text: str
    images: List[Dict[str, Any]]  # serialised ImageMeta dicts
    image_count: int
    modalities: List[str]

    # Intent classification outputs
    intent_scores: Dict[str, float]
    primary_intent: str

    # Validation outputs
    is_valid: bool
    validation_error: Optional[str]
    candidate_task: Optional[str]

    # Dispatch outputs
    final_task: Optional[str]
    sub_tasks: List[str]
    reason: str
    rl_q_values: Optional[Dict[str, float]]


# ---------------------------------------------------------------------------
# Intent keyword banks (richer than the old router's simple token check)
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS = {
    "change": [
        "change", "changed", "difference", "differ", "delta", "before",
        "after", "compare", "comparison", "temporal", "evolution",
        "transition", "transform", "converted", "built", "demolished",
        "deforest", "flood", "urban sprawl", "encroach",
    ],
    "grounding": [
        "where", "locate", "location", "find", "point to", "bounding box",
        "bbox", "region", "position", "highlight", "mark", "show me",
    ],
    "fusion": [
        "combine", "fuse", "fusion", "sar", "radar", "optical", "cloud",
        "cross-modal", "dual", "sentinel-1", "sentinel-2", "s1", "s2",
        "multi-modal", "multimodal",
    ],
    "vqa": [
        "what", "describe", "classify", "identify", "type", "land cover",
        "crop", "vegetation", "water", "urban", "built-up", "how many",
        "count", "area", "extent", "caption",
    ],
}


def _score_intents(query_text: str) -> Dict[str, float]:
    """Score each intent bucket by counting keyword hits, normalised to [0, 1]."""
    q = query_text.lower()
    scores: Dict[str, float] = {}
    max_hits = 1  # avoid div-by-zero

    for intent, keywords in _INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        scores[intent] = float(hits)
        if hits > max_hits:
            max_hits = hits

    # Normalise
    for k in scores:
        scores[k] = round(scores[k] / max_hits, 3) if max_hits > 0 else 0.0

    return scores


# ---------------------------------------------------------------------------
# Graph Node Functions
# ---------------------------------------------------------------------------

def intent_classifier_node(state: RouterState) -> Dict[str, Any]:
    """
    Node 1: Analyse query text to produce intent scores.
    Pure function — no side effects.
    """
    query_text = state["query_text"]
    scores = _score_intents(query_text)
    primary = max(scores, key=lambda k: scores[k]) if scores else "vqa"

    return {
        "intent_scores": scores,
        "primary_intent": primary,
    }


def input_validation_node(state: RouterState) -> Dict[str, Any]:
    """
    Node 2: Validate that the image count × modality combination is legal.
    Determines the candidate TaskType based on structural rules.
    """
    n = state["image_count"]
    modalities = state["modalities"]

    if n == 1:
        return {
            "is_valid": True,
            "validation_error": None,
            "candidate_task": TaskType.SINGLE_IMAGE.value,
        }

    if n == 2:
        mod_set = set(modalities)

        # Same modality (both optical or both SAR) → change detection
        if len(mod_set) == 1 and "unknown" not in mod_set:
            return {
                "is_valid": True,
                "validation_error": None,
                "candidate_task": TaskType.BI_TEMPORAL_CHANGE.value,
            }

        # One optical + one SAR → fusion
        if mod_set == {"optical", "sar"}:
            return {
                "is_valid": True,
                "validation_error": None,
                "candidate_task": TaskType.CROSS_MODAL_FUSION.value,
            }

    # Everything else is invalid
    return {
        "is_valid": False,
        "validation_error": (
            f"Cannot route {n} image(s) with modalities {modalities}. "
            "Valid combos: 1 image; 2 same-modality images; 2 images (optical+SAR)."
        ),
        "candidate_task": None,
    }


def specialist_dispatch_node(state: RouterState) -> Dict[str, Any]:
    """
    Node 3: Finalise TaskType and sub-tasks using intent + validation + RL.

    If the RL router is available and trained, it can override the candidate
    (with a learning penalty for disagreement). Otherwise the rule-based
    candidate is used directly.
    """
    if not state.get("is_valid"):
        return {
            "final_task": None,
            "sub_tasks": [],
            "reason": state.get("validation_error", "Invalid input"),
            "rl_q_values": None,
        }

    candidate = state["candidate_task"]
    primary_intent = state.get("primary_intent", "vqa")
    query_text = state["query_text"]

    # --- RL refinement (optional) ---
    rl_q_values = None
    final_task = candidate
    reason_parts = []

    try:
        from core.rl_router import RLRouterAgent

        rl_agent = RLRouterAgent()

        # Reconstruct ImageMeta objects for the RL agent's state extraction
        image_dicts = state.get("images", [])
        image_metas = []
        for d in image_dicts:
            image_metas.append(ImageMeta(**d))

        fallback_task = TaskType(candidate)
        chosen, q_vals = rl_agent.select_action(query_text, image_metas, fallback_task)
        rl_q_values = q_vals

        # Only allow RL to override the structural rule if it has a *strictly
        # higher* Q-value for its chosen action than for the rule-based fallback.
        # This prevents epsilon-greedy exploration noise from overriding correct
        # structural routing during tests and low-data startup.
        fallback_q = q_vals.get(fallback_task.value, 0.0)
        chosen_q = q_vals.get(chosen.value, 0.0)
        if chosen == fallback_task or chosen_q <= fallback_q:
            # Structural rule wins — RL did not find a better option
            chosen = fallback_task
            reward = 1.0
        else:
            reward = -0.5  # Penalise override for disagreement with structure

        rl_agent.update(query_text, image_metas, chosen, reward)

        final_task = chosen.value
        if chosen != fallback_task:
            reason_parts.append(
                f"RL Q-Learning dynamically reassigned from {candidate} to {final_task} "
                f"(Q={chosen_q:.3f} > fallback Q={fallback_q:.3f})"
            )
        else:
            reason_parts.append(f"RL Q-Learning confirmed structural routing to {final_task}")
    except (ImportError, Exception):
        reason_parts.append(f"Rule-based routing to {final_task} (RL unavailable)")

    # --- Determine sub-tasks ---
    task_type = TaskType(final_task)
    if task_type == TaskType.SINGLE_IMAGE:
        sub_tasks = _single_image_subtasks(query_text)
        reason_parts.append(f"Intent: {primary_intent}, sub-tasks: {[s.value for s in sub_tasks]}")
    elif task_type == TaskType.BI_TEMPORAL_CHANGE:
        sub_tasks = [SubTask.CHANGE_VQA, SubTask.CHANGE_DESCRIPTION]
        reason_parts.append(f"Bi-temporal pair detected, intent: {primary_intent}")
    else:
        sub_tasks = [SubTask.FUSION_ANALYSIS]
        reason_parts.append(f"Optical+SAR fusion, intent: {primary_intent}")

    return {
        "final_task": final_task,
        "sub_tasks": [s.value for s in sub_tasks],
        "reason": "; ".join(reason_parts),
        "rl_q_values": rl_q_values,
    }


def _single_image_subtasks(query_text: str) -> List[SubTask]:
    """Pick grounding when the query implies localisation, else captioning."""
    q = query_text.lower()
    localisation_terms = ("where", "locate", "bounding box", "point to", "find the")
    if any(t in q for t in localisation_terms):
        return [SubTask.VQA, SubTask.GROUNDING]
    return [SubTask.VQA, SubTask.CAPTION]


# ---------------------------------------------------------------------------
# LangGraph Assembly
# ---------------------------------------------------------------------------

def _build_graph():
    """
    Build and compile the LangGraph StateGraph.
    Returns the compiled graph, or None if langgraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None

    graph = StateGraph(RouterState)

    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("input_validator", input_validation_node)
    graph.add_node("specialist_dispatcher", specialist_dispatch_node)

    # Set entry point and linear flow
    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "input_validator")
    graph.add_edge("input_validator", "specialist_dispatcher")
    graph.add_edge("specialist_dispatcher", END)

    return graph.compile()


class LangGraphRouter:
    """
    Agentic router using a LangGraph StateGraph.

    Usage:
        router = LangGraphRouter()
        decision = router.route(query_text, images, trace)
    """

    def __init__(self):
        self._graph = _build_graph()

    @property
    def is_available(self) -> bool:
        return self._graph is not None

    def route(
        self,
        query_text: str,
        images: List[ImageMeta],
        trace: ExecutionTrace,
    ) -> RouteDecision:
        """
        Run the full LangGraph pipeline and return a RouteDecision.
        Raises RoutingError on validation failure.
        """
        from core.router import RoutingError

        if not self._graph:
            raise ImportError("langgraph not installed — cannot use LangGraphRouter")

        # Build initial state
        initial_state: RouterState = {
            "query_text": query_text,
            "images": [img.model_dump() for img in images],
            "image_count": len(images),
            "modalities": [img.modality.value for img in images],
        }

        # Execute graph
        result = self._graph.invoke(initial_state)

        # --- Trace: record every node's contribution ---
        trace.add(
            step="langgraph_intent_classification",
            component="LangGraphRouter.IntentClassifier",
            parameters={"query_text": query_text},
            output_summary=(
                f"Intent scores: {result.get('intent_scores', {})}, "
                f"primary: {result.get('primary_intent', 'unknown')}"
            ),
        )

        trace.add(
            step="langgraph_input_validation",
            component="LangGraphRouter.InputValidator",
            parameters={
                "image_count": len(images),
                "modalities": [img.modality.value for img in images],
            },
            output_summary=(
                f"valid={result.get('is_valid')}, "
                f"candidate_task={result.get('candidate_task')}"
            ),
        )

        # Check validation
        if not result.get("is_valid"):
            trace.add(
                step="langgraph_dispatch",
                component="LangGraphRouter.SpecialistDispatcher",
                parameters={},
                output_summary=f"REJECTED: {result.get('validation_error', 'unknown error')}",
            )
            raise RoutingError(result.get("validation_error", "Invalid input combination"))

        # Build RouteDecision
        final_task = TaskType(result["final_task"])
        sub_tasks = [SubTask(s) for s in result.get("sub_tasks", [])]

        decision = RouteDecision(
            task_type=final_task,
            sub_tasks=sub_tasks,
            reason=result.get("reason", ""),
            image_count=len(images),
            modalities=[img.modality for img in images],
        )

        trace.add(
            step="langgraph_dispatch",
            component="LangGraphRouter.SpecialistDispatcher",
            parameters={"rl_q_values": result.get("rl_q_values")},
            output_summary=f"Routed to {decision.task_type.value} — {decision.reason}",
        )

        return decision
