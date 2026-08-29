"""
SatQuery AI - Agentic Router.

Primary routing uses a LangGraph StateGraph (if installed) with three nodes:
  1. IntentClassifier — scores query text for intent tokens
  2. InputValidator — validates image count × modality combinations
  3. SpecialistDispatcher — selects TaskType + sub-tasks with RL refinement

Fallback: a rule-based table on (image_count, modality) — inherently
inspectable, free, instant, and injection-proof (PRD section 4.2).

Routing rules (from PRD section 4.2):
    1 image                              -> single-image branch
    2 images, same modality              -> bi-temporal change branch
    2 images, mixed modality (opt + sar) -> cross-modal fusion branch
"""
from __future__ import annotations

from typing import List

from models.schemas import (
    ImageMeta,
    Modality,
    RouteDecision,
    SubTask,
    TaskType,
    ExecutionTrace,
)


class RoutingError(ValueError):
    """Raised when the given set of images cannot be routed to any task."""


# ---------------------------------------------------------------------------
# Try to import LangGraph router — graceful fallback if not available
# ---------------------------------------------------------------------------

try:
    from core.langgraph_router import LangGraphRouter
    _LANGGRAPH_ROUTER = LangGraphRouter()
    HAS_LANGGRAPH = _LANGGRAPH_ROUTER.is_available
except (ImportError, Exception):
    _LANGGRAPH_ROUTER = None
    HAS_LANGGRAPH = False


class AgenticRouter:
    def route(self, query_text: str, images: List[ImageMeta], trace: ExecutionTrace) -> RouteDecision:
        # --- Primary: LangGraph StateGraph router ---
        if HAS_LANGGRAPH and _LANGGRAPH_ROUTER is not None:
            try:
                return _LANGGRAPH_ROUTER.route(query_text, images, trace)
            except RoutingError:
                raise  # Propagate validation errors
            except Exception:
                # LangGraph internal failure — fall through to rule-based
                trace.add(
                    step="route_fallback",
                    component="AgenticRouter",
                    parameters={},
                    output_summary="LangGraph router failed; falling back to rule-based routing.",
                )

        # --- Fallback: rule-based router with RL refinement ---
        return self._rule_based_route(query_text, images, trace)

    def _rule_based_route(self, query_text: str, images: List[ImageMeta], trace: ExecutionTrace) -> RouteDecision:
        """Original rule-based routing logic with RL integration."""
        n = len(images)
        modalities = [img.modality for img in images]

        try:
            from core.rl_router import RLRouterAgent
            rl_agent = RLRouterAgent()
        except ImportError:
            rl_agent = None

        if n == 1:
            fallback_task = TaskType.SINGLE_IMAGE
            reason = "Exactly one image supplied -> single-image VQA/caption/grounding branch."
        elif n == 2 and modalities[0] == modalities[1] and modalities[0] != Modality.UNKNOWN:
            fallback_task = TaskType.BI_TEMPORAL_CHANGE
            reason = (
                f"Two images of the same modality ({modalities[0].value}) supplied "
                "-> treated as a bi-temporal pair for change detection."
            )
        elif n == 2 and {Modality.OPTICAL, Modality.SAR} == set(modalities):
            fallback_task = TaskType.CROSS_MODAL_FUSION
            reason = "One optical and one SAR image supplied -> cross-modal fusion branch."
        else:
            trace.add(
                step="route",
                component="AgenticRouter (rule-based fallback)",
                parameters={"image_count": n, "modalities": [m.value for m in modalities]},
                output_summary="REJECTED: invalid image combination for any known task.",
            )
            raise RoutingError(
                f"Cannot route {n} image(s) with modalities {[m.value for m in modalities]}. "
                "Valid combos: 1 image; 2 images same modality; 2 images optical+SAR."
            )
            
        # RL Integration: Use RL to refine the routing decision
        task_type = fallback_task
        if rl_agent:
            chosen, q_vals = rl_agent.select_action(query_text, images, fallback_task)

            # Only override the structural rule if RL has a strictly higher
            # Q-value for its chosen action than for the fallback action.
            fallback_q = q_vals.get(fallback_task.value, 0.0)
            chosen_q = q_vals.get(chosen.value, 0.0)
            if chosen != fallback_task and chosen_q > fallback_q:
                task_type = chosen
                reward = -0.5
            else:
                task_type = fallback_task
                reward = 1.0

            rl_agent.update(query_text, images, task_type, reward)

            if task_type != fallback_task:
                reason = f"RL Router dynamically reassigned to {task_type.value} (overriding rules, Q={chosen_q:.3f} > {fallback_q:.3f})."
            else:
                reason += " (RL Q-Learning confirmed)"

        if task_type == TaskType.SINGLE_IMAGE:
            sub_tasks = self._single_image_subtasks(query_text)
        elif task_type == TaskType.BI_TEMPORAL_CHANGE:
            sub_tasks = [SubTask.CHANGE_VQA, SubTask.CHANGE_DESCRIPTION]
        else:
            sub_tasks = [SubTask.FUSION_ANALYSIS]

        decision = RouteDecision(
            task_type=task_type,
            sub_tasks=sub_tasks,
            reason=reason,
            image_count=n,
            modalities=modalities,
        )

        trace.add(
            step="route",
            component="AgenticRouter (rule-based fallback with RL)",
            parameters={"image_count": n, "modalities": [m.value for m in modalities]},
            output_summary=f"Routed to {decision.task_type.value} ({decision.reason})",
        )
        return decision

    @staticmethod
    def _single_image_subtasks(query_text: str) -> List[SubTask]:
        """
        Single-image branch is ALWAYS VQA, plus either captioning or grounding
        per the PS's mandatory requirement. We pick grounding when the query
        implies localisation (where/locate/bounding box), else captioning.
        """
        q = query_text.lower()
        localisation_terms = ("where", "locate", "bounding box", "point to", "find the")
        if any(t in q for t in localisation_terms):
            return [SubTask.VQA, SubTask.GROUNDING]
        return [SubTask.VQA, SubTask.CAPTION]
