"""
SatQuery AI - Contextual Bandit & Q-Learning Agentic Router.

Learns optimal tool-routing policies mapping:
  State: (query_intent_features, image_count, modalities, optical_cloud_cover)
  Action: TaskType (SINGLE_IMAGE, BI_TEMPORAL_CHANGE, CROSS_MODAL_FUSION)
Updates Q-table based on execution feedback and user interaction.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

from models.schemas import ImageMeta, Modality, TaskType


class RLRouterAgent:
    """
    Contextual Bandit & Q-Learning agent for satellite task routing.
    """
    def __init__(self, q_table_path: str = "storage/q_table.json"):
        self.q_table_path = q_table_path
        self.epsilon = 0.05  # Exploration rate (decaying)
        self.alpha = 0.15   # Learning rate
        self.gamma = 0.90   # Discount factor
        self.q_table: Dict[str, Dict[str, float]] = self._load_q_table()

    def _load_q_table(self) -> Dict[str, Dict[str, float]]:
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_q_table(self) -> None:
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        try:
            with open(self.q_table_path, "w") as f:
                json.dump(self.q_table, f, indent=2)
        except Exception:
            pass

    def _extract_state(self, query_text: str, images: List[ImageMeta]) -> str:
        q = query_text.lower()

        # Query Intent Tokens
        has_change = 1 if any(w in q for w in ("change", "difference", "delta", "before", "after", "compare")) else 0
        has_grounding = 1 if any(w in q for w in ("where", "locate", "bbox", "bounding box", "find")) else 0
        has_fusion = 1 if any(w in q for w in ("combine", "fuse", "sar", "optical", "cloud")) else 0

        # Modality features
        img_count = len(images)
        mods = sorted([img.modality.value for img in images])
        mod_key = "+".join(mods) if mods else "none"

        # Cloud Cover indicator
        optical_imgs = [img for img in images if img.modality == Modality.OPTICAL]
        max_cloud = max([img.cloud_cover_fraction or 0.0 for img in optical_imgs], default=0.0)
        cloud_high = 1 if max_cloud > 0.4 else 0

        return f"n:{img_count}_mods:{mod_key}_chg:{has_change}_grd:{has_grounding}_fus:{has_fusion}_cld:{cloud_high}"

    def select_action(
        self, query_text: str, images: List[ImageMeta], fallback_action: TaskType
    ) -> Tuple[TaskType, Dict[str, float]]:
        """
        Selects TaskType using epsilon-greedy action selection.
        Initializes unvisited states biased toward rule-based fallback.
        """
        state = self._extract_state(query_text, images)

        if state not in self.q_table:
            self.q_table[state] = {
                TaskType.SINGLE_IMAGE.value: 0.1,
                TaskType.BI_TEMPORAL_CHANGE.value: 0.1,
                TaskType.CROSS_MODAL_FUSION.value: 0.1,
            }
            # Strongly bias initially toward rule-based ground truth
            self.q_table[state][fallback_action.value] = 1.0

        q_values = self.q_table[state]

        if random.random() < self.epsilon:
            chosen_action = random.choice(list(TaskType))
        else:
            best_action_str = max(q_values.items(), key=lambda x: x[1])[0]
            chosen_action = TaskType(best_action_str)

        return chosen_action, q_values

    def update(
        self,
        query_text: str,
        images: List[ImageMeta],
        action: TaskType,
        reward: float,
    ) -> None:
        """Updates Q-values based on execution trace rewards."""
        state = self._extract_state(query_text, images)
        if state not in self.q_table:
            return

        old_val = self.q_table[state].get(action.value, 0.1)
        new_val = (1.0 - self.alpha) * old_val + self.alpha * reward
        self.q_table[state][action.value] = round(new_val, 4)
        self._save_q_table()
