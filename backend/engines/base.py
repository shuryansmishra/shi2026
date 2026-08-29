"""Shared helpers for the three vision engines."""
from __future__ import annotations

import hashlib
from typing import List


def deterministic_seed(*parts: str) -> int:
    """
    Turn arbitrary strings (query text, file ids) into a stable integer seed,
    so mock-mode outputs are deterministic per-input instead of random noise
    -- useful for demos and for writing tests against fixed expected output.
    """
    joined = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(joined).hexdigest(), 16) % (2**31)


MOCK_LAND_COVER_CLASSES: List[str] = [
    "urban/built-up", "agricultural land", "forest cover",
    "water body", "bare soil", "wetland",
]

MOCK_CHANGE_CLASSES: List[str] = [
    "new construction", "vegetation loss", "flooding",
    "deforestation", "no significant change",
]
