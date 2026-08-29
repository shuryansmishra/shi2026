"""
SatQuery AI - Bhoonidhi (ISRO) data access wrapper.

IMPORTANT (PRD section 6): the previously-drafted REST endpoint/JSON-auth
schema for Bhoonidhi was UNVERIFIED and should be treated as fabricated.
What's actually real and used here instead:

  - Bhoonidhi (https://bhoonidhi.nrsc.gov.in) is ISRO's real Earth
    Observation data hub. An API exists; request access via
    bhoonidhi@nrsc.gov.in.
  - There is a real community Python client on PyPI: `pip install bhoonidhi`,
    which exposes an NLP-driven "smart search" plus a download function that
    needs credentials from Bhoonidhi / uops.nrsc.gov.in registration.

This module is a thin, defensive wrapper around that package so the rest of
the codebase never has to import an unverified API shape directly, and so
the app degrades cleanly (with a clear error) if the package or credentials
aren't available yet -- registration approval isn't instant, so most of the
team will be building against demo_data/ for a while.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import get_settings


class BhoonidhiUnavailable(RuntimeError):
    """Raised when the bhoonidhi package isn't installed or credentials are missing."""


class BhoonidhiClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _require_ready(self) -> None:
        if not self.settings.BHOONIDHI_USER or not self.settings.BHOONIDHI_PASSWORD:
            raise BhoonidhiUnavailable(
                "BHOONIDHI_USER / BHOONIDHI_PASSWORD not set in .env. Register at "
                "https://bhoonidhi.nrsc.gov.in and https://uops.nrsc.gov.in, then "
                "set both values. Until then, use cached scenes in demo_data/."
            )
        try:
            import bhoonidhi  # noqa: F401
        except ImportError as exc:
            raise BhoonidhiUnavailable(
                "The `bhoonidhi` package is not installed. Run `pip install bhoonidhi`."
            ) from exc

    def smart_search(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """
        Thin pass-through to the community package's NLP smart search, e.g.
        "Get me Cartosat-2S data from the region of Chennai from the last 1 month".
        Returns a list of scene metadata dicts.
        """
        self._require_ready()
        import bhoonidhi

        return bhoonidhi.bhoonidhiSmartSearch(natural_language_query)  # type: ignore[attr-defined]

    def download_scene(self, scene_id: str, output_dir: str) -> Optional[str]:
        """
        Downloads a single scene by id. Requires valid Bhoonidhi/UOPS
        credentials to already be set in the environment for the underlying
        package to pick up.
        """
        self._require_ready()
        import bhoonidhi

        return bhoonidhi.download(  # type: ignore[attr-defined]
            scene_id=scene_id,
            username=self.settings.BHOONIDHI_USER,
            password=self.settings.BHOONIDHI_PASSWORD,
            output_dir=output_dir,
        )
