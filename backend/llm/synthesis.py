"""
SatQuery AI - LLM Synthesis Layer.

This is the ONLY component allowed to produce prose, and it is deliberately
kept "dumb": its system prompt instructs it to answer using nothing but the
locked EvidenceObject (PRD section 4.2). It cannot compute a new number,
invent a coordinate, or contradict the evidence -- if the evidence engine
didn't put a number in, the LLM cannot report it.

Two modes:
  - LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY set -> calls the real API.
  - No key set / LLM_PROVIDER=none -> deterministic template renderer, so the
    whole pipeline (including this layer) runs with zero external dependencies
    or cost while the team builds everything else.
"""
from __future__ import annotations

import os
from typing import Optional

from config import get_settings
from models.schemas import EvidenceObject, ExecutionTrace, RouteDecision

SYSTEM_PROMPT = """You are the answer-synthesis layer of SatQuery AI, an ISRO \
remote-sensing assistant. You will be given a structured Evidence object \
produced by deterministic computer-vision and GIS code. Answer the user's \
question using ONLY the values in that object. Do not invent, estimate, or \
adjust any number, coordinate, class name, or confidence value that is not \
already present in the Evidence object. If the Evidence object does not \
contain enough information to answer part of the question, say so plainly \
instead of guessing. Keep the answer to 2-4 sentences, factual and precise."""


class LLMSynthesis:
    def __init__(self) -> None:
        self.settings = get_settings()

    def synthesize(
        self,
        query_text: str,
        route: RouteDecision,
        evidence: EvidenceObject,
        trace: ExecutionTrace,
    ) -> str:
        if self.settings.LLM_PROVIDER == "anthropic" and self.settings.ANTHROPIC_API_KEY:
            try:
                answer = self._call_anthropic(query_text, evidence)
                method = "anthropic_api"
            except Exception:
                answer = self._template_answer(query_text, route, evidence)
                method = "grounded_template_fallback"
        else:
            # Try local HF pipeline if available, else use grounded template renderer
            try:
                answer = self._call_local_llm(query_text, evidence)
                method = "local_hf_pipeline"
            except Exception:
                answer = self._template_answer(query_text, route, evidence)
                method = "grounded_template_synthesis"

        trace.add(
            step="synthesize_answer",
            component=f"LLMSynthesis ({method})",
            parameters={"query_text": query_text},
            output_summary=answer,
        )
        return answer

    # -- real LLM call --------------------------------------------------------

    def _call_anthropic(self, query_text: str, evidence: EvidenceObject) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. Run `pip install anthropic` "
                "or set LLM_PROVIDER=none to use the template fallback."
            ) from exc

        client = anthropic.Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=self.settings.ANTHROPIC_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"User question: {query_text}\n\n"
                        f"Evidence object (JSON):\n{evidence.model_dump_json(indent=2)}"
                    ),
                }
            ],
        )
        text_blocks = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        return " ".join(text_blocks).strip()

    # -- local HF pipeline ----------------------------------------------------

    def _call_local_llm(self, query_text: str, evidence: EvidenceObject) -> str:
        model_path = self.settings.VQA_MODEL_PATH
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError("Local model path not found. Falling back to grounded template synthesis.")

        import torch
        from transformers import pipeline

        if not hasattr(self, '_hf_pipeline'):
            self._hf_pipeline = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\nUser question: {query_text}\n"
            f"Evidence: {evidence.model_dump_json()}\n"
            f"<|assistant|>\n"
        )

        outputs = self._hf_pipeline(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self._hf_pipeline.tokenizer.eos_token_id
        )
        
        generated = outputs[0]["generated_text"]
        answer = generated.split("<|assistant|>\n")[-1].strip()
        return answer

    # -- zero-dependency fallback ---------------------------------------------

    def _template_answer(self, query_text: str, route: RouteDecision, evidence: EvidenceObject) -> str:
        parts = []
        classes = evidence.land_cover_classes or evidence.change_classes
        if classes:
            parts.append(f"Analysis identified the following land cover/change categories: {', '.join(classes)}.")
        
        if evidence.area_ha is not None:
            parts.append(f"The estimated spatial area is {evidence.area_ha:.2f} hectares.")
            
        if evidence.change_area_ha is not None:
            parts.append(f"The estimated bi-temporal change area is {evidence.change_area_ha:.2f} hectares.")
            
        if evidence.object_count is not None:
            parts.append(f"Detected object count: {evidence.object_count}.")

        if evidence.bbox_latlon:
            b = evidence.bbox_latlon
            parts.append(f"Geographic Bounding Box: [{b.min_lon:.4f}, {b.min_lat:.4f}, {b.max_lon:.4f}, {b.max_lat:.4f}].")

        if evidence.confidence is not None:
            parts.append(f"Model confidence: {evidence.confidence:.0%}.")

        if evidence.notes:
            ssim_notes = [n for n in evidence.notes if "SSIM" in n or "Structural" in n]
            if ssim_notes:
                parts.append(f"({ssim_notes[0]})")

        if not parts:
            parts.append("No structured evidence was available to answer this query.")
            
        return " ".join(parts)
