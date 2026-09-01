"""
SatQuery AI - Single-Image Engine (Qwen 2.5-VL Powered).

Supports:
1. Remote Inference: If QWEN_REMOTE_URL is set (Google Colab ngrok tunnel, Modal, HF Space)
2. Local Inference: If running on local GPU using Qwen2.5-VL-7B-Instruct / LoRA weights
3. Deterministic Mock Fallback: When VQA_MOCK_MODE=True or if models/tunnels are offline
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
import requests
from PIL import Image

from config import get_settings
from engines.base import deterministic_seed, MOCK_LAND_COVER_CLASSES
from models.schemas import ExecutionTrace, ImageMeta, SubTask


class SingleImageEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._local_model = None
        self._local_processor = None

    def run(
        self,
        query_text: str,
        image: ImageMeta,
        sub_tasks: List[SubTask],
        trace: ExecutionTrace,
    ) -> Dict[str, Any]:
        # 1. Check if remote Colab / Cloud endpoint is configured
        if self.settings.QWEN_REMOTE_URL and not self.settings.VQA_MOCK_MODE:
            try:
                result = self._run_remote_qwen(query_text, image, sub_tasks)
                trace.add(
                    step="single_image_inference",
                    component="SingleImageEngine (Remote Qwen2.5-VL API)",
                    parameters={"remote_url": self.settings.QWEN_REMOTE_URL, "file_id": image.file_id},
                    output_summary=f"Qwen Answer: '{result.get('generated_answer', '')[:80]}...', confidence={result.get('confidence', 0.9):.2f}",
                )
                return result
            except Exception as exc:
                print(f"[SingleImageEngine] Remote Qwen API failed: {exc}. Falling back to local/mock.")

        # 2. Check if local model weights or pipeline is active
        if not self.settings.VQA_MOCK_MODE and (self.settings.VQA_MODEL_PATH or self.settings.QWEN_MODEL_ID):
            try:
                result = self._run_local_qwen(query_text, image, sub_tasks)
                trace.add(
                    step="single_image_inference",
                    component="SingleImageEngine (Local Qwen2.5-VL)",
                    parameters={"model_id": self.settings.QWEN_MODEL_ID, "file_id": image.file_id},
                    output_summary=f"Qwen Answer: '{result.get('generated_answer', '')[:80]}...', confidence={result.get('confidence', 0.9):.2f}",
                )
                return result
            except Exception as exc:
                print(f"[SingleImageEngine] Local Qwen inference failed: {exc}. Falling back to mock.")

        # 3. Mock Fallback
        result = self._run_mock(query_text, image, sub_tasks)
        trace.add(
            step="single_image_inference",
            component="SingleImageEngine (Mock)",
            parameters={"sub_tasks": [s.value for s in sub_tasks], "file_id": image.file_id},
            output_summary=f"land_cover={result['land_cover_classes']}, confidence={result['confidence']:.2f}",
        )
        return result

    # -- Prompt Builder matching SatQuery Training Notebook ------------------

    def _build_qwen_prompt(self, query_text: str, sub_tasks: List[SubTask]) -> str:
        query_lower = query_text.lower().strip()

        # 1. Binary YES/NO prompt
        if any(w in query_lower for w in ["is there", "are there", "does it have", "does it contain", "is this", "present?"]) or "yes or no" in query_lower:
            return (
                "Answer the following satellite-image question.\n\n"
                f"Question:\n{query_text}\n\n"
                "This is a YES/NO question.\n"
                "Reply with exactly one word:\n"
                "yes\n"
                "or\n"
                "no\n\n"
                "Do not explain your answer."
            )

        # 2. Multiple Choice Questions (MCQ)
        if any(opt in query_text for opt in ["a)", "b)", "A.", "B."]) or "multiple-choice" in query_lower:
            return (
                "Answer the following multiple-choice question about the satellite image.\n\n"
                f"{query_text}\n\n"
                "Reply with ONLY the letter of the correct answer:\n"
                "a\n"
                "b\n"
                "c\n"
                "or\n"
                "d\n\n"
                "Do not provide an explanation."
            )

        # 3. Bounding-box / Grounding
        if SubTask.GROUNDING in sub_tasks or any(w in query_lower for w in ["bounding box", "coordinates", "localize", "detect", "bbox"]):
            return (
                "Look at the satellite image and answer the following bounding-box request.\n\n"
                f"{query_text}\n\n"
                "Return ONLY the bounding box coordinates in this exact format:\n\n"
                "[x1 y1, x2 y2]\n\n"
                "All coordinates must be normalized between 0 and 1.\n"
                "Do not provide any explanation or additional text."
            )

        # 4. Captioning
        if SubTask.CAPTION in sub_tasks or any(w in query_lower for w in ["describe", "caption", "summary", "summarize"]):
            return (
                "Look at the satellite image and answer the following request.\n\n"
                f"{query_text}\n\n"
                "Provide a concise factual description based only on what can be inferred from the image.\n"
                "Do not mention that you are an AI."
            )

        # 5. Default VQA
        return (
            "Look at the satellite image and answer the following question.\n\n"
            f"Question:\n{query_text}\n\n"
            "Provide a concise factual answer based strictly on the image content."
        )

    # -- Remote Inference (Colab / Modal / HuggingFace Tunnel) ----------------

    def _run_remote_qwen(self, query_text: str, image: ImageMeta, sub_tasks: List[SubTask]) -> Dict[str, Any]:
        prompt = self._build_qwen_prompt(query_text, sub_tasks)
        url = self.settings.QWEN_REMOTE_URL.rstrip("/")
        endpoint = f"{url}/predict" if not url.endswith("/predict") else url

        with open(image.path, "rb") as f:
            files = {"image": (os.path.basename(image.path), f, "image/png")}
            data = {"prompt": prompt, "query_text": query_text}
            resp = requests.post(endpoint, files=files, data=data, timeout=45)
            resp.raise_for_status()
            payload = resp.json()

        raw_answer = payload.get("answer") or payload.get("prediction") or payload.get("text", "")
        return self._parse_qwen_output(raw_answer, query_text, image, sub_tasks, source="Remote Qwen2.5-VL")

    # -- Local PyTorch Qwen2.5-VL Inference ----------------------------------

    def _load_local_qwen(self):
        if self._local_model is not None and self._local_processor is not None:
            return self._local_model, self._local_processor

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_id = self.settings.VQA_MODEL_PATH or self.settings.QWEN_MODEL_ID
        print(f"[SingleImageEngine] Loading local Qwen model from: {model_id}")

        if self.settings.QWEN_USE_4BIT and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.settings.QWEN_DEVICE if torch.cuda.is_available() else None
            )

        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28
        )

        self._local_model = model
        self._local_processor = processor
        return model, processor

    def _run_local_qwen(self, query_text: str, image: ImageMeta, sub_tasks: List[SubTask]) -> Dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = self._load_local_qwen()
        prompt = self._build_qwen_prompt(query_text, sub_tasks)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return self._parse_qwen_output(output_text, query_text, image, sub_tasks, source="Local Qwen2.5-VL")

    # -- Parsing and Structuring Output --------------------------------------

    def _parse_qwen_output(
        self, raw_text: str, query_text: str, image: ImageMeta, sub_tasks: List[SubTask], source: str = "Qwen"
    ) -> Dict[str, Any]:
        raw_text_clean = raw_text.strip()
        confidence = 0.92
        bbox_coords = None
        classes = []

        # Check for bounding box pattern [x1 y1, x2 y2]
        bbox_match = re.search(r"\[\s*([0-9.]+)\s+([0-9.]+),\s*([0-9.]+)\s+([0-9.]+)\s*\]", raw_text_clean)
        if bbox_match:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox_match.groups()]
                bbox_coords = [x1, y1, x2, y2]
            except Exception:
                pass

        # Identify land cover keywords from answer
        query_and_ans = (query_text + " " + raw_text_clean).lower()
        for known_class in MOCK_LAND_COVER_CLASSES:
            if known_class.lower() in query_and_ans:
                classes.append(known_class)
        if not classes:
            classes = ["Satellite Scene Features"]

        computed_area_ha = 15.0
        try:
            with Image.open(image.path) as img:
                w, h = img.size
                computed_area_ha = round((w * h * 100) / 10000.0, 2)
        except Exception:
            pass

        result: Dict[str, Any] = {
            "generated_answer": raw_text_clean,
            "land_cover_classes": classes,
            "confidence": confidence,
            "area_ha": computed_area_ha,
            "object_count": 1 if bbox_coords else (len(classes) if SubTask.GROUNDING in sub_tasks else None),
            "raw_scores": {c: 0.95 for c in classes},
            "notes": [f"Synthesized by {source}"],
        }

        if bbox_coords:
            result["bbox_pixel"] = bbox_coords
        if image.bbox_latlon:
            result["bbox_latlon"] = image.bbox_latlon

        return result

    # -- Mock Fallback --------------------------------------------------------

    def _run_mock(self, query_text: str, image: ImageMeta, sub_tasks: List[SubTask]) -> Dict[str, Any]:
        seed = deterministic_seed(query_text, image.file_id)
        n_classes = 1 + (seed % 3)
        classes = [MOCK_LAND_COVER_CLASSES[(seed + i) % len(MOCK_LAND_COVER_CLASSES)] for i in range(n_classes)]
        confidence = 0.6 + (seed % 35) / 100.0

        area_ha = round(5.0 + (seed % 500) / 10.0, 2)
        result: Dict[str, Any] = {
            "land_cover_classes": classes,
            "confidence": round(confidence, 2),
            "area_ha": area_ha,
            "object_count": (seed % 12) if SubTask.GROUNDING in sub_tasks else None,
            "raw_scores": {c: round(0.5 + ((seed + i) % 50) / 100.0, 2) for i, c in enumerate(classes)},
            "notes": [
                "MOCK OUTPUT -- Set QWEN_REMOTE_URL in .env to point to your Google Colab tunnel, "
                "or place local weights and set VQA_MOCK_MODE=False."
            ],
        }
        if SubTask.GROUNDING in sub_tasks and image.bbox_latlon:
            result["bbox_latlon"] = image.bbox_latlon
        return result
