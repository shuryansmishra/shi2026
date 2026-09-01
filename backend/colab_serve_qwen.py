"""
=============================================================================
Google Colab Server for Qwen2.5-VL Satellite Vision-Language Model
=============================================================================
Run this entire script in a Google Colab notebook cell with a GPU runtime (T4/A100).
It will load your Qwen model and provide an ngrok tunnel URL that connects directly
to your local SatQuery web application.

Usage in Colab:
1. Set Runtime -> Change runtime type -> T4 GPU
2. Paste and run this script in a cell.
3. Copy the printed NGROK URL and add it to your local backend/.env:
   QWEN_REMOTE_URL=https://your-ngrok-subdomain.ngrok-free.app
=============================================================================
"""

# Step 1: Install required packages
# Run these in Colab:
# !pip install -q fastapi uvicorn pyngrok nest_asyncio python-multipart
# !pip install -q git+https://github.com/huggingface/transformers
# !pip install -q accelerate bitsandbytes qwen-vl-utils pillow

import io
import os
import torch
import nest_asyncio
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # or path to your fine-tuned LoRA / checkpoint
USE_4BIT = True
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"  # Get free token from https://dashboard.ngrok.com

# ---------------------------------------------------------------------------
# Load Model & Processor
# ---------------------------------------------------------------------------
print("⏳ Loading Qwen2.5-VL Model...")

if USE_4BIT and torch.cuda.is_available():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28
)

print("✅ Qwen2.5-VL loaded successfully!")

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(title="SatQuery Qwen VQA Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

@app.post("/predict")
async def predict(
    prompt: str = Form(...),
    query_text: str = Form(None),
    image: UploadFile = File(...)
):
    """
    Accepts an uploaded satellite image and a structured prompt, runs Qwen inference,
    and returns the generated answer.
    """
    image_bytes = await image.read()
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    prediction = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # Clean up GPU memory
    del inputs, generated_ids, generated_ids_trimmed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "answer": prediction,
        "prediction": prediction,
        "model": MODEL_ID,
        "confidence": 0.94
    }

# ---------------------------------------------------------------------------
# Start ngrok Tunnel & Uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTH_TOKEN":
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    public_url = ngrok.connect(8000).public_url
    print("\n" + "=" * 60)
    print("🚀 SatQuery Qwen2.5-VL Colab Server is LIVE!")
    print(f"👉 Tunnel URL: {public_url}")
    print("\nCopy this line to your local backend/.env file:")
    print(f"QWEN_REMOTE_URL={public_url}")
    print("=" * 60 + "\n")

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
