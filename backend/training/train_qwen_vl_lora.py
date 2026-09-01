"""
SatQuery AI - Qwen 2.5-VL Fine-Tuning Script (QLoRA 4-bit)

Fine-tunes Qwen2.5-VL or Qwen2-VL on BigEarthNet.txt or custom satellite VQA data.
Designed to run on Google Colab (T4/A100), Kaggle GPU, or local CUDA server.

Usage:
    python backend/training/train_qwen_vl_lora.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --data_path ./demo_data/vqa_dataset.json \
        --output_dir ./checkpoints/qwen2.5-vl-sat-lora
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image

# Helper to generate placeholder image if dataset specifies non-existent paths (for tests/dry-runs)
def ensure_image_exists(path: str) -> None:
    if not path or os.path.exists(path):
        return
    try:
        import numpy as np
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Create a small 224x224 dummy raster patch
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(path)
        print(f"[!] Created placeholder image at: {path}")
    except Exception as e:
        print(f"[!] Could not create placeholder image at {path}: {e}")


def format_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format standard dataset loader dictionaries into standard Qwen2-VL multi-modal messages.
    Supports single-image, bi-temporal, and optical-SAR fusion inputs.
    """
    content = []
    task_type = sample.get("task_type")

    if task_type == "change":
        img1 = sample.get("image_t1")
        img2 = sample.get("image_t2")
        if img1:
            ensure_image_exists(img1)
            content.append({"type": "image", "image": img1})
        if img2:
            ensure_image_exists(img2)
            content.append({"type": "image", "image": img2})
    elif task_type == "fusion":
        img_opt = sample.get("image")
        img_sar = sample.get("image_sar")
        if img_opt:
            ensure_image_exists(img_opt)
            content.append({"type": "image", "image": img_opt})
        if img_sar:
            ensure_image_exists(img_sar)
            content.append({"type": "image", "image": img_sar})
    else:
        img = sample.get("image")
        if img:
            ensure_image_exists(img)
            content.append({"type": "image", "image": img})

    content.append({"type": "text", "text": sample["question"]})

    messages = [
        {
            "role": "user",
            "content": content
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["answer"]}
            ]
        }
    ]
    return {"messages": messages}


def train_lora(model_id: str, data_path: str, output_dir: str, epochs: int = 3, lr: float = 2e-4):
    print("==========================================")
    print("  SatQuery AI - Qwen2.5-VL Fine-Tuner     ")
    print("==========================================")
    print(f"Base Model:    {model_id}")
    print(f"Dataset:       {data_path}")
    print(f"Output Dir:    {output_dir}")
    print(f"Epochs:        {epochs}")
    print(f"Learning Rate: {lr}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import TrainingArguments, Trainer
    except ImportError:
        print("\n[!] Dependencies missing for training. Please install:")
        print("    pip install transformers peft bitsandbytes accelerate datasets trl")
        print("\nDummy checkpoint directory created for testing.")
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            f.write('{"peft_type": "LORA", "task_type": "CAUSAL_LM"}')
        return

    # 1. Load dataset VQA samples
    print("\n[1/5] Loading and formatting dataset...")
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} samples from {data_path}")
    else:
        print(f"⚠️ Dataset path {data_path} not found. Generating dummy samples for dry run...")
        # Create a small dummy sample set
        raw_data = [
            {
                "image": "./demo_data/optical_single.tif",
                "question": "What land cover types are visible in this satellite image?",
                "answer": "The scene contains broad-leaved forest and water bodies.",
                "task_type": "vqa"
            },
            {
                "image_t1": "./demo_data/optical_t1.tif",
                "image_t2": "./demo_data/optical_t2.tif",
                "question": "Describe the change between these two acquisitions.",
                "answer": "New built-up urban structures have replaced agricultural pastures.",
                "task_type": "change"
            }
        ]
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        with open(data_path, "w") as f:
            json.dump(raw_data, f, indent=2)

    formatted_dataset = [format_sample(sample) for sample in raw_data]

    # Split into train/validation
    val_size = max(1, int(len(formatted_dataset) * 0.1)) if len(formatted_dataset) > 5 else 0
    if val_size > 0:
        train_data = formatted_dataset[val_size:]
        val_data = formatted_dataset[:val_size]
    else:
        train_data = formatted_dataset
        val_data = []
    print(f"Training split: {len(train_data)} samples, Validation split: {len(val_data)} samples")

    # 2. 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Processor and Model
    print("\n[2/5] Loading base vision-language model...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Qwen2-VL models require specifying min/max pixels to limit memory footprint
    # Especially important on T4 or limited VRAM GPUs
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 4. Prepare for PEFT/LoRA
    print("[3/5] Injecting LoRA adapters...")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Define Collator & Trainer
    print("\n[4/5] Preparing trainer configuration...")

    def extract_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
        images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        img_path = part.get("image")
                        if img_path and os.path.exists(img_path):
                            try:
                                images.append(Image.open(img_path).convert("RGB"))
                            except Exception as e:
                                print(f"Error loading image {img_path}: {e}")
        return images

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        batch_images = []
        
        for item in batch:
            messages = item["messages"]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(prompt)
            batch_images.append(extract_images(messages))

        # Check if batch has any images to avoid processing errors
        has_images = any(len(imgs) > 0 for imgs in batch_images)

        inputs = processor(
            text=texts,
            images=batch_images if has_images else None,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()
        # Set pad token labels to -100 so they are ignored by the loss function
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # batch size 1 to fit on typical GPUs
        gradient_accumulation_steps=4,  # accumulate gradients to emulate batch size 4
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=5,
        evaluation_strategy="epoch" if val_data else "no",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,  # CRITICAL: prevents removing Qwen vision inputs
        label_names=["labels"],
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        data_collator=collate_fn
    )

    # 6. Execute Trainer Loop
    print("\n[5/5] Executing fine-tuning loop...")
    trainer.train()

    # Save trained adapters
    print(f"\nSaving fine-tuning adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\n✅ Fine-tuning complete! Checkpoint saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5-VL for SatQuery AI")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="./demo_data/vqa_dataset.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen2.5-vl-sat-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()
    train_lora(args.model_id, args.data_path, args.output_dir, args.epochs, args.lr)
