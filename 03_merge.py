import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from model_assets import MODEL_PATH, build_missing_assets_error

TRAINING_ROOT = "./translategemma-finetuned"
ADAPTER_OUTPUT_DIR = os.path.join(TRAINING_ROOT, "adapter")
ARTIFACT_METADATA_PATH = os.path.join(TRAINING_ROOT, "training_artifact.json")
MERGED_MODEL_DIR = "./final_merged_model"


def load_metadata():
    if not os.path.exists(ARTIFACT_METADATA_PATH):
        return None

    with open(ARTIFACT_METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_weights():
    metadata = load_metadata()

    if metadata and metadata.get("training_mode") == "full":
        print(
            "Error: the latest training run produced a full fine-tuned model. "
            "03_merge.py only applies to LoRA adapter outputs."
        )
        sys.exit(1)

    if not os.path.exists(ADAPTER_OUTPUT_DIR):
        print(
            "Error: ./translategemma-finetuned/adapter not found. "
            "Please run 02_finetune.py with a LoRA profile first."
        )
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("Error: ./local_model not found. Pre-downloaded base model is required.")
        sys.exit(1)

    missing_assets_error = build_missing_assets_error(MODEL_PATH)
    if missing_assets_error:
        print(f"Error: {missing_assets_error}")
        sys.exit(1)

    print("Loading base processor and model from local path...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print("Loading LoRA adapter weights...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_OUTPUT_DIR)

    print("Merging adapters into base model (this may take a lot of RAM)...")
    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(torch.bfloat16)

    print("Saving the final merged model and processor...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    processor.save_pretrained(MERGED_MODEL_DIR)

    print(f"Done! Merged model ready in {MERGED_MODEL_DIR}.")


if __name__ == "__main__":
    merge_weights()