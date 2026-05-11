import json
import os
import sys

import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from model_assets import MODEL_PATH, build_missing_assets_error

TRAINING_ROOT = "./translategemma-finetuned"
ADAPTER_OUTPUT_DIR = os.path.join(TRAINING_ROOT, "adapter")
ARTIFACT_METADATA_PATH = os.path.join(TRAINING_ROOT, "training_artifact.json")
MERGED_MODEL_DIR = "./final_merged_model"

PROFILE_CONFIGS = {
    "auto": {
        "description": "Auto device selection (GPU if available, otherwise CPU)",
    },
    "gpu": {
        "description": "Prefer GPU and use bfloat16 where supported",
    },
    "cpu": {
        "description": "Force CPU-only merge and use float32 dtype",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into the base model.")
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(PROFILE_CONFIGS.keys()),
        default="auto",
        help="Device profile to use for merging (auto/gpu/cpu).",
    )
    return parser.parse_args()


def load_metadata():
    if not os.path.exists(ARTIFACT_METADATA_PATH):
        return None

    with open(ARTIFACT_METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_weights(args):
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

    # Resolve device_map and dtype based on profile
    profile = args.profile
    if profile == "cpu":
        device_map = "cpu"
        dtype = torch.float32
    else:
        # auto or gpu
        if profile == "gpu" and not torch.cuda.is_available():
            print("Warning: --profile gpu requested but no CUDA available; falling back to CPU.")
            device_map = "cpu"
            dtype = torch.float32
        else:
            device_map = "auto"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model with device_map={device_map}, dtype={dtype}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        dtype=dtype,
    )

    print("Loading LoRA adapter weights...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_OUTPUT_DIR)

    print("Merging adapters into base model (this may take a lot of RAM)...")
    merged_model = model.merge_and_unload()
    # Move merged model to a compact dtype suitable for the chosen device
    if profile == "cpu":
        # keep float32 on CPU
        merged_model = merged_model.to(torch.float32)
    else:
        # prefer bfloat16 on GPU if available
        merged_model = merged_model.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    print("Saving the final merged model and processor...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    processor.save_pretrained(MERGED_MODEL_DIR)

    print(f"Done! Merged model ready in {MERGED_MODEL_DIR}.")


if __name__ == "__main__":
    args = parse_args()
    merge_weights(args)