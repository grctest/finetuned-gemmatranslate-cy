import argparse
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import logging as transformers_logging
import shutil

from model_assets import build_missing_assets_error, MODEL_PATH, find_missing_model_assets, REQUIRED_PROCESSOR_FILES

MERGED_MODEL_DIR = "./final_merged_model"
FULL_MODEL_OUTPUT_DIR = "./translategemma-finetuned/full_model"
RESPONSE_TEMPLATE = "<start_of_turn>model\n"
USER_TURN_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN_TEMPLATE = "<end_of_turn>\n"

PROFILE_CONFIGS = {
    "auto": {"description": "Auto device selection (GPU if available, otherwise CPU)"},
    "gpu": {"description": "Prefer GPU and use bfloat16 where supported"},
    "cpu": {"description": "Force CPU-only inference and use float32 dtype"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TranslateGemma English/Welsh inference checks."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model path. Defaults to merged model, then full fine-tune output.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate for each translation.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(PROFILE_CONFIGS.keys()),
        default="auto",
        help="Device profile to use for inference (auto/gpu/cpu).",
    )
    return parser.parse_args()


def resolve_model_path(explicit_path=None):
    if explicit_path:
        return explicit_path

    for candidate in (MERGED_MODEL_DIR, FULL_MODEL_OUTPUT_DIR):
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No trained model was found. Run 03_merge.py for LoRA outputs or use the full fine-tuned model from 02_finetune.py --profile max_vram."
    )


def run_inference(args):
    # suppress non-critical transformers info messages
    transformers_logging.set_verbosity_error()

    try:
        finetuned_path = resolve_model_path(args.model_path)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)

    def ensure_processor_files(target_path):
        missing = find_missing_model_assets(target_path)
        processor_missing = [m for m in missing if any(p in m for p in REQUIRED_PROCESSOR_FILES)]
        if processor_missing and os.path.exists(MODEL_PATH) and target_path != MODEL_PATH:
            print(f"Copying missing processor files from {MODEL_PATH} into {target_path}...")
            os.makedirs(target_path, exist_ok=True)
            for fname in REQUIRED_PROCESSOR_FILES:
                src = os.path.join(MODEL_PATH, fname)
                dst = os.path.join(target_path, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

    def load_processor_and_model(target_path, profile):
        ensure_processor_files(target_path)
        missing_assets_error = build_missing_assets_error(target_path)
        if missing_assets_error:
            raise RuntimeError(missing_assets_error)

        processor = AutoProcessor.from_pretrained(target_path)
        tokenizer = processor.tokenizer

        if profile == "cpu":
            device_map = "cpu"
            dtype = torch.float32
        else:
            if profile == "gpu" and not torch.cuda.is_available():
                print("Warning: --profile gpu requested but no CUDA available; falling back to CPU.")
                device_map = "cpu"
                dtype = torch.float32
            else:
                device_map = "auto"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading model from {target_path} with device_map={device_map}, dtype={dtype}...")
        model = AutoModelForImageTextToText.from_pretrained(
            target_path,
            device_map=device_map,
            dtype=dtype,
        )
        return processor, tokenizer, model

    # Load baseline (original pre-finetuned) model first
    base_path = MODEL_PATH
    try:
        base_proc, base_tok, base_model = load_processor_and_model(base_path, args.profile)
    except RuntimeError as err:
        print(f"Error loading base model: {err}")
        sys.exit(1)

    # Load finetuned/merged model
    try:
        ft_proc, ft_tok, ft_model = load_processor_and_model(finetuned_path, args.profile)
    except RuntimeError as err:
        print(f"Error loading finetuned model: {err}")
        sys.exit(1)

    def translate_with(tokenizer, model, text, source_code="en", target_code="cy"):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_code,
                        "target_lang_code": target_code,
                        "text": text,
                    }
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Test translations only: baseline then finetuned
    en_text = "The train from Cardiff arrives just after noon."
    cy_text = "Bore da, sut wyt ti heddiw?"

    print("\n=== Baseline (pre-finetuned) translations ===")
    print(f"Input (EN): {en_text}")
    print("Output (CY):", translate_with(base_tok, base_model, en_text, 'en', 'cy'))
    print("\n-----------------------------")
    print(f"Input (CY): {cy_text}")
    print("Output (EN):", translate_with(base_tok, base_model, cy_text, 'cy', 'en'))

    print("\n=== Finetuned model translations ===")
    print(f"Input (EN): {en_text}")
    print("Output (CY):", translate_with(ft_tok, ft_model, en_text, 'en', 'cy'))
    print("\n-----------------------------")
    print(f"Input (CY): {cy_text}")
    print("Output (EN):", translate_with(ft_tok, ft_model, cy_text, 'cy', 'en'))


if __name__ == "__main__":
    run_inference(parse_args())