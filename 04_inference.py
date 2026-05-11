import argparse
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from model_assets import build_missing_assets_error

MERGED_MODEL_DIR = "./final_merged_model"
FULL_MODEL_OUTPUT_DIR = "./translategemma-finetuned/full_model"
RESPONSE_TEMPLATE = "<start_of_turn>model\n"
USER_TURN_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN_TEMPLATE = "<end_of_turn>\n"


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
    try:
        model_path = resolve_model_path(args.model_path)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"Loading model and processor for inference from {model_path}...")
    missing_assets_error = build_missing_assets_error(model_path)
    if missing_assets_error:
        print(f"Error: {missing_assets_error}")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    def generate_from_prompt(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def translate(text, source_code="en", target_code="cy"):
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

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return generate_from_prompt(prompt)

    def ask_assistant(prompt_text):
        bos_token = tokenizer.bos_token or ""
        prompt = (
            f"{bos_token}{USER_TURN_TEMPLATE}"
            f"{prompt_text.strip()}"
            f"{END_OF_TURN_TEMPLATE}"
            f"{RESPONSE_TEMPLATE}"
        )
        return generate_from_prompt(prompt)

    print("\n--- Testing Translations ---")

    english_text = "The train from Cardiff arrives just after noon."
    print(f"Input (EN): {english_text}")
    print(f"Output (CY): {translate(english_text, 'en', 'cy')}")

    print("\n-----------------------------")

    welsh_text = "Bore da, sut wyt ti heddiw?"
    print(f"Input (CY): {welsh_text}")
    print(f"Output (EN): {translate(welsh_text, 'cy', 'en')}")

    print("\n--- Testing Assistant Prompts ---")

    english_prompt = "Write two short sentences about why bilingual education matters."
    print(f"Prompt (EN): {english_prompt}")
    print(f"Response: {ask_assistant(english_prompt)}")

    print("\n-----------------------------")

    welsh_prompt = "Ysgrifennwch ddwy frawddeg fer am bwysigrwydd addysg ddwyieithog."
    print(f"Prompt (CY): {welsh_prompt}")
    print(f"Response: {ask_assistant(welsh_prompt)}")


if __name__ == "__main__":
    run_inference(parse_args())