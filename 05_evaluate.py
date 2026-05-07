import argparse
import json
import os
import statistics
import subprocess
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor

DATASET_PATH = "./processed_data"
EVALUATION_DIR = "./evaluation"
MERGED_MODEL_DIR = "./final_merged_model"
FULL_MODEL_OUTPUT_DIR = "./translategemma-finetuned/full_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate English/Welsh TranslateGemma checkpoints with MetricX."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["validation", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--metricx-model",
        type=str,
        default="google/metricx-24-hybrid-large-v2p6-bfloat16",
        help="MetricX model checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--metricx-tokenizer",
        type=str,
        default="google/mt5-xl",
        help="Tokenizer required by MetricX.",
    )
    parser.add_argument("--metricx-max-input-length", type=int, default=1536)
    parser.add_argument("--metricx-batch-size", type=int, default=1)
    parser.add_argument(
        "--qe",
        action="store_true",
        help="Run MetricX in quality-estimation mode without references.",
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


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_messages(example):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": example["source_lang_code"],
                    "target_lang_code": example["target_lang_code"],
                    "text": example["source_text"],
                }
            ],
        }
    ]


def generate_predictions(args, model_path):
    dataset = load_from_disk(DATASET_PATH)[args.split]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    rows = []
    for example in dataset:
        prompt = tokenizer.apply_chat_template(
            build_messages(example),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        hypothesis = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        rows.append(
            {
                "source_text": example["source_text"],
                "target_text": example["target_text"],
                "source_lang_code": example["source_lang_code"],
                "target_lang_code": example["target_lang_code"],
                "hypothesis": hypothesis,
            }
        )

    return rows


def run_metricx(args, metricx_input_path, metricx_output_path):
    command = [
        sys.executable,
        "-m",
        "metricx24.predict",
        "--tokenizer",
        args.metricx_tokenizer,
        "--model_name_or_path",
        args.metricx_model,
        "--max_input_length",
        str(args.metricx_max_input_length),
        "--batch_size",
        str(args.metricx_batch_size),
        "--input_file",
        metricx_input_path,
        "--output_file",
        metricx_output_path,
    ]

    if args.qe:
        command.append("--qe")

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "MetricX evaluation failed. Install MetricX with `pip install git+https://github.com/google-research/metricx.git` and retry.\n"
            f"stderr:\n{result.stderr.strip()}"
        )


def main(args):
    try:
        model_path = resolve_model_path(args.model_path)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)

    if not os.path.exists(DATASET_PATH):
        print("Error: ./processed_data not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    os.makedirs(EVALUATION_DIR, exist_ok=True)

    print(f"Generating predictions from {model_path} on the {args.split} split...")
    predictions = generate_predictions(args, model_path)

    predictions_path = os.path.join(EVALUATION_DIR, f"{args.split}_predictions.jsonl")
    write_jsonl(predictions_path, predictions)

    metricx_input_path = os.path.join(EVALUATION_DIR, f"{args.split}_metricx_input.jsonl")
    metricx_output_path = os.path.join(EVALUATION_DIR, f"{args.split}_metricx_output.jsonl")

    metricx_rows = []
    for row in predictions:
        metricx_rows.append(
            {
                "source": row["source_text"],
                "hypothesis": row["hypothesis"],
                "reference": "" if args.qe else row["target_text"],
            }
        )
    write_jsonl(metricx_input_path, metricx_rows)

    print("Running MetricX scoring...")
    run_metricx(args, metricx_input_path, metricx_output_path)
    metricx_output = read_jsonl(metricx_output_path)

    scores = [row["prediction"] for row in metricx_output]
    summary = {
        "split": args.split,
        "samples": len(scores),
        "metricx_model": args.metricx_model,
        "mode": "qe" if args.qe else "reference",
        "average_metricx": statistics.mean(scores) if scores else None,
        "median_metricx": statistics.median(scores) if scores else None,
        "lower_is_better": True,
        "predictions_path": predictions_path,
        "metricx_output_path": metricx_output_path,
    }

    summary_path = os.path.join(EVALUATION_DIR, f"{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
