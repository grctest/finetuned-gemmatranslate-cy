import argparse
import json
import os
import shutil

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "google/translategemma-4b-it"
DATASET_ID = "Helsinki-NLP/opus-100"
DATASET_CONFIG = "cy-en"
SOURCE_LANG_CODE = "en"
TARGET_LANG_CODE = "cy"
SOURCE_LANG_NAME = "English"
TARGET_LANG_NAME = "Welsh"
PROCESSED_DATA_DIR = "./processed_data"
LOCAL_MODEL_DIR = "./local_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare English/Welsh TranslateGemma fine-tuning data."
    )
    parser.add_argument(
        "--textblob-token-budget",
        type=int,
        default=512,
        help="Concatenate adjacent sentence pairs up to this rendered token budget. Use 0 to disable.",
    )
    return parser.parse_args()


def build_translation_record(source_text, target_text, src_code, tgt_code):
    return {
        "task": "translation",
        "source_text": source_text,
        "target_text": target_text,
        "source_lang_code": src_code,
        "target_lang_code": tgt_code,
    }


def build_messages(record):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": record["source_lang_code"],
                    "target_lang_code": record["target_lang_code"],
                    "text": record["source_text"],
                }
            ],
        },
        {
            "role": "assistant",
            "content": record["target_text"],
        },
    ]


def is_quality_pair(source_text, target_text):
    if not source_text or not target_text:
        return False

    source_length = len(source_text)
    target_length = len(target_text)

    if source_length == 0 or target_length == 0:
        return False

    return not (
        source_length > 3 * target_length or target_length > 3 * source_length
    )


def estimate_rendered_tokens(tokenizer, record):
    rendered = tokenizer.apply_chat_template(
        build_messages(record),
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def merge_records(records):
    first = records[0]
    return {
        "task": first["task"],
        "source_text": "\n".join(record["source_text"] for record in records),
        "target_text": "\n".join(record["target_text"] for record in records),
        "source_lang_code": first["source_lang_code"],
        "target_lang_code": first["target_lang_code"],
    }


def group_into_textblobs(records, tokenizer, token_budget):
    if token_budget <= 0:
        return records

    grouped_records = []
    buffer = []

    for record in records:
        candidate_records = buffer + [record]
        merged_candidate = merge_records(candidate_records)
        candidate_length = estimate_rendered_tokens(tokenizer, merged_candidate)

        if buffer and candidate_length > token_budget:
            grouped_records.append(merge_records(buffer))
            buffer = [record]
            continue

        buffer = candidate_records

    if buffer:
        grouped_records.append(merge_records(buffer))

    return grouped_records


def ensure_directory_path(path):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)


def process_and_save(args):
    print(f"Step 1: Downloading processor ({MODEL_ID}) for token-aware data shaping...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    print(f"Step 2: Downloading OPUS-100 {DATASET_CONFIG} dataset from Hugging Face...")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    processed_splits = {}

    for split_name, split_dataset in dataset.items():
        print(f"Processing split: {split_name} (Original size: {len(split_dataset)})")
        english_to_welsh = []
        welsh_to_english = []

        for item in split_dataset:
            translations = item["translation"]
            source_text = translations.get(SOURCE_LANG_CODE, "").strip()
            target_text = translations.get(TARGET_LANG_CODE, "").strip()

            if not is_quality_pair(source_text, target_text):
                continue

            english_to_welsh.append(
                build_translation_record(
                    source_text,
                    target_text,
                    SOURCE_LANG_CODE,
                    TARGET_LANG_CODE,
                )
            )
            welsh_to_english.append(
                build_translation_record(
                    target_text,
                    source_text,
                    TARGET_LANG_CODE,
                    SOURCE_LANG_CODE,
                )
            )

        grouped_records = group_into_textblobs(
            english_to_welsh,
            tokenizer,
            args.textblob_token_budget,
        )
        grouped_records.extend(
            group_into_textblobs(
                welsh_to_english,
                tokenizer,
                args.textblob_token_budget,
            )
        )

        processed_splits[split_name] = grouped_records
        print(f" -> Processed {split_name} (New bidirectional size: {len(grouped_records)})")

    print("Saving processed data to disk...")
    if os.path.isdir(PROCESSED_DATA_DIR):
        print(" -> Removing stale processed_data directory before saving regenerated flat records")
        shutil.rmtree(PROCESSED_DATA_DIR)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    dataset_dict = DatasetDict(
        {
            split_name: Dataset.from_list(records)
            for split_name, records in processed_splits.items()
        }
    )
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

    with open(
        os.path.join(PROCESSED_DATA_DIR, "prepare_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "dataset_id": DATASET_ID,
                "dataset_config": DATASET_CONFIG,
                "source_lang_code": SOURCE_LANG_CODE,
                "target_lang_code": TARGET_LANG_CODE,
                "source_language_name": SOURCE_LANG_NAME,
                "target_language_name": TARGET_LANG_NAME,
                "textblob_token_budget": args.textblob_token_budget,
                "row_format": "flat_translation_records",
            },
            handle,
            indent=2,
        )
    print(f"Done! Data prepared and saved to {PROCESSED_DATA_DIR}.")

    print(f"\nStep 3: Downloading tokenizer and model ({MODEL_ID})...")
    ensure_directory_path(LOCAL_MODEL_DIR)
    processor.save_pretrained(LOCAL_MODEL_DIR)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(LOCAL_MODEL_DIR)

    print("\nStep 4: Post-download configuration patching...")
    config_path = os.path.join(LOCAL_MODEL_DIR, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)

        tokenizer_config["clean_up_tokenization_spaces"] = False

        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(tokenizer_config, handle, indent=2)

        print(" -> Patched tokenizer_config.json")

    template_path = os.path.join(LOCAL_MODEL_DIR, "chat_template.jinja")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as handle:
            template_content = handle.read()

        if '"cy":' in template_content:
            print(" -> Verified: 'cy' is already supported in chat_template.jinja")
        else:
            print(" -> Warning: 'cy' not found in chat_template.jinja. Injecting Welsh support...")
            updated_content = template_content.replace(
                '"en": "English",',
                '"en": "English",\n    "cy": "Welsh",',
            )
            with open(template_path, "w", encoding="utf-8") as handle:
                handle.write(updated_content)

    print(
        f"\nDone! Model, tokenizer, and English/Welsh dataset are ready in {LOCAL_MODEL_DIR} and {PROCESSED_DATA_DIR}."
    )


if __name__ == "__main__":
    process_and_save(parse_args())
