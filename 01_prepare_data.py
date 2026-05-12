import argparse
import json
import os
import random
import sys

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from model_assets import MODEL_ID, MODEL_PATH, ensure_local_model_snapshot

PROCESSED_DATA_DIR = "./processed_data"
DEFAULT_TRANSLATION_DIRECTIONS = ["en-cy"]
SUPPORTED_TRANSLATION_DIRECTIONS = {"en-cy", "cy-en"}
RESPONSE_TEMPLATE = "<start_of_turn>model\n"
USER_TURN_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN_TEMPLATE = "<end_of_turn>\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Welsh fine-tuning data.")
    parser.add_argument(
        "--recipe",
        type=str,
        default="data_recipe.json",
        help="Path to the JSON data recipe.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass interactive prompt.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip checking or downloading the local TranslateGemma model snapshot.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of CPU cores to use for dataset processing.",
    )
    args = parser.parse_args()

    max_cores = os.cpu_count() or 1
    args.num_proc = max(1, min(args.num_proc, max_cores))
    
    return args


def ensure_local_model(args):
    if args.skip_model_download:
        print(f"Skipping local model sync for {MODEL_ID}.")
        return

    print(f"Checking local multimodal model snapshot in {MODEL_PATH}...")
    missing_assets = ensure_local_model_snapshot(MODEL_PATH, MODEL_ID)
    if missing_assets:
        rendered = "\n  - ".join(missing_assets)
        raise RuntimeError(
            "TranslateGemma snapshot is still incomplete after download attempt. Missing:\n"
            f"  - {rendered}"
        )
    print("Local TranslateGemma snapshot is ready.")

def load_and_slice(name, path, usage_factor, config=None, split="train"):
    ds = load_dataset(path, config, split=split)
    ds = ds.shuffle(seed=42)
    num_rows = int(len(ds) * usage_factor)
    return ds.select(range(num_rows)), len(ds), num_rows

def get_en_to_cy_templates(en, cy, cy_def):
    templates = [
        (f"How do you say '{en}' in Welsh? Please provide a definition.", f"The Welsh term for '{en}' is '{cy}'. It refers to {cy_def}"),
        (f"What is the Welsh equivalent of the term '{en}'? Explain what it means.", f"In Welsh, '{en}' is translated as '{cy}'. This term is used to describe {cy_def}"),
        (f"I'm looking for the Welsh word for '{en}'. Could you also explain its meaning?", f"Certainly! The Welsh word is '{cy}', which means {cy_def}"),
        (f"Can you give me the Welsh translation for '{en}' along with its definition?", f"The translation is '{cy}'. To define it: {cy_def}"),
        (f"Define '{en}' and give me its Welsh terminology.", f"In Welsh, it's called '{cy}'. The definition provided is: {cy_def}")
    ]
    return random.choice(templates)

def get_cy_to_en_templates(cy, en, en_def):
    templates = [
        (f"Beth yw'r gair Saesneg am '{cy}', a beth yw'r diffiniad?", f"Y term Saesneg ar gyfer '{cy}' yw '{en}'. Mae'n golygu {en_def}"),
        (f"Sut ydych chi'n dweud '{cy}' yn Saesneg? Eglurwch yr ystyr hefyd.", f"'{en}' yw'r cyfystyron Saesneg ar gyfer '{cy}'. Dyma'r esboniad: {en_def}"),
        (f"Rhowch y term Saesneg ar gyfer '{cy}' ynghyd â'i ddiffiniad.", f"Y gair Saesneg yw '{en}', ac fe'i diffinnir fel a ganlyn: {en_def}"),
        (f"Oes gennych chi'r cyfieithiad Saesneg ar gyfer '{cy}' a chyd-destun?", f"Wrth gwrs. '{en}' yw'r gair yn Saesneg. O ran yr ystyr: {en_def}"),
        (f"Beth yw ystyr '{cy}' a beth yw'r term Saesneg cyfatebol?", f"Y term Saesneg cyfatebol yw '{en}'. Eglurhad o hyn yw: {en_def}")
    ]
    return random.choice(templates)


def normalize_translation_directions(info):
    directions = info.get("directions", DEFAULT_TRANSLATION_DIRECTIONS)
    normalized = []

    for direction in directions:
        cleaned_direction = str(direction).strip().lower().replace("_", "-")
        if cleaned_direction not in SUPPORTED_TRANSLATION_DIRECTIONS:
            raise ValueError(
                f"Unsupported translation direction '{direction}'. "
                f"Expected one of: {sorted(SUPPORTED_TRANSLATION_DIRECTIONS)}"
            )
        if cleaned_direction not in normalized:
            normalized.append(cleaned_direction)

    if not normalized:
        raise ValueError("At least one translation direction must be configured.")

    return normalized


def build_translation_records(en_text, cy_text, directions):
    records = []

    en_text = str(en_text).strip() if en_text else ""
    cy_text = str(cy_text).strip() if cy_text else ""

    if not en_text or not cy_text:
        return records

    for direction in directions:
        if direction == "en-cy":
            records.append(
                {
                    "task": "translation",
                    "source_text": en_text,
                    "target_text": cy_text,
                    "source_lang_code": "en",
                    "target_lang_code": "cy",
                }
            )
        elif direction == "cy-en":
            records.append(
                {
                    "task": "translation",
                    "source_text": cy_text,
                    "target_text": en_text,
                    "source_lang_code": "cy",
                    "target_lang_code": "en",
                }
            )

    return records


def build_empty_processed_dataset():
    return Dataset.from_dict(
        {
            "task": [],
            "source_text": [],
            "target_text": [],
            "source_lang_code": [],
            "target_lang_code": [],
        }
    )


def build_training_prompt(example, tokenizer):
    if example["task"] == "translation":
        prompt = tokenizer.apply_chat_template(
            [
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
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    elif example["task"] == "instruction":
        bos_token = tokenizer.bos_token or ""
        prompt = (
            f"{bos_token}{USER_TURN_TEMPLATE}"
            f"{str(example['source_text']).strip()}"
            f"{END_OF_TURN_TEMPLATE}"
            f"{RESPONSE_TEMPLATE}"
        )
    else:
        raise ValueError(f"Unsupported task type '{example['task']}'.")

    if not prompt.endswith(RESPONSE_TEMPLATE):
        raise ValueError("Prompt no longer ends at the assistant response boundary.")

    return prompt


def build_training_completion(target_text):
    return f"{str(target_text).strip()}{END_OF_TURN_TEMPLATE}"


def parse_target_ratio(recipe):
    target_ratio = recipe.get("meta_strategy", {}).get("target_ratio", "70:30")

    if isinstance(target_ratio, str):
        parts = target_ratio.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Unsupported target_ratio '{target_ratio}'. Expected format like '70:30'."
            )
        translation_ratio, instruction_ratio = (float(part.strip()) for part in parts)
    elif isinstance(target_ratio, (list, tuple)) and len(target_ratio) == 2:
        translation_ratio, instruction_ratio = (float(part) for part in target_ratio)
    else:
        raise ValueError(
            f"Unsupported target_ratio '{target_ratio}'. Expected '70:30' or a two-item list."
        )

    ratio_sum = translation_ratio + instruction_ratio
    if translation_ratio <= 0 or instruction_ratio <= 0 or ratio_sum <= 0:
        raise ValueError("target_ratio must contain two positive values.")

    return translation_ratio / ratio_sum, instruction_ratio / ratio_sum


import re
import re
from transformers import AutoTokenizer

import concurrent.futures

def enforce_sequence_lengths(ds, tokenizer, max_tokens=2048, num_proc=1):
    if len(ds) == 0:
        return ds
    # Conservative policy:
    # - Drop rows whose source alone is too long to leave useful room for a completion
    # - Do not split rows; keep the full prompt side intact
    # - Truncate the target using the exact training prompt/completion format from 02_finetune.py

    # 1. Fast, Rust-level batched tokenization to get src/tgt token lengths
    def compute_lengths(batch):
        src_enc = tokenizer(batch["source_text"], add_special_tokens=False)
        tgt_enc = tokenizer(batch["target_text"], add_special_tokens=False)
        return {
            "src_len": [len(x) for x in src_enc["input_ids"]],
            "tgt_len": [len(x) for x in tgt_enc["input_ids"]]
        }

    ds = ds.map(compute_lengths, batched=True, num_proc=num_proc, desc="Tokenizing")

    # 2. Drop rows whose SOURCE leaves no realistic room for a target.
    #    User-requested cutoff: 1337 tokens for source (apply to both translations and instructions).
    SRC_MAX_ALLOWED = 1337
    def is_src_too_long(x):
        return x.get("src_len", 0) > SRC_MAX_ALLOWED

    too_long_src_ds = ds.filter(is_src_too_long, num_proc=num_proc, desc="Filter: Too-long sources")
    if len(too_long_src_ds) > 0:
        print(f"      -> Dropping {len(too_long_src_ds):,} rows whose source > {SRC_MAX_ALLOWED} tokens (no room for target).")
        # Remove these rows from the working dataset
        ds = ds.filter(lambda x: not is_src_too_long(x), num_proc=num_proc, desc="Remove too-long sources")

    # 3. Identify oversized rows (either side exceeds max_tokens)
    def is_oversized(x):
        return x["src_len"] > max_tokens or x["tgt_len"] > max_tokens

    good_ds = ds.filter(lambda x: not is_oversized(x), num_proc=num_proc, desc="Bucket: Under Limit")
    long_ds = ds.filter(is_oversized, num_proc=num_proc, desc="Bucket: Over Limit")

    # Remove temp length columns from good bucket
    good_ds = good_ds.remove_columns([c for c in ["src_len", "tgt_len"] if c in good_ds.column_names])

    if len(long_ds) == 0:
        # Final truncation pass (ensure targets fit); see step 5 below
        final_ds = good_ds
    else:
        print(f"      -> Analyzed {len(ds):,} rows: {len(good_ds):,} OK, {len(long_ds):,} oversized.")

        # For long bucket: we'll truncate targets for ALL oversized rows rather than attempt splitting.
        final_pieces = [good_ds]
        long_ds = long_ds.remove_columns([c for c in ["src_len", "tgt_len"] if c in long_ds.column_names])
        final_pieces.append(long_ds)
        final_ds = concatenate_datasets(final_pieces)

    # 4. Final pass: truncate targets so the exact finetune prompt+completion fit max_tokens.

    def truncate_target(example):
        tgt = example.get("target_text", "") or ""

        prompt = build_training_prompt(example, tokenizer)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_suffix_ids = tokenizer.encode(
            build_training_completion(""),
            add_special_tokens=False,
        )
        pad_len = 1 if getattr(tokenizer, "pad_token_id", None) is not None else 0
        allowed = max(0, max_tokens - pad_len - len(prompt_ids) - len(completion_suffix_ids))

        if allowed <= 0:
            example["target_text"] = ""
            return example

        tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
        if len(tgt_ids) > allowed:
            example["target_text"] = tokenizer.decode(
                tgt_ids[:allowed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

        return example

    final_ds = final_ds.map(truncate_target, num_proc=num_proc, desc="Truncating targets")
    rows_before_empty_filter = len(final_ds)
    final_ds = final_ds.filter(
        lambda x: bool((x.get("target_text", "") or "").strip()),
        num_proc=num_proc,
        desc="Filter: Empty targets",
    )

    removed_empty_targets = rows_before_empty_filter - len(final_ds)
    if removed_empty_targets > 0:
        print(f"      -> Dropped {removed_empty_targets:,} rows after truncation produced an empty target.")

    return final_ds


def split_long_sequences(df, max_tokens, tokenizer):
    def get_chunks(text, limit):
        if not text:
            return []
        
        # Check actual token length instead of character estimate
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= limit:
            return [text.strip()]
            
        # Find a smart split point near the middle of the text characters
        mid = len(text) // 2
        search_start = max(0, mid - (len(text) // 4))
        search_end = min(len(text), mid + (len(text) // 4))
        search_area = text[search_start:search_end]
        
        split_idx = -1
        for delimiter in ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ']:
            idx = search_area.rfind(delimiter)
            if idx != -1:
                split_idx = search_start + idx + len(delimiter)
                break
                
        if split_idx == -1:
            split_idx = mid
            
        part1 = text[:split_idx].strip()
        part2 = text[split_idx:].strip()
        
        chunks = []
        if len(tokenizer.encode(part1, add_special_tokens=False)) > limit:
            chunks.extend(get_chunks(part1, limit))
        elif part1:
            chunks.append(part1)
            
        if len(tokenizer.encode(part2, add_special_tokens=False)) > limit:
            chunks.extend(get_chunks(part2, limit))
        elif part2:
            chunks.append(part2)
            
        return chunks

    def process_row(row_tuple):
        _, row = row_tuple
        src = str(row.get("source_text", ""))
        tgt = str(row.get("target_text", ""))
        task = row.get("task", "")
        
        src_tokens = len(tokenizer.encode(src, add_special_tokens=False))
        tgt_tokens = len(tokenizer.encode(tgt, add_special_tokens=False))
        
        if src_tokens <= max_tokens and tgt_tokens <= max_tokens:
            return [row]
            
        src_chunks = get_chunks(src, max_tokens)
        tgt_chunks = get_chunks(tgt, max_tokens)
        
        local_new_rows = []
        if task == "translation":
            limit = min(len(src_chunks), len(tgt_chunks))
            for i in range(limit):
                new_row = row.copy()
                new_row['source_text'] = src_chunks[i]
                new_row['target_text'] = tgt_chunks[i]
                local_new_rows.append(new_row)
        else:
            limit = max(len(src_chunks), len(tgt_chunks))
            for i in range(limit):
                new_row = row.copy()
                s = src_chunks[i] if i < len(src_chunks) else (src_chunks[-1] if len(src_chunks) > 0 else "")
                t = tgt_chunks[i] if i < len(tgt_chunks) else (tgt_chunks[-1] if len(tgt_chunks) > 0 else "")
                new_row['source_text'] = s
                new_row['target_text'] = t
                local_new_rows.append(new_row)
        return local_new_rows

    total_rows = len(df)
    new_rows = []
    
    max_workers = (os.cpu_count() or 1) * 2
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row_tuple) for row_tuple in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            new_rows.extend(future.result())
                
    return pd.DataFrame(new_rows)


def summarize_translation_directions(df):
    translation_df = df[df["task"] == "translation"].copy()
    if translation_df.empty:
        return {}

    translation_df["direction"] = (
        translation_df["source_lang_code"] + "->" + translation_df["target_lang_code"]
    )
    direction_counts = translation_df["direction"].value_counts().sort_index()
    return {direction: int(count) for direction, count in direction_counts.items()}


def rebalance_dataset(df, recipe):
    translation_ratio, instruction_ratio = parse_target_ratio(recipe)
    translation_df = df[df["task"] == "translation"].copy()
    instruction_df = df[df["task"] == "instruction"].copy()

    if translation_df.empty or instruction_df.empty:
        return (
            df.sample(frac=1, random_state=42).reset_index(drop=True),
            {
                "rebalanced": False,
                "translation_target_ratio": translation_ratio,
                "instruction_target_ratio": instruction_ratio,
                "rows_removed_by_rebalancing": 0,
                "translation_direction_counts": summarize_translation_directions(df),
            },
        )

    translation_df["direction"] = (
        translation_df["source_lang_code"] + "->" + translation_df["target_lang_code"]
    )
    balanced_translation_df = translation_df
    expected_directions = ["en->cy", "cy->en"]
    if all(direction in translation_df["direction"].values for direction in expected_directions):
        target_per_direction = min(
            len(translation_df[translation_df["direction"] == direction])
            for direction in expected_directions
        )
        balanced_translation_df = (
            translation_df.groupby("direction", group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=target_per_direction,
                    random_state=42,
                )
                if group.name in expected_directions and len(group) > target_per_direction
                else group
            )
            .reset_index(drop=True)
        )

    translation_available = len(balanced_translation_df)
    instruction_available = len(instruction_df)

    if translation_available / translation_ratio <= instruction_available / instruction_ratio:
        target_translation_rows = translation_available
        target_instruction_rows = min(
            instruction_available,
            int(round(target_translation_rows * instruction_ratio / translation_ratio)),
        )
    else:
        target_instruction_rows = instruction_available
        target_translation_rows = min(
            translation_available,
            int(round(target_instruction_rows * translation_ratio / instruction_ratio)),
        )

    sampled_translation_df = (
        balanced_translation_df.sample(n=target_translation_rows, random_state=42)
        if len(balanced_translation_df) > target_translation_rows
        else balanced_translation_df
    )
    sampled_instruction_df = (
        instruction_df.sample(n=target_instruction_rows, random_state=42)
        if len(instruction_df) > target_instruction_rows
        else instruction_df
    )

    rebalanced_df = (
        pd.concat([sampled_translation_df, sampled_instruction_df], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return (
        rebalanced_df,
        {
            "rebalanced": True,
            "translation_target_ratio": translation_ratio,
            "instruction_target_ratio": instruction_ratio,
            "rows_removed_by_rebalancing": int(len(df) - len(rebalanced_df)),
            "translation_direction_counts": summarize_translation_directions(rebalanced_df),
        },
    )


def build_stratify_key(df):
    return df["task"] + ":" + df["source_lang_code"] + "->" + df["target_lang_code"]


def resolve_eval_size(recipe, total_rows):
    configured_eval_size = recipe.get("meta_strategy", {}).get("eval_size", 1000)

    if isinstance(configured_eval_size, float):
        if not 0 < configured_eval_size < 1:
            raise ValueError("meta_strategy.eval_size as a float must be between 0 and 1.")
        eval_size = int(round(total_rows * configured_eval_size))
    else:
        eval_size = int(configured_eval_size)

    if total_rows < 2:
        raise ValueError("At least two rows are required to create a train/eval split.")

    return max(1, min(eval_size, total_rows - 1))


def compute_stratified_eval_counts(stratum_counts, eval_size):
    max_eval_by_group = {key: max(count - 1, 0) for key, count in stratum_counts.items()}
    eligible_groups = [key for key, count in max_eval_by_group.items() if count > 0]

    if not eligible_groups:
        raise ValueError("No strata contain enough rows to support a held-out split.")

    capped_eval_size = min(eval_size, sum(max_eval_by_group.values()))
    allocations = {key: 0 for key in stratum_counts}
    ordered_groups = sorted(eligible_groups, key=lambda key: (-stratum_counts[key], key))

    for key in ordered_groups[: min(capped_eval_size, len(eligible_groups))]:
        allocations[key] = 1

    remaining = capped_eval_size - sum(allocations.values())
    total_rows = sum(stratum_counts.values())
    raw_targets = {
        key: (count / total_rows) * capped_eval_size for key, count in stratum_counts.items()
    }

    while remaining > 0:
        candidates = [
            key for key in eligible_groups if allocations[key] < max_eval_by_group[key]
        ]
        if not candidates:
            break

        candidates.sort(
            key=lambda key: (
                raw_targets[key] - allocations[key],
                stratum_counts[key] - allocations[key],
                key,
            ),
            reverse=True,
        )
        allocations[candidates[0]] += 1
        remaining -= 1

    return allocations


def split_dataset_stratified(df, recipe):
    eval_size = resolve_eval_size(recipe, len(df))
    df = df.copy()
    df["stratify_key"] = build_stratify_key(df)
    stratum_counts = df["stratify_key"].value_counts().sort_index().to_dict()
    eval_counts = compute_stratified_eval_counts(stratum_counts, eval_size)

    eval_parts = []
    for stratify_key, sample_count in eval_counts.items():
        if sample_count <= 0:
            continue
        stratum_df = df[df["stratify_key"] == stratify_key]
        eval_parts.append(stratum_df.sample(n=sample_count, random_state=42))

    if not eval_parts:
        raise ValueError("Failed to sample any held-out evaluation rows.")

    eval_df = pd.concat(eval_parts).sort_index()
    train_df = df.drop(index=eval_df.index)

    train_df = train_df.drop(columns=["stratify_key"]).sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df = eval_df.drop(columns=["stratify_key"]).sample(frac=1, random_state=42).reset_index(drop=True)

    split_summary = {
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "eval_stratification": {
            key: int(count) for key, count in sorted(eval_counts.items()) if count > 0
        },
    }
    return train_df, eval_df, split_summary

def parse_termcymru(ds, directions):
    records = []
    counts = {
        "translations": 0,
        "translation_directions": {"en->cy": 0, "cy->en": 0},
        "en_instructions": 0,
        "cy_instructions": 0,
    }
    
    for row in ds:
        # Columns in TermCymru are in Welsh
        en = row.get("Saesneg") or ""
        cy = row.get("Cymraeg") or ""
        en_def = row.get("Diffiniad Saesneg") or ""
        cy_def = row.get("Diffiniad Cymraeg") or ""

        # Normalize any whitespace or purely empty values
        en, cy = str(en).strip(), str(cy).strip()
        en_def, cy_def = str(en_def).strip(), str(cy_def).strip()
        
        null_vals = ["none", "null", ""]

        if en and cy and en.lower() not in null_vals and cy.lower() not in null_vals:
            translation_records = build_translation_records(en, cy, directions)
            records.extend(translation_records)
            counts["translations"] += len(translation_records)
            for record in translation_records:
                direction = f"{record['source_lang_code']}->{record['target_lang_code']}"
                counts["translation_directions"][direction] = (
                    counts["translation_directions"].get(direction, 0) + 1
                )
            
            # Inject deep context if available, enriching definition mapping
            ctx_en = str(row.get("Cyd-destun Saesneg", "")).strip()
            ctx_cy = str(row.get("Cyd-destun Cymraeg", "")).strip()
            
            en_def_enriched = f"{en_def}\nContext: {ctx_en}" if ctx_en and ctx_en.lower() not in null_vals else en_def
            cy_def_enriched = f"{cy_def}\nCyd-destun: {ctx_cy}" if ctx_cy and ctx_cy.lower() not in null_vals else cy_def
            
            if cy_def and cy_def.lower() not in null_vals:
                prompt, response = get_en_to_cy_templates(en, cy, cy_def_enriched)
                records.append({
                    "task": "instruction",
                    "source_text": prompt,
                    "target_text": response,
                    "source_lang_code": "en",
                    "target_lang_code": "cy",
                })
                counts["en_instructions"] += 1
            if en_def and en_def.lower() not in null_vals:
                prompt, response = get_cy_to_en_templates(cy, en, en_def_enriched)
                records.append({
                    "task": "instruction",
                    "source_text": prompt,
                    "target_text": response,
                    "source_lang_code": "cy",
                    "target_lang_code": "en",
                })
                counts["cy_instructions"] += 1

    print(f"      TermCymru Breakdown:")
    for direction, count in counts["translation_directions"].items():
        if count > 0:
            print(f"        - Direct Translations ({direction}): {count:,}")
    print(f"        - EN->CY Instructions: {counts['en_instructions']:,}")
    print(f"        - CY->EN Instructions: {counts['cy_instructions']:,}")

    if not records:
        return build_empty_processed_dataset(), 0, 0

    return Dataset.from_list(records), counts["translations"], counts["en_instructions"] + counts["cy_instructions"]

def process_translation_ds(ds, directions):
    cols = ds.column_names
    records = []

    for row in ds:
        en, cy = "", ""
        
        # 1. Handle nested 'translation' dicts (OPUS-100 format)
        if "translation" in cols:
            trans_obj = row.get("translation", {})
            if isinstance(trans_obj, str):
                try: trans_obj = json.loads(trans_obj)
                except: trans_obj = {}
            if isinstance(trans_obj, dict):
                en = trans_obj.get("en", "")
                cy = trans_obj.get("cy", "")
        else:
            # 2. Dynamic column sniffing for flat structures
            for k in ["text_en", "en", "english", "English", "source", "Saesneg"]:
                if k in cols and row.get(k): 
                    en = row[k]
                    break
            for k in ["text_cy", "cy", "welsh", "Welsh", "cymraeg", "Cymraeg", "target"]:
                if k in cols and row.get(k): 
                    cy = row[k]
                    break

        records.extend(build_translation_records(en, cy, directions))

    if not records:
        return build_empty_processed_dataset()

    return Dataset.from_list(records)

def resolve_instruction_language(info, dataset_name):
    source_lang_code = info.get("source_lang_code")
    target_lang_code = info.get("target_lang_code")

    if source_lang_code and target_lang_code:
        return source_lang_code, target_lang_code

    language_code = info.get("language")
    if language_code:
        return language_code, language_code

    normalized_hint = f"{dataset_name} {info.get('config', '')}".lower()
    if "cym" in normalized_hint or "welsh" in normalized_hint or normalized_hint.endswith(" cy"):
        return "cy", "cy"

    return "en", "en"


def process_instruction_ds(ds, source_lang_code, target_lang_code, num_proc=1):
    cols = ds.column_names
    def map_fn(row):
        src, tgt = "", ""
        
        # Check for standard multi-turn chat schemas (e.g., Nemotron)
        if "messages" in cols and isinstance(row.get("messages"), list):
            for msg in row["messages"]:
                if msg.get("role") in ["user", "human"] and not src:
                    src = msg.get("content", "")
                elif msg.get("role") in ["assistant", "gpt", "model"] and not tgt:
                    tgt = msg.get("content", "")
        # ShareGPT style conversations
        elif "conversations" in cols and isinstance(row.get("conversations"), list):
            for msg in row["conversations"]:
                if msg.get("from") == "human" and not src:
                    src = msg.get("value", "")
                elif msg.get("from") == "gpt" and not tgt:
                    tgt = msg.get("value", "")
        # Standard flat instruct schema (e.g., Muri uses input/output)
        else:
            for k in ["instruction", "prompt", "input", "text_en"]:
                if k in cols and row.get(k): 
                    src = row[k]
                    break
            for k in ["output", "response", "completion", "text_cy"]:
                if k in cols and row.get(k): 
                    tgt = row[k]
                    break

        return {
            "task": "instruction",
            "source_text": str(src).strip() if src else "",
            "target_text": str(tgt).strip() if tgt else "",
            "source_lang_code": source_lang_code,
            "target_lang_code": target_lang_code,
        }
    return ds.map(map_fn, remove_columns=cols, num_proc=num_proc)

def main():
    args = parse_args()

    ensure_local_model(args)

    with open(args.recipe, "r") as f:
        recipe = json.load(f)

    # Initialize Tokenizer globally so it can be passed around
    print(f"Loading tokenizer from {MODEL_PATH} for sequence length filtering...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"[DRY RUN] Loading Data Recipe: {recipe.get('profile_name', 'Unknown')}")
    print("-" * 60)
    print(f"{'CATEGORY':<15} | {'DATASET SOURCE':<30} | {'ROWS USED'}")
    print("-" * 60)

    total_rows = 0
    total_translation_rows = 0
    total_instruction_rows = 0
    loaded_datasets = []

    # --- TRANSLATION LOOP ---
    for name, info in recipe.get("translation_data", {}).items():
        if float(info.get("usage", 1.0)) == 0.0:
            continue
        try:
            ds, orig_len, used_len = load_and_slice(name, info["path"], info["usage"], info.get("config"))
            directions = normalize_translation_directions(info)
            processed_ds = process_translation_ds(ds, directions)
            
            # 1. Filter out null/empty strings immediately
            processed_ds = processed_ds.filter(lambda x: bool(x['source_text']) and bool(x['target_text']), num_proc=args.num_proc)
            # 2. Enforce token limits upstream
            processed_ds = enforce_sequence_lengths(processed_ds, tokenizer, max_tokens=2048, num_proc=args.num_proc)

            constructed_rows = len(processed_ds)
            print(f"TRANSLATION     | {name:<30} | {constructed_rows:,}")
            
            info["rows_used"] = constructed_rows
            info["source_rows_used"] = used_len
            info["rows_available"] = orig_len
            total_translation_rows += constructed_rows
            total_rows += constructed_rows
            
            processed_ds = processed_ds.add_column("dataset_source", [name] * constructed_rows)
            loaded_datasets.append(processed_ds)
        except Exception as e:
            print(f"WARNING: Failed to load translation dataset '{name}': {e}")
            continue

    # --- DICTIONARY LOOP ---
    for name, info in recipe.get("dictionary_data", {}).items():
        if float(info.get("usage", 1.0)) == 0.0:
            continue
        try:
            ds, orig_len, used_len = load_and_slice(name, info["path"], info["usage"], info.get("config"))
            directions = normalize_translation_directions(info)
            parsed_ds, _, _ = parse_termcymru(ds, directions)
            
            # Filter and Enforce
            parsed_ds = parsed_ds.filter(lambda x: bool(x['source_text']) and bool(x['target_text']), num_proc=args.num_proc)
            parsed_ds = enforce_sequence_lengths(parsed_ds, tokenizer, max_tokens=2048, num_proc=args.num_proc)

            # Recalculate true counts post-split
            tasks = parsed_ds['task']
            t_cnt = tasks.count("translation")
            i_cnt = tasks.count("instruction")

            print(f"DICTIONARY      | {name:<30} | {len(parsed_ds):,}")
            
            info["rows_used"] = len(parsed_ds)
            info["source_rows_used"] = used_len
            info["rows_available"] = orig_len
            total_translation_rows += t_cnt
            total_instruction_rows += i_cnt
            total_rows += len(parsed_ds)
            
            parsed_ds = parsed_ds.add_column("dataset_source", [name] * len(parsed_ds))
            loaded_datasets.append(parsed_ds)
        except Exception as e:
            print(f"WARNING: Failed to load dictionary dataset '{name}': {e}")
            continue
            
    # --- INSTRUCTION LOOP ---
    for name, info in recipe.get("instruction_data", {}).items():
        if float(info.get("usage", 1.0)) == 0.0:
            continue
        try:
            ds, orig_len, used_len = load_and_slice(name, info["path"], info["usage"], info.get("config"))
            source_lang_code, target_lang_code = resolve_instruction_language(info, name)
            processed_ds = process_instruction_ds(ds, source_lang_code, target_lang_code, num_proc=args.num_proc)
            
            # Filter and enforce the same prompt-aware sequence limits used in training.
            processed_ds = processed_ds.filter(lambda x: bool(x['source_text']) and bool(x['target_text']), num_proc=args.num_proc)
            processed_ds = enforce_sequence_lengths(processed_ds, tokenizer, max_tokens=2048, num_proc=args.num_proc)
            
            constructed_rows = len(processed_ds)
            print(f"INSTRUCTION     | {name:<30} | {constructed_rows:,}")
            
            info["rows_used"] = constructed_rows
            info["source_rows_used"] = used_len
            info["rows_available"] = orig_len
            total_instruction_rows += constructed_rows
            total_rows += constructed_rows
            
            processed_ds = processed_ds.add_column("dataset_source", [name] * constructed_rows)
            loaded_datasets.append(processed_ds)
        except Exception as e:
            print(f"WARNING: Failed to load instruction dataset '{name}': {e}")
            continue

    print("-" * 60)
    print(f"TOTAL TRANSLATIONS: {total_translation_rows:,}")
    print(f"TOTAL INSTRUCTIONS: {total_instruction_rows:,}")
    print(f"TOTAL MERGED ROWS:  {total_rows:,}")
    print("-" * 60)
    
    # Write summary to recipe
    recipe["_run_summary"] = {
        "total_translation_rows": total_translation_rows,
        "total_instruction_rows": total_instruction_rows,
        "total_rows": total_rows,
        "translation_percentage": f"{(total_translation_rows/total_rows)*100:.1f}%" if total_rows > 0 else "0%",
        "instruction_percentage": f"{(total_instruction_rows/total_rows)*100:.1f}%" if total_rows > 0 else "0%"
    }
    with open(args.recipe, "w") as f:
        json.dump(recipe, f, indent=4)

    if not args.yes:
        print("Summary before prompt:")
        print(f"  - Model snapshot path: {MODEL_PATH}")
        print(f"  - Total merged rows planned: {total_rows:,}")
        print(f"  - Translation rows planned: {total_translation_rows:,}")
        print(f"  - Instruction rows planned: {total_instruction_rows:,}")
        ans = input("Proceed with construction? [y/N] ")
        if ans.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    print("\n--- Starting Construction Phase ---")
    print("1. Concatenating all processed and filtered datasets into a unified pool...")
    final_ds = concatenate_datasets(loaded_datasets)
    
    print("2. Performing global deduplication (dropping identical source-target pairs)...")
    df = final_ds.to_pandas()
    initial_count = len(df)
    
    # 1. Sort by source to ensure deterministic "first" choice (optional but good for consistency)
    df = df.sort_values("dataset_source")
    
    # 2. Identify the 'survivor' source for every row
    # Transform gives us the dataset_source of the 'first' instance for every row in that group
    df["clashed_with"] = df.groupby(["source_text", "target_text"])["dataset_source"].transform("first")
    
    # 3. Identify duplicates
    duplicate_mask = df.duplicated(subset=['source_text', 'target_text'], keep='first')
    df_duplicates = df[duplicate_mask].copy()
    
    # Update 'clashed_with' for the dropped rows: 
    # if the survivor is the same as the current row, it's a 'self' clash
    df_duplicates["clashed_with"] = df_duplicates.apply(
        lambda x: "self" if x["dataset_source"] == x["clashed_with"] else x["clashed_with"], 
        axis=1
    )
    
    df = df.drop_duplicates(subset=['source_text', 'target_text'])
    deduped_count = len(df)
    print(f"   -> Dropped {initial_count - deduped_count:,} duplicate rows.")
    
    # Segment dropped duplicates by source dataset
    duplicates_by_source = {}
    for source, group in df_duplicates.groupby("dataset_source"):
        # We keep 'clashed_with' in the JSON so user knows who 'won' the deduplication
        duplicates_by_source[source] = group.drop(columns=["dataset_source"]).to_dict(orient="records")

    # Save segmented duplicates to a JSON file
    duplicates_file = os.path.join(PROCESSED_DATA_DIR, "dropped_duplicates.json")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(duplicates_file, "w", encoding="utf-8") as f:
        json.dump(duplicates_by_source, f, indent=4, ensure_ascii=False)
    
    print(f"   -> Logged segmented duplicates to {duplicates_file}")

    print("4. Rebalancing task mix and translation directions...")
    df, rebalance_summary = rebalance_dataset(df, recipe)
    print(
        "   -> Final task mix target:",
        f"{rebalance_summary['translation_target_ratio'] * 100:.1f}% translation /",
        f"{rebalance_summary['instruction_target_ratio'] * 100:.1f}% instruction",
    )
    if rebalance_summary["translation_direction_counts"]:
        print("   -> Translation directions after rebalance:")
        for direction, count in sorted(rebalance_summary["translation_direction_counts"].items()):
            print(f"      - {direction}: {count:,}")
    print(
        f"   -> Removed {rebalance_summary['rows_removed_by_rebalancing']:,} rows during rebalancing."
    )
    
    # Remove helper columns before saving the final dataset
    df = df.drop(columns=["dataset_source", "clashed_with"])
    final_ds = Dataset.from_pandas(df, preserve_index=False)
    
    print("5. Splitting into Train and Validation sets with stratification...")
    train_df, eval_df, split_summary = split_dataset_stratified(df, recipe)
    final_split = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(eval_df, preserve_index=False),
        }
    )
    
    print(f"6. Saving formatted data records to {PROCESSED_DATA_DIR}...")
    final_split.save_to_disk(PROCESSED_DATA_DIR)
    
    # Calculate constructed summary
    task_counts = df['task'].value_counts()
    transl_cnt = int(task_counts.get('translation', 0))
    instr_cnt = int(task_counts.get('instruction', 0))
    translation_direction_counts = summarize_translation_directions(df)
    
    recipe["_constructed_summary"] = {
        "total_translation_rows": transl_cnt,
        "total_instruction_rows": instr_cnt,
        "total_rows": len(df),
        "translation_percentage": f"{(transl_cnt/len(df))*100:.1f}%" if len(df) > 0 else "0%",
        "instruction_percentage": f"{(instr_cnt/len(df))*100:.1f}%" if len(df) > 0 else "0%",
        "training_split_rows": len(final_split['train']),
        "eval_split_rows": len(final_split['test']),
        "eval_stratification": split_summary["eval_stratification"],
        "duplicates_removed": int(initial_count - deduped_count),
        "rows_removed_by_rebalancing": rebalance_summary["rows_removed_by_rebalancing"],
        "translation_direction_counts": translation_direction_counts,
    }
    
    with open(args.recipe, "w") as f:
        json.dump(recipe, f, indent=4)
        
    print(f"\nConstruction Complete. Constructed summary appended to {args.recipe}.")

if __name__ == "__main__":
    main()
