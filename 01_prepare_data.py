import argparse
import json
import os
import sys
import gc
import random
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from model_assets import MODEL_ID, MODEL_PATH, ensure_local_model_snapshot
from data_preparation.utils import get_pair_hash, PROCESSED_DATA_DIR
from data_preparation.loaders import (
    parse_termcymru, process_translation_ds, 
    process_instruction_row
)
from data_preparation.processing import enforce_sequence_lengths

SUPPORTED_TRANSLATION_DIRECTIONS = {"en-cy", "cy-en"}

def parse_args():
    """
    Parses command-line arguments for the data preparation pipeline.
    
    Includes flags for data recipes, bypassing prompts, skipping model downloads,
    refreshing token caches, and multi-processing configuration.
    """
    parser = argparse.ArgumentParser(description="Prepare Welsh fine-tuning data.")
    parser.add_argument("--recipe", type=str, default="data_recipe.json")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Clear the token cache and re-process everything.")
    parser.add_argument("--num-proc", type=int, default=max(1, os.cpu_count() - 1))
    return parser.parse_args()

def ensure_local_model(args):
    """
    Validates the local presence of the TranslateGemma model assets.
    
    If assets are missing and --skip-model-download isn't set, it triggers 
    a download/sync attempt via model_assets.py.
    """
    if args.skip_model_download: return
    missing = ensure_local_model_snapshot(MODEL_PATH, MODEL_ID)
    if missing: raise RuntimeError(f"Missing assets: {missing}")

def load_and_slice(path, usage_factor, config=None):
    """
    Loads a dataset from Hugging Face and applies a randomized usage slice.
    
    Args:
        path: HF dataset path or identifier.
        usage_factor: Float [0, 1] representing the fraction of data to use.
        config: Optional configuration name for the dataset.
        
    Returns:
        A tuple of (selected_dataset, total_available_rows, num_rows_selected).
    """
    ds = load_dataset(path, config, split="train").shuffle(seed=42)
    num = int(len(ds) * usage_factor)
    return ds.select(range(num)), len(ds), num

def normalize_directions(info):
    """
    Standardizes translation direction strings (e.g., 'en_cy' -> 'en-cy').
    
    Validates that the requested directions are within the supported set
    defined in SUPPORTED_TRANSLATION_DIRECTIONS.
    """
    dirs = info.get("directions", ["en-cy"])
    norm = []
    for d in dirs:
        c = str(d).strip().lower().replace("_", "-")
        if c not in SUPPORTED_TRANSLATION_DIRECTIONS:
            raise ValueError(f"Unsupported direction: {c}")
        if c not in norm: norm.append(c)
    return norm

def parse_target_ratio(recipe):
    """
    Parses the 'target_ratio' from the recipe into probabilities.
    
    Accepts string formats like "70:30" or lists like [0.7, 0.3].
    Normalized to sum to 1.0 for use in rebalancing logic.
    """
    tr = recipe.get("meta_strategy", {}).get("target_ratio", "70:30")
    if isinstance(tr, str): p = [float(x.strip()) for x in tr.split(":")]
    else: p = [float(x) for x in tr]
    s = sum(p)
    return p[0]/s, p[1]/s

def resolve_eval_size(recipe, total):
    """
    Calculates the absolute number of evaluation rows based on the recipe.
    
    If 'eval_size' is a float < 1, it's treated as a percentage of 'total'.
    Otherwise, it is treated as a fixed integer count of rows.
    """
    es = recipe.get("meta_strategy", {}).get("eval_size", 1000)
    val = int(round(total * es)) if isinstance(es, float) else int(es)
    return max(1, min(val, total - 1))

def main():
    args = parse_args()
    ensure_local_model(args)

    if args.refresh:
        cache_path = os.path.join(PROCESSED_DATA_DIR, "token_cache.sqlite")
        if os.path.exists(cache_path):
            print(f"Refreshing: Removing token cache at {cache_path}")
            os.remove(cache_path)

    with open(args.recipe, "r") as f: recipe = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    total_rows = 0
    total_t_rows = 0
    total_i_rows = 0
    loaded = []

    # --- Translation ---
    for name, info in recipe.get("translation_data", {}).items():
        if float(info.get("usage", 1.0)) == 0.0: continue
        ds, orig, used = load_and_slice(info["path"], info["usage"], info.get("config"))
        p_ds = process_translation_ds(ds, normalize_directions(info))
        p_ds = p_ds.filter(lambda x: bool(x["source_text"] and x["target_text"]))
        p_ds = enforce_sequence_lengths(p_ds, MODEL_PATH, num_proc=args.num_proc)
        print(f"TRANSLATION | {name:<30} | {len(p_ds):,}")
        total_t_rows += len(p_ds); total_rows += len(p_ds)
        loaded.append(p_ds.add_column("dataset_source", [name]*len(p_ds)))

    # --- Dictionary ---
    for name, info in recipe.get("dictionary_data", {}).items():
        if float(info.get("usage", 1.0)) == 0.0: continue
        ds, orig, used = load_and_slice(info["path"], info["usage"], info.get("config"))
        p_ds, counts = parse_termcymru(ds, normalize_directions(info))
        p_ds = p_ds.filter(lambda x: bool(x["source_text"] and x["target_text"]))
        p_ds = enforce_sequence_lengths(p_ds, MODEL_PATH, num_proc=args.num_proc)
        print(f"DICTIONARY  | {name:<30} | {len(p_ds):,}")
        total_t_rows += sum(1 for x in p_ds["task"] if x=="translation")
        total_i_rows += sum(1 for x in p_ds["task"] if x=="instruction")
        total_rows += len(p_ds)
        loaded.append(p_ds.add_column("dataset_source", [name]*len(p_ds)))

    # --- Instruction ---
    for name, info in recipe.get("instruction_data", {}).items():
        if float(info.get("usage", 1.0)) == 0.0: continue
        ds, orig, used = load_and_slice(info["path"], info["usage"], info.get("config"))
        p_ds = ds.map(process_instruction_row, remove_columns=ds.column_names, num_proc=args.num_proc)
        p_ds = p_ds.filter(lambda x: bool(x["source_text"] and x["target_text"]))
        p_ds = enforce_sequence_lengths(p_ds, MODEL_PATH, num_proc=args.num_proc)
        print(f"INSTRUCTION | {name:<30} | {len(p_ds):,}")
        total_i_rows += len(p_ds); total_rows += len(p_ds)
        loaded.append(p_ds.add_column("dataset_source", [name]*len(p_ds)))

    print(f"TOTAL: {total_rows:,} (T: {total_t_rows:,}, I: {total_i_rows:,})")
    if not args.yes and input("Proceed? [y/N] ").lower() != 'y': sys.exit(0)

    print("--- Construction ---")
    final_ds = concatenate_datasets(loaded); del loaded; gc.collect()
    
    # Deduplication
    print("2. Deduplicating...")
    def hash_gen(ds, size=10000):
        for i in range(0, len(ds), size):
            b = ds.select(range(i, min(i+size, len(ds))))
            for s, t in zip(b["source_text"], b["target_text"]): yield get_pair_hash(s, t)
    
    hashes = list(hash_gen(final_ds))
    sources = final_ds["dataset_source"]
    seen, keep = {}, []
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DATA_DIR, "dropped_duplicates.jsonl"), "w") as f:
        for idx, (h, src) in enumerate(zip(hashes, sources)):
            if h not in seen: seen[h] = str(src); keep.append(idx)
            else: f.write(json.dumps({"source": str(src), "winner": seen[h], "hash": h}) + "\n")
    final_ds = final_ds.select(keep); del hashes, sources, seen; gc.collect()

    # Rebalancing
    print("3. Rebalancing...")
    t_ratio, i_ratio = parse_target_ratio(recipe)
    t_idx = [i for i, t in enumerate(final_ds["task"]) if str(t)=="translation"]
    i_idx = [i for i, t in enumerate(final_ds["task"]) if str(t)=="instruction"]
    
    if t_idx and i_idx:
        t_avail, i_avail = len(t_idx), len(i_idx)
        if t_avail/t_ratio <= i_avail/i_ratio:
            nT, nI = t_avail, min(i_avail, int(round(t_avail*i_ratio/t_ratio)))
        else:
            nI, nT = i_avail, min(t_avail, int(round(i_avail*t_ratio/i_ratio)))
        sel = random.sample(t_idx, nT) + random.sample(i_idx, nI)
        random.shuffle(sel)
        final_ds = final_ds.select(sel)

    # Split
    print("4. Splitting...")
    eval_sz = resolve_eval_size(recipe, len(final_ds))
    final_ds = final_ds.map(lambda b: {"stratify_key": [f"{t}:{s}->{tg}" for t,s,tg in zip(b["task"], b["source_lang_code"], b["target_lang_code"])]}, batched=True)
    final_ds = final_ds.class_encode_column("stratify_key")
    split = final_ds.train_test_split(test_size=eval_sz, stratify_by_column="stratify_key", seed=42)
    
    # Save
    split.save_to_disk(PROCESSED_DATA_DIR)
    print(f"Done. Saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
