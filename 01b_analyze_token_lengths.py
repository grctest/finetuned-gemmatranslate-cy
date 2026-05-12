import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from model_assets import MODEL_PATH

PROCESSED_DATA_DIR = "./processed_data"

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token lengths of the processed dataset.")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of CPU cores to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to analyze (e.g. 0.05 for 5%). Matches the finetuning script behavior.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset from {PROCESSED_DATA_DIR}...")
    dataset_dict = load_from_disk(PROCESSED_DATA_DIR)
    
    if args.split == "all":
        dataset = dataset_dict["train"] # If merging is needed we'd concat, but usually train is representative enough
        if "test" in dataset_dict:
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["test"]])
    else:
        dataset = dataset_dict[args.split]
    
    # Apply dataset fraction if requested
    if args.dataset_fraction < 1.0:
        original_size = len(dataset)
        new_size = int(original_size * args.dataset_fraction)
        print(f"Applying dataset fraction {args.dataset_fraction:.1%}: {original_size:,} -> {new_size:,} samples.")
        dataset = dataset.select(range(new_size))

    num_samples = len(dataset)
    print(f"Loaded {num_samples} records. Loading tokenizer from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # We will measure the token length of the conversational prompt formatting, same as TRL uses
    # The dataset has 'source_text' and 'target_text' based on 01_prepare_data output, wait:
    # 02_finetune usually formats it into a chat.
    
    def calculate_length(example):
        source = example.get("source_text", "")
        target = example.get("target_text", "")
        task = example.get("task", "translation")
        
        # Consistent with how the model sees the data
        formatted_text = f"<start_of_turn>user\nTranslate this: {source}<end_of_turn>\n<start_of_turn>model\n{target}<end_of_turn>"
        
        tokens = tokenizer(formatted_text, truncation=False, padding=False)
        total = len(tokens["input_ids"])
        return {
            "total_tokens": total,
            "is_translation": 1 if task == "translation" else 0
        }

    print("Mapping dataset to calculate lengths...")
    res_ds = dataset.map(
        calculate_length,
        num_proc=args.num_proc,
        desc="Tokenizing",
        remove_columns=dataset.column_names
    )
    
    def print_stats(data, label):
        if len(data) == 0:
            print(f"\n--- {label} Statistics: No Data ---")
            return
        mean_v = np.mean(data)
        med_v = np.median(data)
        max_v = np.max(data)
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        p99_9 = np.percentile(data, 99.9)

        print(f"\n--- {label} Statistics ---")
        print(f"Count:           {len(data):,}")
        print(f"Mean Tokens:     {mean_v:.2f}")
        print(f"Median Tokens:   {med_v:.2f}")
        print(f"Maximum Tokens:  {max_v}")
        print(f"95th Percentile: {p95:.0f}")
        print(f"99th Percentile: {p99:.0f}")
        print(f"99.9th Percentil:{p99_9:.0f}")

    all_lengths = np.array(res_ds["total_tokens"])
    is_trans = np.array(res_ds["is_translation"])
    
    print_stats(all_lengths, "Global Token Length Distribution")
    print_stats(all_lengths[is_trans == 1], "TRANSLATION Segment")
    print_stats(all_lengths[is_trans == 0], "INSTRUCTION Segment")
    
    print("\nTraining Analysis:")
    print("When `packing=True`, lengths are squashed together up to `max_seq_length`, so the max length matters less for clipping.")

if __name__ == "__main__":
    main()
