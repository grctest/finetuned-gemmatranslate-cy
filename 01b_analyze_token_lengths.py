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
    
    num_samples = len(dataset)
    print(f"Loaded {num_samples} records. Loading tokenizer from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # We will measure the token length of the conversational prompt formatting, same as TRL uses
    # The dataset has 'source_text' and 'target_text' based on 01_prepare_data output, wait:
    # 02_finetune usually formats it into a chat.
    
    def calculate_length(example):
        # Approximating typical Gemma Instruct formatting for translation:
        # <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>
        source = example.get("source_text", "")
        target = example.get("target_text", "")
        
        # Simple string concatenation to estimate text length passed to tokenizer
        formatted_text = f"<start_of_turn>user\nTranslate this to another language: {source}<end_of_turn>\n<start_of_turn>model\n{target}<end_of_turn>"
        
        # We only need the length of the input_ids
        tokens = tokenizer(formatted_text, truncation=False, padding=False)
        return {"total_tokens": len(tokens["input_ids"])}

    print("Mapping dataset to calculate lengths...")
    lengths_dataset = dataset.map(
        calculate_length,
        num_proc=args.num_proc,
        desc="Tokenizing",
        remove_columns=dataset.column_names
    )
    
    lengths = np.array(lengths_dataset["total_tokens"])
    
    # Compute Statistics
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    max_length = np.max(lengths)
    min_length = np.min(lengths)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    p99_9 = np.percentile(lengths, 99.9)

    print("\n--- Token Length Distribution Statistics ---")
    print(f"Total Sequences: {num_samples:,}")
    print(f"Mean Tokens:     {mean_length:.2f}")
    print(f"Median Tokens:   {median_length:.2f}")
    print(f"Minimum Tokens:  {min_length}")
    print(f"Maximum Tokens:  {max_length}")
    print(f"95th Percentile: {p95:.0f}")
    print(f"99th Percentile: {p99:.0f}")
    print(f"99.9th Percentil:{p99_9:.0f}")
    
    print("\nBased on these stats, if you look at the 99th or 99.9th percentile, that is your ideal `max_seq_length` when `packing=False`.")
    print("When `packing=True`, lengths are squashed together up to `max_seq_length`, so the max length matters less for clipping.")

if __name__ == "__main__":
    main()
