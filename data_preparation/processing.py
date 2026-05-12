import random
from datasets import Dataset, concatenate_datasets
from .utils import get_worker_tokenizer

RESPONSE_TEMPLATE = "<start_of_turn>model\n"
USER_TURN_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN_TEMPLATE = "<end_of_turn>\n"

def build_training_prompt(example, tokenizer):
    """
    Constructs the model-specific training prompt (e.g. ChatML/Gemma format).
    
    Handles both structured translation tasks (using apply_chat_template)
    and flattened instruction tasks.
    """
    if example["task"] == "translation":
        prompt = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": example["source_lang_code"],
                    "target_lang_code": example["target_lang_code"],
                    "text": example["source_text"],
                }],
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
    elif example["task"] == "instruction":
        bos_token = tokenizer.bos_token or ""
        prompt = f"{bos_token}{USER_TURN_TEMPLATE}{str(example['source_text']).strip()}{END_OF_TURN_TEMPLATE}{RESPONSE_TEMPLATE}"
    else:
        raise ValueError(f"Unsupported task type '{example['task']}'.")
    return prompt

def build_training_completion(target_text):
    return f"{str(target_text).strip()}{END_OF_TURN_TEMPLATE}"

def compute_lengths_batch(batch, tokenizer_path):
    """
    Batched mapping function to calculate src/tgt lengths.
    
    Processes text in batches to leverage Rust-based FastTokenizer 
    parallelism and HF internal caching.
    """
    tok = get_worker_tokenizer(tokenizer_path)
    src_texts = [str(x) if x is not None else "" for x in batch.get("source_text", [])]
    tgt_texts = [str(x) if x is not None else "" for x in batch.get("target_text", [])]
    
    # Use the Rust tokenizer's batched call for maximum efficiency
    src_enc = tok(src_texts, add_special_tokens=False, padding=False)
    tgt_enc = tok(tgt_texts, add_special_tokens=False, padding=False)
    
    return {
        "src_len": [len(ids) for ids in src_enc["input_ids"]],
        "tgt_len": [len(ids) for ids in tgt_enc["input_ids"]]
    }

def truncate_target_example(example, tokenizer_path, max_tokens):
    """
    Truncates the target_text if the total sequence (prompt + target)
    exceeds the max_tokens limit.
    
    This ensures that the model isn't trained on incomplete sequences during 
    fine-tuning and maintains strict budget compliance.
    """
    tok = get_worker_tokenizer(tokenizer_path)
    tgt = example.get("target_text", "") or ""
    try:
        prompt = build_training_prompt(example, tok)
    except Exception:
        example["target_text"] = ""
        return example

    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    completion_suffix_ids = tok.encode(build_training_completion(""), add_special_tokens=False)
    pad_len = 1 if getattr(tok, "pad_token_id", None) is not None else 0
    allowed = max(0, max_tokens - pad_len - len(prompt_ids) - len(completion_suffix_ids))

    if allowed <= 0:
        example["target_text"] = ""
        return example

    tgt_ids = tok.encode(tgt, add_special_tokens=False)
    if len(tgt_ids) > allowed:
        example["target_text"] = tok.decode(
            tgt_ids[:allowed], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
    return example

def enforce_sequence_lengths(ds, tokenizer_path, max_tokens=2048, num_proc=1):
    """
    Orchestrates the sequence length enforcement pipeline.
    
    1. Tokenizes and gets lengths (using cache).
    2. Filters out sources that are too long for any target response.
    3. Buckets data into 'Under Limit' and 'Over Limit'.
    4. Truncates 'Over Limit' targets to fit within the budget.
    """
    if len(ds) == 0: return ds
    ds = ds.map(compute_lengths_batch, batched=True, fn_kwargs={"tokenizer_path": tokenizer_path}, num_proc=num_proc, desc="Tokenizing")
    
    SRC_MAX_ALLOWED = 1266
    ds = ds.filter(lambda x: x["src_len"] <= SRC_MAX_ALLOWED, num_proc=num_proc, desc="Filter: Too-long sources")
    
    good_ds = ds.filter(lambda x: x["src_len"] <= max_tokens and x["tgt_len"] <= max_tokens, num_proc=num_proc, desc="Bucket: Under Limit")
    long_ds = ds.filter(lambda x: x["src_len"] > max_tokens or x["tgt_len"] > max_tokens, num_proc=num_proc, desc="Bucket: Over Limit")
    
    good_ds = good_ds.remove_columns([c for c in ["src_len", "tgt_len"] if c in good_ds.column_names])
    if len(long_ds) == 0: return good_ds

    long_ds = long_ds.map(truncate_target_example, fn_kwargs={"tokenizer_path": tokenizer_path, "max_tokens": max_tokens}, num_proc=num_proc, desc="Truncating targets")
    long_ds = long_ds.filter(lambda x: bool((x.get("target_text", "") or "").strip()), num_proc=num_proc).remove_columns(["src_len", "tgt_len"])
    
    return concatenate_datasets([good_ds, long_ds])
