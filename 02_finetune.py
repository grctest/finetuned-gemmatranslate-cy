import os
import sys
import json
import re
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForImageTextToText, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

MODEL_ID = "./local_model"
MAX_LENGTH = 2000
LEGACY_PROMPT_PATTERN = re.compile(r"^Translate from ([^\s]+) to ([^:]+):\n(.*)$", re.DOTALL)

def normalize_messages(raw_messages):
    normalized_messages = []

    for message in raw_messages:
        normalized_message = dict(message)
        role = normalized_message.get("role")
        content = normalized_message.get("content")

        if role == "assistant":
            if isinstance(content, list):
                if len(content) != 1 or content[0].get("type") != "text":
                    raise ValueError("Assistant message content must be a single text item.")
                normalized_message["content"] = content[0].get("text", "")
            elif not isinstance(content, str):
                raise ValueError("Assistant message content must be a string.")
        elif role == "user":
            if not isinstance(content, list) or len(content) != 1:
                raise ValueError("User message content must be a single-item list.")

            normalized_content = dict(content[0])
            if "source_lang_code" not in normalized_content or "target_lang_code" not in normalized_content:
                match = LEGACY_PROMPT_PATTERN.match(normalized_content.get("text", ""))
                if not match:
                    raise ValueError("Legacy user prompt is missing language codes and could not be parsed.")

                normalized_content["source_lang_code"] = match.group(1)
                normalized_content["target_lang_code"] = match.group(2)
                normalized_content["text"] = match.group(3)

            normalized_message["content"] = [normalized_content]
        else:
            raise ValueError(f"Unsupported role: {role}")

        normalized_messages.append(normalized_message)

    return normalized_messages

def tokenize_example(example, tokenizer):
    messages = normalize_messages(json.loads(example["messages"]))
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Training examples must end with an assistant response.")

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    prompt_length = min(len(tokenized_prompt["input_ids"]), len(tokenized_full["input_ids"]))
    labels = [-100] * prompt_length + tokenized_full["input_ids"][prompt_length:]

    return {
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
        "supervised_tokens": len(tokenized_full["input_ids"]) - prompt_length,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TranslateGemma 4B")
    parser.add_argument(
        "--profile", 
        type=str, 
        choices=["cpu", "3090", "high_vram"], 
        default="3090",
        help="Hardware profile to optimize training settings."
    )
    return parser.parse_args()

def run_finetune(args):
    
    if not os.path.exists("./processed_data"):
        print("Error: ./processed_data not found. Please run 01_prepare_data.py first.")
        sys.exit(1)
    
    if not os.path.exists("./local_model"):
        print("Error: ./local_model not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    print("Loading processed dataset...")
    dataset = load_from_disk("./processed_data")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Rendering chat prompts and tokenizing data...")
    tokenized_train = dataset["train"].map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=["messages"],
    )
    tokenized_val = dataset["validation"].map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=["messages"],
    )

    tokenized_train = tokenized_train.filter(lambda example: example["supervised_tokens"] > 0)
    tokenized_val = tokenized_val.filter(lambda example: example["supervised_tokens"] > 0)

    tokenized_train = tokenized_train.remove_columns(["supervised_tokens"])
    tokenized_val = tokenized_val.remove_columns(["supervised_tokens"])

    # Configure hardware-specific settings based on the selected profile
    if args.profile == "cpu":
        print("Profile [CPU]: Loading in float32. Training will be extremely slow.")
        model_device_map = "cpu"
        model_dtype = torch.float32
        batch_size = 1
        grad_accum = 8
        use_bf16 = False
        grad_ckpt = False
        use_cpu = True
    elif args.profile == "3090":
        print("Profile [3090]: Loading in bfloat16 for 24GB VRAM target.")
        model_device_map = "auto"
        model_dtype = torch.bfloat16
        batch_size = 4
        grad_accum = 4
        use_bf16 = True
        grad_ckpt = True
        use_cpu = False
    elif args.profile == "high_vram":
        print("Profile [high_vram]: Loading in bfloat16 optimized for 48GB+ (A6000/A100/H100).")
        model_device_map = "auto"
        model_dtype = torch.bfloat16
        batch_size = 8
        grad_accum = 2
        use_bf16 = True
        grad_ckpt = True
        use_cpu = False

    print(f"Loading Base Model onto {model_device_map}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map=model_device_map,
        dtype=model_dtype
    )

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Initializing Trainer...")
    training_args = TrainingArguments(
        output_dir="./translategemma-finetuned",
        use_cpu=use_cpu,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        fp16=False,
        bf16=use_bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={'use_reentrant': False} if grad_ckpt else None,
        save_strategy="no", # Don't waste time saving until the end
        eval_strategy="no", # Skip evaluation during training to focus on speed
        report_to="none"
    )

    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
    )

    sample_count = min(32, len(tokenized_train))
    if sample_count == 0:
        raise ValueError("No train examples retained supervised target tokens after tokenization.")

    avg_supervised_tokens = sum(
        sum(label != -100 for label in tokenized_train[index]["labels"])
        for index in range(sample_count)
    ) / sample_count
    first_target_ids = [label for label in tokenized_train[0]["labels"] if label != -100]

    print("Supervised label tokens in first example:", len(first_target_ids))
    print("Average supervised label tokens in first", sample_count, "train examples:", round(avg_supervised_tokens, 2))
    print("First supervised target preview:", tokenizer.decode(first_target_ids[:80], skip_special_tokens=False))

    print("Starting fine-tuning! Progress will be printed below...")
    trainer.train()
    
    print("Saving adapter weights...")
    trainer.model.save_pretrained("./translategemma-finetuned")
    print("Done! Weights saved to ./translategemma-finetuned")

if __name__ == "__main__":
    args = parse_args()
    run_finetune(args)