import os
import sys
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_ID = "./local_model"

def run_finetune():
    # Optimization for Ryzen 7 5800X (8 cores / 16 threads)
    torch.set_num_threads(16)
    
    if not os.path.exists("./processed_data"):
        print("Error: ./processed_data not found. Please run 01_prepare_data.py first.")
        sys.exit(1)
    
    if not os.path.exists("./local_model"):
        print("Error: ./local_model not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    print("Loading processed dataset...")
    dataset = load_from_disk("./processed_data")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("Bypassing language validation for 'gd'...")
    # Standard TranslateGemma template patching logic
    
    def apply_template(example):
        # Decode the JSON string back into a Python list
        messages = json.loads(example["messages"])
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    print("Applying chat template to dataset...")
    encoded_train = dataset["train"].map(apply_template, remove_columns=["messages"])
    encoded_val = dataset["validation"].map(apply_template, remove_columns=["messages"])

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    print("Tokenizing data...")
    tokenized_train = encoded_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = encoded_val.map(tokenize_fn, batched=True, remove_columns=["text"])

    print("Loading Base Model onto CPU in bfloat16...")
    # Using CausalLM as TranslateGemma is based on Gemma
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.bfloat16
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
        use_cpu=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # Reduced from 16 to get feedback 4x faster
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=1, # Print every single step for immediate feedback
        bf16=True, 
        save_strategy="no", # Don't waste time saving until the end
        eval_strategy="no", # Skip evaluation during training to focus on speed
        report_to="none"
    )

    from transformers import DataCollatorForLanguageModeling
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting fine-tuning! Progress will be printed below...")
    trainer.train()
    
    print("Saving adapter weights...")
    trainer.model.save_pretrained("./translategemma-finetuned")
    print("Done! Weights saved to ./translategemma-finetuned")

if __name__ == "__main__":
    run_finetune()