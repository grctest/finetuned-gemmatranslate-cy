import os
import sys
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForImageTextToText, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MODEL_ID = "./local_model"
MAX_LENGTH = 1024

def format_prompts_func(example):
    return [example["messages"]]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TranslateGemma 4B")
    parser.add_argument(
        "--profile", 
        type=str, 
        choices=["cpu", "3090", "high_vram", "max_vram"], 
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

    # Configure hardware-specific settings based on the selected profile
    use_flash_attn = False
    optim_type = "adamw_torch"

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
    elif args.profile == "max_vram":
        print("Profile [max_vram]: Enabling Full Fine-Tuning & Flash Attention 2.")
        model_device_map = None # Defer to DeepSpeed/FSDP
        model_dtype = torch.bfloat16
        batch_size = 16 
        grad_accum = 1
        use_bf16 = True
        grad_ckpt = False # VRAM is plenty; disable checkpointing for 20% speedup
        use_cpu = False
        use_flash_attn = True
        optim_type = "adamw_torch_fused"

    print(f"Loading Base Model onto {model_device_map}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map=model_device_map,
        dtype=model_dtype,
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa"
    )

    if args.profile != "max_vram":
        print("Configuring LoRA...")
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        print("Configuring Full Fine-Tuning (LoRA Disabled)...")
        peft_config = None

    print("Initializing Trainer...")
    training_args = TrainingArguments(
        output_dir="./translategemma-finetuned",
        use_cpu=use_cpu,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        fp16=False,
        bf16=use_bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={'use_reentrant': False} if grad_ckpt else None,
        save_strategy="no", # Don't waste time saving until the end
        eval_strategy="no", # Skip evaluation during training to focus on speed
        report_to="none",
        optim=optim_type,
        neftune_noise_alpha=5.0 if args.profile == "max_vram" else None
    )

    model.config.use_cache = False

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<start_of_turn>model\n",
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=4096 if args.profile == "max_vram" else MAX_LENGTH,
        packing=True if args.profile == "max_vram" else False
    )

    print("Starting fine-tuning! Progress will be printed below...")
    trainer.train()
    
    print("Saving adapter weights...")
    trainer.model.save_pretrained("./translategemma-finetuned")
    print("Done! Weights saved to ./translategemma-finetuned")

if __name__ == "__main__":
    args = parse_args()
    run_finetune(args)