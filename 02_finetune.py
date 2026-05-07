import argparse
import json
import os
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

MODEL_PATH = "./local_model"
DATASET_PATH = "./processed_data"
TRAINING_ROOT = "./translategemma-finetuned"
ADAPTER_OUTPUT_DIR = os.path.join(TRAINING_ROOT, "adapter")
FULL_MODEL_OUTPUT_DIR = os.path.join(TRAINING_ROOT, "full_model")
ARTIFACT_METADATA_PATH = os.path.join(TRAINING_ROOT, "training_artifact.json")
RESPONSE_TEMPLATE = "<start_of_turn>model\n"

PROFILE_CONFIGS = {
    "cpu": {
        "description": "CPU-only fallback with LoRA and 512-token context.",
        "device_map": None,
        "torch_dtype": torch.float32,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 64,
        "max_seq_length": 512,
        "bf16": False,
        "gradient_checkpointing": False,
        "use_flash_attention": False,
        "packing": False,
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": True,
    },
    "3090": {
        "description": "24GB VRAM LoRA profile targeting effective batch size 64.",
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "max_seq_length": 512,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": False,
        "packing": False,
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": False,
    },
    "high_vram": {
        "description": "48GB+ VRAM LoRA profile with longer context and packing.",
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1024,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": False,
        "packing": True,
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": False,
    },
    "max_vram": {
        "description": "Distributed full fine-tuning profile with Flash Attention 2 and DeepSpeed.",
        "device_map": None,
        "torch_dtype": torch.bfloat16,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "bf16": True,
        "gradient_checkpointing": False,
        "use_flash_attention": True,
        "packing": True,
        "training_mode": "full",
        "deepspeed": "./ds_config.json",
        "use_cpu": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune TranslateGemma for English/Welsh translation."
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(PROFILE_CONFIGS.keys()),
        default="3090",
        help="Hardware profile to optimize training settings.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for train samples, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for validation samples, useful for smoke tests.",
    )
    parser.add_argument(
        "--instruction-mix-ratio",
        type=float,
        default=0.0,
        help="Reserved scaffold for future generic instruction data mixing.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Set to 'wandb' to enable Weights & Biases reporting.",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Override the DeepSpeed config path used by the max_vram profile.",
    )
    return parser.parse_args()


def validate_processed_dataset(dataset):
    required_columns = {
        "task",
        "source_text",
        "target_text",
        "source_lang_code",
        "target_lang_code",
    }
    missing_columns = required_columns.difference(dataset["train"].column_names)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "processed_data is missing expected flat translation columns: "
            f"{missing}. Re-run 01_prepare_data.py to regenerate the Welsh dataset."
        )


def build_messages(example):
    if example["task"] != "translation":
        raise ValueError(
            f"Unsupported task type '{example['task']}'. Only translation rows are implemented."
        )

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
        },
        {
            "role": "assistant",
            "content": example["target_text"],
        },
    ]


def render_training_text(example, tokenizer):
    return tokenizer.apply_chat_template(
        build_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_text_dataset(split_dataset, tokenizer, max_samples):
    if max_samples is not None:
        split_dataset = split_dataset.select(range(min(max_samples, len(split_dataset))))

    return split_dataset.map(
        lambda example: {"text": render_training_text(example, tokenizer)},
        desc="Rendering TranslateGemma chat texts",
    )


def build_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def freeze_embeddings(model):
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        input_embeddings.weight.requires_grad = False

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_embeddings.weight.requires_grad = False


def validate_response_template(tokenizer):
    sample_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "cy",
                        "text": "Hello world",
                    }
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    if RESPONSE_TEMPLATE not in sample_prompt:
        raise ValueError(
            "TranslateGemma chat template no longer contains the expected assistant marker. "
            "Update RESPONSE_TEMPLATE before training."
        )


def save_artifact_metadata(profile_name, profile, artifact_path):
    os.makedirs(TRAINING_ROOT, exist_ok=True)
    with open(ARTIFACT_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "profile": profile_name,
                "training_mode": profile["training_mode"],
                "artifact_path": artifact_path,
                "max_seq_length": profile["max_seq_length"],
                "effective_batch_size": (
                    profile["per_device_train_batch_size"]
                    * profile["gradient_accumulation_steps"]
                ),
                "optimizer": "adafactor",
                "learning_rate": 1e-4,
                "embedding_freeze": True,
                "source_lang_code": "en",
                "target_lang_code": "cy",
            },
            handle,
            indent=2,
        )


def run_finetune(args):
    if args.instruction_mix_ratio > 0:
        raise ValueError(
            "Instruction-mix wiring is only scaffolded right now. Keep --instruction-mix-ratio at 0 until an external instruction dataset is integrated."
        )

    if not os.path.exists(DATASET_PATH):
        print("Error: ./processed_data not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("Error: ./local_model not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    profile = PROFILE_CONFIGS[args.profile]
    deepspeed_config = args.deepspeed_config or profile["deepspeed"]

    print("Loading processed dataset...")
    dataset = load_from_disk(DATASET_PATH)
    validate_processed_dataset(dataset)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_response_template(tokenizer)

    print("Rendering training texts from flat Welsh translation records...")
    train_dataset = prepare_text_dataset(dataset["train"], tokenizer, args.max_train_samples)
    eval_dataset = prepare_text_dataset(dataset["validation"], tokenizer, args.max_eval_samples)

    print(f"Profile [{args.profile}]: {profile['description']}")
    print(
        "Effective batch size:",
        profile["per_device_train_batch_size"] * profile["gradient_accumulation_steps"],
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map=profile["device_map"],
        torch_dtype=profile["torch_dtype"],
        attn_implementation=(
            "flash_attention_2" if profile["use_flash_attention"] else "sdpa"
        ),
        low_cpu_mem_usage=True,
    )
    freeze_embeddings(model)

    if hasattr(model, "enable_input_require_grads") and profile["gradient_checkpointing"]:
        model.enable_input_require_grads()

    model.config.use_cache = False

    peft_config = None
    if profile["training_mode"] == "lora":
        print("Configuring LoRA adapters...")
        peft_config = build_lora_config()
    else:
        print("Configuring full fine-tuning path...")

    training_args = TrainingArguments(
        output_dir=TRAINING_ROOT,
        overwrite_output_dir=True,
        use_cpu=profile["use_cpu"],
        per_device_train_batch_size=profile["per_device_train_batch_size"],
        per_device_eval_batch_size=profile["per_device_train_batch_size"],
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        fp16=False,
        bf16=profile["bf16"],
        gradient_checkpointing=profile["gradient_checkpointing"],
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if profile["gradient_checkpointing"] else None
        ),
        save_strategy="no",
        eval_strategy="no",
        report_to=args.report_to,
        optim="adafactor",
        remove_unused_columns=False,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False if deepspeed_config else None,
        max_grad_norm=1.0,
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=profile["max_seq_length"],
        packing=profile["packing"],
    )

    print("Starting fine-tuning...")
    trainer.train()

    os.makedirs(TRAINING_ROOT, exist_ok=True)
    if profile["training_mode"] == "lora":
        os.makedirs(ADAPTER_OUTPUT_DIR, exist_ok=True)
        trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
        processor.save_pretrained(ADAPTER_OUTPUT_DIR)
        save_artifact_metadata(args.profile, profile, ADAPTER_OUTPUT_DIR)
        print(f"Done! Adapter weights saved to {ADAPTER_OUTPUT_DIR}")
    else:
        os.makedirs(FULL_MODEL_OUTPUT_DIR, exist_ok=True)
        trainer.save_model(FULL_MODEL_OUTPUT_DIR)
        processor.save_pretrained(FULL_MODEL_OUTPUT_DIR)
        save_artifact_metadata(args.profile, profile, FULL_MODEL_OUTPUT_DIR)
        print(f"Done! Full fine-tuned model saved to {FULL_MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    run_finetune(parse_args())