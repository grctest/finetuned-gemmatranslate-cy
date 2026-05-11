import argparse
import importlib.util
import json
import os
import sys

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from model_assets import MODEL_PATH, build_missing_assets_error

DATASET_PATH = "./processed_data"
TRAINING_ROOT = "./translategemma-finetuned"
ADAPTER_OUTPUT_DIR = os.path.join(TRAINING_ROOT, "adapter")
FULL_MODEL_OUTPUT_DIR = os.path.join(TRAINING_ROOT, "full_model")
ARTIFACT_METADATA_PATH = os.path.join(TRAINING_ROOT, "training_artifact.json")
RESPONSE_TEMPLATE = "<start_of_turn>model\n"
USER_TURN_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN_TEMPLATE = "<end_of_turn>\n"

PROFILE_CONFIGS = {
    "cpu": {
        "description": "CPU-only fallback with LoRA and 512-token context.",
        "device_map": None,
        "dtype": torch.float32,
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
        "learning_rate": 1e-4,
        "optimizer": "adafactor",
    },
    "3090": {
        "description": "24GB VRAM LoRA profile targeting effective batch size 64.",
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "max_seq_length": 512,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": False,
        "packing": False, # Disabled until flash attention is used
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": False,
        "learning_rate": 1e-4,
        "optimizer": "adafactor",
    },
    "high_vram": {
        "description": "48GB+ VRAM LoRA profile with longer context and packing.",
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 2048,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": True,
        "packing": False, # Disabled until flash attention is used
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": False,
        "learning_rate": 1e-4,
        "optimizer": "adafactor",
    },
    "H100": {
        "description": "1x H100 full fine-tuning profile relying on native PyTorch SDPA.",
        "device_map": None,
        "dtype": torch.bfloat16,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 2048,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": False,
        "packing": False, # Disabled until flash attention is used
        "training_mode": "full",
        "deepspeed": None,
        "use_cpu": False,
        "learning_rate": 1e-4,
        "optimizer": "adafactor",
        "dataset_fraction": 1.0,
    },
    "H200": {
        "description": "1x H200 High-Density Profile. Maximizing 141GB VRAM and Hopper core throughput.",
        "device_map": None,
        "dtype": torch.bfloat16,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": False,
        "packing": False, # Disabled until flash attention is used
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": False,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch_fused",
        "dataset_fraction": 1.0,
    },
    "H200F": {
        "description": "1x H200 - Flash Attention 3 & Packing Enabled.",
        "device_map": None,
        "dtype": torch.bfloat16,
        "per_device_train_batch_size": 12,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_flash_attention": True,
        "packing": True, # Packing combines multiple short rows into robust 8192 blocks
        "training_mode": "lora",
        "deepspeed": None,
        "use_cpu": False,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch_fused",
        "dataset_fraction": 0.05,
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
        "--disable-packing",
        action="store_true",
        help="Disable sequence packing, regardless of the chosen hardware profile.",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Override the DeepSpeed config path used by the max_vram profile.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of CPU cores to use for initial dataset mapping. Defaults to all minus 1.",
    )
    parser.add_argument(
        "--sft-num-proc",
        type=int,
        default=min(4, max(1, os.cpu_count() - 1)),
        help="Number of CPU cores to use for SFTTrainer's heavy tokenization step. Capped default to avoid WSL/I/O PyArrow errors.",
    )
    args = parser.parse_args()

    max_cores = os.cpu_count() or 1
    args.num_proc = max(1, min(args.num_proc, max_cores))
    args.sft_num_proc = max(1, min(args.sft_num_proc, max_cores))
    
    return args


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


def resolve_eval_split(dataset):
    if "validation" in dataset:
        return "validation"
    if "test" in dataset:
        return "test"

    available_splits = ", ".join(sorted(dataset.keys()))
    raise ValueError(
        "processed_data must contain either a validation or test split for evaluation. "
        f"Available splits: {available_splits}"
    )


def build_translation_prompt(example, tokenizer):
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

    if not prompt.endswith(RESPONSE_TEMPLATE):
        raise ValueError(
            "TranslateGemma translation prompt no longer ends at the assistant response boundary."
        )

    return prompt


def build_instruction_prompt(example, tokenizer):
    bos_token = tokenizer.bos_token or ""
    prompt = (
        f"{bos_token}{USER_TURN_TEMPLATE}"
        f"{str(example['source_text']).strip()}"
        f"{END_OF_TURN_TEMPLATE}"
        f"{RESPONSE_TEMPLATE}"
    )

    if not prompt.endswith(RESPONSE_TEMPLATE):
        raise ValueError("Instruction prompt no longer ends at the assistant response boundary.")

    return prompt


def build_completion(example):
    return f"{str(example['target_text']).strip()}{END_OF_TURN_TEMPLATE}"


def render_prompt_completion(example, tokenizer):
    if example["task"] == "translation":
        prompt = build_translation_prompt(example, tokenizer)
    elif example["task"] == "instruction":
        prompt = build_instruction_prompt(example, tokenizer)
    else:
        raise ValueError(f"Unsupported task type '{example['task']}'.")

    return {
        "prompt": prompt,
        "completion": build_completion(example),
    }


def prepare_prompt_completion_dataset(split_dataset, tokenizer, max_samples, num_proc):
    if max_samples is not None:
        split_dataset = split_dataset.select(range(min(max_samples, len(split_dataset))))

    return split_dataset.map(
        lambda example: render_prompt_completion(example, tokenizer),
        remove_columns=split_dataset.column_names,
        desc="Rendering TranslateGemma prompt-completion rows",
        num_proc=num_proc,
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


def save_artifact_metadata(profile_name, profile, artifact_path, packing_enabled, deepspeed_config):
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
                "optimizer": profile.get("optimizer", "adafactor"),
                "learning_rate": profile.get("learning_rate", 1e-4),
                "completion_only_loss": True,
                "packing_enabled": packing_enabled,
                "deepspeed_config": deepspeed_config,
                "embedding_freeze": True,
                "task_mix": {
                    "translation": "70% target",
                    "instruction": "30% target",
                },
                "translation_directions": ["en->cy", "cy->en"],
            },
            handle,
            indent=2,
        )


def resolve_attention_backends(profile):
    if not profile["use_flash_attention"]:
        return ["sdpa"]

    compute_capability_major = None
    if torch.cuda.is_available():
        try:
            compute_capability_major, _ = torch.cuda.get_device_capability()
        except RuntimeError:
            compute_capability_major = None

    is_blackwell = compute_capability_major is not None and compute_capability_major >= 10
    backends = []

    if is_blackwell:
        if importlib.util.find_spec("flash_attn.cute") is not None:
            backends.append("flash_attention_4")
        else:
            print(
                "Blackwell GPU detected. Flash Attention 3 is Hopper-only, so this run will "
                "skip it unless Flash Attention 4 is installed."
            )
    else:
        if importlib.util.find_spec("flash_attn_interface") is not None:
            backends.append("flash_attention_3")
        if importlib.util.find_spec("flash_attn") is not None:
            backends.append("flash_attention_2")

    backends.append("sdpa")
    return backends


def resolve_deepspeed_config(args, profile):
    requested_config = args.deepspeed_config or profile["deepspeed"]
    if requested_config is None:
        return None

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_with_distributed_runtime = any(
        key in os.environ
        for key in (
            "ACCELERATE_PROCESS_INDEX",
            "ACCELERATE_USE_DEEPSPEED",
            "LOCAL_RANK",
            "RANK",
        )
    )

    if world_size <= 1 and not launched_with_distributed_runtime:
        print(
            "Single-GPU run detected; disabling DeepSpeed for the max_vram profile. "
            "This is the recommended path for a single B200 when launching with plain python."
        )
        return None

    return requested_config


def run_finetune(args):
    # Enforce PyTorch native NCCL distributed backend to prevent DeepSpeed from 
    # falling back to MPI (mpi4py) when executed without accelerate launch.
    # This is the highest-performance path for NVIDIA GPUs like the B200.
    #if "WORLD_SIZE" not in os.environ:
    #    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    #    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    #    os.environ["RANK"] = os.environ.get("RANK", "0")
    #    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
    #    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")

    """
    if args.instruction_mix_ratio > 0:
        raise ValueError(
            "Instruction-mix wiring is only scaffolded right now. Keep --instruction-mix-ratio at 0 until an external instruction dataset is integrated."
        )
    """

    if not os.path.exists(DATASET_PATH):
        print("Error: ./processed_data not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("Error: ./local_model not found. Please run 01_prepare_data.py first.")
        sys.exit(1)

    missing_assets_error = build_missing_assets_error(MODEL_PATH)
    if missing_assets_error:
        print(f"Error: {missing_assets_error}")
        sys.exit(1)

    profile = PROFILE_CONFIGS[args.profile]
    deepspeed_config = resolve_deepspeed_config(args, profile)
    
    packing_enabled = profile.get("packing", False)
    if args.disable_packing:
        packing_enabled = False

    print("Loading processed dataset...")
    dataset = load_from_disk(DATASET_PATH)
    validate_processed_dataset(dataset)
    eval_split_name = resolve_eval_split(dataset)

    # Apply dataset fraction if defined in profile
    dataset_fraction = profile.get("dataset_fraction", 1.0)
    if dataset_fraction < 1.0:
        original_size = len(dataset["train"])
        new_size = int(original_size * dataset_fraction)
        print(f"Applying profile dataset fraction {dataset_fraction:.1%}: {original_size:,} -> {new_size:,} samples.")
        dataset["train"] = dataset["train"].select(range(new_size))

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_response_template(tokenizer)

    original_add_bos_token = getattr(tokenizer, "add_bos_token", None)
    if original_add_bos_token is not None:
        tokenizer.add_bos_token = False

    print("Rendering prompt-completion training records from flat Welsh data...")
    accelerator = Accelerator()
    with accelerator.main_process_first():
        train_dataset = prepare_prompt_completion_dataset(
            dataset["train"], tokenizer, args.max_train_samples, args.num_proc
        )
        # Evaluation is disabled in SFTConfig, so skip rendering it to save time
        eval_dataset = None

    print(f"Profile [{args.profile}]: {profile['description']}")
    print(
        "Effective batch size:",
        profile["per_device_train_batch_size"] * profile["gradient_accumulation_steps"],
    )

    attention_backends = resolve_attention_backends(profile)
    selected_attention_backend = None
    model = None
    last_attention_error = None

    for attention_backend in attention_backends:
        try:
            print(f"Loading model with attention backend: {attention_backend}")
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH,
                device_map=profile["device_map"],
                dtype=profile["dtype"],
                attn_implementation=attention_backend,
                low_cpu_mem_usage=True,
            )
            selected_attention_backend = attention_backend
            break
        except (ImportError, ValueError) as exc:
            last_attention_error = exc
            if attention_backend == attention_backends[-1]:
                raise
            print(
                f"Attention backend '{attention_backend}' is unavailable ({exc}). "
                "Trying the next fallback..."
            )

    if model is None:
        raise RuntimeError("Failed to initialize the model attention backend.") from last_attention_error

    if selected_attention_backend == "flash_attention_4":
        print("Using Flash Attention 4.")
    elif selected_attention_backend == "flash_attention_3":
        print("Using Flash Attention 3.")
    elif profile["use_flash_attention"]:
        print(f"Using fallback attention backend: {selected_attention_backend}")

    freeze_embeddings(model)

    if hasattr(model, "enable_input_require_grads") and profile["gradient_checkpointing"]:
        model.enable_input_require_grads()

    model.config.use_cache = False

    # Explicitly align the model config with the tokenizer to prevent warnings
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    peft_config = None
    if profile["training_mode"] == "lora":
        print("Configuring LoRA adapters...")
        peft_config = build_lora_config()
    else:
        print("Configuring full fine-tuning path...")

    training_args = SFTConfig(
        output_dir=TRAINING_ROOT,
        use_cpu=profile["use_cpu"],
        per_device_train_batch_size=profile["per_device_train_batch_size"],
        per_device_eval_batch_size=profile["per_device_train_batch_size"],
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        learning_rate=profile.get("learning_rate", 1e-4),
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
        optim=profile.get("optimizer", "adafactor"),
        remove_unused_columns=False,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False if deepspeed_config else None,
        max_grad_norm=1.0,
        max_length=profile["max_seq_length"],
        packing=packing_enabled,
        eval_packing=packing_enabled,
        completion_only_loss=True,
        dataset_num_proc=args.sft_num_proc,
        dataloader_num_workers=args.sft_num_proc,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()

    if original_add_bos_token is not None:
        tokenizer.add_bos_token = original_add_bos_token

    os.makedirs(TRAINING_ROOT, exist_ok=True)
    if profile["training_mode"] == "lora":
        os.makedirs(ADAPTER_OUTPUT_DIR, exist_ok=True)
        trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
        processor.save_pretrained(ADAPTER_OUTPUT_DIR)
        save_artifact_metadata(
            args.profile,
            profile,
            ADAPTER_OUTPUT_DIR,
            packing_enabled,
            deepspeed_config,
        )
        print(f"Done! Adapter weights saved to {ADAPTER_OUTPUT_DIR}")
    else:
        os.makedirs(FULL_MODEL_OUTPUT_DIR, exist_ok=True)
        trainer.save_model(FULL_MODEL_OUTPUT_DIR)
        processor.save_pretrained(FULL_MODEL_OUTPUT_DIR)
        save_artifact_metadata(
            args.profile,
            profile,
            FULL_MODEL_OUTPUT_DIR,
            packing_enabled,
            deepspeed_config,
        )
        print(f"Done! Full fine-tuned model saved to {FULL_MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    run_finetune(parse_args())