import os

from huggingface_hub import snapshot_download


MODEL_ID = "google/translategemma-4b-it"
MODEL_PATH = "./local_model"

REQUIRED_PROCESSOR_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

OPTIONAL_TOKENIZER_FILES = [
    "added_tokens.json",
]


def _model_weights_present(model_path):
    direct_weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
    ]
    if any(os.path.exists(os.path.join(model_path, name)) for name in direct_weight_files):
        return True

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False

    return any(
        name.startswith("model-") and name.endswith(".safetensors")
        for name in os.listdir(model_path)
    )


def find_missing_model_assets(model_path):
    missing = [
        name for name in REQUIRED_PROCESSOR_FILES if not os.path.exists(os.path.join(model_path, name))
    ]

    if not _model_weights_present(model_path):
        missing.append(
            "model weights (model.safetensors or model.safetensors.index.json + model-*.safetensors)"
        )

    return missing


def ensure_local_model_snapshot(model_path=MODEL_PATH, model_id=MODEL_ID, force_download=False):
    missing = find_missing_model_assets(model_path)
    if not force_download and not missing:
        return []

    os.makedirs(model_path, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        resume_download=True,
    )
    return find_missing_model_assets(model_path)


def build_missing_assets_error(model_path=MODEL_PATH):
    missing = find_missing_model_assets(model_path)
    if not missing:
        return None

    rendered = "\n  - ".join(missing)
    return (
        f"{model_path} is missing required TranslateGemma multimodal assets:\n"
        f"  - {rendered}\n"
        "Run `python 01_prepare_data.py --yes` to download the complete Hugging Face snapshot, "
        "including processor metadata and model weights."
    )