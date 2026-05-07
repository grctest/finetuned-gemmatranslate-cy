# TranslateGemma-4b Fine-Tuning for Scottish Gaelic (GD)

This project provides a complete, sequential pipeline for fine-tuning the **TranslateGemma-4b-it** model to support Scottish Gaelic (Gàidhlig) using LoRA (Low-Rank Adaptation).

## ⚠️ Prerequisites

* **OS:** Windows/Linux/macOS
* **RAM:** 32GB Minimum (64GB Recommended for the merging phase).
* **Disk Space:** ~20GB for model weights and dataset shards.
* **CPU:** Modern CPU with AVX-512 or AMX support is highly recommended for `bfloat16` performance.

## 🚀 Setup (WSL + venv)

1. **Create and Activate Virtual Environment:**
   Run these commands in your WSL terminal:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Hugging Face Login:**
   Since TranslateGemma is a gated model, ensure you have accepted the terms on Hugging Face and are logged in using the new `hf` CLI:
   ```bash
   hf auth login
   ```

## 🛠️ Execution Pipeline

The implementation is split into four scripts. Each script validates that the previous step was completed successfully. **Ensure your venv is active (`source venv/bin/activate`) before running.**

### Step 1: Data Preparation & Model Download
Downloads the OPUS-100 `en-gd` dataset, filters for quality, and downloads the base **TranslateGemma-4b-it** model weights to `./local_model` so everything is available offline for the next steps.
```bash
python 01_prepare_data.py
```

### Step 2: Fine-Tuning
Performs LoRA training with the shipped TranslateGemma chat template.

We support different hardware profiles natively. If you don't supply a flag, it defaults to the `3090` (24GB VRAM) profile. 

```bash
# Recommended for 24GB GPUs (RTX 3090, 4090)
python 02_finetune.py --profile 3090

# Recommended for 48GB+ GPUs (A6000, A100, H100) - Larger batches
python 02_finetune.py --profile high_vram

# If you only have CPU (will be very slow)
python 02_finetune.py --profile cpu
```
*Note: This will output progress every 10 steps to ensure the terminal doesn't appear frozen.*

### Step 3: Weight Merging
Merges the LoRA adapter weights back into the base Safetensors model to create a standalone model.
```bash
python 03_merge.py
```

### Step 4: Inference Testing
Loads the final merged model and runs test translations for both English → Gaelic and Gaelic → English.
```bash
python 04_inference.py
```

## 📂 Project Structure

- [01_prepare_data.py](01_prepare_data.py): Dataset acquisition and cleaning.
- [02_finetune.py](02_finetune.py): CPU-optimized LoRA training script.
- [03_merge.py](03_merge.py): Script to consolidate adapters and base weights.
- [04_inference.py](04_inference.py): Interactive/Sample test script.
- [requirements.txt](requirements.txt): Necessary Python libraries.

## 💡 Pro-Tips

- **Swap Memory:** If `03_merge.py` crashes due to RAM limitations, ensure you have a large Pagefile (Windows) or Swap partition (Linux) enabled.
- **Sequence Length:** Training is configured for a maximum sequence length of 2000 tokens by default.
- **Quantization:** Once `04_inference.py` is successful, you can use `llama.cpp` to convert the `./final_merged_model` to GGUF format for even faster local inference.
