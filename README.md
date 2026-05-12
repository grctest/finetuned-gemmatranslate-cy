# TranslateGemma Fine-Tuning for English and Welsh

This repository fine-tunes `google/translategemma-4b-it` for English↔Welsh translation using the OPUS-100 `cy-en` split. The pipeline now supports multiple hardware profiles, a paper-aligned training configuration, and a MetricX-based evaluation path.

## Setup

0. Install python
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Then close and reopen your terminal!

0.5. Create a screen env
```bash
screen -S finetune

# for reconnecting:
screen -ls
screen -r finetune
```

1. Create and activate a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```

1. Install the core dependencies.
```bash
pip install --upgrade pip
pip install -r requirements.txt
#pip install --no-build-isolation transformer_engine[pytorch] # Multi-gpu..
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2110
```

If you need a CUDA or ROCm-specific PyTorch wheel, install a `torch` 2.11.x build first, then run `pip install -r requirements.txt` for the pinned Hugging Face stack.

3. Accept the TranslateGemma license on Hugging Face and log in.
```bash
hf auth login
```

## Pipeline

### Step 1: Prepare Welsh Data and Summary Configuration
This step evaluates the proportions defined in `data_recipe.json`, downloads/caches datasets as required, synthesizes instruction Q&A logic out of dictionary arrays, expands configured high-quality translation sources bidirectionally, rebalances the post-dedup pool toward the target 70:30 translation/instruction mix, and writes summary metrics back dynamically.

The held-out split is now stratified by task and language direction, and its size is configurable through `meta_strategy.eval_size` in `data_recipe.json`.

It also ensures `./local_model` contains the full Hugging Face TranslateGemma snapshot needed for multimodal processor loading, including:

- `preprocessor_config.json`
- `processor_config.json`
- `special_tokens_map.json`
- `tokenizer.model`
- tokenizer/config/chat template files
- model weights, either as `model.safetensors` or the sharded `model.safetensors.index.json` plus `model-*.safetensors`

```bash
python 01_prepare_data.py | tee 01_prep_log.txt
```
*(You will be prompted `[y/N]` visually to inspect the constructed data totals before the merge script allocates array memory).*

#### Optional Step 1b:

```bash
python 01b_analyze_token_lengths.py --num-proc 6 --dataset-fraction 0.05 | tee 01b_token_analysis.txt
python 01b_analyze_token_lengths.py --num-proc 20 | tee 01b_token_analysis.txt
```

### Step 2: Fine-Tune
All profiles use the same flat English/Welsh dataset contract, freeze embeddings, and train on prompt-completion pairs with completion-only loss, matching the TranslateGemma technical report more closely.

Recommended profile commands:
```bash
python 02_finetune.py --profile 3090
python 02_finetune.py --profile high_vram
python 02_finetune.py --profile H200 --num-proc 20 --sft-num-proc 30
python 02_finetune.py --profile H200F --num-proc 20 --sft-num-proc 30 | tee 02_finetune_log.txt
```

Experimental commands:
```
accelerate launch --num_processes 8 --use_deepspeed 02_finetune.py --profile max_vram

accelerate launch --num_processes 1 --use_deepspeed 02_finetune.py \
  --profile max_vram \
  --num-proc 16 \
  --sft-num-proc 4

accelerate launch --multi_gpu --num_processes 2 02_finetune.py --profile high_vram --num-proc 16 --sft-num-proc 4

accelerate launch --num_processes 4 --use_deepspeed 02_finetune.py --profile max_vram --deepspeed-config ./ds_config_no_offload.json
```

### Step 3: Merge LoRA Adapters
Only run this after LoRA profiles.
```bash
python 03_merge.py --profile cpu
```

If you trained with `--profile max_vram`, skip this step.

### Step 4: Testing Inference

```bash
python 04_inference.py --profile cpu
```

### Step 5: Convert to GGUF

```
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py ./final_merged_model --outfile 4B_cy_q8_0.gguf --outtype q8_0 | tee 06_conversion_log.txt
```

## Output Layout

- `./processed_data`: regenerated flat English/Welsh dataset.
- `./local_model`: downloaded TranslateGemma processor and base model.
- `./translategemma-finetuned/adapter`: LoRA adapters for lighter profiles.
- `./translategemma-finetuned/full_model`: full fine-tuned model for `max_vram`.
- `./translategemma-finetuned/training_artifact.json`: metadata describing the latest training artifact.
- `./final_merged_model`: merged standalone model produced by `03_merge.py`.
- `./evaluation`: generated predictions and MetricX summaries.