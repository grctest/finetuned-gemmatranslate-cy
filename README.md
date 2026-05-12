# TranslateGemma Fine-Tuning for improving bidirectional English and Welsh translations!

This repository fine-tunes `google/translategemma-4b-it` for English↔Welsh translations.

It supports multiple hardware profiles and has a [paper](https://arxiv.org/abs/2601.09012)-aligned training configuration.

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

Use this script to evaluate your training dataset for token length distribution.

Since our script and the paper use a 2048 max sequence length you should aim for your data to be within this limit.

```bash
python 01b_analyze_token_lengths.py --num-proc 6 --dataset-fraction 0.05 | tee 01b_token_analysis.txt
python 01b_analyze_token_lengths.py --num-proc 20 | tee 01b_token_analysis.txt
```

### Step 2: Fine-Tune

All profiles use the same flat English/Welsh dataset contract, freeze embeddings, and train on prompt-completion pairs with completion-only loss, matching the TranslateGemma technical report more closely.

Recommended profile commands:
```bash
python 02_finetune.py --profile H200 --num-proc 20 --sft-num-proc 30
python 02_finetune.py --profile H200F --num-proc 20 --sft-num-proc 30 | tee 02_finetune_log.txt
```

### Step 3: Merge LoRA Adapters

Once you've finetuned your model you have to merge the output with the original model, creating the final output model for inference/quantitization.

```bash
python 03_merge.py --profile cpu
```

### Step 4: Testing Inference

If you want to quickly test out inference of translations run the following script!

```bash
python 04_inference.py --profile cpu
```

### Step 5: Convert to GGUF

To use our finetuned model on translategemma tools you'll likely need to convert the model safetensors to a GGUF format - run these commands:

```
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py ./final_merged_model --outfile 4B_cy_q8_0.gguf --outtype q8_0 | tee 06_conversion_log.txt
```