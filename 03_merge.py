import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel

MODEL_ID = "./local_model"

def merge_weights():
    if not os.path.exists("./translategemma-finetuned"):
        print("Error: ./translategemma-finetuned not found. Please run 02_finetune.py first.")
        sys.exit(1)
    
    if not os.path.exists("./local_model"):
        print("Error: ./local_model not found. Pre-downloaded base model is required.")
        sys.exit(1)

    print("Loading base tokenizer and model from local path...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16
    )

    print("Loading LoRA adapter weights...")
    model = PeftModel.from_pretrained(base_model, "./translategemma-finetuned")

    print("Merging adapters into base model (this may take a lot of RAM)...")
    merged_model = model.merge_and_unload()
    # Ensure the merged model remains in bfloat16
    merged_model = merged_model.to(torch.bfloat16)

    print("Saving the final merged model and tokenizer...")
    os.makedirs("./final_merged_model", exist_ok=True)
    merged_model.save_pretrained("./final_merged_model")
    tokenizer.save_pretrained("./final_merged_model")

    print("Done! Merged model ready in ./final_merged_model.")

if __name__ == "__main__":
    merge_weights()