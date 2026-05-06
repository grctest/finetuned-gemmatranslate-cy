import os
import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "google/translategemma-4b-it"

def format_example(source_text, target_text, src_code="en", tgt_code="gd"):
    return {
        "text": f"User: Translate from {src_code} to {tgt_code}:\n{source_text}\n\nAssistant: {target_text}"
        # Simplified instruction template for broad compatibility if we bypass actual chat templates
    }

def format_example_chat(source_text, target_text, src_code="en", tgt_code="gd"):
    return {
        "messages": [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": src_code,
                    "target_lang_code": tgt_code,
                    "text": source_text
                }]
            },
            {
                "role": "assistant",
                "content": target_text
            }
        ]
    }

def is_quality_pair(en_text, gd_text):
    if not en_text or not gd_text:
        return False
    len_en = len(en_text)
    len_gd = len(gd_text)
    if len_en == 0 or len_gd == 0:
        return False
    # Filter 3x length discrepancy
    if len_en > 3 * len_gd or len_gd > 3 * len_en:
        return False
    return True

def process_and_save():
    print("Step 1: Downloading OPUS-100 en-gd dataset from Hugging Face...")
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-gd")
    
    processed_dataset = {}
    
    for split in dataset.keys():
        print(f"Processing split: {split} (Original size: {len(dataset[split])})")
        new_data = {"messages": []}
        
        for item in dataset[split]:
            translations = item["translation"]
            en_text = translations.get("en", "").strip()
            gd_text = translations.get("gd", "").strip()
            
            if not is_quality_pair(en_text, gd_text):
                continue
                
            formatted_en_gd = format_example_chat(en_text, gd_text, "en", "gd")
            formatted_gd_en = format_example_chat(gd_text, en_text, "gd", "en")

            # Store as JSON strings to avoid PyArrow schema mixing list/string errors
            new_data["messages"].append(json.dumps(formatted_en_gd["messages"]))
            new_data["messages"].append(json.dumps(formatted_gd_en["messages"]))
            
        print(f" -> Processed {split} (New bidirectional size: {len(new_data['messages'])})")
        
        # We will map it to a format the datasets library can save
        processed_dataset[split] = new_data
        
    print("Saving processed data to disk...")
    os.makedirs("./processed_data", exist_ok=True)
    
    from datasets import Dataset
    train_ds = Dataset.from_dict(processed_dataset["train"])
    val_ds = Dataset.from_dict(processed_dataset["validation"])
    test_ds = Dataset.from_dict(processed_dataset["test"])
    
    DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds}).save_to_disk("./processed_data")
    print("Done! Data prepared and saved to ./processed_data.")

    print(f"\nStep 2: Downloading Tokenizer and Model ({MODEL_ID})...")
    print("This may take a while depending on your internet connection.")
    
    # Download tokenizer (TranslateGemma uses a specialized tokenizer but we load it via AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained("./local_model")
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model.save_pretrained("./local_model")
    
    print("\nStep 3: Post-Download Configuration Patching...")
    # 1. Update tokenizer_config.json to ensure pad_token is handled correctly
    config_path = "./local_model/tokenizer_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            t_config = json.load(f)
        
        # Ensure padding side is correct for training (usually right for packing, but depends on model)
        # Gemma traditionally prefers left padding for inference, but Trainer handles batching.
        # We'll stick to model defaults unless specific issues arise.
        t_config["clean_up_tokenization_spaces"] = True
        
        with open(config_path, "w") as f:
            json.dump(t_config, f, indent=2)
        print(" -> Patched tokenizer_config.json")

    # 2. Verify chat_template.jinja contains 'gd'
    template_path = "./local_model/chat_template.jinja"
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            template_content = f.read()
        
        if '"gd":' in template_content:
            print(" -> Verified: 'gd' is already supported in chat_template.jinja")
        else:
            print(" -> Warning: 'gd' not found in chat_template.jinja. Injecting...")
            # Simple injection after English if missing
            new_content = template_content.replace('"en": "English",', '"en": "English",\n    "gd": "Scottish Gaelic",')
            with open(template_path, "w") as f:
                f.write(new_content)
    
    print(f"\nDone! Model, tokenizer, and dataset are ready in ./local_model and ./processed_data")

if __name__ == "__main__":
    process_and_save()
