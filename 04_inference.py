import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

def run_inference():
    if not os.path.exists("./final_merged_model"):
        print("Error: ./final_merged_model not found. Please run 03_merge.py first.")
        sys.exit(1)

    print("Loading merged model and tokenizer for inference...")
    tokenizer = AutoTokenizer.from_pretrained("./final_merged_model", clean_up_tokenization_spaces=False)
    model = AutoModelForImageTextToText.from_pretrained(
        "./final_merged_model",
        device_map="auto",
        dtype=torch.bfloat16
    )

    def translate(text, source_code="en", target_code="gd"):
        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": text
                }]
            }
        ]
        
        print(f"\nFormatting prompt for {source_code} -> {target_code}...")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        print("Generating response...")
        outputs = model.generate(**inputs, max_new_tokens=100)
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response

    print("\n--- Testing Translations ---")
    
    # Test English -> Scottish Gaelic
    en_text = "The quick brown fox jumps over the lazy dog."
    print(f"Input (EN): {en_text}")
    print(f"Output (GD): {translate(en_text, 'en', 'gd')}")
    
    print("\n-----------------------------")
    
    # Test Scottish Gaelic -> English
    # "Hello, how are you today?" in Gaelic: "Halò, ciamar a tha thu an-diugh?"
    gd_text = "Halò, ciamar a tha thu an-diugh?"
    print(f"Input (GD): {gd_text}")
    print(f"Output (EN): {translate(gd_text, 'gd', 'en')}")

if __name__ == "__main__":
    run_inference()