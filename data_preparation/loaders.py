import json
import random
from datasets import Dataset
from .utils import get_pair_hash

def build_translation_records(en_text, cy_text, directions):
    """
    Expands a raw English/Welsh pair into specific directional records.
    
    Generates 'en-cy' and/or 'cy-en' tasks based on the enabled directions.
    """
    records = []
    en_text, cy_text = str(en_text).strip(), str(cy_text).strip()
    if not en_text or not cy_text: return records

    for d in directions:
        if d == "en-cy":
            records.append({"task": "translation", "source_text": en_text, "target_text": cy_text, "source_lang_code": "en", "target_lang_code": "cy"})
        elif d == "cy-en":
            records.append({"task": "translation", "source_text": cy_text, "target_text": en_text, "source_lang_code": "cy", "target_lang_code": "en"})
    return records

def get_en_to_cy_templates(en, cy, cy_def):
    templates = [
        (f"How do you say '{en}' in Welsh? Please provide a definition.", f"The Welsh term for '{en}' is '{cy}'. It refers to {cy_def}"),
        (f"What is the Welsh equivalent of the term '{en}'? Explain what it means.", f"In Welsh, '{en}' is translated as '{cy}'. This term is used to describe {cy_def}"),
        (f"I'm looking for the Welsh word for '{en}'. Could you also explain its meaning?", f"Certainly! The Welsh word is '{cy}', which means {cy_def}")
    ]
    return random.choice(templates)

def get_cy_to_en_templates(cy, en, en_def):
    templates = [
        (f"Beth yw'r gair Saesneg am '{cy}', a beth yw'r diffiniad?", f"Y term Saesneg ar gyfer '{cy}' yw '{en}'. Mae'n golygu {en_def}"),
        (f"Sut ydych chi'n dweud '{cy}' yn Saesneg? Eglurwch yr ystyr hefyd.", f"'{en}' yw'r cyfystyron Saesneg ar gyfer '{cy}'. Dyma'r esboniad: {en_def}")
    ]
    return random.choice(templates)

def parse_termcymru(ds, directions):
    """
    Parses the specialized TermCymru dictionary format.
    
    Extracts direct translations and generates instruction records using 
    definitions and context from the Welsh terminology metadata.
    """
    records = []
    counts = {"translations": 0, "translation_directions": {"en->cy": 0, "cy->en": 0}, "en_instructions": 0, "cy_instructions": 0}
    null_vals = ["none", "null", ""]
    
    for row in ds:
        en, cy = str(row.get("Saesneg", "")).strip(), str(row.get("Cymraeg", "")).strip()
        en_def, cy_def = str(row.get("Diffiniad Saesneg", "")).strip(), str(row.get("Diffiniad Cymraeg", "")).strip()
        
        if en and cy and en.lower() not in null_vals and cy.lower() not in null_vals:
            t_recs = build_translation_records(en, cy, directions)
            records.extend(t_recs)
            counts["translations"] += len(t_recs)
            for r in t_recs:
                d = f"{r['source_lang_code']}->{r['target_lang_code']}"
                counts["translation_directions"][d] = counts["translation_directions"].get(d, 0) + 1
            
            ctx_en, ctx_cy = str(row.get("Cyd-destun Saesneg", "")).strip(), str(row.get("Cyd-destun Cymraeg", "")).strip()
            en_def_e = f"{en_def}\nContext: {ctx_en}" if ctx_en and ctx_en.lower() not in null_vals else en_def
            cy_def_e = f"{cy_def}\nCyd-destun: {ctx_cy}" if ctx_cy and ctx_cy.lower() not in null_vals else cy_def
            
            if cy_def and cy_def.lower() not in null_vals:
                p, r = get_en_to_cy_templates(en, cy, cy_def_e)
                records.append({"task": "instruction", "source_text": p, "target_text": r, "source_lang_code": "en", "target_lang_code": "cy"})
                counts["en_instructions"] += 1
            if en_def and en_def.lower() not in null_vals:
                p, r = get_cy_to_en_templates(cy, en, en_def_e)
                records.append({"task": "instruction", "source_text": p, "target_text": r, "source_lang_code": "cy", "target_lang_code": "en"})
                counts["cy_instructions"] += 1
    return Dataset.from_list(records), counts

def process_translation_ds(ds, directions):
    """
    Standardizes generic translation datasets (OPUS-100, etc.) into 
    the internal unified format.
    
    Includes heuristic column mapping for common dataset schemas.
    """
    cols = ds.column_names
    records = []
    for row in ds:
        en, cy = "", ""
        if "translation" in cols:
            t = row["translation"]
            if isinstance(t, str): t = json.loads(t)
            en, cy = t.get("en", ""), t.get("cy", "")
        else:
            for k in ["text_en", "en", "english", "source", "Saesneg"]:
                if k in cols and row.get(k): en = row[k]; break
            for k in ["text_cy", "cy", "welsh", "cymraeg", "target"]:
                if k in cols and row.get(k): cy = row[k]; break
        records.extend(build_translation_records(en, cy, directions))
    return Dataset.from_list(records)

def process_instruction_row(row, source_lang_code="en", target_lang_code="en"):
    """
    Parses complex instruction formats (ShareGPT/Messages) into 
    source/target pairs.
    
    Extracts the first user turn as the source and the subsequent 
    assistant response as the target.
    """
    src, tgt = "", ""
    if isinstance(row.get("messages"), list):
        for m in row["messages"]:
            r, c = str(m.get("role", "")).lower(), m.get("content", "")
            if isinstance(c, list): c = " ".join(p.get("text", str(p)) for p in c)
            if r in ("user", "human") and not src: src = str(c).strip()
            elif r == "assistant" and not tgt: tgt = str(c).strip()
    elif isinstance(row.get("conversations"), list):
        for m in row["conversations"]:
            r = str(m.get("role", "") if isinstance(m, dict) else "").lower()
            c = m.get("content", m) if isinstance(m, dict) else m
            if r in ("user", "human") and not src: src = str(c).strip()
            elif r == "assistant" and not tgt: tgt = str(c).strip()
    else:
        for k in ["instruction", "prompt", "input", "text_en"]:
            if row.get(k): src = str(row[k]).strip(); break
        for k in ["output", "response", "completion", "text_cy"]:
            if row.get(k): tgt = str(row[k]).strip(); break
    return {"task": "instruction", "source_text": src, "target_text": tgt, "source_lang_code": source_lang_code, "target_lang_code": target_lang_code}
