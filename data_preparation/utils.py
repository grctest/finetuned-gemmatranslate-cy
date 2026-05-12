import os
import hashlib
from transformers import AutoTokenizer

PROCESSED_DATA_DIR = "./processed_data"
_WORKER_TOKENIZERS = {}

def get_worker_tokenizer(tokenizer_path):
    """
    Initializes and caches a tokenizer instance for the current process.
    
    Used primarily in multi-processing workers to avoid redundant I/O 
    and pickling overhead.
    """
    if tokenizer_path in _WORKER_TOKENIZERS:
        return _WORKER_TOKENIZERS[tokenizer_path]
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    _WORKER_TOKENIZERS[tokenizer_path] = tok
    return tok

def get_pair_hash(s, t):
    """
    Generates a unique SHA1 hex digest for a source-target pair.
    
    Used for streaming deduplication to identify identical translation 
    units across different datasets.
    """
    s_str, t_str = str(s), str(t)
    return hashlib.sha1(f"{s_str}|||{t_str}".encode("utf-8")).hexdigest()

def get_pair_hash(s, t):
    """
    Generates a unique SHA1 hex digest for a source-target string pair.
    
    Used for global deduplication across multiple input datasets.
    """
    s = "" if s is None else str(s)
    t = "" if t is None else str(t)
    return hashlib.sha1((s + "\0" + t).encode("utf-8")).digest().hex()
