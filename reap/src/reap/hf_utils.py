import logging
import os

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger("reap")


def is_local_dir(path: str) -> bool:
    try:
        return os.path.isdir(path)
    except Exception:
        return False


def check_tokenizer_dir(model_dir: str) -> bool:
    if not is_local_dir(model_dir):
        logger.info("Tokenizer will be loaded from remote or cache: %s", model_dir)
        return True
    files = set(os.listdir(model_dir))
    has_tokenizer = (
        ("tokenizer.json" in files)
        or (("vocab.txt" in files) and ("tokenizer_config.json" in files))
        or (("vocab.json" in files) and ("merges.txt" in files))
        or ("spiece.model" in files)
    )
    if not has_tokenizer:
        logger.error("Tokenizer files are missing at %s", model_dir)
        return False
    return True


def check_model_weights_or_repo(src: str) -> bool:
    if not is_local_dir(src):
        logger.info("Model weights will be loaded from remote or cache: %s", src)
        return True
    files = set(os.listdir(src))
    has_cfg = "config.json" in files
    has_weights = ("pytorch_model.bin" in files) or ("model.safetensors" in files)
    if not (has_cfg and has_weights):
        logger.error("Model config or weights are missing at %s", src)
        return False
    return True


def load_tokenizer(model_dir: str) -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    except Exception as e:
        msg = str(e).lower()
        if "sentencepiece" in msg and is_local_dir(model_dir) and os.path.exists(os.path.join(model_dir, "tokenizer.json")):
            tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
        else:
            raise
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok
