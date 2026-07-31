import bz2
import gzip
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .sequence import clean_seq


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_df_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_json_safely(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def to_numpy_1d(x):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, list):
        x = np.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def batch_iterator(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size], i, min(i + batch_size, len(arr))


def dataset_stem(input_path: str) -> str:
    base = os.path.basename(input_path)
    while True:
        root, ext = os.path.splitext(base)
        if ext.lower() in (".gz", ".bz2", ".fa", ".fasta", ".csv", ".tsv", ".txt"):
            base = root
        else:
            break
    return base


def open_text_maybe_compressed(path: str):
    lower = path.lower()
    if lower.endswith(".gz"):
        return gzip.open(path, "rt")
    if lower.endswith(".bz2"):
        return bz2.open(path, "rt")
    return open(path, "r", encoding="utf-8")


def is_fasta_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fa.bz2", ".fasta.bz2"))


def sniff_fasta(path: str) -> bool:
    try:
        with open_text_maybe_compressed(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    return line.startswith(">")
    except Exception:
        return False
    return False


def parse_fasta(path: str) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    cur_id, parts = None, []
    with open_text_maybe_compressed(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append(clean_seq("".join(parts)))
                cur_id = line[1:].split()[0] or f"seq_{len(ids) + 1}"
                parts = []
            else:
                parts.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append(clean_seq("".join(parts)))
    return ids, seqs


def detect_seq_col(df: pd.DataFrame) -> str:
    for col in ["sequence", "seq", "Sequence", "dna", "DNA"]:
        if col in df.columns:
            return col
    raise ValueError("No sequence column found. Expected one of: sequence, seq, Sequence, dna, DNA.")


def read_input_any(input_path: str) -> Tuple[List[str], Optional[np.ndarray], List[str], str]:
    if is_fasta_path(input_path) or sniff_fasta(input_path):
        ids, seqs = parse_fasta(input_path)
        return seqs, None, ids, "fasta"

    sep = "\t" if input_path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(input_path, sep=sep)
    seq_col = detect_seq_col(df)
    seqs = [clean_seq(s) for s in df[seq_col].astype(str).tolist()]
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [f"seq_{i + 1}" for i in range(len(seqs))]
    y_true = None
    for col in ["expression", "y_true", "label", "target"]:
        if col in df.columns:
            y_true = df[col].astype(float).values
            break
    return seqs, y_true, ids, "csv"
