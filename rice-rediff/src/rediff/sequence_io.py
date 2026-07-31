from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ALPHABET = set("ACGTN")
BASES = np.array(list("ACGT"))
BASE_TO_IDX = {b: i for i, b in enumerate("ACGT")}
RC_MAP = str.maketrans("ACGTacgt", "TGCAtgca")


def detect_sequence_column(df: pd.DataFrame, sequence_col: Optional[str] = None) -> str:
    if sequence_col:
        if sequence_col not in df.columns:
            raise ValueError(f"Column '{sequence_col}' was not found in {list(df.columns)}")
        return sequence_col

    candidates = ["sequence", "Sequence", "seq", "Seq", "enhancer", "Enhancer", "dna", "DNA"]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        values = df[col].dropna().astype(str)
        if values.empty:
            continue
        probe = values.iloc[0].strip().upper()
        if probe and all(ch in ALPHABET for ch in probe[:50]):
            return col
    raise ValueError("Could not infer a DNA sequence column.")


def read_csv_sequences(path: str | Path, sequence_col: Optional[str] = None) -> List[str]:
    df = pd.read_csv(path)
    col = detect_sequence_column(df, sequence_col)
    return df[col].astype(str).tolist()


def clean_sequences(
    seqs: Iterable[str],
    max_n_frac: float = 0.2,
    dedup: bool = True,
) -> List[str]:
    out: List[str] = []
    seen = set()
    for seq in seqs:
        s = str(seq).strip().upper()
        if not s:
            continue
        if any(ch not in ALPHABET for ch in s):
            continue
        if s.count("N") / max(1, len(s)) > max_n_frac:
            continue
        if dedup and s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def rev_comp(seq: str) -> str:
    return seq.translate(RC_MAP)[::-1]


def center_crop(seq: str, length: int) -> Tuple[str, int, int]:
    start = (len(seq) - length) // 2
    end = start + length
    return seq[start:end], start, end


def center_pad(seq: str, length: int, pad_base: str = "A") -> str:
    pad = length - len(seq)
    left = pad // 2
    right = pad - left
    return (pad_base * left) + seq + (pad_base * right)


def to_fixed_length(seq: str, length: int, pad_base: str = "A") -> Tuple[str, bool, int, int]:
    if len(seq) >= length:
        fixed, start, end = center_crop(seq, length)
        return fixed, False, start, end
    return center_pad(seq, length, pad_base), True, -1, -1


def one_hot_01(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    eye = np.eye(4, dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        idx = BASE_TO_IDX.get(ch)
        arr[i, :] = 0.25 if idx is None else eye[idx]
    return arr


def scale_input(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "pm1":
        return x * 2.0 - 1.0
    if mode == "01":
        return x
    raise ValueError(f"Unknown input scale: {mode}")


def decode_indices(indices: np.ndarray) -> List[str]:
    idx = np.asarray(indices)
    if idx.ndim == 1:
        return ["".join(BASES[idx])]
    return ["".join(BASES[row]) for row in idx]


def read_fasta(path: str | Path, min_len: int = 1) -> List[str]:
    seqs: List[str] = []
    buf: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf).upper())
                    buf = []
            else:
                buf.append(line)
        if buf:
            seqs.append("".join(buf).upper())

    clean = []
    for seq in seqs:
        s = "".join(ch if ch in "ACGT" else "A" for ch in seq)
        if len(s) >= min_len:
            clean.append(s)
    return clean


def write_fasta(path: str | Path, seqs: Sequence[str], prefix: str = "seq") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for i, seq in enumerate(seqs, start=1):
            handle.write(f">{prefix}_{i}\n{str(seq).upper()}\n")


def write_split_csv(
    path: str | Path,
    seqs: Sequence[str],
    split_labels: Sequence[str],
    orig_lens: Sequence[int],
    was_padded: Sequence[bool],
    crop_starts: Sequence[int],
    crop_ends: Sequence[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "sequence_512", "split", "orig_len", "was_padded", "crop_start", "crop_end"])
        for i, row in enumerate(zip(seqs, split_labels, orig_lens, was_padded, crop_starts, crop_ends)):
            writer.writerow([i, *row])
