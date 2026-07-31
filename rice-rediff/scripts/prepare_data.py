from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rediff.sequence_io import (
    clean_sequences,
    read_csv_sequences,
    to_fixed_length,
    write_fasta,
    write_split_csv,
)


def fingerprint(seqs: list[str]) -> str:
    digest = hashlib.sha256()
    for seq in seqs:
        digest.update((seq + "\n").encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 512 bp rice enhancer FASTA files for REDiff.")
    parser.add_argument("--input", required=True, help="Input CSV containing enhancer sequences.")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--sequence-col", default=None, help="Sequence column name. If omitted, infer automatically.")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n-frac", type=float, default=0.20)
    parser.add_argument("--pad-base", default="A")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = clean_sequences(
        read_csv_sequences(args.input, args.sequence_col),
        max_n_frac=args.max_n_frac,
        dedup=True,
    )
    fixed, padded, starts, ends, orig_lens = [], [], [], [], []
    for seq in raw:
        seq_fixed, was_padded, start, end = to_fixed_length(seq, args.seq_len, args.pad_base)
        fixed.append(seq_fixed)
        padded.append(was_padded)
        starts.append(start)
        ends.append(end)
        orig_lens.append(len(seq))

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(fixed))
    rng.shuffle(indices)
    n_val = int(len(indices) * args.val_frac)
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])

    labels = np.array(["train"] * len(fixed), dtype=object)
    labels[val_idx] = "val"

    train = [fixed[i] for i in train_idx]
    val = [fixed[i] for i in val_idx]
    write_fasta(out_dir / "train_512.fasta", train, prefix="train")
    write_fasta(out_dir / "val_512.fasta", val, prefix="val")
    write_split_csv(out_dir / "split_ddpm_v1.csv", fixed, labels, orig_lens, padded, starts, ends)

    metadata = {
        "input": str(Path(args.input)),
        "seq_len": args.seq_len,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "max_n_frac": args.max_n_frac,
        "pad_base": args.pad_base,
        "n_total": len(fixed),
        "n_train": len(train),
        "n_val": len(val),
        "fingerprint_raw": fingerprint(raw),
        "fingerprint_512": fingerprint(fixed),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }
    with open(out_dir / "split_ddpm_v1.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Prepared {len(train)} train and {len(val)} validation sequences in {out_dir}")


if __name__ == "__main__":
    main()
