import glob
import os
from typing import List


def subdirs(path: str) -> List[str]:
    try:
        return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except Exception:
        return []


def auto_discover_folds(models_root: str) -> List[int]:
    folds = set()
    for d in subdirs(models_root):
        base = os.path.basename(d)
        if not base.startswith("fold"):
            continue
        rest = base[4:]
        num = ""
        for ch in rest:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            folds.add(int(num))
    if folds:
        return sorted(folds)

    fallback = []
    for k in [1, 2, 3, 4, 5]:
        if glob.glob(os.path.join(models_root, f"fold{k}_*")) or os.path.isdir(os.path.join(models_root, f"fold{k}")):
            fallback.append(k)
    return fallback if fallback else [1, 2, 3, 4, 5]


def find_latest_fold_dir(models_root: str, k: int) -> str:
    exact = os.path.join(models_root, f"fold{k}")
    pattern = os.path.join(models_root, f"fold{k}_*")
    cands = []
    if os.path.isdir(exact):
        cands.append(exact)
    cands.extend(glob.glob(pattern))
    cands = sorted(set(cands), key=lambda p: os.path.getmtime(p), reverse=True)
    if not cands:
        raise FileNotFoundError(f"[Fold {k}] no fold directory found: {exact} or {pattern}")
    return cands[0]


def find_merged_dir(fold_root: str) -> str:
    merged_dir = os.path.join(fold_root, "merged")
    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(f"Missing merged directory: {merged_dir}")
    has_reg = os.path.isfile(os.path.join(merged_dir, "regularized_state.pt"))
    has_base = os.path.isfile(os.path.join(merged_dir, "pytorch_model.bin"))
    if not (has_reg or has_base):
        raise FileNotFoundError(f"No regularized_state.pt or pytorch_model.bin under {merged_dir}")
    return merged_dir
