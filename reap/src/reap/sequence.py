import random
from typing import List


def tokenizer_looks_like_6mer(tok) -> bool:
    vocab = getattr(tok, "get_vocab", lambda: {})()
    if not vocab:
        return False
    sample = ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"]
    return sum(1 for k in sample if k in vocab) >= 3


def seq_to_6mers(seq: str, k: int = 6) -> str:
    seq = "".join(ch if ch in "ACGT" else "N" for ch in str(seq).upper().replace("U", "T"))
    if len(seq) < k:
        return seq
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def rc_seq(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return str(seq).translate(table)[::-1].upper()


def clean_seq(seq: str) -> str:
    return "".join(ch if ch in "ACGT" else "N" for ch in str(seq).upper().replace("U", "T"))


def crop_center(seq: str, length: int) -> str:
    seq = clean_seq(seq)
    if len(seq) <= length:
        return seq
    start = (len(seq) - length) // 2
    return seq[start:start + length]


def crop_with_shift(seq: str, length: int, shift: int) -> str:
    seq = clean_seq(seq)
    if len(seq) <= length:
        return seq
    start = (len(seq) - length) // 2 + shift
    start = max(0, min(start, len(seq) - length))
    return seq[start:start + length]


def crop_sequence_center(seq: str, length: int) -> str:
    return crop_center(seq, length)


class SequenceAugmentation:
    def __init__(self, mask_prob: float = 0.0, rc_prob: float = 0.0, center_jitter: int = 0):
        self.mask_prob = float(mask_prob)
        self.rc_prob = float(rc_prob)
        self.center_jitter = int(center_jitter)

    def mask(self, seq: str) -> str:
        if self.mask_prob <= 0:
            return seq
        chars = list(seq)
        for i, ch in enumerate(chars):
            if ch in "ACGT" and random.random() < self.mask_prob:
                chars[i] = "N"
        return "".join(chars)

    def jitter_center_crop(self, seq: str, length: int) -> str:
        seq = clean_seq(seq)
        if len(seq) <= length:
            out = seq
        else:
            base = (len(seq) - length) // 2
            jitter = random.randint(-self.center_jitter, self.center_jitter) if self.center_jitter > 0 else 0
            start = max(0, min(base + jitter, len(seq) - length))
            out = seq[start:start + length]
        if self.rc_prob > 0 and random.random() < self.rc_prob:
            out = rc_seq(out)
        return self.mask(out)
