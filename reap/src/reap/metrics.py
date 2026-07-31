from typing import Dict

import numpy as np


def safe_np(x):
    return np.asarray(x, dtype=float).reshape(-1)


def pearson_np(a, b):
    a = safe_np(a)
    b = safe_np(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman_np(a, b):
    a = safe_np(a)
    b = safe_np(b)
    if len(a) < 2:
        return 0.0
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return pearson_np(ar, br)


def linreg_calibration(y_true, y_pred):
    y_true = safe_np(y_true)
    y_pred = safe_np(y_pred)
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        return 1.0, 0.0, 0.0
    a, b = np.polyfit(y_true, y_pred, 1)
    r2 = pearson_np(y_true, y_pred) ** 2
    return float(a), float(b), float(r2)


def fit_true_on_pred(y_true, y_pred):
    y_true = safe_np(y_true)
    y_pred = safe_np(y_pred)
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def ndcg_at_k(y_true, y_score, k):
    y_true = safe_np(y_true)
    y_score = safe_np(y_score)
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(int(k), len(y_true)))
    order = np.argsort(-y_score)[:k]
    ideal = np.argsort(-y_true)[:k]
    gains = (2 ** y_true[order] - 1) / np.log2(np.arange(2, k + 2))
    ideal_gains = (2 ** y_true[ideal] - 1) / np.log2(np.arange(2, k + 2))
    denom = float(np.sum(ideal_gains))
    return float(np.sum(gains) / denom) if denom > 0 else 0.0


def enrich_top_frac(y_true, y_score, frac=0.05):
    y_true = safe_np(y_true)
    y_score = safe_np(y_score)
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(round(len(y_true) * frac)))
    top_mean = float(np.mean(y_true[np.argsort(-y_score)[:k]]))
    base_mean = float(np.mean(y_true))
    return float(top_mean / base_mean) if abs(base_mean) > 1e-12 else 0.0


def summarize_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = safe_np(y_true)
    y_pred = safe_np(y_pred)
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(diff ** 2)) / denom if denom > 0 else 0.0
    a, b, calib_r2 = linreg_calibration(y_true, y_pred)
    out = {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "pearson": pearson_np(y_true, y_pred),
        "spearman": spearman_np(y_true, y_pred),
        "calib_slope": a,
        "calib_intercept": b,
        "calib_r2": calib_r2,
        "pred_std": float(np.std(y_pred)),
        "true_std": float(np.std(y_true)),
    }
    out["std_ratio"] = out["pred_std"] / out["true_std"] if out["true_std"] > 1e-12 else 0.0
    out["ndcg@5%"] = ndcg_at_k(y_true, y_pred, max(1, int(round(0.05 * len(y_true)))))
    out["ndcg@10%"] = ndcg_at_k(y_true, y_pred, max(1, int(round(0.10 * len(y_true)))))
    out["enrich@5%"] = enrich_top_frac(y_true, y_pred, 0.05)
    out["enrich@10%"] = enrich_top_frac(y_true, y_pred, 0.10)
    return out
