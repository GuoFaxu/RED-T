import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import glob
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed
)

from reap.io_utils import (
    batch_iterator,
    dataset_stem,
    ensure_dir,
    load_json_safely as _load_json_safely,
    read_input_any,
)
from reap.metrics import summarize_metrics
from reap.model_io import auto_discover_folds, find_latest_fold_dir, find_merged_dir
from reap.modeling import (
    ModelConfig as InferCfg,
    RegularizedModel,
    infer_mlp_head_from_state_dict as _infer_mlp_head_from_state_dict,
)
from reap.scaling import TargetScaler
from reap.sequence import (
    crop_center,
    crop_with_shift,
    rc_seq,
    seq_to_6mers,
    tokenizer_looks_like_6mer as _tokenizer_looks_like_6mer,
)

VERSION = "2025-11-21"

class Cfg:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    INPUT       = os.path.join(BASE_DIR, "examples", "example_sequences.fasta")
    BASE_MODEL  = os.path.join(BASE_DIR, "models", "pretrained", "plant-dnabert-6mer")
    MODELS_ROOT = os.path.join(BASE_DIR, "models", "finetuned")
    RESULTS_ROOT= os.path.join(BASE_DIR, "results")

    FOLDS = "auto"

    MAX_LEN     = 512
    BATCH_SIZE  = 256
    TTA_WINDOWS = 1
    TTA_SHIFT   = 128
    RC_TTA      = True
    CALIBRATION = "auto"
    AGGREGATE   = "both"
    SEED        = 42
    SUFFIX      = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
L = logging.getLogger("inference-ensemble")

def load_global_scaler(results_root: str) -> Optional[TargetScaler]:
    extras = os.path.join(results_root, "extras")
    cands = sorted(glob.glob(os.path.join(extras, "global_scaler_*.json")), key=os.path.getmtime, reverse=True)
    for p in cands:
        obj = _load_json_safely(p)
        if not obj:
            continue
        prm = obj.get("global_scaler") or obj
        try:
            sc = TargetScaler(**prm)
            L.info("[Scaler] global: %s", os.path.basename(p))
            return sc
        except Exception:
            continue
    return None

def load_fold_scaler(results_root: str, fold_id: int) -> Optional[TargetScaler]:
    extras = os.path.join(results_root, "extras")
    p2 = os.path.join(extras, f"scaler_fold{fold_id}.json")
    for cand in [p2] + sorted(glob.glob(os.path.join(extras, f"*fold{fold_id}*scaler*.json")), key=os.path.getmtime, reverse=True):
        obj = _load_json_safely(cand)
        if not obj:
            continue
        prm = obj.get("scaler") or obj
        try:
            sc = TargetScaler(**prm)
            L.info("[Scaler] fold %d: %s", fold_id, os.path.basename(cand))
            return sc
        except Exception:
            continue
    return None

def load_base_model_and_tokenizer(base_model_dir: str, device):
    tok = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir, trust_remote_code=True, local_files_only=True,
        num_labels=1, problem_type="regression"
    ).to(device).eval()
    return model, tok

def load_regularized_wrapper(base_model_dir: str, merged_dir: str, device):
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir, trust_remote_code=True, local_files_only=True,
        num_labels=1, problem_type="regression"
    )
    wrapper = RegularizedModel(base, InferCfg).to(device)
    pt = os.path.join(merged_dir, "regularized_state.pt")
    sd = torch.load(pt, map_location="cpu")
    wrapper.load_state_dict(sd, strict=True)
    wrapper.eval()
    return wrapper

def load_merged_weights_into_base(model: nn.Module, merged_dir: str, device) -> nn.Module:
    pt = os.path.join(merged_dir, "pytorch_model.bin")
    if not os.path.isfile(pt):
        raise FileNotFoundError(f"Missing weights: {pt}")
    sd = torch.load(pt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    inferred = _infer_mlp_head_from_state_dict(sd, init_drop=0.20, mid_drop=0.10)
    if inferred is not None:
        try:
            model.classifier = inferred.to(device)
            L.info("Classifier rebuilt from checkpoint.")
        except Exception as e:
            L.warning("Classifier rebuild failed: %s", e)
    target_keys = set(model.state_dict().keys())
    new_sd = {}
    for k, v in sd.items():
        kk = k
        for p in ("base_model.model.", "base_model.", "model."):
            if kk.startswith(p):
                kk = kk[len(p):]
        if kk in target_keys:
            new_sd[kk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        L.warning("Missing keys: %d", len(missing))
    if unexpected:
        L.warning("Unexpected keys: %d", len(unexpected))
    return model

def find_latest_calibration_linear(results_root: str, fold_id: int) -> Optional[Tuple[float, float]]:
    metrics_dir = os.path.join(results_root, "metrics")
    if not os.path.isdir(metrics_dir):
        return None
    patt1 = os.path.join(metrics_dir, f"fold{fold_id}_calibration.json")
    patt2 = os.path.join(metrics_dir, f"*fold{fold_id}*calibration*.json")
    cands = sorted(set(glob.glob(patt1) + glob.glob(patt2)), key=lambda p: os.path.getmtime(p), reverse=True)
    for p in cands:
        try:
            with open(p, "r") as f:
                obj = json.load(f)
            cal = obj.get("calibration", obj)
            a = float(cal.get("a")); b = float(cal.get("b"))
            return a, b
        except Exception:
            continue
    return None

def try_apply_isotonic(results_root: str, fold_id: int, pred: np.ndarray) -> Tuple[np.ndarray, bool]:
    grid_path = os.path.join(results_root, "metrics", f"fold{fold_id}_isotonic_grid.csv")
    if os.path.exists(grid_path):
        df = pd.read_csv(grid_path)
        x, y = df["pred_grid"].values, df["mapped_true"].values
        pred = np.interp(pred, x, y, left=y[0], right=y[-1])
        L.info("[Fold %d] isotonic calibration applied.", fold_id)
        return pred, True
    return pred, False

def apply_linear_calibration(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    if abs(a) < 1e-8:
        L.warning("Linear calibration skipped.")
        return pred
    return a * pred + b

@torch.no_grad()
def predict_scores(model: nn.Module, tok: AutoTokenizer, seqs: List[str],
                   max_len: int, batch_size: int, device,
                   is_kmer_tok: bool, kmer_k: int,
                   tta_windows: int = 0, tta_shift: int = 128,
                   return_logvar: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    N = len(seqs)
    if tta_windows and tta_windows > 0:
        shifts = [(w - (tta_windows // 2)) * tta_shift for w in range(tta_windows)]
    else:
        shifts = [0]

    preds_all = []
    logvars_all = [] if return_logvar else None

    for sh in shifts:
        cropped = [crop_center(s, max_len) if sh == 0 else crop_with_shift(s, max_len, sh) for s in seqs]

        if is_kmer_tok:
            max_len_for_tok = (max_len - kmer_k + 1) + 2
            processed_seqs = [seq_to_6mers(s, k=kmer_k) for s in cropped]
            enc = tok(processed_seqs, padding="max_length", truncation=True, max_length=max_len_for_tok)
        else:
            enc = tok(cropped, padding="max_length", truncation=True, max_length=max_len)

        ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

        preds = np.zeros(N, dtype=np.float32)
        logvars = np.zeros(N, dtype=np.float32) if return_logvar else None

        for _, s, e in batch_iterator(list(range(N)), batch_size):
            batch_ids = torch.tensor(ds["input_ids"][s:e], dtype=torch.long, device=device)
            batch_mask= torch.tensor(ds["attention_mask"][s:e], dtype=torch.long, device=device)
            out = model(input_ids=batch_ids, attention_mask=batch_mask)

            if isinstance(out, dict):
                logits = out["logits"]
                if return_logvar and ("pred_logvar" in out):
                    lv = out["pred_logvar"].detach().float().view(-1).cpu().numpy()
                    logvars[s:e] = lv
            else:
                logits = out.logits

            preds[s:e] = logits.detach().float().view(-1).cpu().numpy()

        preds_all.append(preds)
        if return_logvar:
            if logvars is None:
                logvars = np.zeros(N, dtype=np.float32)
            logvars_all.append(logvars)

    preds_avg = np.mean(np.stack(preds_all, axis=0), axis=0)
    if return_logvar:
        lv_avg = np.mean(np.stack(logvars_all, axis=0), axis=0)
        return preds_avg, lv_avg
    else:
        return preds_avg, None

def run_inference(
    input_path: str,
    base_model_dir: str,
    models_root: str,
    results_root: str,
    folds: List[int],
    max_len: int = 512,
    batch_size: int = 256,
    tta_windows: int = 3,
    tta_shift: int = 128,
    rc_tta: bool = True,
    calibration: str = "auto",
    aggregate: str = "both",
    seed: int = 42,
    suffix: str = ""
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L.info("Device: %s | Version: %s", device, VERSION)

    seqs, labels, ids, input_type = read_input_any(input_path)

    ts = time.strftime("%Y%m%d_%H%M%S")
    data_name = dataset_stem(input_path)
    job_root  = ensure_dir(os.path.join(results_root, data_name))
    preds_dir   = ensure_dir(os.path.join(job_root, "preds"))
    metrics_dir = ensure_dir(os.path.join(job_root, "metrics"))
    runs_dir    = ensure_dir(os.path.join(job_root, "runs"))
    L.info("[Outputs] %s", job_root)

    _, tokenizer = load_base_model_and_tokenizer(base_model_dir, device)

    IS_KMER_TOK = _tokenizer_looks_like_6mer(tokenizer)
    KMER_K = 6
    if IS_KMER_TOK:
        L.info("Detected k-mer tokenizer. Will apply k-mer preprocessing.")

    global_scaler = load_global_scaler(results_root)
    if global_scaler is None:
        L.warning("Global scaler not found.")

    fold_pred_list = []
    fold_sigma_list = []
    for k in folds:
        fold_root = find_latest_fold_dir(models_root, k)
        merged_dir = find_merged_dir(fold_root)
        reg_pt = os.path.join(merged_dir, "regularized_state.pt")

        if os.path.exists(reg_pt):
            model = load_regularized_wrapper(base_model_dir, merged_dir, device)
            want_logvar = True
            L.info("[Fold %d] loaded RegularizedModel.", k)
        else:
            base = AutoModelForSequenceClassification.from_pretrained(
                base_model_dir, trust_remote_code=True, local_files_only=True,
                num_labels=1, problem_type="regression"
            ).to(device).eval()
            model = load_merged_weights_into_base(base, merged_dir, device)
            want_logvar = False
            L.warning("[Fold %d] regularized_state.pt not found.", k)

        pred_fwd, lv_fwd = predict_scores(model, tokenizer, seqs, max_len, batch_size, device,
                                          IS_KMER_TOK, KMER_K,
                                          tta_windows, tta_shift, return_logvar=want_logvar)
        if rc_tta:
            seqs_rc = [rc_seq(s) for s in seqs]
            pred_rc, lv_rc = predict_scores(model, tokenizer, seqs_rc, max_len, batch_size, device,
                                            IS_KMER_TOK, KMER_K,
                                            tta_windows, tta_shift, return_logvar=want_logvar)
            pred_scaled = 0.5 * (pred_fwd + pred_rc)
            if want_logvar and (lv_fwd is not None) and (lv_rc is not None):
                lv_avg = 0.5 * (lv_fwd + lv_rc)
            else:
                lv_avg = None
        else:
            pred_scaled = pred_fwd
            lv_avg = lv_fwd

        scaler_used = None
        if global_scaler is not None:
            pred = global_scaler.inverse(pred_scaled); scaler_used = "global"
        else:
            fold_scaler = load_fold_scaler(results_root, k)
            if fold_scaler is not None:
                pred = fold_scaler.inverse(pred_scaled); scaler_used = f"fold{k}"
            else:
                pred = pred_scaled; scaler_used = "none"
        L.info("[Fold %d] scaler: %s", k, scaler_used)

        if calibration == "auto":
            pred, ok_iso = try_apply_isotonic(results_root, k, pred)
            if not ok_iso:
                ab = find_latest_calibration_linear(results_root, k)
                if ab is not None:
                    a, b = ab
                    pred = apply_linear_calibration(pred, a, b)
                    L.info("[Fold %d] linear calibration applied.", k)
                else:
                    L.info("[Fold %d] calibration not found.", k)

        fold_pred_list.append(pred)

        sigma = None
        if want_logvar and (lv_avg is not None):
            sigma = np.exp(0.5 * lv_avg).astype(np.float32)
            fold_sigma_list.append(sigma)

        df_single = {"id": ids, "y_pred": pred}
        if sigma is not None:
            df_single["sigma"] = sigma
        if labels is not None:
            df_single["y_true"] = labels
        
        cols_order = ["id", "y_true", "y_pred", "sigma"]
        final_cols = [c for c in cols_order if c in df_single]
        df_single_pd = pd.DataFrame(df_single)[final_cols]

        out_single = os.path.join(preds_dir, f"single_fold{k}_preds_{ts}{('_'+suffix) if suffix else ''}.csv")
        df_single_pd.to_csv(out_single, index=False)
        L.info("[Fold %d] %s", k, out_single)

    mat = np.stack(fold_pred_list, axis=0)
    preds_std = mat.std(axis=0).astype(np.float32)
    sigma_mean = None
    if fold_sigma_list:
        sigma_mean = np.mean(np.stack(fold_sigma_list, axis=0), axis=0).astype(np.float32)

    outs = {}
    if aggregate in ("mean", "both"):
        outs["mean"] = mat.mean(axis=0)
    if aggregate in ("median", "both"):
        outs["median"] = np.median(mat, axis=0)

    for agg, arr in outs.items():
        cols = {"id": ids, "y_pred_ensemble_"+agg: arr, "pred_std_across_folds": preds_std}
        if sigma_mean is not None:
            cols["sigma_mean"] = sigma_mean
        if labels is not None:
            cols["y_true"] = labels

        cols_order_agg = ["id", "y_true", "y_pred_ensemble_"+agg, "sigma_mean", "pred_std_across_folds"]
        final_cols_agg = [c for c in cols_order_agg if c in cols]
        df_agg_pd = pd.DataFrame(cols)[final_cols_agg]
        
        out_path = os.path.join(preds_dir, f"test_preds_ensemble_{agg}_{ts}{('_'+suffix) if suffix else ''}.csv")
        df_agg_pd.to_csv(out_path, index=False)
        L.info("[Ensemble %s] %s", agg, out_path)

        if labels is not None:
            metrics = summarize_metrics(labels, arr)
            with open(os.path.join(metrics_dir, f"metrics_ensemble_{agg}_{ts}{('_'+suffix) if suffix else ''}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            L.info("[Metrics %s] %s", agg, metrics)

    manifest = {
        "timestamp": ts,
        "args": {
            "input": input_path,
            "input_type": input_type,
            "base_model_dir": base_model_dir,
            "models_root": models_root,
            "folds": folds,
            "max_len": max_len,
            "batch_size": batch_size,
            "tta_windows": tta_windows,
            "tta_shift": tta_shift,
            "rc_tta": rc_tta,
            "calibration": calibration,
            "aggregate": aggregate,
            "seed": seed,
            "suffix": suffix
        },
        "run": {
            "version": VERSION,
            "job_root": job_root,
            "data_name": data_name
        },
        "outputs": {
            "single_folds": [os.path.join(preds_dir, f"single_fold{k}_preds_{ts}{('_'+suffix) if suffix else ''}.csv") for k in folds],
            "ensembles": {agg: os.path.join(preds_dir, f"test_preds_ensemble_{agg}_{ts}{('_'+suffix) if suffix else ''}.csv") for agg in outs.keys()}
        }
    }
    with open(os.path.join(runs_dir, f"inference_run_{ts}{('_'+suffix) if suffix else ''}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def resolve_input_path(arg_input: Optional[str], arg_input_csv: Optional[str]) -> str:
    cand = arg_input or arg_input_csv or Cfg.INPUT
    cand = os.path.expanduser(str(cand))
    abs1 = os.path.abspath(cand)
    if os.path.exists(abs1):
        return abs1
    abs2 = os.path.abspath(os.path.join(Cfg.BASE_DIR, cand))
    if os.path.exists(abs2):
        return abs2
    raise FileNotFoundError(f"Input file not found: {abs1} or {abs2}")

def parse_args_with_defaults():
    ap = argparse.ArgumentParser(description="REAP inference")
    ap.add_argument("--input",        default=None)
    ap.add_argument("--input_csv",    default=None)

    ap.add_argument("--base_model",   default=Cfg.BASE_MODEL)
    ap.add_argument("--models_root",  default=Cfg.MODELS_ROOT)
    ap.add_argument("--results_root", default=Cfg.RESULTS_ROOT)
    ap.add_argument("--folds",        default=Cfg.FOLDS)
    ap.add_argument("--max_len",      type=int, default=Cfg.MAX_LEN)
    ap.add_argument("--batch_size",   type=int, default=Cfg.BATCH_SIZE)
    ap.add_argument("--tta_windows",  type=int, default=Cfg.TTA_WINDOWS)
    ap.add_argument("--tta_shift",    type=int, default=Cfg.TTA_SHIFT)
    ap.add_argument("--rc_tta",       type=lambda x: str(x).lower() in ["1","true","yes","y"], default=Cfg.RC_TTA)
    ap.add_argument("--seed",         type=int, default=Cfg.SEED)
    ap.add_argument("--calibration",  choices=["none","auto"], default=Cfg.CALIBRATION)
    ap.add_argument("--aggregate",    choices=["mean","median","both"], default=Cfg.AGGREGATE)
    ap.add_argument("--suffix",       default=Cfg.SUFFIX)
    args = ap.parse_args()

    if isinstance(args.folds, str) and args.folds.strip().lower() == "auto":
        folds = auto_discover_folds(args.models_root)
        L.info("[Folds] %s", folds)
    else:
        folds = [int(x) for x in str(args.folds).split(",") if x.strip()]

    input_path = resolve_input_path(args.input, args.input_csv)
    L.info("[Input] %s", input_path)

    args.input = input_path
    return args, folds

def main():
    args, folds = parse_args_with_defaults()
    run_inference(
        input_path=args.input,
        base_model_dir=args.base_model,
        models_root=args.models_root,
        results_root=args.results_root,
        folds=folds,
        max_len=args.max_len,
        batch_size=args.batch_size,
        tta_windows=args.tta_windows,
        tta_shift=args.tta_shift,
        rc_tta=args.rc_tta,
        calibration=args.calibration,
        aggregate=args.aggregate,
        seed=args.seed,
        suffix=args.suffix
    )

if __name__ == "__main__":
    main()
