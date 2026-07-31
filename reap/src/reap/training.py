import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging, random, shutil, datetime, json, math, copy, warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
)
from transformers.optimization import get_scheduler

from reap.hf_utils import check_model_weights_or_repo, check_tokenizer_dir, load_tokenizer
from reap.io_utils import save_df_csv, save_json, to_numpy_1d as _to_numpy_1d
from reap.metrics import fit_true_on_pred, summarize_metrics
from reap.modeling import RegularizedModel
from reap.scaling import TargetScaler
from reap.sequence import (
    SequenceAugmentation,
    crop_sequence_center,
    rc_seq,
    seq_to_6mers,
    tokenizer_looks_like_6mer as _tokenizer_looks_like_6mer,
)

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

HAVE_PEFT = True
try:
    from peft import LoraConfig, get_peft_model
    try:
        from peft import PeftModel
    except Exception:
        PeftModel = None
except Exception:
    HAVE_PEFT = False
    PeftModel = None

try:
    from sklearn.model_selection import GroupKFold, StratifiedKFold
except Exception as e:
    raise RuntimeError("scikit-learn is required.") from e

HAVE_ISOTONIC = True
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    HAVE_ISOTONIC = False


class Cfg:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    ALL_CSV = "all_data.csv"
    TEST_CSV = "test_data.csv"

    PRETRAINED = os.path.join(BASE_DIR, "models", "pretrained", "plant-dnabert-6mer")
    PRETRAINED_WEIGHTS = ""

    OUT_ROOT = os.path.join(BASE_DIR, "models")
    RESULTS_ROOT = os.path.join(BASE_DIR, "results")

    EPOCHS = 50
    BATCH = 8
    GRAD_ACCUM = 2
    MAX_LEN = 512
    LR_LORA = 1.0e-5
    LR_HEAD = 1.0e-4
    WARMUP = 0.10
    WEIGHT_DECAY = 0.02
    FP16 = False
    SEED = 42

    DROPOUT = 0.15
    GRAD_CLIP = 1.0
    LOSS = "huber"
    HUBER_DELTA = 0.35
    L1_ON_HEAD = 0.0
    MLP_HIDDEN1 = 256
    MLP_HIDDEN2 = 64

    LOGVAR_CLAMP_MIN = -2.0
    LOGVAR_CLAMP_MAX = 1.5
    LOGVAR_REG_LAMBDA = 0.01

    RANK_LOSS_LAMBDA = 0.0
    RANK_PAIR_SAMPLES = 1024
    RANK_SKIP_EQUAL = True
    CORR_LOSS_LAMBDA = 0.0
    STD_MATCH_LAMBDA = 0.0
    RC_EQUIVARIANT = True
    RC_CONSIST_LAMBDA = 0.01
    CLS_TOPK_FRAC = 0.15
    CLS_LOSS_LAMBDA = 0.0
    QUANTILE_TAUS = (0.5, 0.8)
    QUANTILE_LAMBDA = 0.0

    TARGET_TRANSFORM = "zscore"
    USE_GLOBAL_SCALER = True

    USE_AUG = True
    MASK_PROB = 0.0
    RC_PROB = 0.0
    CENTER_JITTER = 0

    EXTREME_OVERSAMPLE = False
    EXTREME_Q = 0.15
    EXTREME_MULT = 2

    PATIENCE = 10
    METRIC = "pearson"

    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.10
    LORA_TARGETS = "query,key,value,dense"

    UNFREEZE_LAST_N_BLOCKS = 2

    STAGE2_ENABLE = False
    STAGE2_UNFREEZE_LAST_N_BLOCKS = 2
    STAGE2_EPOCHS = 6
    STAGE2_LR_LORA = 8e-6
    STAGE2_LR_HEAD = 2e-5
    STAGE2_PATIENCE = 2
    STAGE2_BATCH = 32
    STAGE2_EVAL_BATCH = 32
    STAGE2_GRAD_ACCUM = 2
    STAGE2_FP16 = True
    STAGE2_RANK_PAIR_SAMPLES = 256
    STAGE2_DISABLE_RC = False
    STAGE2_RC_NO_GRAD = False

    POOL_WINS = (8, 16, 32, 64)
    LAYER_MIX_K = 4

    CALIBRATE_ON_VAL = True
    CALIBRATION = "isotonic"

    CALIB_SHRINK = 0.7
    CALIB_STD_SHRINK = True


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhancer-5fold")


class DetailedEarlyStopping(EarlyStoppingCallback):
    def __init__(self, patience=8, threshold=0.0, metric="pearson"):
        super().__init__(early_stopping_patience=patience, early_stopping_threshold=threshold)
        self.metric_name = f"eval_{metric}" if not metric.startswith("eval_") else metric

class RegressionTrainer(Trainer):
    def __init__(self, loss_type="nll", huber_delta=0.5, l1_lambda=0.0,
                 head_keys=None, rank_lambda: float = 0.0,
                 rank_pair_samples: int = 1024, rank_skip_equal: bool = True,
                 corr_loss_lambda: float = 0.0, std_match_lambda: float = 0.0,
                 cls_loss_lambda: float = 0.0, rc_consist_lambda: float = 0.0,
                 quantile_taus: Tuple[float,...] = (0.5,0.8), quantile_lambda: float = 0.0,
                 logvar_reg_lambda: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.l1_lambda = l1_lambda
        self.head_keys = head_keys or ["classifier","regress","score","lm_head","head_trunk","mu_head","logvar_head","cls_head","quant_head","layer_mix"]
        self.rank_lambda = float(rank_lambda)
        self.rank_pair_samples = int(rank_pair_samples)
        self.rank_skip_equal = bool(rank_skip_equal)
        self.corr_loss_lambda = float(corr_loss_lambda)
        self.std_match_lambda = float(std_match_lambda)
        self.cls_loss_lambda = float(cls_loss_lambda)
        self.rc_consist_lambda = float(rc_consist_lambda)
        self.quantile_taus = quantile_taus
        self.quantile_lambda = float(quantile_lambda)
        self.logvar_reg_lambda = float(logvar_reg_lambda)
        self._last = {}

    @staticmethod
    def _to_float(x):
        if torch.is_tensor(x): return x.detach().mean().cpu().item()
        return float(x)

    def _rank_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.rank_lambda <= 0.0: return torch.tensor(0.0, device=logits.device)
        B = logits.numel()
        if B < 2: return torch.tensor(0.0, device=logits.device)
        M = min(self.rank_pair_samples, max(1, B * (B - 1) // 2))
        idx_i = torch.randint(0, B, (M,), device=logits.device)
        idx_j = torch.randint(0, B, (M,), device=logits.device)
        mask = (idx_i != idx_j)
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        if idx_i.numel() == 0: return torch.tensor(0.0, device=logits.device)
        li, lj = labels[idx_i], labels[idx_j]
        si = torch.sign(li - lj)
        if self.rank_skip_equal:
            nz = (si != 0)
            if nz.sum() == 0: return torch.tensor(0.0, device=logits.device)
            idx_i, idx_j = idx_i[nz], idx_j[nz]; si, li, lj = si[nz], li[nz], lj[nz]
        yi, yj = logits[idx_i], logits[idx_j]
        w = torch.abs(li - lj); w = w / (w.mean().clamp_min(1e-8))
        pair = F.softplus(-si * (yi - yj))
        return (w * pair).mean() * self.rank_lambda

    def _corr_loss(self, preds, target):
        px = preds - preds.mean(); py = target - target.mean()
        vx = torch.sqrt((px ** 2).mean() + 1e-8); vy = torch.sqrt((py ** 2).mean() + 1e-8)
        corr = (px * py).mean() / (vx * vy)
        return 1.0 - corr

    def _std_match_loss(self, preds, target):
        sp = torch.sqrt(preds.var(unbiased=False) + 1e-8)
        st = torch.sqrt(target.var(unbiased=False) + 1e-8)
        return ((sp / st) - 1.0) ** 2

    def _pinball(self, pred, target, tau):
        e = target - pred
        return torch.where(e>=0, tau*e, (tau-1)*e).mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        cls_labels = inputs.pop("cls_labels", None)

        outputs = model(**inputs)
        mu = outputs["logits"].view(-1)
        labels = labels.view(-1)

        loss_logvar_reg = torch.tensor(0.0, device=mu.device)
        if self.loss_type == "nll" and outputs.get("pred_logvar", None) is not None:
            logv = outputs["pred_logvar"].view(-1)
            base_loss = 0.5 * ( (labels - mu)**2 * torch.exp(-logv) + logv )
            base_loss = base_loss.mean()
            if self.logvar_reg_lambda > 0:
                loss_logvar_reg = self.logvar_reg_lambda * (logv**2).mean()
                base_loss = base_loss + loss_logvar_reg
        elif self.loss_type == "huber":
            base_loss = F.huber_loss(mu, labels, delta=self.huber_delta, reduction="mean")
        else:
            base_loss = F.mse_loss(mu, labels, reduction="mean")

        if self.l1_lambda > 0:
            l1 = 0.0
            for n,p in model.named_parameters():
                if p.requires_grad and any(k in n for k in self.head_keys):
                    l1 += p.abs().sum()
            base_loss = base_loss + self.l1_lambda * l1

        rank_loss = self._rank_loss(mu, labels)
        corr_loss = self._corr_loss(mu, labels) * self.corr_loss_lambda
        stdm_loss = self._std_match_loss(mu, labels) * self.std_match_lambda

        cls_loss = torch.tensor(0.0, device=mu.device)
        if self.cls_loss_lambda > 0 and cls_labels is not None and ("cls_logits" in outputs):
            logits_c = outputs["cls_logits"].view(-1)
            cls_loss = F.binary_cross_entropy_with_logits(logits_c, cls_labels.float()) * self.cls_loss_lambda

        q_loss = torch.tensor(0.0, device=mu.device)
        aux = outputs.get("aux", {})
        if self.quantile_lambda > 0 and ("q_preds" in aux):
            qpred = aux["q_preds"]
            for i, tau in enumerate(self.quantile_taus):
                q_loss = q_loss + self._pinball(qpred[:,i].view(-1), labels, tau)
            q_loss = q_loss * self.quantile_lambda

        rc_loss = torch.tensor(0.0, device=mu.device)
        if self.rc_consist_lambda > 0 and ("rc_consistency" in aux):
            rc = aux["rc_consistency"]
            if torch.is_tensor(rc) and rc.ndim > 0: rc = rc.mean()
            rc_loss = rc * self.rc_consist_lambda

        loss = base_loss + rank_loss + corr_loss + stdm_loss + cls_loss + q_loss + rc_loss

        self._last = {"loss_base": self._to_float(base_loss),
                      "loss_rank": self._to_float(rank_loss),
                      "loss_corr": self._to_float(corr_loss),
                      "loss_stdm": self._to_float(stdm_loss),
                      "loss_cls":  self._to_float(cls_loss),
                      "loss_q":    self._to_float(q_loss),
                      "loss_rc":   self._to_float(rc_loss),
                      "loss_logvar": self._to_float(loss_logvar_reg)}
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        logs = {**logs, **self._last}
        super().log(logs)


def select_lora_targets_for_bert(regularized_model: nn.Module, include_tokens: List[str]) -> List[str]:
    include_tokens = [t.strip().lower() for t in include_tokens if t.strip()]
    bad_tokens = ["classifier", "head_trunk", "mu_head", "logvar_head", "cls_head", "quant_head", "attn_score", "layer_mix"]
    names: List[str] = []
    for name, module in regularized_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        low = name.lower()
        if not low.startswith("base."):
            continue
        if any(bt in low for bt in bad_tokens):
            continue
        if any(tok in low for tok in include_tokens):
            names.append(name)
    names = sorted(set(names))
    return names


def _offset_log_history(logs: List[Dict], epoch_offset: float, step_offset: int) -> List[Dict]:
    out = []
    for e in logs or []:
        d = dict(e)
        if "epoch" in d:
            try: d["epoch"] = float(d["epoch"]) + float(epoch_offset)
            except Exception: pass
        if "step" in d:
            try: d["step"] = int(d["step"]) + int(step_offset)
            except Exception: pass
        out.append(d)
    return out

def _last_value_in_logs(logs: List[Dict], key: str, default: float = 0.0) -> float:
    v = default
    for e in logs or []:
        if key in e:
            try:
                vv = float(e[key])
                if vv > v: v = vv
            except Exception:
                pass
    return v

def _unwrap_base_like(m: nn.Module) -> nn.Module:
    try:
        if HAVE_PEFT and PeftModel is not None and isinstance(m, PeftModel):
            return getattr(m, "model", m)
    except Exception:
        pass
    return m

def unfreeze_last_n_blocks(model: nn.Module, n_blocks: int):
    if n_blocks <= 0:
        return 0
    try:
        mm = _unwrap_base_like(model)
        host = getattr(mm, "base", mm)
        candidates = []

        for path in [
            "bert.encoder.layer",
            "roberta.encoder.layer",
            "electra.encoder.layer",
            "deberta.encoder.layer",
            "deberta.encoder.layers",
            "transformer.h",
        ]:
            node = host
            ok = True
            for attr in path.split("."):
                if hasattr(node, attr):
                    node = getattr(node, attr)
                else:
                    ok = False; break
            if ok and isinstance(node, (nn.ModuleList, list)) and len(node) > 0:
                candidates.append((path, node))

        if not candidates:
            def find_lists(root, root_name=""):
                hits = []
                for name, mod in root.named_modules():
                    full = f"{root_name}.{name}" if root_name else name
                    if any(key in full for key in (".encoder.layer", ".layers", ".block", ".h")):
                        if isinstance(mod, (nn.ModuleList, list)) and len(mod) > 0:
                            hits.append((full, mod))
                return hits
            candidates = find_lists(host, "base")

        if not candidates:
            logger.warning("Backbone blocks not found.")
            return 0

        path, layers = candidates[0]
        total = len(layers)
        n = max(0, min(n_blocks, total))
        for i in range(total - n, total):
            for p in layers[i].parameters():
                p.requires_grad = True
        logger.info("Unfroze last %d backbone blocks.", n)
        return n
    except Exception as e:
        logger.warning("Backbone unfreeze failed: %s", e)
        return 0


def export_merged_if_possible(trainer: Trainer, tokenizer: AutoTokenizer, out_dir: str, fold_id: int) -> bool:
    merged_dir = os.path.join(out_dir, "merged"); os.makedirs(merged_dir, exist_ok=True)
    m = trainer.model
    saved_any = False

    if HAVE_PEFT and PeftModel is not None and isinstance(m, PeftModel):
        try:
            m.eval()
            merged = m.merge_and_unload()
            base_only = getattr(merged, "base", None)
            if base_only is not None and hasattr(base_only, "save_pretrained"):
                base_only.save_pretrained(merged_dir, safe_serialization=False)
                tokenizer.save_pretrained(merged_dir)
                saved_any = True
            torch.save(merged.state_dict(), os.path.join(merged_dir, "regularized_state.pt"))
            save_json({"fold": fold_id, "merged": True, "method": "peft.merge_and_unload",
                       "state": "regularized_state.pt"},
                      os.path.join(merged_dir, "merge_manifest.json"))
            logger.info("[Fold %d] merged weights exported.", fold_id)
            return True
        except Exception as e:
            logger.warning("[Fold %d] merge_and_unload failed: %s", fold_id, e)

    try:
        host = _unwrap_base_like(m)
        base_only = getattr(host, "base", None)
        if base_only is not None and hasattr(base_only, "save_pretrained"):
            base_only.save_pretrained(merged_dir, safe_serialization=False)
            tokenizer.save_pretrained(merged_dir)
            saved_any = True
        torch.save(host.state_dict(), os.path.join(merged_dir, "regularized_state.pt"))
        save_json({"fold": fold_id, "merged": False, "method": "direct_save",
                   "state": "regularized_state.pt"},
                  os.path.join(merged_dir, "merge_manifest.json"))
        logger.info("[Fold %d] merged weights exported.", fold_id)
        return True
    except Exception as e:
        logger.warning("[Fold %d] merged export failed: %s", fold_id, e)

    return saved_any


def build_trainer(model, tokenizer, tokenized, cfg, scaler, head_keys, optimizer, lr_scheduler,
                  patience, metric_name):
    greater = metric_name not in ["loss","eval_rmse","eval_mae","eval_mse"]
    eval_bs = getattr(cfg, "EVAL_BATCH", None) or (cfg.BATCH * 2)
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.OUT_ROOT, "checkpoints"),
        per_device_train_batch_size=cfg.BATCH,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LR_LORA, num_train_epochs=cfg.EPOCHS,
        warmup_ratio=cfg.WARMUP, weight_decay=cfg.WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        metric_for_best_model=metric_name, greater_is_better=greater,
        load_best_model_at_end=True,
        logging_dir=os.path.join(cfg.RESULTS_ROOT, "logs"), logging_steps=50, report_to="none",
        fp16=cfg.FP16, max_grad_norm=cfg.GRAD_CLIP,
        dataloader_num_workers=4, dataloader_pin_memory=True, dataloader_persistent_workers=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    early_cb = DetailedEarlyStopping(patience=patience, threshold=0.0, metric=cfg.METRIC)

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            preds = eval_pred.predictions; labels = eval_pred.label_ids
        else:
            preds, labels = eval_pred
        preds = _to_numpy_1d(preds); labels = _to_numpy_1d(labels)
        pred = scaler.inverse(preds); true = scaler.inverse(labels)
        return summarize_metrics(true, pred)

    trainer = RegressionTrainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
        loss_type=cfg.LOSS, huber_delta=cfg.HUBER_DELTA,
        l1_lambda=cfg.L1_ON_HEAD, head_keys=head_keys,
        rank_lambda=cfg.RANK_LOSS_LAMBDA, rank_pair_samples=cfg.RANK_PAIR_SAMPLES,
        rank_skip_equal=cfg.RANK_SKIP_EQUAL,
        corr_loss_lambda=cfg.CORR_LOSS_LAMBDA, std_match_lambda=cfg.STD_MATCH_LAMBDA,
        cls_loss_lambda=cfg.CLS_LOSS_LAMBDA, rc_consist_lambda=cfg.RC_CONSIST_LAMBDA,
        quantile_taus=cfg.QUANTILE_TAUS, quantile_lambda=cfg.QUANTILE_LAMBDA,
        logvar_reg_lambda=cfg.LOGVAR_REG_LAMBDA,
        callbacks=[early_cb]
    )
    return trainer


def train_one_fold(cfg: Cfg, fold_id: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame, global_ts: str,
                   global_scaler_params: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    set_seed(cfg.SEED + fold_id)
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    assert check_tokenizer_dir(cfg.PRETRAINED)
    weights_src = cfg.PRETRAINED_WEIGHTS.strip() or cfg.PRETRAINED
    assert check_model_weights_or_repo(weights_src)

    out_dir     = os.path.join(cfg.OUT_ROOT, f"finetuned/fold{fold_id}_{global_ts}")
    ckpt_dir    = os.path.join(cfg.OUT_ROOT, "checkpoints")
    results_dir = os.path.join(cfg.RESULTS_ROOT, "metrics")
    preds_dir   = os.path.join(cfg.RESULTS_ROOT, "preds")
    logs_dir    = os.path.join(cfg.RESULTS_ROOT, "logs_csv")
    extras_dir  = os.path.join(cfg.RESULTS_ROOT, "extras")
    for d in [out_dir, ckpt_dir, results_dir, preds_dir, logs_dir, extras_dir]:
        os.makedirs(d, exist_ok=True)

    train_csv = os.path.join(cfg.DATA_DIR, f"_tmp_train_fold{fold_id}.csv")
    val_csv   = os.path.join(cfg.DATA_DIR, f"_tmp_val_fold{fold_id}.csv")
    test_csv  = os.path.join(cfg.DATA_DIR, f"_tmp_test_fold{fold_id}.csv")
    train_df.to_csv(train_csv, index=False); val_df.to_csv(val_csv, index=False); test_df.to_csv(test_csv, index=False)

    dataset = load_dataset("csv", data_files={"train":train_csv, "validation":val_csv, "test":test_csv})

    tokenizer = load_tokenizer(cfg.PRETRAINED)
    is_kmer_tok = _tokenizer_looks_like_6mer(tokenizer)
    kmer_k = 6
    if is_kmer_tok:
        logger.info("[Fold %d] 6-mer tokenizer detected.", fold_id)
    else:
        logger.info("[Fold %d] non-6-mer tokenizer detected.", fold_id)

    aug = SequenceAugmentation(cfg.MASK_PROB, cfg.RC_PROB, cfg.CENTER_JITTER) if cfg.USE_AUG else None

    scaler = TargetScaler(cfg.TARGET_TRANSFORM)
    if global_scaler_params is not None:
        scaler.set_params(global_scaler_params)
        logger.info("[Fold %d] using global scaler.", fold_id)
    else:
        y_train_all = np.array(dataset["train"]["expression"], dtype=float)
        scaler.fit(y_train_all)
        logger.info(f"[Fold {fold_id}] Target Scaling(LOCAL): {cfg.TARGET_TRANSFORM}")

    topk_thr = float(train_df["expression"].quantile(1 - cfg.CLS_TOPK_FRAC)) if cfg.CLS_LOSS_LAMBDA > 0 else None

    def preprocess(batch, split="train"):
        raw = batch["sequence"]
        if split == "train" and aug and cfg.CENTER_JITTER > 0:
            seqs = [aug.jitter_center_crop(s.upper().replace('U','T'), cfg.MAX_LEN) for s in raw]
        else:
            seqs = [crop_sequence_center(s, cfg.MAX_LEN) for s in raw]

        if is_kmer_tok:
            proc = [seq_to_6mers(s, k=kmer_k) for s in seqs]
            enc = tokenizer(proc, padding="max_length", truncation=True,
                            max_length=(cfg.MAX_LEN - kmer_k + 1 + 2))
        else:
            enc = tokenizer(seqs, padding="max_length", truncation=True, max_length=cfg.MAX_LEN)

        labs = np.asarray(batch["expression"], dtype=float)
        enc["labels"] = scaler.transform(labs).tolist()

        if cfg.CLS_LOSS_LAMBDA > 0 and topk_thr is not None:
            enc["cls_labels"] = (labs >= topk_thr).astype(np.int64).tolist()

        if cfg.RC_EQUIVARIANT and split == "train":
            rc_raw = [rc_seq(s) for s in seqs]
            if is_kmer_tok:
                rc_proc = [seq_to_6mers(s, k=kmer_k) for s in rc_raw]
                enc_rc = tokenizer(rc_proc, padding="max_length", truncation=True,
                                   max_length=(cfg.MAX_LEN - kmer_k + 1 + 2))
            else:
                enc_rc = tokenizer(rc_raw, padding="max_length", truncation=True, max_length=cfg.MAX_LEN)
            enc["input_ids_rc"] = enc_rc["input_ids"]
            enc["attention_mask_rc"] = enc_rc["attention_mask"]
        return enc

    tokenized = {}
    for split in ["train","validation","test"]:
        remove_cols = [c for c in ["sequence","expression","gene_id","chromosome"] if c in dataset[split].column_names]
        tokenized[split] = dataset[split].map(lambda e, s=split: preprocess(e, s), batched=True,
                                              remove_columns=remove_cols, num_proc=1)

    base_src = weights_src
    base = AutoModelForSequenceClassification.from_pretrained(
        base_src, num_labels=1, problem_type="regression", trust_remote_code=True
    )
    try:
        if base.config.vocab_size != len(tokenizer):
            base.resize_token_embeddings(len(tokenizer))
            logger.info("[Fold %d] resized token embeddings.", fold_id)
        if getattr(base.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            base.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        logger.warning("[Fold %d] embedding update failed: %s", fold_id, e)

    model = RegularizedModel(base, cfg)

    if hasattr(model, "gradient_checkpointing_enable"):
        try: model.gradient_checkpointing_enable()
        except Exception: pass

    for p in model.parameters(): p.requires_grad = False

    if cfg.USE_LORA and HAVE_PEFT:
        include_tokens = [s.strip() for s in cfg.LORA_TARGETS.split(",") if s.strip()]
        lora_targets = select_lora_targets_for_bert(model, include_tokens)
        if lora_targets:
            logger.info("[Fold %d] LoRA targets: %s", fold_id, lora_targets[:8])
            lcfg = LoraConfig(
                r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT,
                target_modules=lora_targets, bias="none", task_type="SEQ_CLS"
            )
            model = get_peft_model(model, lcfg)
        else:
            logger.warning("[Fold %d] LoRA targets not found.", fold_id)
    elif cfg.USE_LORA and not HAVE_PEFT:
        logger.warning("[Fold %d] peft not installed.", fold_id)

    for n,p in model.named_parameters():
        if ("lora_" in n) or ("head_trunk" in n) or ("mu_head" in n) or ("logvar_head" in n) or ("cls_head" in n) or ("quant_head" in n) or ("layer_mix" in n):
            p.requires_grad = True

    unfreeze_last_n_blocks(model, cfg.UNFREEZE_LAST_N_BLOCKS)

    head_keys = ["head_trunk","mu_head","logvar_head","cls_head","quant_head","layer_mix"]
    def is_head_param(n): return any(k in n for k in head_keys)
    opt_groups = [
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and not is_head_param(n)],
         "lr":cfg.LR_LORA, "weight_decay":cfg.WEIGHT_DECAY},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and is_head_param(n)],
         "lr":cfg.LR_HEAD, "weight_decay":cfg.WEIGHT_DECAY},
    ]
    optimizer = torch.optim.AdamW(opt_groups, eps=1e-8)

    steps_per_epoch = math.ceil(len(tokenized["train"]) / max(1, cfg.BATCH))
    steps_per_epoch = math.ceil(steps_per_epoch / max(1, cfg.GRAD_ACCUM))
    num_training_steps = steps_per_epoch * int(cfg.EPOCHS)
    warmup_steps = int(cfg.WARMUP * num_training_steps)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer,
                                 num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    metric_for_best = "loss" if cfg.METRIC=="loss" else f"eval_{cfg.METRIC}"
    trainer = build_trainer(model, tokenizer, tokenized, cfg, scaler, head_keys,
                            optimizer, lr_scheduler, patience=cfg.PATIENCE, metric_name=metric_for_best)

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        logger.info(f"[Fold {fold_id}] GPU: {dev.name} | VRAM: {dev.total_memory/1024**3:.1f} GB")

    logger.info(f"[Fold {fold_id}] Stage-1 training (RC-equivariant={cfg.RC_EQUIVARIANT}, rank={cfg.RANK_LOSS_LAMBDA}, corr={cfg.CORR_LOSS_LAMBDA}, stdm={cfg.STD_MATCH_LAMBDA})")
    trainer.train()

    stage1_logs = list(getattr(trainer.state, "log_history", []))
    all_logs = list(stage1_logs)

    out_dir     = os.path.join(cfg.OUT_ROOT, f"finetuned/fold{fold_id}_{global_ts}")
    trainer.save_model(out_dir); tokenizer.save_pretrained(out_dir)
    if trainer.state.best_model_checkpoint and os.path.isdir(trainer.state.best_model_checkpoint):
        best_out = os.path.join(out_dir, "best"); os.makedirs(best_out, exist_ok=True)
        for fn in os.listdir(trainer.state.best_model_checkpoint):
            src=os.path.join(trainer.state.best_model_checkpoint, fn)
            if os.path.isfile(src): shutil.copy2(src, os.path.join(best_out, fn))
        logger.info("[Fold %d] best checkpoint copied.", fold_id)

    _ok_merged = export_merged_if_possible(trainer, tokenizer, out_dir, fold_id)
    if not _ok_merged: logger.warning("[Fold %d] merged export failed.", fold_id)

    val_out  = trainer.predict(tokenized["validation"])
    test_out = trainer.predict(tokenized["test"])
    val_pred_raw  = scaler.inverse(_to_numpy_1d(val_out.predictions))
    test_pred_raw = scaler.inverse(_to_numpy_1d(test_out.predictions))
    val_true  = scaler.inverse(_to_numpy_1d(val_out.label_ids))
    test_true = scaler.inverse(_to_numpy_1d(test_out.label_ids))

    save_json({"fold": fold_id, "val_metrics": val_out.metrics, "test_metrics": test_out.metrics},
              os.path.join(results_dir, f"fold{fold_id}_predict_metrics.json"))

    try:
        seqs = test_df["sequence"].tolist()
        labels = np.array(test_df["expression"], dtype=float)

        base_seqs = [crop_sequence_center(s, cfg.MAX_LEN) for s in seqs]
        if is_kmer_tok:
            proc0 = [seq_to_6mers(s, k=kmer_k) for s in base_seqs]
            enc0 = tokenizer(proc0, padding="max_length", truncation=True,
                             max_length=(cfg.MAX_LEN - kmer_k + 1 + 2))
        else:
            enc0 = tokenizer(base_seqs, padding="max_length", truncation=True, max_length=cfg.MAX_LEN)
        ds0 = Dataset.from_dict({"input_ids": enc0["input_ids"], "attention_mask": enc0["attention_mask"],
                                 "labels": scaler.transform(labels).tolist()})
        out0 = _to_numpy_1d(trainer.predict(ds0).predictions)

        rc_seqs = [rc_seq(s) for s in base_seqs]
        if is_kmer_tok:
            proc1 = [seq_to_6mers(s, k=kmer_k) for s in rc_seqs]
            enc1 = tokenizer(proc1, padding="max_length", truncation=True,
                             max_length=(cfg.MAX_LEN - kmer_k + 1 + 2))
        else:
            enc1 = tokenizer(rc_seqs, padding="max_length", truncation=True, max_length=cfg.MAX_LEN)
        ds1 = Dataset.from_dict({"input_ids": enc1["input_ids"], "attention_mask": enc1["attention_mask"],
                                 "labels": scaler.transform(labels).tolist()})
        out1 = _to_numpy_1d(trainer.predict(ds1).predictions)

        pred_z = 0.5 * (out0 + out1)
        test_pred_rc = scaler.inverse(pred_z)
        save_df_csv(pd.DataFrame({"y_true": test_true, "y_pred_rc_tta": test_pred_rc}),
                    os.path.join(results_dir, f"fold{fold_id}_test_rc_tta_raw.csv"))
        test_pred_raw = test_pred_rc
    except Exception as e:
        logger.warning("[Fold %d] RC-TTA failed: %s", fold_id, e)

    metrics_val_raw  = summarize_metrics(val_true, val_pred_raw)
    metrics_test_raw = summarize_metrics(test_true, test_pred_raw)

    calib = {"type": cfg.CALIBRATION, "a": None, "b": None}
    base_pred_for_cal = test_pred_raw
    if cfg.CALIBRATE_ON_VAL and cfg.CALIBRATION != "none":
        if cfg.CALIBRATION == "linear":
            a,b = fit_true_on_pred(val_true, val_pred_raw)
            if cfg.CALIB_SHRINK is not None:
                a = float(cfg.CALIB_SHRINK) * a + (1.0 - float(cfg.CALIB_SHRINK)) * 1.0
            calib.update({"a": float(a), "b": float(b)})
            save_json({"fold": fold_id, "calibration_true_on_pred": calib},
                      os.path.join(results_dir, f"fold{fold_id}_calibration.json"))
            test_pred_raw = a * base_pred_for_cal + b
            metrics_test_raw = summarize_metrics(test_true, test_pred_raw)
            logger.info("[Fold %d] linear calibration applied.", fold_id)
            if cfg.CALIB_STD_SHRINK:
                try:
                    val_std_true = float(np.std(val_true))
                    val_std_pred = float(np.std(val_pred_raw))
                    if val_std_true > 0 and val_std_pred > 1e-8:
                        r = min(1.0, val_std_true / val_std_pred)
                        mu = float(np.mean(test_pred_raw))
                        test_pred_raw = mu + r * (test_pred_raw - mu)
                        metrics_test_raw = summarize_metrics(test_true, test_pred_raw)
                        logger.info(f"[Fold {fold_id}] Std-shrink applied: r={r:.3f}")
                except Exception:
                    pass
        elif cfg.CALIBRATION == "isotonic":
            if not HAVE_ISOTONIC:
                logger.warning("[Fold %d] isotonic calibration unavailable.", fold_id)
            else:
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(val_pred_raw, val_true)
                grid_x = np.linspace(val_pred_raw.min(), val_pred_raw.max(), 256)
                grid_y = ir.predict(grid_x)
                save_df_csv(pd.DataFrame({"pred_grid": grid_x, "mapped_true": grid_y}),
                            os.path.join(results_dir, f"fold{fold_id}_isotonic_grid.csv"))
                test_pred_raw = ir.predict(base_pred_for_cal)
                metrics_test_raw = summarize_metrics(test_true, test_pred_raw)
                logger.info("[Fold %d] isotonic calibration applied.", fold_id)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_df_csv(pd.DataFrame({"y_true": val_true, "y_pred": val_pred_raw}),
                os.path.join(preds_dir, f"val_preds_fold{fold_id}_{ts}.csv"))
    save_df_csv(pd.DataFrame({"y_true": test_true, "y_pred": test_pred_raw}),
                os.path.join(preds_dir, f"test_preds_fold{fold_id}_{ts}.csv"))

    if all_logs:
        df_log = pd.DataFrame(all_logs)
        save_df_csv(df_log, os.path.join(logs_dir, f"log_history_fold{fold_id}_{ts}.csv"))
        cols = [c for c in ["epoch","step","loss","loss_base","loss_logvar","loss_rank","loss_corr","loss_stdm",
                            "loss_cls","loss_q","loss_rc","eval_loss","eval_pearson","eval_r2","eval_rmse"] if c in df_log.columns]
        if cols:
            save_df_csv(df_log[cols], os.path.join(results_dir, f"fold{fold_id}_loss_decomposition_{ts}.csv"))

    for p in [train_csv, val_csv, test_csv]:
        try: os.remove(p)
        except Exception: pass

    logger.info("[Fold %d] done. Val pearson=%.3f | std_ratio=%.3f", fold_id, metrics_val_raw.get("pearson"), metrics_val_raw.get("std_ratio"))
    metrics_val_out = dict(metrics_val_raw)
    metrics_val_out.update({"calib_type": cfg.CALIBRATION})
    return pd.DataFrame({"y_true": test_true, "y_pred": test_pred_raw}), metrics_val_out


def parse_args():
    ap = argparse.ArgumentParser(description="Train the 5-fold REAP regression model.")
    ap.add_argument("--data-dir", default=Cfg.DATA_DIR)
    ap.add_argument("--all-csv", default=Cfg.ALL_CSV)
    ap.add_argument("--test-csv", default=Cfg.TEST_CSV)
    ap.add_argument("--pretrained", default=Cfg.PRETRAINED)
    ap.add_argument("--pretrained-weights", default=Cfg.PRETRAINED_WEIGHTS)
    ap.add_argument("--out-root", default=Cfg.OUT_ROOT)
    ap.add_argument("--results-root", default=Cfg.RESULTS_ROOT)
    ap.add_argument("--epochs", type=int, default=Cfg.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=Cfg.BATCH)
    ap.add_argument("--grad-accum", type=int, default=Cfg.GRAD_ACCUM)
    ap.add_argument("--max-len", type=int, default=Cfg.MAX_LEN)
    ap.add_argument("--seed", type=int, default=Cfg.SEED)
    return ap.parse_args()


def apply_args(cfg: Cfg, args):
    cfg.DATA_DIR = os.path.abspath(args.data_dir)
    cfg.ALL_CSV = args.all_csv
    cfg.TEST_CSV = args.test_csv
    cfg.PRETRAINED = os.path.abspath(args.pretrained)
    cfg.PRETRAINED_WEIGHTS = os.path.abspath(args.pretrained_weights) if args.pretrained_weights else ""
    cfg.OUT_ROOT = os.path.abspath(args.out_root)
    cfg.RESULTS_ROOT = os.path.abspath(args.results_root)
    cfg.EPOCHS = args.epochs
    cfg.BATCH = args.batch_size
    cfg.GRAD_ACCUM = args.grad_accum
    cfg.MAX_LEN = args.max_len
    cfg.SEED = args.seed
    return cfg


def main():
    cfg = apply_args(Cfg(), parse_args())
    set_seed(cfg.SEED)

    all_csv  = os.path.join(cfg.DATA_DIR, cfg.ALL_CSV)
    test_csv = os.path.join(cfg.DATA_DIR, cfg.TEST_CSV)
    if not os.path.exists(all_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(f"Missing data files: {all_csv}, {test_csv}")

    all_df  = pd.read_csv(all_csv)
    test_df = pd.read_csv(test_csv)

    for d in [cfg.OUT_ROOT, cfg.RESULTS_ROOT]:
        os.makedirs(d, exist_ok=True)

    global_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    global_scaler_params = None
    if cfg.USE_GLOBAL_SCALER:
        logger.info("[Global Scaler] fitting on all_data.csv")
        g_scaler = TargetScaler(cfg.TARGET_TRANSFORM)
        g_scaler.fit(all_df["expression"].values.astype(float))
        global_scaler_params = g_scaler.get_params()
        save_json({"global_scaler": global_scaler_params},
                  os.path.join(cfg.RESULTS_ROOT, "extras", f"global_scaler_{global_ts}.json"))

    def stratify_bins(y, n_bins=10):
        y = pd.Series(y).astype(float)
        try:    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        except Exception: bins = pd.cut(y, bins=n_bins, labels=False)
        return bins.values

    if "chromosome" in all_df.columns and all_df["chromosome"].nunique() >= 5:
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(all_df, groups=all_df["chromosome"].astype(str)))
        logger.info("Using GroupKFold.")
    else:
        y_bins = stratify_bins(all_df["expression"].values, n_bins=10)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
        splits = list(skf.split(all_df, y_bins))
        logger.info("Using StratifiedKFold.")

    fold_preds, fold_val_metrics = [], []

    for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
        train_df = all_df.iloc[tr_idx].reset_index(drop=True)
        val_df   = all_df.iloc[va_idx].reset_index(drop=True)
        test_df_ = test_df.copy().reset_index(drop=True)

        preds_df, val_metrics = train_one_fold(cfg, fold_id, train_df, val_df, test_df_, global_ts,
                                               global_scaler_params=global_scaler_params)
        fold_preds.append(preds_df); fold_val_metrics.append(val_metrics)

    logger.info("Ensembling fold predictions.")
    y_true = fold_preds[0]["y_true"].values
    mat = np.stack([df["y_pred"].values for df in fold_preds], axis=0)

    ens_equal = mat.mean(axis=0)
    pears = np.array([m.get("pearson", 0.0) for m in fold_val_metrics], dtype=float)
    w = np.log1p(np.exp(pears))
    if not np.isfinite(w).any() or w.sum() <= 0: w = np.ones_like(w)
    w = w / w.sum()
    ens_weighted = (w[:,None] * mat).sum(axis=0)

    final_equal   = summarize_metrics(y_true, ens_equal)
    final_weighted = summarize_metrics(y_true, ens_weighted)
    logger.info("[Ensemble Test - equal ] " + " | ".join([f"{k}={v:.4f}" for k,v in final_equal.items() if isinstance(v,(int,float))]))
    logger.info("[Ensemble Test - weight] " + " | ".join([f"{k}={v:.4f}" for k,v in final_weighted.items() if isinstance(v,(int,float))]))

    os.makedirs(os.path.join(cfg.RESULTS_ROOT, "preds"), exist_ok=True)
    ens_path_eq = os.path.join(cfg.RESULTS_ROOT, "preds", f"test_preds_ensemble_equal_{global_ts}.csv")
    ens_path_wt = os.path.join(cfg.RESULTS_ROOT, "preds", f"test_preds_ensemble_weighted_{global_ts}.csv")
    save_df_csv(pd.DataFrame({"y_true": y_true, "y_pred_ensemble_equal": ens_equal}), ens_path_eq)
    save_df_csv(pd.DataFrame({"y_true": y_true, "y_pred_ensemble_weighted": ens_weighted}), ens_path_wt)
    save_df_csv(pd.DataFrame(mat.T, columns=[f"fold{i}" for i in range(1, len(fold_preds)+1)]),
                os.path.join(cfg.RESULTS_ROOT, "preds", f"test_preds_ensemble_members_{global_ts}.csv"))

    report = {
        "config": {k:getattr(cfg,k) for k in dir(cfg) if k.isupper()},
        "val_folds": fold_val_metrics,
        "weights_val_pearson": w.tolist(),
        "ensemble_test_equal": final_equal,
        "ensemble_test_weighted": final_weighted
    }
    os.makedirs(os.path.join(cfg.RESULTS_ROOT, "metrics"), exist_ok=True)
    report_path = os.path.join(cfg.RESULTS_ROOT, "metrics", f"report_5fold_{global_ts}.json")
    save_json(report, report_path)

    manifest = {
        "timestamp": global_ts,
        "paths": {
            "report": report_path,
            "ensemble_equal_csv": ens_path_eq,
            "ensemble_weighted_csv": ens_path_wt,
            "ensemble_members_csv": os.path.join(cfg.RESULTS_ROOT, "preds", f"test_preds_ensemble_members_{global_ts}.csv"),
            "metrics_dir": os.path.join(cfg.RESULTS_ROOT, "metrics"),
            "preds_dir": os.path.join(cfg.RESULTS_ROOT, "preds"),
            "logs_csv_dir": os.path.join(cfg.RESULTS_ROOT, "logs_csv"),
            "extras_dir": os.path.join(cfg.RESULTS_ROOT, "extras"),
        }
    }
    save_json(manifest, os.path.join(cfg.RESULTS_ROOT, f"run_manifest_{global_ts}.json"))
    logger.info(f"Done. Ensemble outputs:\n- equal:   {ens_path_eq}\n- weighted:{ens_path_wt}")

if __name__ == "__main__":
    main()
