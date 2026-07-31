from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rediff import script_utils
from rediff.sequence_io import (
    clean_sequences,
    one_hot_01,
    read_csv_sequences,
    rev_comp,
    scale_input,
    to_fixed_length,
    write_fasta,
)


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: list[str],
        split: str,
        seq_len: int,
        pad_base: str,
        input_scale: str,
        jitter: int = 0,
        rc_prob: float = 0.0,
    ) -> None:
        self.seqs = seqs
        self.split = split
        self.seq_len = seq_len
        self.pad_base = pad_base
        self.input_scale = input_scale
        self.jitter = jitter
        self.rc_prob = rc_prob

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw = self.seqs[idx]
        if self.split == "train" and len(raw) > self.seq_len:
            center = (len(raw) - self.seq_len) // 2
            delta = random.randint(-self.jitter, self.jitter) if self.jitter > 0 else 0
            start = max(0, min(len(raw) - self.seq_len, center + delta))
            seq = raw[start : start + self.seq_len]
        else:
            seq, _, _, _ = to_fixed_length(raw, self.seq_len, self.pad_base)

        if self.split == "train" and random.random() < self.rc_prob:
            seq = rev_comp(seq)

        x = one_hot_01(seq)
        x = scale_input(x, self.input_scale)
        x = np.transpose(x, (1, 0))[:, :, None]
        return torch.tensor(x, dtype=torch.float32)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def mono_target(seqs: list[str]) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    total = 0
    for seq in seqs:
        for ch in seq:
            if ch in idx:
                counts[idx[ch]] += 1
                total += 1
    return (counts / total).astype(np.float32) if total else np.full(4, 0.25, dtype=np.float32)


def aux_weights_for_epoch(cfg: dict, epoch: int) -> dict[str, float]:
    schedule = cfg.get("aux_schedule", [])
    for entry in schedule:
        if epoch <= int(entry["end_epoch"]):
            return {k: float(v) for k, v in entry.get("weights", {}).items()}
    return {k: float(v) for k, v in schedule[-1].get("weights", {}).items()} if schedule else {}


def build_diffusion_config(cfg: dict, device: str, run_name: str) -> dict:
    defaults = script_utils.diffusion_defaults()
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("train", {})
    defaults.update(
        dict(
            device=device,
            run_name=run_name,
            project_name=cfg.get("project_name", "rice_enhancer_ddpm"),
            log_dir=train_cfg.get("checkpoint_dir", "outputs/checkpoints"),
            log_rate=int(train_cfg.get("log_rate", 1)),
            checkpoint_rate=int(train_cfg.get("checkpoint_rate", 100)),
            img_channels=4,
            img_size=(int(data_cfg.get("seq_len", 512)), 1),
            base_channels=int(model_cfg.get("base_channels", 96)),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 8])),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            time_emb_dim=int(model_cfg.get("time_emb_dim", 512)),
            attention_resolutions=tuple(model_cfg.get("attention_resolutions", [1, 2])),
            norm=model_cfg.get("norm", "gn"),
            num_groups=int(model_cfg.get("num_groups", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            initial_pad=int(model_cfg.get("initial_pad", 0)),
            out_init_conv_padding=int(model_cfg.get("out_init_conv_padding", 1)),
            timesteps=int(diffusion_cfg.get("timesteps", 1000)),
            schedule_type=diffusion_cfg.get("schedule_type", "cosine"),
            schedule_low=float(diffusion_cfg.get("schedule_low", 1e-4)),
            schedule_high=float(diffusion_cfg.get("schedule_high", 0.02)),
            loss_type=diffusion_cfg.get("loss_type", "l2"),
            pred_type=diffusion_cfg.get("pred_type", "v"),
            loss_weight_type=diffusion_cfg.get("loss_weight_type", "min_snr"),
            min_snr_gamma=float(diffusion_cfg.get("min_snr_gamma", 2.0)),
            ema_decay=float(diffusion_cfg.get("ema_decay", 0.999)),
            ema_start=int(diffusion_cfg.get("ema_start", 0)),
            ema_update_rate=int(diffusion_cfg.get("ema_update_rate", 1)),
            learning_rate=float(train_cfg.get("learning_rate", 1.5e-4)),
            batch_size=int(train_cfg.get("batch_size", 512)),
            iterations=int(train_cfg.get("epochs", 3000)),
            model_checkpoint=None,
            optim_checkpoint=None,
        )
    )
    return defaults


def init_loss_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["run_name", "epoch", "train_loss", "val_loss", "best_val", "lr", "timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train REDiff for rice enhancer generation.")
    parser.add_argument("--config", default="configs/rediff_rice_512.yaml")
    parser.add_argument("--data", default=None, help="Optional input CSV override.")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(args.device or train_cfg.get("device", "auto"))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    split_dir = out_dir / "data"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    input_csv = args.data or data_cfg.get("input_csv")
    if not input_csv:
        raise ValueError("Set data.input_csv in the config or pass --data.")

    raw = clean_sequences(
        read_csv_sequences(input_csv, data_cfg.get("sequence_col")),
        max_n_frac=float(data_cfg.get("max_n_frac", 0.2)),
        dedup=True,
    )
    rng = np.random.RandomState(int(data_cfg.get("split_seed", 42)))
    indices = np.arange(len(raw))
    rng.shuffle(indices)
    n_val = int(len(raw) * float(data_cfg.get("val_frac", 0.10)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_seqs = [raw[i] for i in train_idx]
    val_seqs = [raw[i] for i in val_idx]
    print(f"Using {len(train_seqs)} train and {len(val_seqs)} validation sequences.")

    write_fasta(split_dir / "train_512.fasta", [to_fixed_length(s, int(data_cfg.get("seq_len", 512)))[0] for s in train_seqs], "train")
    write_fasta(split_dir / "val_512.fasta", [to_fixed_length(s, int(data_cfg.get("seq_len", 512)))[0] for s in val_seqs], "val")

    seq_len = int(data_cfg.get("seq_len", 512))
    input_scale = train_cfg.get("input_scale", "pm1")
    train_loader = DataLoader(
        SequenceDataset(
            train_seqs,
            "train",
            seq_len,
            data_cfg.get("pad_base", "A"),
            input_scale,
            jitter=int(data_cfg.get("jitter", 8)),
            rc_prob=float(data_cfg.get("rc_prob", 0.5)),
        ),
        batch_size=int(train_cfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        SequenceDataset(val_seqs, "val", seq_len, data_cfg.get("pad_base", "A"), input_scale),
        batch_size=int(train_cfg.get("batch_size", 512)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    run_name = train_cfg.get("run_name") or dt.datetime.now().strftime("rediff-%Y-%m-%d-%H-%M")
    diffusion_args = build_diffusion_config(cfg, device, run_name)
    with open(ckpt_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(diffusion_args, handle, indent=2)

    diffusion = script_utils.get_diffusion_from_args(diffusion_args).to(device)
    diffusion.register_buffer("mono_target", torch.tensor(mono_target(train_seqs), dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("epochs", 3000)),
        eta_min=float(train_cfg.get("learning_rate", 1.5e-4)) * float(train_cfg.get("eta_min_factor", 0.2)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if train_cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    accum_steps = int(train_cfg.get("accum_steps", 4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    loss_csv = out_dir / "loss_curve.csv"
    init_loss_csv(loss_csv)
    best_val = float("inf")
    milestone_epochs = set(int(x) for x in train_cfg.get("milestone_epochs", []))

    for epoch in range(1, int(train_cfg.get("epochs", 3000)) + 1):
        diffusion.train()
        train_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        aux_weights = aux_weights_for_epoch(cfg, epoch)

        for step, x in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True).to(next(diffusion.parameters()).dtype)
            with autocast():
                if hasattr(diffusion, "training_step"):
                    loss, _ = diffusion.training_step(
                        x,
                        aux_weights=aux_weights,
                        aux_temp=float(train_cfg.get("aux_temp", 1.0)),
                    )
                else:
                    loss = diffusion(x)
            train_loss_sum += float(loss.item())
            loss = loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % accum_steps == 0 or step == len(train_loader):
                if grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if hasattr(diffusion, "update_ema"):
                    diffusion.update_ema()

        scheduler.step()
        train_loss = train_loss_sum / max(1, len(train_loader))

        diffusion.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True).to(next(diffusion.parameters()).dtype)
                with autocast():
                    val_sum += float(diffusion(x).item())
        val_loss = val_sum / max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(diffusion.state_dict(), ckpt_dir / f"{run_name}-best-model.pt")
            if hasattr(diffusion, "ema_model"):
                torch.save(diffusion.ema_model.state_dict(), ckpt_dir / f"{run_name}-best-ema.pt")

        lr = optimizer.param_groups[0]["lr"]
        with open(loss_csv, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [run_name, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{best_val:.6f}", f"{lr:.6e}", dt.datetime.now().isoformat()]
            )
        print(f"[Epoch {epoch:04d}] train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

        if epoch % int(train_cfg.get("checkpoint_rate", 100)) == 0:
            torch.save(diffusion.state_dict(), ckpt_dir / f"{run_name}-epoch-{epoch:04d}-model.pt")
            torch.save(optimizer.state_dict(), ckpt_dir / f"{run_name}-epoch-{epoch:04d}-optimizer.pt")
        if epoch in milestone_epochs:
            torch.save(diffusion.state_dict(), ckpt_dir / f"{run_name}-epoch-{epoch:04d}-milestone.pt")

    print(f"Training complete. Outputs are in {out_dir}")


if __name__ == "__main__":
    main()
