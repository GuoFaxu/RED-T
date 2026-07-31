from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rediff import script_utils
from rediff.sequence_io import write_fasta


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_diffusion(cfg: dict, device: str):
    defaults = script_utils.diffusion_defaults()
    seq_len = int(cfg["data"].get("seq_len", 512))
    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("train", {})
    defaults.update(
        dict(
            device=device,
            project_name=cfg.get("project_name", "rice_enhancer_ddpm"),
            img_channels=4,
            img_size=(seq_len, 1),
            learning_rate=float(train_cfg.get("learning_rate", 1.5e-4)),
            batch_size=int(train_cfg.get("batch_size", 512)),
            iterations=int(train_cfg.get("epochs", 3000)),
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
        )
    )
    return script_utils.get_diffusion_from_args(defaults).to(device), defaults


def smart_load(diffusion, state_dict: dict) -> None:
    keys = list(state_dict.keys())
    has_prefix = any(k.startswith(("model.", "ema_model.")) for k in keys)
    looks_unet = any(k.startswith(("init_conv.", "downs.", "mid.", "ups.", "out_conv.")) for k in keys)
    if has_prefix:
        diffusion.load_state_dict(state_dict, strict=False)
    elif looks_unet:
        diffusion.model.load_state_dict(state_dict, strict=False)
        if hasattr(diffusion, "ema_model"):
            diffusion.ema_model.load_state_dict(state_dict, strict=False)
    else:
        diffusion.load_state_dict(state_dict, strict=False)


def probs_to_seqs(samples: torch.Tensor, strategy: str, temp: float) -> list[str]:
    if samples.ndim != 4 or samples.shape[1] != 4:
        raise ValueError(f"Expected [N,4,L,1], got {tuple(samples.shape)}")
    probs = torch.clamp(samples, 1e-8, 1.0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    probs = probs[:, :, :, 0].permute(0, 2, 1)
    if strategy == "argmax":
        idx = probs.argmax(dim=-1)
    else:
        p = probs
        if temp != 1.0:
            p = p.pow(1.0 / max(1e-6, temp))
            p = p / p.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(p.reshape(-1, 4), 1).reshape(p.shape[:-1])
    table = np.array(list("ACGT"))
    return ["".join(table[row]) for row in idx.cpu().numpy()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample rice enhancer sequences from a trained REDiff checkpoint.")
    parser.add_argument("--config", default="configs/rediff_rice_512.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/rediff_epoch2400.pth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_cfg = cfg.get("sample", {})
    device = resolve_device(args.device)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    diffusion, build_cfg = build_diffusion(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu")
    smart_load(diffusion, state)
    diffusion.eval()
    if hasattr(diffusion, "use_ema"):
        diffusion.use_ema = bool(args.use_ema)

    num_samples = args.num_samples or int(sample_cfg.get("num_samples", 500))
    chunk_size = args.chunk_size or int(sample_cfg.get("chunk_size", 1024))
    decode_strategy = sample_cfg.get("decode_strategy", "sample")
    decode_temp = float(sample_cfg.get("decode_temp", 0.90))
    input_scale = cfg.get("train", {}).get("input_scale", "pm1")
    amp = bool(sample_cfg.get("amp", True)) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if sample_cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp
        else contextlib.nullcontext()
    )

    if args.output:
        output = Path(args.output)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path("sequences") / f"{cfg.get('project_name', 'rice_enhancer_ddpm')}-gen{num_samples}-{stamp}.fasta"

    all_seqs: list[str] = []
    remain = num_samples
    with torch.no_grad():
        while remain > 0:
            n = min(chunk_size, remain)
            print(f"Sampling {n} sequences...")
            with autocast():
                y = diffusion.sample(n, torch.device(device), use_ema=args.use_ema)
            y = y.to(dtype=torch.float32)
            y01 = (y + 1.0) / 2.0 if input_scale == "pm1" else y
            y01 = torch.clamp(y01, 0.0, 1.0)
            all_seqs.extend(probs_to_seqs(y01, decode_strategy, decode_temp))
            remain -= n

    write_fasta(output, all_seqs, prefix="gen")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "output": str(output),
        "num_samples": num_samples,
        "chunk_size": chunk_size,
        "use_ema": args.use_ema,
        "decode_strategy": decode_strategy,
        "decode_temp": decode_temp,
        "build_config": build_cfg,
    }
    with open(output.with_suffix(output.suffix + ".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {len(all_seqs)} sequences to {output}")


if __name__ == "__main__":
    main()
