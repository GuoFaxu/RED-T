from types import SimpleNamespace

import torch

from .diffusion import GaussianDiffusion, generate_cosine_schedule, generate_linear_schedule
from .unet import UNet


def diffusion_defaults():
    return dict(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        run_name="run",
        project_name="proj",
        log_dir="./model",
        log_rate=1,
        checkpoint_rate=200,
        img_channels=4,
        img_size=(512, 1),
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=512,
        time_emb_scale=1.0,
        attention_resolutions=(1, 2),
        norm="gn",
        num_groups=32,
        dropout=0.1,
        initial_pad=0,
        out_init_conv_padding=1,
        timesteps=1000,
        schedule_type="cosine",
        schedule_low=1e-4,
        schedule_high=0.02,
        loss_type="l2",
        pred_type="v",
        loss_weight_type="min_snr",
        min_snr_gamma=5.0,
        ema_decay=0.9999,
        ema_start=0,
        ema_update_rate=1,
        learning_rate=2e-4,
        batch_size=64,
        iterations=2000,
        model_checkpoint=None,
        optim_checkpoint=None,
    )


def get_diffusion_from_args(args) -> GaussianDiffusion:
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    base_channels = getattr(args, "base_channels", 128)
    time_emb_dim = getattr(args, "time_emb_dim", None) or base_channels * 4

    net = UNet(
        img_channels=args.img_channels,
        base_channels=base_channels,
        channel_mults=getattr(args, "channel_mults", (1, 2, 4, 8)),
        num_res_blocks=getattr(args, "num_res_blocks", 2),
        time_emb_dim=time_emb_dim,
        time_emb_scale=getattr(args, "time_emb_scale", 1.0),
        num_classes=None,
        activation=torch.nn.functional.silu,
        dropout=getattr(args, "dropout", 0.1),
        attention_resolutions=getattr(args, "attention_resolutions", (1, 2)),
        norm=getattr(args, "norm", "gn"),
        num_groups=getattr(args, "num_groups", 32),
        initial_pad=getattr(args, "initial_pad", 0),
        out_init_conv_padding=getattr(args, "out_init_conv_padding", 1),
    )

    num_timesteps = int(getattr(args, "timesteps", 1000))
    if getattr(args, "schedule_type", "cosine") == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            getattr(args, "schedule_low", 1e-4),
            getattr(args, "schedule_high", 0.02),
        )

    return GaussianDiffusion(
        model=net,
        img_size=tuple(getattr(args, "img_size", (512, 1))),
        img_channels=int(getattr(args, "img_channels", 4)),
        num_classes=0,
        betas=betas,
        loss_type=getattr(args, "loss_type", "l2"),
        ema_decay=getattr(args, "ema_decay", 0.9999),
        ema_start=getattr(args, "ema_start", 0),
        ema_update_rate=getattr(args, "ema_update_rate", 1),
        use_ema=True,
        pred_type=getattr(args, "pred_type", "v"),
        loss_weight_type=getattr(args, "loss_weight_type", "min_snr"),
        min_snr_gamma=float(getattr(args, "min_snr_gamma", 5.0)),
    )
