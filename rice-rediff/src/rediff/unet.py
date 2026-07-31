import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True, eps=1e-6)
    if norm == "bn":
        return nn.BatchNorm2d(num_channels, eps=1e-5)
    if norm == "gn":
        return nn.GroupNorm(num_groups, num_channels, eps=1e-6)
    if norm is None:
        return nn.Identity()
    raise ValueError("unknown normalization type")


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
        )

    def forward(self, x, time_emb):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x, time_emb):
        x = F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        return self.upsample_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        attn = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        out = torch.bmm(attn, v).view(b, h, w, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        time_emb_dim=None,
        num_classes=None,
        activation=F.silu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()
        self.activation = activation
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0))
        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
        )
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
        nn.init.kaiming_normal_(self.conv_1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv_1.bias)
        nn.init.kaiming_normal_(self.conv_2[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.conv_2[1].bias)

    def forward(self, x, time_emb=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        return self.attention(out)


class UNet(nn.Module):
    def __init__(
        self,
        img_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=None,
        time_emb_scale=1.0,
        num_classes=None,
        activation=F.silu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
        out_init_conv_padding=1,
    ):
        super().__init__()
        self.activation = activation
        self.initial_pad = initial_pad
        self.num_classes = num_classes
        self.time_mlp = (
            nn.Sequential(
                PositionalEmbedding(base_channels, time_emb_scale),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            if time_emb_dim is not None
            else None
        )

        kernel_size = (out_init_conv_padding * 2 + 1, 1)
        padding = (out_init_conv_padding, 0)
        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=kernel_size, padding=padding)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels,
                        out_channels,
                        dropout,
                        time_emb_dim=time_emb_dim,
                        num_classes=num_classes,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(
                    nn.Conv2d(now_channels, now_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
                )
                channels.append(now_channels)

        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=True,
                ),
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=False,
                ),
            ]
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    ResidualBlock(
                        channels.pop() + now_channels,
                        out_channels,
                        dropout,
                        time_emb_dim=time_emb_dim,
                        num_classes=num_classes,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
            if i != 0:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=(2, 1), mode="nearest"),
                        nn.Conv2d(now_channels, now_channels, kernel_size=(3, 1), padding=(1, 0)),
                    )
                )
        assert len(channels) == 0

    def forward(self, x, time=None):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (0, 0, ip, ip))
        time_emb = self.time_mlp(time) if self.time_mlp is not None else None

        x = self.init_conv(x)
        skips = [x]
        for layer in self.downs:
            if hasattr(layer, "forward") and "time_emb" in layer.forward.__code__.co_varnames:
                x = layer(x, time_emb)
            else:
                x = layer(x)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                skip = skips.pop()
                if skip.shape[2:] != x.shape[2:]:
                    delta_h = skip.shape[2] - x.shape[2]
                    if delta_h > 0:
                        skip = skip[:, :, : x.shape[2], :]
                    elif delta_h < 0:
                        x = x[:, :, : skip.shape[2], :]
                x = torch.cat([x, skip], dim=1)
            x = layer(x, time_emb) if isinstance(layer, ResidualBlock) else layer(x)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        return x[:, :, ip:-ip, :] if ip != 0 else x
