from typing import Dict, Optional

import torch
import torch.nn as nn


class ModelConfig:
    DROPOUT = 0.15
    POOL_WINS = (8, 16, 32, 64)
    LAYER_MIX_K = 4
    LOGVAR_CLAMP_MIN = -2.0
    LOGVAR_CLAMP_MAX = 1.5
    MLP_HIDDEN1 = 256
    MLP_HIDDEN2 = 64
    HEAD_DROPOUT = 0.15
    QUANTILE_TAUS = (0.5, 0.8)


class RegularizedModel(nn.Module):
    def __init__(self, base, cfg=ModelConfig):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.config = getattr(base, "config", None)

        d_model = getattr(
            self.base.config,
            "hidden_size",
            getattr(self.base.config, "d_model", getattr(self.base.config, "model_dim", 512)),
        )
        self.K = int(cfg.LAYER_MIX_K)
        self.layer_mix = nn.Parameter(torch.zeros(self.K))
        self.attn_score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.pools = nn.ModuleList([nn.AvgPool1d(k, stride=k) for k in cfg.POOL_WINS])

        feat_mult = 3 + len(cfg.POOL_WINS)
        in_dim = d_model * feat_mult
        h1 = cfg.MLP_HIDDEN1
        h2 = cfg.MLP_HIDDEN2
        head_dropout = getattr(cfg, "HEAD_DROPOUT", 0.15)

        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(head_dropout)]
        if h2 > 0:
            layers += [nn.Linear(h1, h2), nn.GELU(), nn.Dropout(head_dropout)]
            head_in = h2
        else:
            head_in = h1
        self.head_trunk = nn.Sequential(*layers)

        self.mu_head = nn.Linear(head_in, 1)
        self.logvar_head = nn.Linear(head_in, 1)
        self.cls_head = nn.Linear(head_in, 1)
        self.quant_head = nn.Linear(head_in, len(getattr(cfg, "QUANTILE_TAUS", (0.5, 0.8))))

    @property
    def hf_config(self):
        return getattr(self, "config", getattr(self.base, "config", None))

    def _masked_mean(self, x, mask):
        m = mask.unsqueeze(-1).float()
        s = (x * m).sum(dim=1)
        d = m.sum(dim=1).clamp_min(1e-6)
        return s / d

    def _masked_max(self, x, mask):
        m = mask.bool()
        neg_inf = torch.finfo(x.dtype).min
        x_mask = x.masked_fill(~m.unsqueeze(-1), neg_inf)
        return x_mask.max(dim=1).values

    def _attn_pool(self, x, mask):
        score = self.attn_score(x).squeeze(-1)
        score = score.masked_fill(~mask.bool(), -1e9)
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)

    def _pyramid_pool(self, x, mask):
        x_t = x.transpose(1, 2)
        outs = []
        m = mask.float().unsqueeze(1)
        for pool in self.pools:
            y = pool(x_t).transpose(1, 2)
            denom = pool(m).squeeze(1).clamp_min(1e-6)
            y = y / denom.unsqueeze(-1)
            outs.append(y.max(dim=1).values)
        return torch.cat(outs, dim=-1) if outs else x.mean(dim=1)

    def _fuse_layers(self, hidden_states):
        hs = [hidden_states[-i] for i in range(1, min(self.K, len(hidden_states)) + 1)]
        hs = torch.stack(hs, dim=0)
        w = torch.softmax(self.layer_mix[: hs.size(0)], dim=0).view(-1, 1, 1, 1)
        return (w * hs).sum(0)

    def encode_once(self, input_ids, attention_mask, **extra):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        for key in ("token_type_ids", "position_ids", "inputs_embeds", "head_mask"):
            if key in extra:
                kwargs[key] = extra[key]
        outputs = self.base(**kwargs, output_hidden_states=True, return_dict=True)
        last = self._fuse_layers(outputs.hidden_states)
        return last, attention_mask

    def pool_multiscale(self, tok, mask):
        mean_pool = self._masked_mean(tok, mask)
        max_pool = self._masked_max(tok, mask)
        attn_pool = self._attn_pool(tok, mask)
        pyr_pool = self._pyramid_pool(tok, mask)
        return torch.cat([mean_pool, max_pool, attn_pool, pyr_pool], dim=-1)

    def forward(self, input_ids=None, attention_mask=None, input_ids_rc=None, attention_mask_rc=None, **kw):
        tok_fwd, m_fwd = self.encode_once(input_ids=input_ids, attention_mask=attention_mask, **kw)
        feat_fwd = self.pool_multiscale(tok_fwd, m_fwd)

        aux = {}
        if input_ids_rc is not None and attention_mask_rc is not None:
            tok_rc, m_rc = self.encode_once(input_ids=input_ids_rc, attention_mask=attention_mask_rc)
            feat_rc = self.pool_multiscale(tok_rc, m_rc)
            feat = 0.5 * (feat_fwd + feat_rc)
            aux["rc_feature_mse"] = torch.mean((feat_fwd - feat_rc) ** 2)
        else:
            feat = feat_fwd

        h = self.head_trunk(self.dropout(feat))
        mu = self.mu_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1).clamp(self.cfg.LOGVAR_CLAMP_MIN, self.cfg.LOGVAR_CLAMP_MAX)
        cls_logits = self.cls_head(h).squeeze(-1)
        quant = self.quant_head(h)

        if input_ids_rc is not None and attention_mask_rc is not None:
            h_f = self.head_trunk(self.dropout(feat_fwd))
            h_r = self.head_trunk(self.dropout(feat_rc))
            aux["rc_consistency"] = torch.mean((self.mu_head(h_f).squeeze(-1) - self.mu_head(h_r).squeeze(-1)) ** 2)

        return {
            "loss": None,
            "logits": mu.unsqueeze(-1),
            "mu": mu,
            "pred_logvar": logvar,
            "logvar": logvar,
            "cls_logits": cls_logits,
            "quantiles": quant,
            "aux": aux,
        }


def infer_mlp_head_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    init_drop: float = 0.20,
    mid_drop: float = 0.10,
) -> Optional[nn.Sequential]:
    keys = [k for k in state_dict if k.startswith("classifier.")]
    if not keys:
        return None
    linear_weights = []
    for key, value in state_dict.items():
        if key.startswith("classifier.") and key.endswith(".weight") and value.ndim == 2:
            linear_weights.append((key, tuple(value.shape)))
    if len(linear_weights) < 1:
        return None

    layers = []
    for i, (_, shape) in enumerate(linear_weights):
        out_dim, in_dim = shape
        if i == 0:
            layers.extend([nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(init_drop)])
        elif i < len(linear_weights) - 1:
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(mid_drop)])
        else:
            layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)
