"""M-fm: zero-shot TSFM wrappers with one interface.

`load(name, device)` returns an object with `.predict(contexts, horizon) -> (median, quantiles)` where
`contexts` is a list of 1-D float arrays (each ending at the issue time tau, i.e. the last value is the hour
ending at tau), `median` has shape (n, horizon) and `quantiles` has shape (n, horizon, 9) for the levels
QLEVELS = 0.1 ... 0.9. No fine-tuning anywhere (zero-shot only).
"""
from __future__ import annotations

import numpy as np
import torch

QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

REVISIONS = {  # immutable HF revisions (results/ENVIRONMENT.md); loaders enforce them
    "amazon/chronos-bolt-base": "5d9f166d69f47aef3401367a7b842e78fe97b121",
    "amazon/chronos-2": "29ec3766d36d6f73f0696f85560a422f50e8498c",
    "google/timesfm-2.5-200m-pytorch": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    "Salesforce/moirai-2.0-R-small": "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
}

CHECKPOINTS = {
    "chronos-bolt": "amazon/chronos-bolt-base",
    "chronos-2": "amazon/chronos-2",
    "timesfm-2.5": "google/timesfm-2.5-200m-pytorch",
    "moirai-2": "Salesforce/moirai-2.0-R-small",
}


class _Chronos:
    def __init__(self, ckpt: str, device: str):
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained(ckpt, revision=REVISIONS[ckpt], device_map=device,
                                                        torch_dtype=torch.float32)

    def predict(self, contexts: list[np.ndarray], horizon: int):
        ctx = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
        q, _mean = self.pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=QLEVELS)
        q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        if q.ndim == 4:  # Chronos-2 returns (n, n_variates, h, q) for univariate input -> squeeze variates
            q = q[:, 0]
        return q[..., QLEVELS.index(0.5)], q


class _TimesFM:
    def __init__(self, ckpt: str, device: str, max_context: int = 2048):
        import timesfm

        torch.set_float32_matmul_precision("high")
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(ckpt, revision=REVISIONS[ckpt])
        self.m.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=128,
                per_core_batch_size=128,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def predict(self, contexts: list[np.ndarray], horizon: int):
        point, qf = self.m.forecast(horizon=horizon, inputs=[np.asarray(c, dtype=np.float32) for c in contexts])
        qf = np.asarray(qf)  # (n, h, 10): index 0 = mean, 1..9 = deciles 0.1..0.9
        q = qf[..., 1:10]
        return q[..., 4], q


class _Chronos2Cov(_Chronos):
    """Chronos-2 with one known-future covariate T: contexts are dicts {target, past_T, future_T}."""

    def predict(self, contexts, horizon: int):
        inputs = [{"target": np.asarray(c["target"], np.float32),
                   "past_covariates": {"T": np.asarray(c["past_T"], np.float32)},
                   "future_covariates": {"T": np.asarray(c["future_T"], np.float32)}} for c in contexts]
        q, _mean = self.pipe.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=QLEVELS)
        q = np.stack([x.detach().cpu().numpy()[0] for x in q])  # (n, h, 9)
        return q[..., QLEVELS.index(0.5)], q


def load(name: str, device: str = "cpu"):
    if name == "chronos-2+T":
        return _Chronos2Cov(CHECKPOINTS["chronos-2"], device)
    ckpt = CHECKPOINTS[name]
    if name.startswith("chronos"):
        return _Chronos(ckpt, device)
    if name.startswith("timesfm"):
        return _TimesFM(ckpt, device)
    raise ValueError(f"model {name} needs its own environment (see src/fm/moirai_env.py)")
