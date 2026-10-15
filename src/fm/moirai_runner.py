"""M-fm: Moirai-2 runner for the separate `.venv-moirai` environment (uni2ts pins torch 2.4).

Interface: `python src/fm/moirai_runner.py <in.npz> <out.npz> <horizon> <device> [--timing]`
in.npz holds `contexts` (n, L) float32; out.npz gets `median` (n, H) and `quantiles` (n, H, 9) at levels 0.1..0.9,
plus `levels` (the model's own quantile levels, checked to be 0.1..0.9). With --timing it prints the same
timing measurements as one CSV line.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

CKPT = "Salesforce/moirai-2.0-R-small"


def build(horizon: int, context_length: int, device: str):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(CKPT, revision="30f43ff08c8494f4943ae1521e9d4e94a0fbb389")
    m = Moirai2Forecast(module=module, prediction_length=horizon, context_length=context_length, target_dim=1,
                        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    return m.to(device).eval()


def predict(m, contexts: np.ndarray):
    out = m.predict([c.astype(np.float32) for c in contexts])  # (n, n_q, H)
    levels = np.asarray(m.module.quantile_levels, dtype=float)
    want = np.round(np.arange(0.1, 0.95, 0.1), 2)
    idx = [int(np.argmin(np.abs(levels - w))) for w in want]
    if not np.allclose(levels[idx], want):
        raise ValueError(f"Moirai-2 quantile levels {levels} do not contain 0.1..0.9")
    q = np.transpose(out[:, idx, :], (0, 2, 1))
    return q[..., 4], q, levels


def main():
    inp, outp, H, dev = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.set_num_threads(4)
    ctx = np.load(inp)["contexts"]
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); m = build(H, ctx.shape[1], dev); t_load = time.perf_counter() - t0
    if "--timing" in sys.argv:
        predict(m, ctx[:1])
        times = []
        for c in ctx:
            t0 = time.perf_counter(); predict(m, c[None]);
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); med, q, lv = predict(m, ctx)
        if dev == "cuda":
            torch.cuda.synchronize()
        tb = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
        print(f"TIMING,moirai-2,{CKPT},{dev},{len(ctx)},{t_load:.2f},{np.median(times):.4f},"
              f"{np.percentile(times, 90):.4f},{tb:.3f},{vram:.1f}", flush=True)
    else:
        med, q, lv = predict(m, ctx)
    np.savez(outp, median=med, quantiles=q, levels=lv)


if __name__ == "__main__":
    main()
