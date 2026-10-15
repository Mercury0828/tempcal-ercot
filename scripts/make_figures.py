"""Paper figures from frozen artifacts only (results/confirm/test/*.csv, results/figdata/*.parquet).

Usage: python figs/make_figures.py   (no test-period access needed; reads stored outputs)
Fig. 1  Elliott and Heather, ERCOT system load with S-CONF and TEMPCAL (guarded) 90 % intervals, cold-bin hours shaded.
Fig. 2  Cold / mild / hot coverage per series: four univariate TSFMs (sigma S-CONF), Chronos-2+T S-CONF, TEMPCAL.
Fig. 3  Cold-bin coverage by winter and horizon band, ERCOT and the 8-zone aggregate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "figs"
MM = 1 / 25.4
COL = 88.9 * MM
TZ = "America/Chicago"
SERIES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]
ZONES = SERIES[:-1]
# colour-blind-safe (Okabe-Ito) + shape/line redundancy for grayscale
C_TEMP, C_SCONF, C_UNIV, C_SHADE = "#0072B2", "#555555", "#999999", "#E69F00"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "lines.linewidth": 0.9, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")  # exact final width (constrained layout), no cropping
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def _event_panel(ax, a: str, b: str, rows: dict) -> None:
    lo_t = pd.Timestamp(a).tz_localize(TZ) + pd.Timedelta(hours=7)
    hi_t = (pd.Timestamp(b) + pd.Timedelta(days=1)).tz_localize(TZ) + pd.Timedelta(hours=6)
    d = {k: v[(v.t > lo_t) & (v.t <= hi_t)].sort_values("t") for k, v in rows.items()}
    g, s = d["GUARDED"], d["S-CONF_rel"]
    t = g["t"].dt.tz_localize(None).to_numpy()
    for i in np.flatnonzero(g["bin"].eq("cold").to_numpy()):
        ax.axvspan(t[i] - np.timedelta64(1, "h"), t[i], color=C_SHADE, alpha=0.18, lw=0)
    ax.fill_between(t, g["lo"] / 1e3, g["hi"] / 1e3, step="pre", color=C_TEMP, alpha=0.25, lw=0,
                    label="TEMPCAL 90% interval")
    ax.step(t, g["lo"].to_numpy() / 1e3, where="pre", color=C_TEMP, lw=0.4)
    ax.step(t, g["hi"].to_numpy() / 1e3, where="pre", color=C_TEMP, lw=0.4)
    ax.step(t, s["lo"].to_numpy() / 1e3, where="pre", color=C_SCONF, ls="--", lw=0.8, label="S-CONF 90% interval")
    ax.step(t, s["hi"].to_numpy() / 1e3, where="pre", color=C_SCONF, ls="--", lw=0.8)
    ax.step(t, g["y"].to_numpy() / 1e3, where="pre", color="black", lw=1.0, label="Actual load")
    ax.set_xlim(t[0] - np.timedelta64(1, "h"), t[-1])
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.set_ylabel("Load (GW)")
    ax.grid(axis="y", lw=0.3, color="#DDDDDD")


def fig1() -> None:
    rows = {}
    for arm in ("S-CONF_rel", "GUARDED"):
        x = pd.read_parquet(RES / "figdata" / f"{arm}.parquet")
        x = x[(x.series == "ERCOT") & (x.h <= 24)].copy()
        x["t"] = x["t_end_utc"].dt.tz_convert(TZ)
        rows[arm] = x
    fig, axes = plt.subplots(2, 1, figsize=(COL, 80 * MM))
    _event_panel(axes[0], "2022-12-21", "2022-12-26", rows)
    _event_panel(axes[1], "2024-01-13", "2024-01-17", rows)
    axes[0].set_title("(a) Winter Storm Elliott, December 2022", loc="left")
    axes[1].set_title("(b) Winter Storm Heather, January 2024", loc="left")
    from matplotlib.patches import Patch
    h, l_ = axes[0].get_legend_handles_labels()
    h.append(Patch(color=C_SHADE, alpha=0.35, lw=0))
    l_.append("Cold forecast hour")
    fig.legend(h, l_, loc="outside upper center", ncol=2, frameon=False, handlelength=1.6, columnspacing=1.0)
    save(fig, "fig1_storms")


def fig2() -> None:
    c1 = pd.read_csv(RES / "confirm" / "test" / "Q1_cells.csv")
    c2 = pd.read_csv(RES / "confirm" / "test" / "Q2_cells.csv")
    c1 = c1[c1.band == "1-24"]
    c2 = c2[c2.band == "1-24"]
    univ = {"B2:chronos-bolt": "Chronos-Bolt", "B2:chronos-2": "Chronos-2", "B2:timesfm-2.5": "TimesFM 2.5",
            "B2:moirai-2": "Moirai-2"}
    fig, axes = plt.subplots(3, 1, figsize=(COL, 80 * MM), sharex=True)
    xs = np.arange(len(SERIES))
    for ax, k, title in zip(axes, ("cold", "mild", "hot"), ("Cold bin", "Mild bin", "Hot bin")):
        for j, (arm, lab) in enumerate(univ.items()):
            v = c1[(c1.arm == arm) & (c1.bin == k)].set_index("series").reindex(SERIES)["coverage"]
            ax.plot(xs - 0.24 + 0.06 * j, v, ls="none", marker="o", ms=2.6, mfc="white", mec=C_UNIV, mew=0.7,
                    label="Load-only models, S-CONF (abs.)" if j == 0 else None)
        v = c2[(c2.arm == "S-CONF_rel") & (c2.bin == k)].set_index("series").reindex(SERIES)["coverage"]
        ax.plot(xs + 0.08, v, ls="none", marker="s", ms=3.0, color="black", label="Chronos-2+T, S-CONF")
        v = c2[(c2.arm == "GUARDED") & (c2.bin == k)].set_index("series").reindex(SERIES)["coverage"]
        ax.plot(xs + 0.2, v, ls="none", marker="D", ms=3.0, color=C_TEMP, label="Chronos-2+T, TEMPCAL")
        ax.axhline(0.9, color="black", lw=0.6, ls=":")
        ax.set_ylabel("Coverage")
        ax.text(0.01, 0.06, title, transform=ax.transAxes, fontsize=8, va="bottom")
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    axes[0].set_ylim(0.4, 1.0)
    axes[1].set_ylim(0.85, 0.95)
    axes[2].set_ylim(0.7, 1.0)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(SERIES, rotation=90)
    hh, ll = axes[0].get_legend_handles_labels()  # order: univariate S-CONF, Chronos-2+T S-CONF, TEMPCAL
    fig.legend(hh, ll, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=0.9)
    fig.align_ylabels(axes)
    save(fig, "fig2_coverage")


def winter(d: pd.Series) -> pd.Series:
    y = d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)
    return y.astype(str).str[-2:] + "/" + (y + 1).astype(str).str[-2:]


def fig3() -> None:
    rows = []
    for arm, lab in (("S-CONF_rel", "S-CONF"), ("GUARDED", "TEMPCAL")):
        x = pd.read_parquet(RES / "figdata" / f"{arm}.parquet")
        x = x[x["bin"] == "cold"].copy()
        x["c"] = (x.y >= x.lo) & (x.y <= x.hi)
        x["winter"] = winter(x["issue_date"])
        x["grp"] = np.where(x.series == "ERCOT", "ERCOT", "8 zones")
        for (gname, w, band), g in x.groupby(["grp", "winter", "band"]):
            rows.append(dict(arm=lab, grp=gname, winter=w, band=band, cov=g.c.mean(), n=len(g)))
    r = pd.DataFrame(rows)
    r.to_csv(RES / "figdata" / "fig3_cold_by_winter.csv", index=False, float_format="%.4f")
    wins = sorted(r.winter.unique())
    bands = ["1-6", "7-24", "25-48"]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 45 * MM), sharey=True)
    xs = np.arange(len(wins))
    mk = {"1-6": "o", "7-24": "s", "25-48": "^"}
    for ax, gname in zip(axes, ("ERCOT", "8 zones")):
        for band in bands:
            for arm, col, ls, fill in (("S-CONF", C_SCONF, "--", "white"), ("TEMPCAL", C_TEMP, "-", C_TEMP)):
                v = r[(r.grp == gname) & (r.band == band) & (r.arm == arm)].set_index("winter").reindex(wins)["cov"]
                off = {"1-6": -0.12, "7-24": 0.0, "25-48": 0.12}[band]
                ax.plot(xs + off, v, ls=ls, color=col, marker=mk[band], ms=3.2, mfc=fill, mec=col, lw=0.8)
        ax.axhline(0.9, color="black", lw=0.6, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(wins)
        ax.set_xlim(-0.35, len(wins) - 0.65)
        ax.set_title(gname)
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    axes[0].set_ylabel("Cold-bin coverage")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=C_TEMP, ls="-", label="TEMPCAL"), Line2D([], [], color=C_SCONF, ls="--", label="S-CONF")]
    hs += [Line2D([], [], color="black", ls="none", marker=mk[b], ms=3.2, mfc="white", label=f"{b} h") for b in bands]
    fig.legend(handles=hs, loc="outside upper center", ncol=5, frameon=False, handlelength=1.5,
               columnspacing=0.8, handletextpad=0.3)
    save(fig, "fig3_winters")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1()
    fig2()
    fig3()
    print("figures written")
