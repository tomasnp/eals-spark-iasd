"""Every figure of the report, regenerated from results/*.csv only.
Re-plotting never re-runs an experiment.

    python plots.py            # all figures that have data
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = os.path.expanduser("~/eals-spark/results")
FIG = f"{RES}/figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "legend.frameon": False,
                     "savefig.bbox": "tight"})
C = {"eALS": "#1f77b4", "ALS": "#d62728"}


def load(name):
    p = f"{RES}/{name}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png")
    plt.close(fig)
    print("->", name)


def slope(x, y):
    """Exponent of the power law y = a x^b, fitted in log-log."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def amdahl_fit(p, t):
    """T(p) = a + b/p  ->  serial fraction s = a/(a+b) with T(1) = a+b."""
    A = np.column_stack([np.ones_like(p, dtype=float), 1.0 / p])
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    return max(0.0, min(1.0, a / (a + b))), a, b


def fig_strong():
    runs = []
    for ds in ("ml-1m", "ml-10m"):
        for tag in ("_il", ""):
            d = load(f"strong_scaling{tag}_{ds}")
            # the interleaved run supersedes the plain one when both exist
            if d is None or (tag == "" and load(f"strong_scaling_il_{ds}") is not None):
                continue
            runs.append((f"{ds}, $B=2p$", d))
    if not runs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for name, d in runs:
        g = d.groupby("cores").t_med
        med, lo, hi = g.median(), g.min(), g.max()
        p = med.index.to_numpy(float)
        ax[0].errorbar(p, med, yerr=[med - lo, hi - med], marker="o",
                       capsize=3, label=name)
        sp = med.iloc[0] / med
        s, _, _ = amdahl_fit(p, med.to_numpy())
        ax[1].plot(p, sp, marker="o", label=f"{name}  ($s={s:.3f}$)")
        ax[2].plot(p, sp / p, marker="o", label=name)
    pp = np.array([1, 2, 4, 8])
    ax[1].plot(pp, pp, ":", c="k", lw=1.2, label="ideal")
    ax[2].axhline(1, ls=":", c="k", lw=1.2)
    for a, t, yl in zip(ax, ("time per iteration", "speedup $S(p)=T_1/T_p$",
                             "efficiency $E(p)=S(p)/p$"), ("seconds", r"$\times$", "")):
        a.set_xlabel("Spark cores $p$ (local[p])")
        a.set_title(t)
        a.set_ylabel(yl)
        a.set_xscale("log", base=2)
        a.set_xticks(pp), a.set_xticklabels(pp)
        a.legend(fontsize=6.5)
    ax[0].set_yscale("log")
    save(fig, "strong_scaling")


def fig_factors():
    dss = [d for d in ("amazon-movies", "ml-1m") if load(f"factor_scaling_{d}") is not None]
    if not dss:
        return
    fig, ax = plt.subplots(1, len(dss), figsize=(4.9 * len(dss), 3.5), squeeze=False)
    for a, ds in zip(ax[0], dss):
        d = load(f"factor_scaling_{ds}")
        ref = None
        for m, sub in d.groupby("method"):
            g = sub[sub.t_med > 0].groupby("K").t_med.median()
            if len(g) < 2:
                continue
            b = slope(g.index.to_numpy(), g.to_numpy())
            hi = g[g.index >= g.index.max() / 4]           # top two octaves
            bh = slope(hi.index.to_numpy(), hi.to_numpy()) if len(hi) > 1 else b
            a.plot(g.index, g.values, marker="o",
                   c=C["eALS"] if "eALS" in m else C["ALS"],
                   label=f"{m}: slope {b:.2f} (top range {bh:.2f})")
            if "eALS" in m:
                ref = g
        lo, hi_ = d[d.t_med > 0].t_med.min(), d.t_med.max()
        if ref is not None:
            k = np.array([ref.index.min(), ref.index.max()], dtype=float)
            for e, st in ((1, "-."), (2, "--"), (3, ":")):
                a.plot(k, ref.iloc[0] * (k / k[0]) ** e, st, c="grey", lw=1,
                       label=f"$K^{e}$")
        a.set_ylim(lo / 2, hi_ * 3)          # keep the guides from rescaling the plot
        a.set_xscale("log", base=2), a.set_yscale("log")
        a.set_xlabel("number of latent factors K")
        a.set_ylabel("seconds / iteration")
        a.set_title(f"{ds}, 8 cores")
        a.legend(fontsize=6.5)
    save(fig, "factor_scaling")


def fig_convergence():
    if all(load(f"convergence_{d}") is None for d in ("ml-1m", "amazon-movies")):
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for j, ds in enumerate(("ml-1m", "amazon-movies")):
        d = load(f"convergence_{ds}")
        if d is None:
            continue
        ax[j].plot(d.iter, d.obj, marker=".", c=C["eALS"])
        ax[j].set_xlabel("iteration"), ax[j].set_ylabel("objective $J$")
        ax[j].set_title(f"{ds}  (K=32, $c_0$=512, $\\alpha$=0.4)")
        ax[j].set_yscale("log")
    save(fig, "convergence")


def fig_quality_time():
    if all(load(f"quality_curve_{d}") is None for d in ("ml-1m", "amazon-movies")):
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for j, ds in enumerate(("ml-1m", "amazon-movies")):
        d = load(f"quality_curve_{ds}")
        if d is None:
            continue
        for m, sub in d.groupby("method"):
            sub = sub.sort_values("wall_s")
            ax[j].plot(sub.wall_s, sub["HR@100"], marker="o", ms=3, label=m)
        ax[j].set_xlabel("wall-clock training time (s)")
        ax[j].set_ylabel("HR@100")
        ax[j].set_title(ds)
        ax[j].set_xscale("log")
        ax[j].legend(fontsize=7)
    save(fig, "quality_vs_time")


if __name__ == "__main__":
    for f in (fig_strong, fig_factors, fig_convergence, fig_quality_time):
        try:
            f()
        except Exception as e:            # a missing experiment must not stop the rest
            print(f"skip {f.__name__}: {type(e).__name__}: {e}")
