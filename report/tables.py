"""Generates report/tables/*.tex and report/tables/macros.tex from results/*.csv.

Every number quoted in the report comes from here, so the text can never drift
away from the measurements.
"""
import os

import numpy as np
import pandas as pd

RES = os.path.expanduser("~/eals-spark/results")
OUT = os.path.dirname(os.path.abspath(__file__)) + "/tables"
os.makedirs(OUT, exist_ok=True)
# Every macro the report may reference. tables.py overwrites the ones whose
# experiment actually ran; the rest keep a visible placeholder so that a partial
# campaign still compiles instead of failing on an undefined control sequence.
MAC = {k: "\\textbf{[n/a]}" for k in (
    "CorrWorst "
    "SpeedupBig SerialBig ToneBig TeightBig SpeedupSmall SerialSmall ToneSmall TeightSmall "
    "SlopeEals SlopeAls "
    "SlopeEalsAmz SlopeAlsAmz SlopeHiEalsAmz SlopeHiAlsAmz "
    "SlopeEalsMl SlopeAlsMl SlopeHiEalsMl SlopeHiAlsMl CrossKAmz CrossKMl "
    "ConvItersAmz ConvItersMl AlsBestHR AlsTimeToBest EalsTimeToAls EalsOverAls "
    "EalsOverAlsBest PopGain PopFloor PopGainMl PopFloorMl EalsOverAlsMl "
    "HRe HRa HRu HReMl HRaMl HRuMl "
    "FitConstEalsAmz FitRtwoEalsAmz FitCrossEalsAmz FitConstAlsAmz FitRtwoAlsAmz "
    "FitCrossAlsAmz FitConstEalsMl FitRtwoEalsMl FitCrossEalsMl FitConstAlsMl "
    "FitRtwoAlsMl FitCrossAlsMl RatioAtMaxAmz KAtRatioAmz RatioAtMinAmz "
    "KAtRatioMinAmz FlipKAmz GrowthEalsAmz GrowthAlsAmz GrowthEalsMl GrowthAlsMl "
    "RatioAtMaxMl KAtRatioMl RatioAtMinMl KAtRatioMinMl FlipKMl").split()}


def load(n):
    p = f"{RES}/{n}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def put(name, body):
    open(f"{OUT}/{name}.tex", "w").write(body.rstrip("\n") + "\n")
    print("->", name)


def mac(k, v):
    MAC[k] = v


def missing(name, what):
    put(name, "\\multicolumn{1}{l}{\\emph{experiment \\texttt{%s} not run}} \\\\\n"
              % what.replace("_", "\\_"))


def t_datasets():
    d = load("datasets")
    if d is None:
        return missing("tab_datasets", "stats")
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{r['dataset']} & {int(r['users']):,} & {int(r['items']):,} & "
                    f"{int(r['interactions']):,} & {100*r['sparsity']:.2f}\\% & "
                    f"{int(r['train']):,} & {int(r['test']):,} \\\\")
    put("tab_datasets", "\n".join(rows).replace(",", "\\,"))


def t_correctness():
    p = f"{RES}/correctness.txt"
    if not os.path.exists(p):
        return missing("tab_correctness", "check_correctness")
    rows, worst = [], 0.0
    for ln in open(p).read().splitlines()[1:]:
        f = ln.split()
        if len(f) != 6 or not f[1].isdigit():
            continue
        rows.append(f"\\texttt{{{f[0]}}} & {f[1]} & {f[2]} & {f[3]} & {f[4]} & {f[5]} \\\\")
        worst = max(worst, float(f[2]), float(f[3]))
    put("tab_correctness", "\n".join(rows))
    mac("CorrWorst", f"{worst:.0e}".replace("e-", "\\cdot 10^{-") + "}")


def _amdahl(p, t):
    A = np.column_stack([np.ones_like(p, dtype=float), 1.0 / p])
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    return max(0.0, min(1.0, a / (a + b)))


def t_strong():
    body = []
    for ds in ("ml-1m", "ml-10m"):
        for tag in ("_il", ""):
            d = load(f"strong_scaling{tag}_{ds}")
            if d is None or (tag == "" and load(f"strong_scaling_il_{ds}") is not None):
                continue
            g = d.groupby("cores").t_med
            med, lo, hi = g.median(), g.min(), g.max()
            p = med.index.to_numpy(float)
            sp = med.iloc[0] / med
            s = _amdahl(p, med.to_numpy())
            body.append("\\multicolumn{5}{l}{\\textit{%s, B = 2p} - fitted Amdahl "
                        "serial fraction $s=%.3f$}\\\\" % (ds, s))
            for c in med.index:
                body.append(f"\\quad {c} & {med[c]:.3f} & {lo[c]:.3f}--{hi[c]:.3f} & "
                            f"{sp[c]:.2f} & {100*sp[c]/c:.0f}\\% \\\\")
            k = "Big" if ds == "ml-10m" else "Small"
            mac("Speedup" + k, f"{sp.iloc[-1]:.2f}")
            mac("Serial" + k, f"{s:.3f}")
            mac("Tone" + k, f"{med.iloc[0]:.2f}")
            mac("Teight" + k, f"{med.iloc[-1]:.2f}")
    if not body:
        return missing("tab_strong", "strong")
    put("tab_strong", "\n".join(body))


def _poly_fit(K, t, powers):
    """Least-squares fit of t = a + sum_p b_p K^p, returning (a, {p: b_p}, R^2)."""
    K = np.asarray(K, float)
    A = np.column_stack([np.ones_like(K)] + [K ** p for p in powers])
    c = np.linalg.lstsq(A, t, rcond=None)[0]
    pred = A @ c
    ss = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return c[0], dict(zip(powers, c[1:])), ss


def t_factors():
    dss = [d for d in ("amazon-movies", "ml-1m") if load(f"factor_scaling_{d}") is not None]
    if not dss:
        return missing("tab_factors", "factors")
    rows = []
    for ds in dss:
        d = load(f"factor_scaling_{ds}")
        piv = d[d.t_med > 0].groupby(["K", "method"]).t_med.median().unstack()
        rows.append("\\multicolumn{4}{l}{\\textit{%s}}\\\\" % ds)
        for K, r in piv.iterrows():
            e = r.get("eALS-rdd", float("nan"))
            a = r.get("MLlib-ALS", float("nan"))
            rows.append(f"\\quad {K} & {e:.2f} & " +
                        (f"{a:.2f} & {a/e:.1f}$\\times$ \\\\" if a == a
                         else "- & - \\\\"))
        key = "Amz" if ds == "amazon-movies" else "Ml"
        # The theory says eALS costs a + b|R|K + c(M+N)K^2 and ALS a + bK^2 + cK^3.
        # Fitting those forms is more informative than a single log-log slope, and it
        # gives the K at which the higher-order term takes over.
        for m, mk, pw in (("eALS-rdd", "Eals", (1, 2)), ("MLlib-ALS", "Als", (2, 3))):
            if m not in piv:
                continue
            g = piv[m].dropna()
            if len(g) < 4:
                continue
            a0, b, r2 = _poly_fit(g.index.to_numpy(), g.to_numpy(), pw)
            lo, hi = pw
            mac(f"FitConst{mk}{key}", f"{a0:.2f}")
            mac(f"FitRtwo{mk}{key}", f"{r2:.4f}")
            if b[hi] > 0 and b[lo] > 0:
                mac(f"FitCross{mk}{key}", f"{b[lo]/b[hi]:.0f}")
        for m, mk in (("eALS-rdd", "Eals"), ("MLlib-ALS", "Als")):
            if m not in piv:
                continue
            g = piv[m].dropna()
            mac(f"Slope{mk}{key}",
                f"{np.polyfit(np.log(g.index), np.log(g.values), 1)[0]:.2f}")
            hi = g[g.index >= g.index.max() / 4]
            if len(hi) > 1:
                mac(f"SlopeHi{mk}{key}",
                    f"{np.polyfit(np.log(hi.index), np.log(hi.values), 1)[0]:.2f}")
    put("tab_factors", "\n".join(rows))
    for ds, key in (("amazon-movies", "Amz"), ("ml-1m", "Ml")):
        d = load(f"factor_scaling_{ds}")
        if d is None:
            continue
        piv = d[d.t_med > 0].groupby(["K", "method"]).t_med.median().unstack()
        if "MLlib-ALS" in piv and "eALS-rdd" in piv:
            r = (piv["MLlib-ALS"] / piv["eALS-rdd"]).dropna()
            mac("RatioAtMax" + key, f"{r.iloc[-1]:.1f}")
            mac("KAtRatio" + key, f"{r.index[-1]:g}")
            mac("RatioAtMin" + key, f"{1/r.iloc[0]:.1f}")
            mac("KAtRatioMin" + key, f"{r.index[0]:g}")
            fl = r[r > 1]
            if len(fl):
                mac("FlipK" + key, f"{fl.index[0]:g}")
        # growth factor over a fixed K window: robust to the additive constant and
        # to the implementation language, unlike a fitted exponent
        for m, mk in (("eALS-rdd", "Eals"), ("MLlib-ALS", "Als")):
            if m in piv and 32 in piv.index and 128 in piv.index:
                g = piv[m]
                if g.get(32) == g.get(32) and g.get(128) == g.get(128):
                    mac("Growth" + mk + key, f"{g[128]/g[32]:.1f}")
    # the report quotes the Amazon numbers as the headline ones
    for a, b in (("SlopeEals", "SlopeHiEalsAmz"), ("SlopeAls", "SlopeHiAlsAmz")):
        if b in MAC and "n/a" not in MAC[b]:
            mac(a, MAC[b])


def t_crossk():
    """K* = 2|R|/(M+N): above it the (M+N)K^2 term of eALS overtakes the
    |R|K term, i.e. this is where the quadratic regime actually starts."""
    d = load("datasets")
    if d is None:
        return
    for ds, key in (("amazon-movies", "Amz"), ("ml-1m", "Ml")):
        r = d[d.dataset == ds]
        if len(r):
            r = r.iloc[0]
            mac("CrossK" + key,
                f"{2*r['train']/(r['users']+r['items']):.0f}")


def t_convergence():
    """How many iterations to get within 1% of the objective at 30 iterations."""
    rows = []
    for ds in ("ml-1m", "amazon-movies"):
        d = load(f"convergence_{ds}")
        if d is None:
            continue
        d = d.sort_values("iter")
        jf = d.obj.iloc[-1]
        k = int(d[d.obj <= 1.01 * jf].iter.min())
        rows.append(f"{ds} & {d.obj.iloc[0]:.0f} & {jf:.0f} & "
                    f"{100*(1 - jf/d.obj.iloc[0]):.1f}\\% & {k} & "
                    f"{d[d.iter == k].wall_s.iloc[0]:.1f} \\\\")
        mac("ConvIters" + ("Amz" if "amazon" in ds else "Ml"), str(k))
    if not rows:
        return missing("tab_convergence", "convergence")
    put("tab_convergence", "\n".join(rows))


def t_qualitytime():
    """How long eALS needs to reach the best accuracy MLlib ALS ever gets to."""
    d = load("quality_curve_amazon-movies")
    if d is None:
        return
    als = d[d.method == "MLlib-ALS"]
    ea = d[d.method == "eALS"].sort_values("wall_s")
    if not len(als) or not len(ea):
        return
    best = als["HR@100"].max()
    mac("AlsBestHR", f"{best:.4f}")
    mac("AlsTimeToBest", f"{als.loc[als['HR@100'].idxmax()].wall_s:.1f}")
    hit = ea[ea["HR@100"] >= best]
    if len(hit):
        mac("EalsTimeToAls", f"{hit.wall_s.iloc[0]:.1f}")
    mac("EalsOverAlsBest", f"{100*(ea['HR@100'].max()/best - 1):.0f}")


def t_quality():
    rows = []
    for ds in ("ml-1m", "amazon-movies"):
        d = load(f"quality_{ds}")
        if d is None:
            continue
        rows.append("\\multicolumn{6}{l}{\\textit{%s}}\\\\" % ds)
        d = d.drop_duplicates("method", keep="last")
        for _, r in d.iterrows():
            rows.append(f"\\quad {r.method} & {r['HR@10']:.4f} & {r['HR@100']:.4f} & "
                        f"{r['NDCG@10']:.4f} & {r['NDCG@100']:.4f} & {r.fit_s:.1f} \\\\")
        g = d.set_index("method")["HR@100"]
        k = "" if ds == "amazon-movies" else "Ml"
        if "eALS" in g and "eALS-uniform(alpha=0)" in g:
            mac("PopGain" + k, f"{100*(g['eALS']/g['eALS-uniform(alpha=0)']-1):.1f}")
        if "eALS" in g and "MostPopular" in g:
            mac("PopFloor" + k, f"{g['eALS']/g['MostPopular']:.1f}")
        if "eALS" in g and "MLlib-ALS" in g:
            mac("EalsOverAls" + k, f"{100*(g['eALS']/g['MLlib-ALS']-1):.0f}")
            mac("HRe" + k, f"{g['eALS']:.4f}")
            mac("HRa" + k, f"{g['MLlib-ALS']:.4f}")
            mac("HRu" + k, f"{g['eALS-uniform(alpha=0)']:.4f}")
    if not rows:
        return missing("tab_quality", "quality")
    put("tab_quality", "\n".join(rows))


if __name__ == "__main__":
    for f in (t_datasets, t_correctness, t_strong, t_factors, t_crossk,
              t_convergence, t_qualitytime, t_quality):
        try:
            f()
        except Exception as e:
            print(f"skip {f.__name__}: {type(e).__name__}: {e}")
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items()))
    open(f"{OUT}/macros.tex", "w").write(body + "\n")
    print(f"-> macros.tex ({len(MAC)} macros)")
    for k, v in sorted(MAC.items()):
        print(f"   \\{k} = {v}")
