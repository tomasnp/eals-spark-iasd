"""All the measurements of the report. One sub-command per experiment, each
writing a raw CSV in results/ ; figures are produced separately by plots.py so
that a re-plot never needs a re-run.

    python experiments.py strong --dataset ml-10m --k 32
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import baselines as B
import eals_rdd as R
import evaluate as EV
from data import load_split
from spark_utils import get_spark

RES = os.environ.get("EALS_RESULTS", os.path.expanduser("~/eals-spark/results"))
DEF = dict(K=32, lam=0.01, c0=512.0, alpha=0.4)


def write(name, rows):
    os.makedirs(RES, exist_ok=True)
    p = f"{RES}/{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {p}  ({len(rows)} rows)")
    return p


def dims(df):
    return (df.agg({"u": "max"}).collect()[0][0] + 1,
            df.agg({"i": "max"}).collect()[0][0] + 1)


def per_iter(hist, warmup=1):
    t = np.array([h["t_total"] for h in hist[warmup:]])
    return dict(t_med=float(np.median(t)), t_min=float(t.min()), t_max=float(t.max()),
                t_user=float(np.median([h["t_user"] for h in hist[warmup:]])),
                t_item=float(np.median([h["t_item"] for h in hist[warmup:]])),
                t_bcast=float(np.median([h["t_bcast"] for h in hist[warmup:]])),
                t_driver=float(np.median([h["t_driver"] for h in hist[warmup:]])))


# ---------------------------------------------------------------- datasets
def exp_stats(a):
    rows = []
    for d in ("ml-100k", "ml-1m", "ml-10m", "amazon-movies"):
        p = os.path.expanduser(f"~/eals-spark/data/processed/{d}/stats.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    write("datasets", rows)


# ------------------------------------------------------------ scalability
def exp_strong(a):
    """(a) Strong scaling: same data, 1/2/4/8 cores.

    With --interleave the repetitions are the *outer* loop, so a slow drift of
    the machine (thermal throttling over a long campaign) hits every core count
    equally instead of penalising whichever one runs last.
    """
    rows = []
    cfgs = [(c, r) for r in range(a.reps) for c in (1, 2, 4, 8)] if a.interleave \
        else [(c, r) for c in (1, 2, 4, 8) for r in range(a.reps)]
    for cores, rep in cfgs:
        if a.cooldown:
            time.sleep(a.cooldown)
        sp = get_spark(f"strong-{cores}", cores=cores)
        tr = load_split(sp, a.dataset, "train").cache()
        M, N = dims(tr)
        # partitions proportional to cores: constant work per task.
        nb = cores * a.pf
        _, _, _, h = R.fit(sp, tr, M, N, iters=a.iters, partitions=nb, **DEF)
        rows.append(dict(dataset=a.dataset, cores=cores, rep=rep,
                         partitions=nb, **per_iter(h)))
        sp.stop()
        print(rows[-1])
    write(f"strong_scaling{'_il' if a.interleave else ''}_{a.dataset}", rows)


def exp_factors(a):
    """(b) The decisive experiment: how the cost of one iteration grows with K.

    Two methodological details matter here, and we learned both the hard way.
    Repeating a heavy configuration three times in a row heats the machine, so the
    configurations that run last are penalised: --interleave makes the repetition
    the outer loop, and --cooldown inserts an idle pause between configurations.
    Without them two identical sweeps an hour apart differed by up to a factor two.
    """
    rows = []
    sp = get_spark("factors", cores=a.cores)
    tr = load_split(sp, a.dataset, "train").cache()
    M, N = dims(tr)
    ks = [int(k) for k in a.ks.split(",")]
    jobs = []
    for K in ks:
        reps = a.reps if K <= a.reps_max_k else 1
        jobs += [(K, r) for r in range(reps)]
    if a.interleave:
        jobs.sort(key=lambda x: (x[1], x[0]))
    for K, rep in jobs:
        if a.cooldown:
            time.sleep(a.cooldown)
        _, _, _, h = R.fit(sp, tr, M, N, iters=a.iters,
                           partitions=a.cores * a.pf, **dict(DEF, K=K))
        rows.append(dict(method="eALS-rdd", dataset=a.dataset, K=K, cores=a.cores,
                         rep=rep, **per_iter(h)))
        print(rows[-1])
    if a.with_als:
        for K in ks:
            if K > a.als_max_k:
                continue
            if a.cooldown:
                time.sleep(a.cooldown)
            # The marginal cost of one ALS iteration is (T_n - T_1)/(n-1). With a
            # small n the fixed block-setup cost dominates both terms and the
            # estimate is pure noise (it can even come out negative), so n must be
            # large enough for n-1 iterations to outweigh the setup.
            t_it, t1, tn = B.als_per_iter_seconds(tr, M, N, K, DEF["lam"],
                                                  iters=a.als_slope_iters,
                                                  blocks=a.cores * a.pf)
            rows.append(dict(method="MLlib-ALS", dataset=a.dataset, K=K, cores=a.cores,
                             rep=0, t_med=t_it, t_min=t_it, t_max=t_it,
                             t_user=0, t_item=0, t_bcast=0, t_driver=0))
            print(rows[-1])
    sp.stop()
    write(f"factor_scaling_{a.dataset}", rows)


# --------------------------------------------------------------- quality
def exp_quality(a):
    """eALS vs MLlib ALS vs eALS(alpha=0) vs MostPopular, and accuracy as a
    function of *wall-clock* time, not of iteration number."""
    rows, curves = [], []
    sp = get_spark("quality", cores=a.cores)
    tr = load_split(sp, a.dataset, "train").cache()
    te = load_split(sp, a.dataset, "test").cache()
    M, N = dims(tr)
    Pp, Qp = EV.most_popular_factors(tr, M, N)
    rows.append(dict(method="MostPopular", dataset=a.dataset, K=0, fit_s=0.0,
                     **EV.rank_metrics(sp, Pp, Qp, tr, te)))
    print(rows[-1])

    for name, alpha in (("eALS", DEF["alpha"]), ("eALS-uniform(alpha=0)", 0.0)):
        d = dict(DEF, alpha=alpha, K=a.k)
        P = Q = None
        t_acc = 0.0
        for step in a.als_iters:
            P, Q, c, h = R.fit(sp, tr, M, N, iters=step,
                               partitions=a.cores * a.pf, **d)
            t_acc = sum(x["t_total"] for x in h)
            m = EV.rank_metrics(sp, P, Q, tr, te)
            curves.append(dict(method=name, dataset=a.dataset, iter=step,
                               wall_s=t_acc, **m))
            print(curves[-1])
        rows.append(dict(method=name, dataset=a.dataset, K=a.k, fit_s=t_acc,
                         **EV.rank_metrics(sp, P, Q, tr, te)))

    for step in a.als_iters:
        Pa, Qa, ts = B.fit_mllib_als(tr, M, N, K=a.k, lam=DEF["lam"], iters=step,
                                     blocks=a.cores * a.pf)
        m = EV.rank_metrics(sp, Pa, Qa, tr, te)
        curves.append(dict(method="MLlib-ALS", dataset=a.dataset, iter=step,
                           wall_s=ts, **m))
        print(curves[-1])
        rows.append(dict(method="MLlib-ALS", dataset=a.dataset, K=a.k, fit_s=ts, **m))
    sp.stop()
    write(f"quality_{a.dataset}", rows)
    write(f"quality_curve_{a.dataset}", curves)


def exp_convergence(a):
    rows = []
    sp = get_spark("conv", cores=a.cores)
    tr = load_split(sp, a.dataset, "train").cache()
    te = load_split(sp, a.dataset, "test").cache()
    M, N = dims(tr)
    P, Q, c, h = R.fit(sp, tr, M, N, iters=a.iters, partitions=a.cores * a.pf,
                       eval_every=1, **dict(DEF, K=a.k))
    wall = 0.0
    for it, x in enumerate(h):
        wall += x["t_total"]
        rows.append(dict(dataset=a.dataset, iter=it + 1, wall_s=wall, obj=x["obj"]))
    write(f"convergence_{a.dataset}", rows)
    sp.stop()


EXPS = dict(stats=exp_stats, strong=exp_strong, factors=exp_factors,
            quality=exp_quality, convergence=exp_convergence)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=sorted(EXPS))
    ap.add_argument("--dataset", default="ml-10m")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--pf", type=int, default=2, help="partitions per core")
    ap.add_argument("--interleave", action="store_true",
                    help="repetitions as the outer loop, to spread machine drift")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--ks", default="8,16,32,64,128,256")
    ap.add_argument("--with-als", action="store_true")
    ap.add_argument("--als-max-k", type=int, default=128)
    ap.add_argument("--als-slope-iters", type=int, default=11)
    ap.add_argument("--reps-max-k", type=int, default=64)
    ap.add_argument("--cooldown", type=int, default=0,
                    help="idle seconds between configurations, to limit thermal drift")
    ap.add_argument("--als-iters", type=int, nargs="*", default=[1, 3, 5, 10, 20])
    a = ap.parse_args()
    EXPS[a.exp](a)
