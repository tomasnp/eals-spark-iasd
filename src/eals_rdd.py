"""eALS on Spark RDDs.

Parallelisation is *exact*, not approximate. Section 4.2 of the paper: given Q
and S^q, the updates of the M user vectors touch disjoint parameters and read
only read-only state, so splitting users across workers reproduces the
sequential result up to the order of the sweep. Same for items given P, S^p.

One iteration:
  1. broadcast Q and S^q               (read-only for the whole user stage)
  2. one task per user block -> new P rows + that block's partial P^T P
  3. reduce the partials into S^p, assemble P on the driver, broadcast it
  4. one task per item block -> new Q rows + that block's partial sum c_i q_i q_i^T
  5. reduce into S^q for the next iteration

Steps 2/4 return the S partials for free, so the caches of Algorithm 1 lines 4
and 12 never need a separate pass over the factors.
"""
import argparse
import json
import time

import numpy as np

from eals_local import item_confidence, update_items, update_users
from spark_utils import get_spark


def build_blocks(df, key, val, n_parts):
    """One RDD element per partition holding the whole block as NumPy arrays:
    (global ids of the rows, local row of each entry, column of each entry).

    Built once and cached: rebuilding it every iteration would re-shuffle the
    whole interaction matrix K times for nothing.
    """
    def to_arrays(it):
        ks, vs = [], []
        for k, v in it:
            ks.append(k)
            vs.append(v)
        if not ks:
            return iter(())
        ks = np.asarray(ks, dtype=np.int64)
        vs = np.asarray(vs, dtype=np.int64)
        ids, rowid = np.unique(ks, return_inverse=True)
        o = np.argsort(rowid, kind="stable")
        return iter([(ids, rowid[o].astype(np.int64), vs[o])])

    rdd = df.select(key, val).rdd.map(lambda r: (r[0], r[1]))
    return rdd.partitionBy(n_parts).mapPartitions(to_arrays, preservesPartitioning=True)


def _user_task(blk, bcP, bcQ, bcC, lam, w, Sq):
    ids, rowid, idx = blk
    PT = np.ascontiguousarray(bcP.value[:, ids])       # this block's rows of P
    update_users(PT, bcQ.value, idx, rowid, bcC.value, Sq, lam, w)
    return ids, PT, PT @ PT.T                          # partial S^p = P_blk^T P_blk


def _item_task(blk, bcP, bcQ, bcC, lam, w, Sp):
    ids, rowid, idx = blk
    QT = np.ascontiguousarray(bcQ.value[:, ids])
    c_blk = bcC.value[ids]
    update_items(QT, bcP.value, idx, rowid, c_blk, Sp, lam, w)
    return ids, QT, (QT * c_blk) @ QT.T                # partial S^q = sum c_i q_i q_i^T


def _objective_task(blk, bcP, bcQ, bcC, Sq, w):
    """Eq. (7) through Eq. (14), split by user block."""
    ids, rowid, idx = blk
    PT = bcP.value[:, ids]
    QT, c = bcQ.value, bcC.value
    rhat = np.zeros(rowid.shape[0])
    for f in range(PT.shape[0]):
        rhat += PT[f][rowid] * QT[f][idx]
    obs = float((w * (1.0 - rhat) ** 2).sum())
    miss = float((PT * (Sq @ PT)).sum() - (c[idx] * rhat ** 2).sum())
    return obs + miss


def fit(spark, train_df, M, N, K=32, lam=0.01, c0=512.0, alpha=0.4, iters=20,
        partitions=8, seed=42, w=1.0, init_std=0.01, s_cache="fused",
        eval_every=0, timing=None):
    """Returns (P, Q, c, history). `history` has one row per iteration with the
    wall time of each phase, used by every scalability experiment."""
    sc = spark.sparkContext
    counts = np.zeros(N)
    for i, n in train_df.groupBy("i").count().collect():
        counts[i] = n
    cvec = item_confidence(counts, c0, alpha)
    bcC = sc.broadcast(cvec)

    ublocks = build_blocks(train_df, "u", "i", partitions).cache()
    iblocks = build_blocks(train_df, "i", "u", partitions).cache()
    ublocks.count(), iblocks.count()                   # force the shuffle once

    rng = np.random.default_rng(seed)
    PT = rng.normal(0, init_std, (K, M))
    QT = rng.normal(0, init_std, (K, N))
    Sq = (QT * cvec) @ QT.T
    bcP = sc.broadcast(PT)

    hist = []
    for it in range(iters):
        t0 = time.perf_counter()
        bcQ = sc.broadcast(QT)
        Squ = Sq
        t_bc1 = time.perf_counter()

        res = ublocks.map(lambda b: _user_task(b, bcP, bcQ, bcC, lam, w, Squ)).collect()
        t_u = time.perf_counter()
        Sp = np.zeros((K, K))
        for ids, blk, sp in res:
            PT[:, ids] = blk
            Sp += sp
        if s_cache == "driver":
            Sp = PT @ PT.T
        bcP.unpersist()
        bcP = sc.broadcast(PT)
        t_bc2 = time.perf_counter()

        res = iblocks.map(lambda b: _item_task(b, bcP, bcQ, bcC, lam, w, Sp)).collect()
        t_i = time.perf_counter()
        Sq = np.zeros((K, K))
        for ids, blk, sq in res:
            QT[:, ids] = blk
            Sq += sq
        if s_cache == "driver":
            Sq = (QT * cvec) @ QT.T
        bcQ.unpersist()
        t_end = time.perf_counter()

        row = dict(iter=it, t_total=t_end - t0, t_user=t_u - t_bc1, t_item=t_i - t_bc2,
                   t_bcast=(t_bc1 - t0) + (t_bc2 - t_u), t_driver=(t_end - t_i))
        if eval_every and (it + 1) % eval_every == 0:
            bcQ2 = sc.broadcast(QT)
            Sq2 = Sq
            j = ublocks.map(lambda b: _objective_task(b, bcP, bcQ2, bcC, Sq2, w)).sum()
            row["obj"] = j + lam * (float((PT ** 2).sum()) + float((QT ** 2).sum()))
            bcQ2.unpersist()
        hist.append(row)
        if timing:
            print(json.dumps(row))

    ublocks.unpersist()
    iblocks.unpersist()
    bcP.unpersist()
    bcC.unpersist()
    return PT.T.copy(), QT.T.copy(), cvec, hist


if __name__ == "__main__":
    from data import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--partitions", type=int, default=None)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--c0", type=float, default=512.0)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=1)
    a = ap.parse_args()

    sp = get_spark(f"eals-{a.dataset}", cores=a.cores)
    tr = load_split(sp, a.dataset, "train").cache()
    M = tr.agg({"u": "max"}).collect()[0][0] + 1
    N = tr.agg({"i": "max"}).collect()[0][0] + 1
    P, Q, c, hist = fit(sp, tr, M, N, K=a.k, lam=a.lam, c0=a.c0, alpha=a.alpha,
                        iters=a.iters, partitions=a.partitions or a.cores * 2,
                        eval_every=a.eval_every, timing=True)
    sp.stop()
