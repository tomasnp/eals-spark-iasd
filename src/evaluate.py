"""Offline evaluation: leave-one-out ranking, HR@k and NDCG@k.

The held-out item is ranked against *all* items the user has not interacted
with in training -- the full protocol of the paper, not the cheap
"100 sampled negatives" variant, which is known to distort the ranking of
recommenders. Cost is O(M N K), so it is done in Spark, block by block.
"""
import numpy as np
from pyspark.sql import functions as F


def _block_eval(rows, bcP, bcQ, ks, block):
    P, Q = bcP.value, bcQ.value           # (M, K) and (N, K)
    N = Q.shape[0]
    buf = []
    out = []

    def flush():
        if not buf:
            return
        us = np.array([r[0] for r in buf], dtype=np.int64)
        tis = np.array([r[2] for r in buf], dtype=np.int64)
        S = P[us] @ Q.T                                   # (b, N) BLAS matmul
        for b, r in enumerate(buf):
            tr = np.asarray(r[1], dtype=np.int64)
            S[b, tr] = -np.inf                            # never re-recommend
        st = S[np.arange(len(buf)), tis].copy()
        rank = (S > st[:, None]).sum(1) + 1               # 1-based rank of the GT item
        for rk in rank:
            out.append(tuple(
                [1.0 if rk <= k else 0.0 for k in ks]
                + [1.0 / np.log2(rk + 1) if rk <= k else 0.0 for k in ks]))
        buf.clear()

    for r in rows:
        buf.append((r[0], r[1], r[2]))
        if len(buf) >= block:
            flush()
            for o in out:
                yield o
            out = []
    flush()
    for o in out:
        yield o


def rank_metrics(spark, P, Q, train_df, test_df, ks=(10, 100), block=256):
    """-> {'HR@10':..., 'NDCG@10':..., ...} averaged over test users."""
    sc = spark.sparkContext
    bcP, bcQ = sc.broadcast(np.ascontiguousarray(P)), sc.broadcast(np.ascontiguousarray(Q))
    seen = train_df.groupBy("u").agg(F.collect_list("i").alias("seen"))
    joined = test_df.select("u", F.col("i").alias("gt")).join(seen, "u", "left")
    rdd = joined.rdd.map(lambda r: (r["u"], r["seen"] or [], r["gt"]))
    n = rdd.count()
    agg = (rdd.mapPartitions(lambda it: _block_eval(it, bcP, bcQ, ks, block))
              .treeAggregate(np.zeros(2 * len(ks)),
                             lambda a, b: a + np.asarray(b), lambda a, b: a + b))
    bcP.unpersist(), bcQ.unpersist()
    names = [f"HR@{k}" for k in ks] + [f"NDCG@{k}" for k in ks]
    return dict(zip(names, (agg / n).tolist()))


def most_popular_factors(train_df, M, N):
    """MostPopular expressed as a rank-1 MF model, so the trivial baseline goes
    through exactly the same ranking code as eALS."""
    cnt = np.zeros((N, 1))
    for i, c in train_df.groupBy("i").count().collect():
        cnt[i, 0] = c
    return np.ones((M, 1)), cnt
