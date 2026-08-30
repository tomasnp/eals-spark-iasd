"""Baselines: MLlib's implicit ALS (the Hu et al. 2008 algorithm eALS is
compared against in the paper) and MostPopular.

MLlib returns its factors as DataFrames; they are pulled back into dense NumPy
matrices so that eALS and ALS are scored by the very same ranking code.
"""
import time

import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F


def fit_mllib_als(train_df, M, N, K=32, lam=0.01, alpha=1.0, iters=10,
                  seed=42, blocks=8):
    t0 = time.perf_counter()
    als = ALS(userCol="u", itemCol="i", ratingCol="r", implicitPrefs=True,
              rank=K, regParam=lam, alpha=alpha, maxIter=iters, seed=seed,
              numUserBlocks=blocks, numItemBlocks=blocks,
              coldStartStrategy="drop", intermediateStorageLevel="MEMORY_AND_DISK")
    m = als.fit(train_df.withColumn("r", F.lit(1.0)))
    fit_s = time.perf_counter() - t0
    P, Q = np.zeros((M, K)), np.zeros((N, K))
    for r in m.userFactors.collect():
        P[r["id"]] = r["features"]
    for r in m.itemFactors.collect():
        Q[r["id"]] = r["features"]
    return P, Q, fit_s


def als_per_iter_seconds(train_df, M, N, K, lam, iters=11, blocks=8, seed=42):
    """Marginal cost of one ALS iteration: fitting with 1 and with `iters`
    iterations and taking the slope cancels the fixed block-setup cost, which
    would otherwise dominate on small data."""
    _, _, t1 = fit_mllib_als(train_df, M, N, K, lam, iters=1, blocks=blocks, seed=seed)
    _, _, tn = fit_mllib_als(train_df, M, N, K, lam, iters=iters, blocks=blocks, seed=seed)
    return (tn - t1) / (iters - 1), t1, tn
