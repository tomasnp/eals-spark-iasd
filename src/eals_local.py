"""Single-machine NumPy eALS (He et al., SIGIR'16).

This module holds the two element-wise update kernels. `eals_rdd.py` imports
them unchanged and only differs in *who* calls them and on which slice of the
data -- so any accuracy difference between the local and the distributed run can
only come from the scheduling, never from the math.

Notation follows the paper: P (M x K) users, Q (N x K) items, r_ui = 1 and
w_ui = 1 on observed entries, w_ui = c_i on missing ones.
Factors are stored *transposed* (K x M / K x N) because the inner loop gathers
one whole coordinate f at a time; a row of the transposed matrix is contiguous.
"""
import numpy as np


def item_confidence(item_counts, c0=512.0, alpha=0.4):
    """Eq. (8): c_i = c0 * f_i^alpha / sum_j f_j^alpha, f_i = |R_i| / sum_j |R_j|."""
    f = item_counts / max(item_counts.sum(), 1.0)
    fa = f ** alpha
    return c0 * fa / fa.sum()


def sort_by_row(rows, cols):
    """Sorting the entries by row turns the per-coordinate gather on the row
    factors into a sequential scan instead of a random one."""
    o = np.argsort(rows, kind="stable")
    return rows[o].astype(np.int64), cols[o].astype(np.int64)


def _predict(FT, GT, rowid, idx):
    """r_hat on the observed entries only, without materialising an (nnz, K)
    gather: K passes of O(nnz) instead of one pass of O(nnz*K) memory."""
    out = np.zeros(rowid.shape[0])
    for f in range(FT.shape[0]):
        out += FT[f][rowid] * GT[f][idx]
    return out


def update_users(PT, QT, idx, rowid, cvec, Sq, lam, w=1.0, rhat=None):
    """Eq. (12) applied to every user of a block, one coordinate f at a time.

    Users are independent given (Q, Sq), so the whole block is updated with
    vector operations: the Gauss-Seidel sweep stays sequential over f, but is
    simultaneous over the users. Cost: O(nnz*K + m*K^2).
    """
    K, m = PT.shape
    c_e = cvec[idx]
    wmc = w - c_e                       # (w_ui - c_i)
    wr = w * 1.0                        # w_ui * r_ui, r_ui = 1
    if rhat is None:
        rhat = _predict(PT, QT, rowid, idx)
    for f in range(K):
        qf = QT[f][idx]
        rhat -= PT[f][rowid] * qf       # r_hat^f = r_hat - p_uf q_if
        num = np.bincount(rowid, (wr - wmc * rhat) * qf, minlength=m)
        den = np.bincount(rowid, wmc * qf * qf, minlength=m)
        num -= Sq[:, f] @ PT - PT[f] * Sq[f, f]     # - sum_{k!=f} p_uk s^q_kf
        den += Sq[f, f] + lam
        PT[f] = num / den
        rhat += PT[f][rowid] * qf
    return rhat


def update_items(QT_blk, PT, idx, rowid, c_blk, Sp, lam, w=1.0, rhat=None):
    """Eq. (13). Same structure as update_users, but the dense term and the
    denominator are scaled by c_i (which is an item property)."""
    K, n = QT_blk.shape
    c_e = c_blk[rowid]
    wmc = w - c_e
    wr = w * 1.0
    if rhat is None:
        rhat = _predict(QT_blk, PT, rowid, idx)
    for f in range(K):
        pf = PT[f][idx]
        rhat -= QT_blk[f][rowid] * pf
        num = np.bincount(rowid, (wr - wmc * rhat) * pf, minlength=n)
        den = np.bincount(rowid, wmc * pf * pf, minlength=n)
        num -= c_blk * (Sp[:, f] @ QT_blk - QT_blk[f] * Sp[f, f])
        den += c_blk * Sp[f, f] + lam
        QT_blk[f] = num / den
        rhat += QT_blk[f][rowid] * pf
    return rhat


def objective(PT, QT, u_idx, i_idx, cvec, lam, w=1.0):
    """Eq. (7) evaluated through Eq. (14) in O(|R| + M K^2) instead of O(MNK):
    the missing-data term becomes sum_u p_u^T S^q p_u - sum_{(u,i) in R} c_i r_ui^2."""
    rhat = _predict(PT, QT, u_idx, i_idx)
    Sq = (QT * cvec) @ QT.T
    obs = float((w * (1.0 - rhat) ** 2).sum())
    miss = float((PT * (Sq @ PT)).sum() - (cvec[i_idx] * rhat ** 2).sum())
    reg = lam * (float((PT ** 2).sum()) + float((QT ** 2).sum()))
    return obs + miss + reg


def fit(u_idx, i_idx, M, N, K=32, lam=0.01, c0=512.0, alpha=0.4, iters=20,
        seed=42, init_std=0.01, w=1.0, trace=None):
    """Algorithm 1, sequential reference implementation."""
    rng = np.random.default_rng(seed)
    PT = rng.normal(0, init_std, (K, M))
    QT = rng.normal(0, init_std, (K, N))
    cvec = item_confidence(np.bincount(i_idx, minlength=N).astype(float), c0, alpha)

    urow, uidx = sort_by_row(u_idx, i_idx)
    irow, iidx = sort_by_row(i_idx, u_idx)

    for it in range(iters):
        Sq = (QT * cvec) @ QT.T                       # Algorithm 1, line 4
        update_users(PT, QT, uidx, urow, cvec, Sq, lam, w)
        Sp = PT @ PT.T                                # Algorithm 1, line 12
        update_items(QT, PT, iidx, irow, cvec, Sp, lam, w)
        if trace is not None:
            trace.append(objective(PT, QT, u_idx, i_idx, cvec, lam, w))
    return PT.T.copy(), QT.T.copy(), cvec


if __name__ == "__main__":
    # Runnable without Spark at all: the reference implementation reads the
    # parquet produced by data.py straight into pandas.
    import argparse
    import glob
    import os
    import time

    import pandas as pd

    ap = argparse.ArgumentParser(description="single-machine NumPy eALS")
    ap.add_argument("--dataset", default="ml-100k")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--c0", type=float, default=512.0)
    ap.add_argument("--lam", type=float, default=0.01)
    a = ap.parse_args()

    proc = os.environ.get("EALS_PROC", os.path.expanduser("~/eals-spark/data/processed"))
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{proc}/{a.dataset}/train/*.parquet"))
    u = df.u.to_numpy(np.int64)
    i = df.i.to_numpy(np.int64)
    tr = []
    t0 = time.perf_counter()
    fit(u, i, int(u.max()) + 1, int(i.max()) + 1, K=a.k, lam=a.lam, c0=a.c0,
        alpha=a.alpha, iters=a.iters, trace=tr)
    print(f"{a.dataset}: |R|={len(u)}  K={a.k}  "
          f"{(time.perf_counter()-t0)/a.iters:.3f} s/iteration")
    for it, j in enumerate(tr, 1):
        print(f"  iter {it:3d}   J = {j:.4f}")
