"""Correctness gate: the Spark implementation must produce the same model as the
sequential NumPy reference, whatever the number of blocks.

Users are independent given (Q, S^q) and items given (P, S^p), so blocking the
data across workers cannot change the result -- only the order in which the
floating-point sums are accumulated. Any deviation beyond ~1e-12 means a real
bug, not numerical noise.
"""
import argparse

import numpy as np

import eals_local as L
import eals_rdd as R
from data import load_split
from spark_utils import get_spark


def brute_force_check(M=20, N=15, K=4, seed=7, lam=0.01, c0=64.0, alpha=0.4):
    """The update rules (12)/(13) are supposed to be the exact coordinate
    minimisers of the objective. Check that against Eq. (5)/(6) evaluated
    literally over the *whole* M x N matrix -- O(MNK), only usable on a toy."""
    rng = np.random.default_rng(seed)
    R = (rng.random((M, N)) < 0.3).astype(float)
    u_idx, i_idx = np.nonzero(R)
    cvec = L.item_confidence(R.sum(0), c0, alpha)
    PT = rng.normal(0, 0.3, (K, M))
    QT = rng.normal(0, 0.3, (K, N))
    W = np.where(R > 0, 1.0, cvec[None, :])           # w_ui

    # Eq. (5) applied literally to every (u, f), no caching, no sparsity trick.
    ref = PT.copy()
    for uu in range(M):
        for f in range(K):
            rhat = QT.T @ ref[:, uu]
            rf = rhat - ref[f, uu] * QT[f]
            num = ((R[uu] - rf) * W[uu] * QT[f]).sum()
            den = (W[uu] * QT[f] ** 2).sum() + lam
            ref[f, uu] = num / den

    fast = PT.copy()
    urow, uidx = L.sort_by_row(u_idx, i_idx)
    Sq = (QT * cvec) @ QT.T
    L.update_users(fast, QT, uidx, urow, cvec, Sq, lam)
    d_u = float(np.abs(fast - ref).max())

    ref = QT.copy()
    for ii in range(N):
        for f in range(K):
            rhat = PT.T @ ref[:, ii]
            rf = rhat - ref[f, ii] * PT[f]
            num = ((R[:, ii] - rf) * W[:, ii] * PT[f]).sum()
            den = (W[:, ii] * PT[f] ** 2).sum() + lam
            ref[f, ii] = num / den
    fast = QT.copy()
    irow, iidx = L.sort_by_row(i_idx, u_idx)
    Sp = PT @ PT.T
    L.update_items(fast, PT, iidx, irow, cvec, Sp, lam)
    d_i = float(np.abs(fast - ref).max())
    print(f"brute force vs Eq.(12): max|dP| = {d_u:.2e}")
    print(f"brute force vs Eq.(13): max|dQ| = {d_i:.2e}")
    return d_u < 1e-10 and d_i < 1e-10


def main(dataset="ml-100k", K=8, iters=5, cores=4):
    ok_bf = brute_force_check()
    print()
    sp = get_spark("check", cores=cores)
    tr = load_split(sp, dataset, "train").cache()
    pdf = tr.toPandas()
    u, i = pdf.u.to_numpy(np.int64), pdf.i.to_numpy(np.int64)
    M, N = int(u.max()) + 1, int(i.max()) + 1
    kw = dict(K=K, iters=iters, c0=512.0, alpha=0.4, lam=0.01)

    Pl, Ql, cl = L.fit(u, i, M, N, **kw)
    Jl = L.objective(np.ascontiguousarray(Pl.T), np.ascontiguousarray(Ql.T), u, i, cl, 0.01)
    rows = [("numpy-local", 1, 0.0, 0.0, Jl)]
    for p in (1, 2, 4, 8):
        P, Q, c, _ = R.fit(sp, tr, M, N, partitions=p, **kw)
        J = L.objective(np.ascontiguousarray(P.T), np.ascontiguousarray(Q.T), u, i, c, 0.01)
        rows.append((f"spark-rdd", p, np.abs(P - Pl).max(), np.abs(Q - Ql).max(), J))

    print(f"{'impl':<14}{'blocks':>7}{'max|dP|':>12}{'max|dQ|':>12}{'objective':>16}{'rel.err':>12}")
    for name, p, dp, dq, J in rows:
        print(f"{name:<14}{p:>7}{dp:>12.2e}{dq:>12.2e}{J:>16.6f}{abs(J - Jl) / Jl:>12.2e}")
    ok = ok_bf and all(r[2] < 1e-10 and r[3] < 1e-10 for r in rows)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sp.stop()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-100k")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--cores", type=int, default=4)
    a = ap.parse_args()
    raise SystemExit(0 if main(a.dataset, a.k, a.iters, a.cores) else 1)
