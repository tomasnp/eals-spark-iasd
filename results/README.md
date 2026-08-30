# Raw results

One CSV per experiment, written by `src/experiments.py`; figures in `figures/` are
produced from these files by `src/plots.py`, and the LaTeX tables of the report by
`report/tables.py`. Nothing here is edited by hand.

Common timing columns, all in seconds and all **medians over the iterations of a run
after discarding the first (warm-up)**: `t_med` (whole iteration), `t_min`/`t_max`
(spread over those iterations), `t_user` / `t_item` (the two update stages),
`t_bcast` (creating the two broadcasts plus scattering the collected `P` blocks),
`t_driver` (scattering `Q` and reducing `S^q`). `rep` indexes the repetitions of the
same configuration.

| File | Produced by | Columns of interest |
|---|---|---|
| `datasets.csv` | `experiments.py stats` | `users, items, interactions, sparsity, train, test` |
| `correctness.txt` | `check_correctness.py` | brute-force check, then max abs. deviation of the Spark run against the NumPy reference |
| `strong_scaling_*.csv` | `experiments.py strong` | `cores, partitions` + timings. Suffix `_il` = repetitions interleaved |
| `factor_scaling_*.csv` | `experiments.py factors --with-als` | `method` (`eALS-rdd` / `MLlib-ALS`), `K` + timings |
| `convergence_*.csv` | `experiments.py convergence` | `iter, wall_s, obj` |
| `quality_*.csv` | `experiments.py quality` | `method, HR@10, HR@100, NDCG@10, NDCG@100, fit_s` |
| `quality_curve_*.csv` | `experiments.py quality` | same, one row per training length: accuracy against wall-clock time |
| `run_all.log` | `scripts/run_all.sh` | the campaign log, including the per-configuration prints |

Machine: Apple M-series, 8 physical cores, 16 GB, macOS 15, Spark 4.1.0 in
`local[p]`, OpenJDK 17, Python 3.11, NumPy 2.3, BLAS pinned to one thread
(`scripts/env.sh`). Seed 42 everywhere.
