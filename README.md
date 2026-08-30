# eALS on Spark — Fast Matrix Factorization for Online Recommendation with Implicit Feedback

### 📄 [**Rapport du projet (PDF, 43 pages, français)**](Rapport_Sinapi_Ernadote_eALS_Spark.pdf)

Thomas Sinapi et Anaël Ernadote — M2 IASD, Université Paris-Dauphine PSL, 2026.

---

PySpark implementation and experimental study of **eALS** (element-wise Alternating
Least Squares), from:

> X. He, H. Zhang, M.-Y. Kan, T.-S. Chua.
> *Fast Matrix Factorization for Online Recommendation with Implicit Feedback.*
> SIGIR 2016. [arXiv:1708.05024](https://arxiv.org/abs/1708.05024)

The paper proves that eALS is *embarrassingly parallel* but ships no
MapReduce/Spark implementation. This repository provides one, alongside the
sequential reference it must agree with:

| module | what it is | why it exists |
|---|---|---|
| `src/eals_local.py` | single-machine NumPy eALS | reference implementation + the two update kernels reused by the Spark version |
| `src/eals_rdd.py`   | Spark **RDD** eALS, broadcast-based | the distributed implementation |

`src/check_correctness.py` asserts that the two produce the **same model** (max
abs. difference `< 1e-13` on ml-100k, bit-identical with a single block) — the
parallelisation is exact, not approximate.

> **Scope.** One Spark implementation (RDD + broadcast), the offline
> leave-one-out protocol, and five experiments: correctness, strong scaling,
> scaling in the number of latent factors $K$, convergence, and accuracy against
> MLlib ALS and a popularity baseline. Section 2.3 of the report explains why the
> RDD + broadcast route was chosen over a DataFrame + shuffle join, and Section 6
> discusses what this choice rules out. The online update of Algorithm 2 is
> described in the report but not implemented.

> **Language.** The report (`Rapport_Sinapi_Ernadote_eALS_Spark.pdf`, 43 pages) and the notebook are in
> **French**. This README and the source comments are in English.

---

## 1. Install

### 1.1 The JDK trap (read this first)

macOS ships no JDK, and the JDK you may already have is very likely **too new**.
Spark 4.x supports **Java 17 or 21 only**. On Java 24+ the Security Manager is
permanently disabled (JEP 486), Hadoop's `UserGroupInformation` calls
`Subject.getSubject(...)`, and every job dies with:

```
UnsupportedOperationException: getSubject is not supported
```

`-Djava.security.manager=allow` does **not** help on JDK 24+ — the JVM refuses
to start at all. Install a supported JDK and point `JAVA_HOME` at it:

```bash
brew install openjdk@17
```

`scripts/env.sh` does that (and pins the BLAS thread count, see below):

```bash
source scripts/env.sh
```

### 1.2 Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1.3 Why `OMP_NUM_THREADS=1`

Each Spark task runs NumPy, and NumPy's BLAS spawns its own threads. On
`local[8]` that is 8 tasks × 8 BLAS threads = 64 threads on 8 cores: the machine
thrashes and every scalability number becomes meaningless. `scripts/env.sh`
pins all BLAS thread pools to 1 so that *Spark* is the only source of
parallelism.

---

## 2. Data

Two families are used, and the report says why:

* **Amazon Movies & TV** — the dataset of the paper (SNAP / McAuley Amazon
  product graph, the source cited in the paper). 4.6M ratings, 10-core filtered
  to 959k interactions. Used for accuracy.
* **MovieLens 100k / 1M / 10M** — a clean ×100 size ladder on a single machine,
  used for the scalability curves.

```bash
mkdir -p data/raw && cd data/raw
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
curl -O https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip -o 'ml-*.zip'
curl -O https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv
cd ../..
```

Then pre-process (10-core filtering, dense re-indexing, leave-one-out split, all
in Spark):

```bash
source scripts/env.sh
cd src
python3 data.py --dataset ml-100k       --kcore 10
python3 data.py --dataset ml-1m         --kcore 10
python3 data.py --dataset ml-10m        --kcore 10
python3 data.py --dataset amazon-movies --kcore 10
```

| dataset | users M | items N | interactions \|R\| | sparsity |
|---|---:|---:|---:|---:|
| ml-100k       |    943 |  1 152 |    97 953 | 90.98 % |
| ml-1m         |  6 040 |  3 260 |   998 539 | 94.93 % |
| ml-10m        | 69 878 |  9 708 | 9 995 471 | 98.53 % |
| amazon-movies | 33 326 | 21 901 |   958 986 | 99.87 % |

---

## 3. Quick start

```bash
source scripts/env.sh && cd src
# the two implementations must produce the same model
python3 check_correctness.py --dataset ml-100k --k 8 --iters 5
# the NumPy reference: no Spark, no JVM, reads the parquet with pandas
python3 eals_local.py --dataset ml-100k --k 32 --iters 20
# the Spark RDD implementation
python3 eals_rdd.py --dataset ml-1m --k 32 --iters 20 --cores 8 --alpha 0.4 --c0 512
```

The notebook `notebooks/demo_small_data.ipynb` runs the whole pipeline
end-to-end on ml-100k in a couple of minutes, with the input and the output of
every cell commented.

---

## 4. Reproducing the report

Everything, in order (~1 h on an 8-core MacBook):

```bash
./scripts/run_all.sh
```

Two rules are baked into that script and are worth knowing before trusting any timing:
experiments run **strictly one at a time** (they share the 8 cores), and the ones that
compare configurations use `--interleave` (repetition as the outer loop) and `--cooldown`
(idle seconds between configurations). Without them, a laptop's thermal drift over a
long campaign silently penalises whichever configuration runs last — we measured a
factor two.

Or one experiment at a time — each writes a raw CSV in `results/`, documented in
[results/README.md](results/README.md):

| figure / table | command |
|---|---|
| dataset table | `python3 experiments.py stats` |
| correctness gate | `python3 check_correctness.py --dataset ml-100k --k 8 --iters 5` |
| strong scaling | `python3 experiments.py strong --dataset ml-10m --iters 4 --reps 3 --interleave` |
| cost vs K, eALS vs MLlib ALS | `python3 experiments.py factors --dataset amazon-movies --with-als --interleave --cooldown 20` |
| convergence | `python3 experiments.py convergence --dataset amazon-movies --iters 30` |
| accuracy vs wall clock | `python3 experiments.py quality --dataset amazon-movies` |

Then the figures, the LaTeX tables and the PDF, in one go:

```bash
./scripts/build_report.sh
```

It runs `src/plots.py` (CSV → `results/figures/`), `report/tables.py`
(CSV → `report/tables/*.tex` and `macros.tex`) and three `pdflatex` passes. **Every
number quoted in the report is produced by `report/tables.py`**, so the text cannot drift
away from the measurements; a figure or a macro whose experiment did not run shows up as a
visible placeholder instead of breaking the build.


## 5. Layout

```
eals-spark/
├── README.md  requirements.txt
├── scripts/   env.sh, run_all.sh, build_report.sh
├── src/
│   ├── spark_utils.py       SparkSession, one place, one configuration
│   ├── data.py              load, k-core, re-index, leave-one-out split (CLI)
│   ├── eals_local.py        NumPy reference + the shared update kernels (CLI)
│   ├── eals_rdd.py          Spark RDD implementation             (CLI)
│   ├── baselines.py         MLlib implicit ALS, MostPopular
│   ├── evaluate.py          HR@k, NDCG@k, full-catalogue ranking
│   ├── check_correctness.py the two implementations must agree   (CLI)
│   ├── experiments.py       every measurement -> results/*.csv   (CLI)
│   └── plots.py             every figure      -> results/figures (CLI)
├── Rapport_Sinapi_Ernadote_eALS_Spark.pdf   the report (43 p, French)
├── notebooks/demo_small_data.ipynb
├── results/                 raw CSVs + figures (see results/README.md)
└── report/                  report.tex, refs.bib, tables.py (LaTeX sources)
```

## 6. Results at a glance

Every number below is measured on the machine described above and produced by
`report/tables.py` from the CSVs in `results/`. Full analysis in
[the report](Rapport_Sinapi_Ernadote_eALS_Spark.pdf).

**Correctness.** The NumPy reference and the Spark RDD version produce the same model to
**7·10⁻¹⁴** (max absolute deviation on the factor matrices); with a single block the
Spark run is bit-identical. The update rules themselves agree with a brute-force
`O(MNK)` evaluation of the paper's Eq. (5) to 3·10⁻¹⁶. eALS parallelises **exactly**,
not approximately.

**Accuracy (Amazon Movies, leave-one-out, ranked against the full catalogue, K=32).**

| method | HR@100 | NDCG@100 |
|---|---:|---:|
| MostPopular | 0.0516 | 0.0127 |
| MLlib ALS (Hu et al. 2008) | 0.1171 | 0.0287 |
| eALS, uniform weights (α=0) | 0.1434 | 0.0345 |
| **eALS (α=0.4)** | **0.1578** | **0.0386** |

eALS beats MLlib's ALS by **+35 %**, of which **+10.0 %** comes
from the popularity-aware weighting alone. It reaches ALS's best-ever accuracy in
3.8 s of training against 17.5 s, and keeps improving afterwards.

**Cost vs K (Amazon Movies, 8 cores).** From K=32 to K=128 one iteration grows
**×2.2** for eALS and **×10.3** for MLlib ALS. eALS is
×2.2 slower at K=16 and ×2.7 faster
at K=128.

**Scalability (ml-10m, 8 cores).** Speedup ×2.07 out of 8 (efficiency 26 %,
Amdahl serial fraction 0.378). The limit is not the algorithm — the parallelisation
is exact and the work splits cleanly — but shared memory bandwidth, broadcast
deserialisation in each Python worker, and per-task Python overhead.


## 7. Report

The full write-up is [the report](Rapport_Sinapi_Ernadote_eALS_Spark.pdf), in French.
Its structure follows the grading grid of the course:

| grid item | section |
|---|---|
| 1. Description of the adopted solution | §2 |
| 2a. Designed algorithms + global description | §3 (derivation, distributed pseudo-code, complexity, exactness) |
| 2b. Comments on the main fragments of code | §4 |
| 3. Experimental analysis, in particular scalability | §5, esp. §5.2 |
| 4. Weak and strong points of the algorithms | §6 |
| 5. Appendix: all the code + a working notebook | Appendix A, Appendix B |

## 8. Authors

Thomas Sinapi and Anaël Ernadote.
M2 IASD — Université Paris-Dauphine PSL, *Machine Learning for Big Data* project, 2026.
Paper: He, Zhang, Kan, Chua, *Fast Matrix Factorization for Online Recommendation with
Implicit Feedback*, SIGIR 2016.
