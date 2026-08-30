"""Dataset loading, k-core filtering, dense re-indexing and the offline
leave-one-out split used by the paper.

All of it runs in Spark DataFrames; the output is parquet so that every
experiment starts from exactly the same materialised training set.
"""
import argparse
import json
import os

from pyspark.sql import Window
from pyspark.sql import functions as F

from spark_utils import get_spark

RAW = os.environ.get("EALS_RAW", os.path.expanduser("~/eals-spark/data/raw"))
PROC = os.environ.get("EALS_PROC", os.path.expanduser("~/eals-spark/data/processed"))

DATASETS = ("ml-100k", "ml-1m", "ml-10m", "amazon-movies")


def load_raw(spark, name):
    """-> DataFrame(user string, item string, ts long). Ratings are dropped:
    implicit feedback only keeps the *existence* of an interaction."""
    if name == "ml-100k":
        p = f"{RAW}/ml-100k/u.data"
        df = spark.read.csv(p, sep="\t", schema="user STRING, item STRING, rating DOUBLE, ts LONG")
    elif name in ("ml-1m", "ml-10m"):
        p = f"{RAW}/ml-1m/ratings.dat" if name == "ml-1m" else f"{RAW}/ml-10M100K/ratings.dat"
        df = (
            spark.read.text(p)
            .select(F.split("value", "::").alias("a"))
            .select(F.col("a")[0].alias("user"), F.col("a")[1].alias("item"),
                    F.col("a")[3].cast("long").alias("ts"))
        )
    elif name == "amazon-movies":
        p = f"{RAW}/ratings_Movies_and_TV.csv"
        df = spark.read.csv(p, schema="user STRING, item STRING, rating DOUBLE, ts LONG")
    else:
        raise ValueError(name)
    return df.select("user", "item", "ts").dropna()


def kcore(df, k):
    """Iterated k-core: dropping sparse users can push items below k and vice
    versa, so both filters are re-applied until the interaction count is
    stable. The k-core of a bipartite graph is unique, so the fixed point does
    not depend on the order in which the two sides are pruned."""
    df = df.cache()
    prev = -1
    while True:
        n = df.count()
        if n == prev:
            return df
        prev = n
        ok_u = df.groupBy("user").count().filter(F.col("count") >= k).select("user")
        ok_i = df.groupBy("item").count().filter(F.col("count") >= k).select("item")
        nxt = df.join(ok_u, "user").join(ok_i, "item").select("user", "item", "ts")
        # localCheckpoint, not cache: cache() keeps the logical plan alive, so
        # after ~8 rounds the plan is deep enough that Spark itself dies with an
        # OutOfMemoryError while *printing* it (TreeNode.generateTreeString).
        # Checkpointing truncates the lineage to a plain scan of the materialised
        # rows. This is the iterative-Spark trap, and it bites in data prep too.
        nxt = nxt.localCheckpoint(eager=True)
        df.unpersist()
        df = nxt


def reindex(df):
    """Dense 0..M-1 / 0..N-1 ids: the eALS kernels index NumPy arrays directly,
    so ids must be contiguous. Ordering by descending frequency also puts the
    heavy users first, which slightly balances the hash partitions."""
    um = (df.groupBy("user").count().orderBy(F.desc("count"), "user")
            .withColumn("u", F.row_number().over(Window.orderBy(F.desc("count"), "user")) - 1)
            .select("user", "u"))
    im = (df.groupBy("item").count().orderBy(F.desc("count"), "item")
            .withColumn("i", F.row_number().over(Window.orderBy(F.desc("count"), "item")) - 1)
            .select("item", "i"))
    out = df.join(um, "user").join(im, "item").select("u", "i", "ts")
    return out, um, im


def leave_one_out(df):
    """Offline protocol: the chronologically last interaction of every user is
    held out. Amazon timestamps are day-granular so ties are frequent; the item
    id is used as a deterministic tie-break."""
    w = Window.partitionBy("u").orderBy(F.desc("ts"), F.desc("i"))
    r = df.withColumn("rk", F.row_number().over(w))
    return r.filter("rk > 1").drop("rk"), r.filter("rk = 1").drop("rk")


def prepare(spark, name, k=10, out_dir=None, sample_frac=None, seed=42):
    out_dir = out_dir or f"{PROC}/{name}"
    df = load_raw(spark, name).dropDuplicates(["user", "item"])
    if sample_frac:
        # Sample *users*, not interactions: sampling interactions would destroy
        # the k-core property and shift the popularity distribution.
        users = df.select("user").distinct().sample(sample_frac, seed=seed)
        df = df.join(users, "user")
    df = kcore(df, k)
    df, um, im = reindex(df)
    df = df.cache()
    tr, te = leave_one_out(df)

    stats = dict(dataset=name, kcore=k, sample_frac=sample_frac,
                 users=df.select("u").distinct().count(),
                 items=df.select("i").distinct().count(),
                 interactions=df.count())
    stats["sparsity"] = 1 - stats["interactions"] / (stats["users"] * stats["items"])
    stats["train"], stats["test"] = tr.count(), te.count()

    for sub, d in (("train", tr), ("test", te)):
        d.write.mode("overwrite").parquet(f"{out_dir}/{sub}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(stats, open(f"{out_dir}/stats.json", "w"), indent=2)
    return stats


def load_split(spark, name, sub="train", proc=None):
    return spark.read.parquet(f"{proc or PROC}/{name}/{sub}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m", choices=DATASETS)
    ap.add_argument("--kcore", type=int, default=10)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sp = get_spark("prepare", cores=a.cores)
    print(json.dumps(prepare(sp, a.dataset, a.kcore, a.out, a.sample_frac), indent=2))
    sp.stop()
