"""Single place where the SparkSession is built, so every script runs with the
same configuration and the scalability experiments only differ by `cores`."""
import os
import shutil
from pyspark.sql import SparkSession

SPARK_TMP = os.environ.get("EALS_SPARK_TMP", "/tmp/eals-spark-tmp")


def get_spark(app="eals", cores=8, driver_mem="10g", shuffle_partitions=None, quiet=True):
    os.makedirs(SPARK_TMP, exist_ok=True)
    # A module the *driver* can import is not automatically importable in the Python
    # workers: they are separate processes and do not inherit sys.path. Every closure
    # that references eals_local / eals_rdd would fail with ModuleNotFoundError. The
    # workers do inherit the environment, so PYTHONPATH is the fix -- and it must be set
    # before the JVM is launched, i.e. before getOrCreate().
    src = os.path.dirname(os.path.abspath(__file__))
    if src not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [src] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    b = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app)
        # local[k] runs everything in the driver JVM, so driver memory is *the*
        # memory knob: the broadcast copies of P and Q live here.
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.local.dir", SPARK_TMP)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", shuffle_partitions or (cores * 2))
        # only used to move a training split to pandas (toPandas) for the NumPy
        # reference and the evaluation.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    s = b.getOrCreate()
    if quiet:
        s.sparkContext.setLogLevel("ERROR")
    return s


def clear_tmp():
    shutil.rmtree(SPARK_TMP, ignore_errors=True)
