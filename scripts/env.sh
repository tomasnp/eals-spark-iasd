# Spark needs a JDK it supports and macOS ships none: point JAVA_HOME at the
# Homebrew JDK before anything else. `source scripts/env.sh` from the repo root.
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON=${PYSPARK_PYTHON:-$(command -v python3)}
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
export EALS_SPARK_TMP=${EALS_SPARK_TMP:-/tmp/eals-spark-tmp}
# NumPy is already the inner parallelism of one Spark task; letting BLAS spawn
# its own threads on top of local[k] oversubscribes the 8 cores and makes every
# scalability measurement meaningless.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
