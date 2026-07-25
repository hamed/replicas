"""The bootstrap primitive: resample a Spark DataFrame, materialize once.

The whole library is built on two functions in this module. Everything in
`replicas.metrics` is just `groupBy('replica').agg(...)` on their output.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def sample(df: DataFrame, by: Sequence[str], fraction: float = 1.0) -> DataFrame:
    """Sample with replacement within each group of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input Spark DataFrame.
    by : sequence of str
        Column names to stratify the sampling over. Pass an empty list for
        unstratified sampling. Under class imbalance, stratifying on the label
        column prevents replicas with zero positives (which would make many
        metrics undefined).
    fraction : float, optional
        Fraction of rows to sample within each group. Defaults to 1.0,
        meaning each replica is the same size as the original.

    Returns
    -------
    DataFrame
        A new Spark DataFrame with the same schema, containing the resampled
        rows.

    Notes
    -----
    No random seed is set inside this function — deliberately. The function is
    called inside a loop by `bootstrap`, once per replica. If the seed were
    fixed, every iteration would produce the *same* "random" sample and all
    your replicas would collapse to one. The lack of a per-call seed is what
    makes the replicas actually different. For run-level reproducibility, seed
    at the SparkContext level instead.
    """
    return df.groupBy(list(by)).applyInPandas(
        lambda pdf: pdf.sample(frac=fraction, replace=True),  # No seed — see docstring.
        schema=df.schema,
    )


def bootstrap(
    df: DataFrame,
    by: Sequence[str] | None = None,
    n_replicas: int = 100,
    checkpoint_dir: str = "/tmp/replicas/",
) -> DataFrame:
    """Generate bootstrap replicas of a Spark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input Spark DataFrame. Typically a predictions table — small
        enough to broadcast — joined ahead of time with the (usually larger)
        feature table if needed.
    by : sequence of str, optional
        Columns to stratify the sampling over. Defaults to no stratification.
        For classifier evaluation, pass the label column to keep class balance
        across replicas.
    n_replicas : int, optional
        Number of bootstrap replicas to generate. Defaults to 100.
    checkpoint_dir : str, optional
        Spark checkpoint directory. Defaults to `/tmp/replicas/`.

    Returns
    -------
    DataFrame
        A long-format DataFrame with the same columns as the input plus a
        `replica` column. Replica `-1` is the original (un-resampled) data;
        replicas `0..n_replicas - 1` are the resamples.

        The result is checkpointed — its random content will not change
        between actions on it.

    Notes
    -----
    Spark is lazy. A naive bootstrap (chained `union` calls without
    materialization) builds a deferred plan, and every terminal action
    re-rolls the random draws. The "replica 7" used to compute precision is
    not the same "replica 7" used to compute recall. Downstream joins go
    haywire; recall > 1 is a common symptom.

    `checkpoint()` forces materialization to disk and replaces the lineage
    with a read from that snapshot. After checkpoint, replica IDs are stable
    across all downstream operations.
    """
    by = list(by or [])

    bts = df.withColumn("replica", F.lit(-1))
    for i in range(n_replicas):
        bts = bts.union(sample(df, by).withColumn("replica", F.lit(i)))

    # Materialize once. Without this, downstream metrics would not align.
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    return bts.checkpoint()
