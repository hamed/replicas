"""Precision-recall metrics on top of bootstrap replicas.

This module is one application of the bootstrap primitive in
`replicas.bootstrap`. The same pattern — compute a metric, grouped by
`replica` — extends to any classifier metric you care to compute.

Data schema
-----------
The functions in this module expect a Spark DataFrame with four columns:

- `prediction` (double): the model's score, higher = more likely positive.
- `positive` (int 0/1): row is a verified positive.
- `negative` (int 0/1): row is a verified negative.
- `unlabeled` (int 0/1): row has a prediction but no verified label.

The three label columns are mutually exclusive: each row has exactly one set
to 1. `unlabeled` rows are tracked separately (as `UP` — unlabeled positives —
in the confusion table) rather than silently treated as negative. This
generalizes to multi-class by adding more label columns; the same logic
applies.
"""

from __future__ import annotations

from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def confusion_table(df: DataFrame, group_by: Sequence[str] | None = None) -> DataFrame:
    """Build a per-threshold confusion table from a predictions DataFrame.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame with columns `prediction`, `positive`, `negative`,
        `unlabeled`. See module docstring for schema details.
    group_by : sequence of str, optional
        Columns to group the calculation over. Typical values include `name`
        (to compare models) and `replica` (to compute bootstrap CIs).

    Returns
    -------
    DataFrame
        One row per (group, threshold), with columns:

        - `threshold` : the prediction threshold θ
        - `TP`, `FP`, `UP` : cumulative true / false / unlabeled positives at θ
        - `dTP`, `dFP`, `dUP` : counts of rows at exactly this score
        - `positives`, `negatives`, `unlabeled` : per-group totals
    """
    group_by = list(group_by or [])

    # Bootstrapping generates many ties: the same row sampled twice has the
    # same prediction score. Aggregate by score first to collapse them.
    df_by_score = df.groupBy(*group_by, "prediction").agg(
        F.sum("positive").alias("dTP"),
        F.sum("negative").alias("dFP"),
        F.sum("unlabeled").alias("dUP"),
    )

    totals = df_by_score.groupBy(*group_by).agg(
        F.sum("dTP").alias("positives"),
        F.sum("dFP").alias("negatives"),
        F.sum("dUP").alias("unlabeled"),
    )

    window = (
        Window.partitionBy(*group_by)
        .orderBy(F.col("prediction").desc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    # An empty group_by list confuses Spark's join, so use crossJoin for the
    # ungrouped case (totals is then a single row, broadcast-safe).
    if group_by:
        joined = df_by_score.join(totals, on=group_by, how="inner")
    else:
        joined = df_by_score.crossJoin(totals)

    return (
        joined
        .withColumns(
            {
                "TP": F.sum("dTP").over(window),
                "FP": F.sum("dFP").over(window),
                "UP": F.sum("dUP").over(window),
            }
        )
        .select(
            *group_by,
            F.col("prediction").alias("threshold"),
            "TP", "FP", "UP",
            "dTP", "dFP", "dUP",
            "positives", "negatives", "unlabeled",
        )
    )


def calculate_pr(df: DataFrame, group_by: Sequence[str] | None = None) -> DataFrame:
    """Compute precision, recall, and cumulative average precision per threshold.

    Parameters
    ----------
    df : DataFrame
        Output of `confusion_table`.
    group_by : sequence of str, optional
        Same grouping columns used in `confusion_table`.

    Returns
    -------
    DataFrame
        Adds columns `precision`, `recall`, `average_precision` to the input.
        Thresholds with no new true positives are filtered out (they do not
        change the curve). The `average_precision` column is a running mean;
        at the highest threshold it equals the standard area under the PR
        curve.
    """
    group_by = list(group_by or [])

    window = (
        Window.partitionBy(*group_by)
        .orderBy(F.col("threshold").desc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    return (
        df.filter(F.col("dTP") > 0)
        .withColumns(
            {
                "precision": F.col("TP") / (F.col("TP") + F.col("FP")),
                "recall": F.col("TP") / F.col("positives"),
            }
        )
        .withColumn(
            "average_precision",
            F.sum(F.col("dTP") * F.col("precision")).over(window) / F.col("TP"),
        )
    )


def at(df: DataFrame, group_by: Sequence[str] | None = None, **kwargs) -> DataFrame:
    """Find the minimum threshold satisfying a target metric condition, per group.

    Parameters
    ----------
    df : DataFrame
        Output of `calculate_pr` (or any DataFrame with a `threshold` column
        and the target metric column).
    group_by : sequence of str, optional
        Grouping columns.
    **kwargs : dict
        Exactly one key-value pair specifying the target, e.g.
        `precision=0.95` or `recall=0.5`.

    Returns
    -------
    DataFrame
        One row per group: the row at the smallest threshold meeting the
        target. With bootstrap replicas in the group, the output is a
        *distribution* of operating points — the bootstrap CI on the
        threshold you would deploy.
    """
    if len(kwargs) != 1:
        raise ValueError(
            f"at() requires exactly one metric=value condition, got {len(kwargs)}"
        )

    group_by = list(group_by or [])
    metric, value = next(iter(kwargs.items()))

    window_spec = Window.partitionBy(*group_by).orderBy(F.asc("threshold"))
    result = (
        df.filter(F.col(metric) >= value)
        .withColumn("_row", F.row_number().over(window_spec))
        .filter(F.col("_row") == 1)
        .drop("_row")
    )
    if group_by:
        result = result.orderBy(*group_by)
    return result
