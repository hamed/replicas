"""Spark implementation of the precision-recall metric pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

_CONFUSION_COLUMNS = [
    "threshold",
    "TP",
    "FP",
    "UP",
    "dTP",
    "dFP",
    "dUP",
    "positives",
    "negatives",
    "unlabeled",
]


def _unique_helper_column(df: DataFrame, stem: str) -> str:
    existing = {column.casefold() for column in df.columns}
    name = stem
    suffix = 0
    while name.casefold() in existing:
        suffix += 1
        name = f"{stem}_{suffix}"
    return name


def confusion_table(df: DataFrame, group_by: Sequence[str]) -> DataFrame:
    by_score = df.groupBy(*group_by, "prediction").agg(
        F.sum("positive").alias("dTP"),
        F.sum("negative").alias("dFP"),
        F.sum("unlabeled").alias("dUP"),
    )
    totals = Window.partitionBy(*group_by).rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )
    cumulative = (
        Window.partitionBy(*group_by)
        .orderBy(F.col("prediction").desc_nulls_last())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    return by_score.withColumns(
        {
            "positives": F.sum("dTP").over(totals),
            "negatives": F.sum("dFP").over(totals),
            "unlabeled": F.sum("dUP").over(totals),
            "TP": F.sum("dTP").over(cumulative),
            "FP": F.sum("dFP").over(cumulative),
            "UP": F.sum("dUP").over(cumulative),
        }
    ).select(
        *group_by,
        F.col("prediction").alias("threshold"),
        *_CONFUSION_COLUMNS[1:],
    )


def calculate_pr(df: DataFrame, group_by: Sequence[str]) -> DataFrame:
    cumulative = (
        Window.partitionBy(*group_by)
        .orderBy(F.col("threshold").desc_nulls_last())
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
            F.sum(F.col("dTP") * F.col("precision")).over(cumulative) / F.col("TP"),
        )
    )


def at(
    df: DataFrame,
    group_by: Sequence[str],
    metric: str,
    value: Any,
) -> DataFrame:
    helper = _unique_helper_column(df, "__replicas_at_row")
    by_threshold = Window.partitionBy(*group_by).orderBy(F.col("threshold").asc_nulls_last())
    return (
        df.filter(F.col(metric) >= value)
        .withColumn(helper, F.row_number().over(by_threshold))
        .filter(F.col(helper) == 1)
        .drop(helper)
    )
