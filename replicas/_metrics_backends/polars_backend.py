"""Polars implementation of the precision-recall metric pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

_COUNTS = {"positive": "dTP", "negative": "dFP", "unlabeled": "dUP"}
_CUMULATIVE = {"dTP": "TP", "dFP": "FP", "dUP": "UP"}
_TOTALS = {"dTP": "positives", "dFP": "negatives", "dUP": "unlabeled"}
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


def _sort(df: pl.DataFrame, group_by: Sequence[str], threshold: str, *, ascending: bool):
    return df.sort(
        [*group_by, threshold],
        descending=[False] * len(group_by) + [not ascending],
        nulls_last=True,
        maintain_order=True,
    )


def confusion_table(df: pl.DataFrame, group_by: Sequence[str]) -> pl.DataFrame:
    result = df.group_by([*group_by, "prediction"]).agg(
        [pl.col(source).sum().alias(target) for source, target in _COUNTS.items()]
    )
    result = _sort(result, group_by, "prediction", ascending=False)

    if group_by:
        totals = [
            pl.col(source).sum().over(list(group_by)).alias(target)
            for source, target in _TOTALS.items()
        ]
        cumulative = [
            pl.col(source).cum_sum().over(list(group_by)).alias(target)
            for source, target in _CUMULATIVE.items()
        ]
    else:
        totals = [pl.col(source).sum().alias(target) for source, target in _TOTALS.items()]
        cumulative = [
            pl.col(source).cum_sum().alias(target) for source, target in _CUMULATIVE.items()
        ]

    return (
        result.with_columns([*totals, *cumulative])
        .rename({"prediction": "threshold"})
        .select([*group_by, *_CONFUSION_COLUMNS])
    )


def calculate_pr(df: pl.DataFrame, group_by: Sequence[str]) -> pl.DataFrame:
    original_columns = list(df.columns)
    result = _sort(df.filter(pl.col("dTP") > 0), group_by, "threshold", ascending=False)
    result = result.with_columns(
        (pl.col("TP") / (pl.col("TP") + pl.col("FP"))).alias("precision"),
        (pl.col("TP") / pl.col("positives")).alias("recall"),
    )
    weighted = pl.col("dTP") * pl.col("precision")
    numerator = weighted.cum_sum().over(list(group_by)) if group_by else weighted.cum_sum()
    result = result.with_columns((numerator / pl.col("TP")).alias("average_precision"))

    metric_columns = ["precision", "recall", "average_precision"]
    retained = [column for column in original_columns if column not in metric_columns]
    return result.select([*retained, *metric_columns])


def at(
    df: pl.DataFrame,
    group_by: Sequence[str],
    metric: str,
    value: Any,
) -> pl.DataFrame:
    columns = list(df.columns)
    result = _sort(df.filter(pl.col(metric) >= value), group_by, "threshold", ascending=True)
    if group_by:
        result = result.unique(subset=list(group_by), keep="first", maintain_order=True)
        result = result.sort(list(group_by), nulls_last=True, maintain_order=True)
    else:
        result = result.head(1)
    return result.select(columns)
