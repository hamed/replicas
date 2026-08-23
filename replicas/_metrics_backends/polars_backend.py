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


def _sort(df: pl.DataFrame, by: Sequence[str], threshold: str, *, ascending: bool):
    return df.sort(
        [*by, threshold],
        descending=[False] * len(by) + [not ascending],
        nulls_last=True,
        maintain_order=True,
    )


def confusion_table(df: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    result = df.group_by([*by, "prediction"]).agg(
        [pl.col(source).sum().alias(target) for source, target in _COUNTS.items()]
    )
    result = _sort(result, by, "prediction", ascending=False)

    if by:
        totals = [
            pl.col(source).sum().over(list(by)).alias(target) for source, target in _TOTALS.items()
        ]
        cumulative = [
            pl.col(source).cum_sum().over(list(by)).alias(target)
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
        .select([*by, *_CONFUSION_COLUMNS])
    )


def calculate_pr(df: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    original_columns = list(df.columns)
    result = _sort(df.filter(pl.col("dTP") > 0), by, "threshold", ascending=False)
    result = result.with_columns(
        (pl.col("TP") / (pl.col("TP") + pl.col("FP"))).alias("precision"),
        (pl.col("TP") / pl.col("positives")).alias("recall"),
    )
    weighted = pl.col("dTP") * pl.col("precision")
    numerator = weighted.cum_sum().over(list(by)) if by else weighted.cum_sum()
    result = result.with_columns((numerator / pl.col("TP")).alias("average_precision"))

    metric_columns = ["precision", "recall", "average_precision"]
    retained = [column for column in original_columns if column not in metric_columns]
    return result.select([*retained, *metric_columns])


def at(
    df: pl.DataFrame,
    by: Sequence[str],
    metric: str,
    value: Any,
) -> pl.DataFrame:
    columns = list(df.columns)
    result = _sort(df.filter(pl.col(metric) >= value), by, "threshold", ascending=True)
    if by:
        result = result.unique(subset=list(by), keep="first", maintain_order=True)
        result = result.sort(list(by), nulls_last=True, maintain_order=True)
    else:
        result = result.head(1)
    return result.select(columns)
