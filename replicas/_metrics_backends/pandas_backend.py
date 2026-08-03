"""pandas implementation of the precision-recall metric pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

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


def _sort(df: pd.DataFrame, group_by: Sequence[str], threshold: str, *, ascending: bool):
    columns = [*group_by, threshold]
    directions = [True] * len(group_by) + [ascending]
    return df.sort_values(
        columns,
        ascending=directions,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _grouped(df: pd.DataFrame, group_by: Sequence[str]):
    key = group_by[0] if len(group_by) == 1 else list(group_by)
    return df.groupby(key, dropna=False, observed=True, sort=False)


def confusion_table(df: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    score_keys = [*group_by, "prediction"]
    result = (
        df.groupby(score_keys, dropna=False, observed=True, sort=False)[list(_COUNTS)]
        .sum()
        .rename(columns=_COUNTS)
        .reset_index()
    )
    result = _sort(result, group_by, "prediction", ascending=False)

    if group_by:
        grouped = _grouped(result, group_by)
        totals = grouped[list(_TOTALS)].transform("sum").rename(columns=_TOTALS)
        cumulative = grouped[list(_CUMULATIVE)].cumsum().rename(columns=_CUMULATIVE)
    else:
        totals = pd.DataFrame(
            {target: [result[source].sum()] * len(result) for source, target in _TOTALS.items()},
            index=result.index,
        )
        cumulative = result[list(_CUMULATIVE)].cumsum().rename(columns=_CUMULATIVE)

    for column in totals:
        result[column] = totals[column]
    for column in cumulative:
        result[column] = cumulative[column]

    result = result.rename(columns={"prediction": "threshold"})
    return result[[*group_by, *_CONFUSION_COLUMNS]]


def calculate_pr(df: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    original_columns = list(df.columns)
    result = df.loc[df["dTP"] > 0].copy()
    result = _sort(result, group_by, "threshold", ascending=False)
    result["precision"] = result["TP"] / (result["TP"] + result["FP"])
    result["recall"] = result["TP"] / result["positives"]
    weighted_precision = result["dTP"] * result["precision"]

    if group_by:
        result["_weighted_precision"] = weighted_precision
        numerator = _grouped(result, group_by)["_weighted_precision"].cumsum()
        result = result.drop(columns="_weighted_precision")
    else:
        numerator = weighted_precision.cumsum()
    result["average_precision"] = numerator / result["TP"]

    metric_columns = ["precision", "recall", "average_precision"]
    retained = [column for column in original_columns if column not in metric_columns]
    return result[[*retained, *metric_columns]]


def at(
    df: pd.DataFrame,
    group_by: Sequence[str],
    metric: str,
    value: Any,
) -> pd.DataFrame:
    columns = list(df.columns)
    result = df.loc[df[metric] >= value].copy()
    result = _sort(result, group_by, "threshold", ascending=True)
    if group_by:
        result = result.drop_duplicates(subset=list(group_by), keep="first")
        result = result.sort_values(
            list(group_by), kind="mergesort", na_position="last"
        ).reset_index(drop=True)
    else:
        result = result.head(1)
    return result[columns]
