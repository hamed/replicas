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


def _sort(df: pd.DataFrame, by: Sequence[str], threshold: str, *, ascending: bool):
    columns = [*by, threshold]
    directions = [True] * len(by) + [ascending]
    return df.sort_values(
        columns,
        ascending=directions,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _grouped(df: pd.DataFrame, by: Sequence[str]):
    key = by[0] if len(by) == 1 else list(by)
    return df.groupby(key, dropna=False, observed=True, sort=False)


def confusion_table(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    score_keys = [*by, "prediction"]
    result = (
        df.groupby(score_keys, dropna=False, observed=True, sort=False)[list(_COUNTS)]
        .sum()
        .rename(columns=_COUNTS)
        .reset_index()
    )
    result = _sort(result, by, "prediction", ascending=False)

    if by:
        grouped = _grouped(result, by)
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
    return result[[*by, *_CONFUSION_COLUMNS]]


def calculate_pr(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    original_columns = list(df.columns)
    result = df.loc[df["dTP"] > 0].copy()
    result = _sort(result, by, "threshold", ascending=False)
    result["precision"] = result["TP"] / (result["TP"] + result["FP"])
    result["recall"] = result["TP"] / result["positives"]
    weighted_precision = result["dTP"] * result["precision"]

    if by:
        grouper = result[by[0]] if len(by) == 1 else [result[column] for column in by]
        numerator = weighted_precision.groupby(
            grouper,
            dropna=False,
            observed=True,
            sort=False,
        ).cumsum()
    else:
        numerator = weighted_precision.cumsum()
    result["average_precision"] = numerator / result["TP"]

    metric_columns = ["precision", "recall", "average_precision"]
    retained = [column for column in original_columns if column not in metric_columns]
    return result[[*retained, *metric_columns]]


def at(
    df: pd.DataFrame,
    by: Sequence[str],
    metric: str,
    value: Any,
) -> pd.DataFrame:
    columns = list(df.columns)
    result = df.loc[df[metric] >= value].copy()
    result = _sort(result, by, "threshold", ascending=True)
    if by:
        result = result.drop_duplicates(subset=list(by), keep="first")
        result = result.sort_values(list(by), kind="mergesort", na_position="last").reset_index(
            drop=True
        )
    else:
        result = result.head(1)
    return result[columns]


def pr_band(
    df: pd.DataFrame,
    by: Sequence[str],
    low: float,
    high: float,
    recall_round: int | None,
) -> pd.DataFrame:
    frame = df
    if recall_round is not None:
        frame = frame.assign(recall=frame["recall"].round(recall_round))

    keys = [*by, "recall"]
    envelope = (
        frame.groupby([*keys, "replica"], dropna=False, observed=True, sort=False)["precision"]
        .max()
        .reset_index()
    )

    # Masked columns rather than two frames and a join: a null grouping value
    # is an ordinary group here, and the three backends do not agree on
    # whether a join matches null keys. One group-by has no such ambiguity.
    envelope = envelope.assign(
        _original=envelope["precision"].where(envelope["replica"] == -1),
        _replica=envelope["precision"].where(envelope["replica"] >= 0),
    )
    result = (
        envelope.groupby(keys, dropna=False, observed=True, sort=False)
        .agg(
            precision=("_original", "max"),
            low=("_replica", lambda values: values.quantile(low)),
            high=("_replica", lambda values: values.quantile(high)),
        )
        .reset_index()
    )
    return result.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)
