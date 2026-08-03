"""Backend-neutral precision-recall metrics for bootstrap replicas.

The public functions in this module accept pandas, Polars, or Spark DataFrames
and return the same kind of DataFrame they receive.  Backend modules are loaded
only when their DataFrame type is used, so importing :mod:`replicas.metrics`
does not import every supported dataframe library.

Data schema
-----------
``confusion_table`` expects four non-null columns:

- ``prediction``: the model score; higher means more likely positive.
- ``positive``: 1 for a verified positive, otherwise 0.
- ``negative``: 1 for a verified negative, otherwise 0.
- ``unlabeled``: 1 when no verified label is available, otherwise 0.

The three indicator columns are mutually exclusive.  Null values in grouping
columns are supported and form an ordinary group.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from importlib import import_module
from typing import Any, Optional, TypeVar

FrameT = TypeVar("FrameT")
# Keep public annotations in pre-PEP 604 form for the Python 3.9 floor.
GroupBy = Optional[Sequence[str]]  # noqa: UP045

_PREDICTION_COLUMNS = ("prediction", "positive", "negative", "unlabeled")
_CONFUSION_COLUMNS = (
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
)
_BACKEND_MODULES = {
    "pandas": "replicas._metrics_backends.pandas_backend",
    "polars": "replicas._metrics_backends.polars_backend",
    "pyspark": "replicas._metrics_backends.spark_backend",
}


def _backend(df: Any):
    """Load the adapter matching *df* without importing optional backends."""
    roots = {cls.__module__.partition(".")[0] for cls in type(df).__mro__}
    for root, module_name in _BACKEND_MODULES.items():
        if root in roots:
            return import_module(module_name)

    supported = "pandas.DataFrame, polars.DataFrame, or pyspark.sql.DataFrame"
    raise TypeError(f"Unsupported dataframe type {type(df)!r}; expected {supported}")


def _groups(group_by: GroupBy) -> list[str]:
    if group_by is None:
        return []
    if isinstance(group_by, (str, bytes)):
        raise TypeError("group_by must be a sequence of column names, not a string")
    try:
        groups = list(group_by)
    except TypeError as error:
        raise TypeError("group_by must be a sequence of column names") from error
    if any(not isinstance(column, str) for column in groups):
        raise TypeError("group_by entries must be column names")

    duplicates = [name for name, count in Counter(groups).items() if count > 1]
    if duplicates:
        raise ValueError(f"group_by contains duplicate columns: {duplicates}")
    return groups


def _validate_columns(df: Any, required: Sequence[str], groups: Sequence[str]) -> None:
    columns = list(df.columns)
    duplicates = [name for name, count in Counter(columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"DataFrame contains duplicate columns: {duplicates}")

    missing = [column for column in (*groups, *required) if column not in columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def confusion_table(df: FrameT, group_by: GroupBy = None) -> FrameT:
    """Build a per-threshold confusion table.

    Scores tied at the same threshold are collapsed before cumulative counts
    are calculated.  The returned columns are ``group_by`` followed by
    ``threshold``, cumulative ``TP``/``FP``/``UP``, per-score
    ``dTP``/``dFP``/``dUP``, and group totals
    ``positives``/``negatives``/``unlabeled``.

    pandas and Polars results are sorted by grouping columns ascending and
    threshold descending.  As usual for Spark DataFrames, Spark row order is
    unspecified unless the caller explicitly orders the result.
    """
    backend = _backend(df)
    groups = _groups(group_by)
    _validate_columns(df, _PREDICTION_COLUMNS, groups)

    reserved = {*_PREDICTION_COLUMNS, *_CONFUSION_COLUMNS}
    conflicts = [column for column in groups if column in reserved]
    if conflicts:
        raise ValueError(f"group_by columns conflict with confusion-table output: {conflicts}")

    return backend.confusion_table(df, groups)


def calculate_pr(df: FrameT, group_by: GroupBy = None) -> FrameT:
    """Add precision, recall, and cumulative average precision.

    Thresholds with no new true positives are omitted.  ``average_precision``
    is a running, true-positive-weighted mean; only its last (lowest-threshold)
    row in each group equals the standard area under the precision-recall
    curve.
    """
    backend = _backend(df)
    groups = _groups(group_by)
    _validate_columns(df, _CONFUSION_COLUMNS, groups)
    conflicts = [column for column in groups if column in _CONFUSION_COLUMNS]
    if conflicts:
        raise ValueError(f"group_by columns conflict with confusion-table metrics: {conflicts}")
    return backend.calculate_pr(df, groups)


def at(df: FrameT, group_by: GroupBy = None, **kwargs: Any) -> FrameT:
    """Return the lowest threshold satisfying one ``metric >= value`` target.

    A group with no qualifying threshold is absent from the result.  With
    replicas in ``group_by``, the result is a distribution of operating points.
    """
    if len(kwargs) != 1:
        raise ValueError(f"at() requires exactly one metric=value condition, got {len(kwargs)}")

    backend = _backend(df)
    groups = _groups(group_by)
    metric, value = next(iter(kwargs.items()))
    _validate_columns(df, ("threshold", metric), groups)
    if "threshold" in groups:
        raise ValueError("threshold cannot also be a group_by column")
    return backend.at(df, groups, metric, value)
