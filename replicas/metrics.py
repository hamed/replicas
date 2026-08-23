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

These two conditions are the caller's responsibility; they are not checked.
Validating them costs a full pass over the data, which on Spark means an
eager job before the lazy plan the caller asked for.  A null in an indicator
column is skipped by every backend's ``sum``, and a row that sets two
indicators is counted twice -- in both cases the counts are silently wrong.
Clean the input before calling ``confusion_table``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from importlib import import_module
from typing import Any, Optional, TypeVar, Union

FrameT = TypeVar("FrameT")
# Keep public annotations in pre-PEP 604 form for the Python 3.9 floor.
ByColumns = Optional[Union[str, Sequence[str]]]  # noqa: UP007, UP045

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
_PR_COLUMNS = ("precision", "recall", "average_precision")
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


def _groups(by: ByColumns) -> list[str]:
    if by is None:
        return []
    if isinstance(by, str):
        return [by]
    if isinstance(by, bytes):
        raise TypeError("by must contain column names as strings, not bytes")
    try:
        groups = list(by)
    except TypeError as error:
        raise TypeError("by must be a column name or a sequence of column names") from error
    if any(not isinstance(column, str) for column in groups):
        raise TypeError("by entries must be column names")

    duplicates = [name for name, count in Counter(groups).items() if count > 1]
    if duplicates:
        raise ValueError(f"by contains duplicate columns: {duplicates}")
    return groups


def _validate_columns(df: Any, required: Sequence[str], groups: Sequence[str]) -> None:
    columns = list(df.columns)
    duplicates = [name for name, count in Counter(columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"DataFrame contains duplicate columns: {duplicates}")

    missing = [column for column in (*groups, *required) if column not in columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def confusion_table(df: FrameT, by: ByColumns = None) -> FrameT:
    """Build a per-threshold confusion table.

    Scores tied at the same threshold are collapsed before cumulative counts
    are calculated.  The returned columns are ``by`` followed by
    ``threshold``, cumulative ``TP``/``FP``/``UP``, per-score
    ``dTP``/``dFP``/``dUP``, and group totals
    ``positives``/``negatives``/``unlabeled``.

    pandas and Polars results are sorted by grouping columns ascending and
    threshold descending.  As usual for Spark DataFrames, Spark row order is
    unspecified unless the caller explicitly orders the result.

    On Spark, omitting ``by`` leaves the cumulative and total windows
    without a partition key, so Spark moves every score to a single partition.
    Group by ``replica`` (the usual bootstrap case) to keep the work spread.
    """
    backend = _backend(df)
    groups = _groups(by)
    _validate_columns(df, _PREDICTION_COLUMNS, groups)

    reserved = {*_PREDICTION_COLUMNS, *_CONFUSION_COLUMNS}
    conflicts = [column for column in groups if column in reserved]
    if conflicts:
        raise ValueError(f"by conflicts with a confusion-table output column: {conflicts}")

    return backend.confusion_table(df, groups)


def calculate_pr(df: FrameT, by: ByColumns = None) -> FrameT:
    """Add precision, recall, and cumulative average precision.

    Thresholds with no new true positives are omitted.  ``average_precision``
    is a running, true-positive-weighted mean; only its last (lowest-threshold)
    row in each group equals the standard area under the precision-recall
    curve.

    As in ``confusion_table``, omitting ``by`` on Spark leaves the
    cumulative window unpartitioned and collapses the data to one partition.
    """
    backend = _backend(df)
    groups = _groups(by)
    _validate_columns(df, _CONFUSION_COLUMNS, groups)
    reserved = {*_CONFUSION_COLUMNS, *_PR_COLUMNS}
    conflicts = [column for column in groups if column in reserved]
    if conflicts:
        raise ValueError(f"by conflicts with a confusion-table or metric column: {conflicts}")
    return backend.calculate_pr(df, groups)


def at(df: FrameT, by: ByColumns = None, **kwargs: Any) -> FrameT:
    """Return the lowest threshold satisfying one ``metric >= value`` target.

    A group with no qualifying threshold is absent from the result.  With
    replicas in ``by``, the result is a distribution of operating points.

    The metric arrives as a keyword argument, so ``by`` is reserved: a column
    of that name cannot be used as a metric target here.

    Use this with a metric that does not *increase* as the threshold falls --
    ``precision`` is the intended one.  Lowering the threshold then trades
    that metric for recall, and the lowest qualifying threshold is the most
    recall the target allows.

    ``recall`` is the opposite: it does not *decrease* as the threshold falls,
    so every threshold below some point clears the target and the lowest of
    them is simply the group's minimum threshold.  ``at(kpi, recall=0.5)``
    therefore returns the last row of each group, at recall ~1.0 and the worst
    precision, whatever the target was.  To pick an operating point by recall,
    take the *highest* qualifying threshold instead.

    On Spark, omitting ``by`` leaves the ranking window without a
    partition key and moves every row to a single partition.
    """
    if len(kwargs) != 1:
        raise ValueError(f"at() requires exactly one metric=value condition, got {len(kwargs)}")

    backend = _backend(df)
    groups = _groups(by)
    metric, value = next(iter(kwargs.items()))
    _validate_columns(df, ("threshold", metric), groups)
    if "threshold" in groups:
        raise ValueError("threshold cannot also appear in by")
    return backend.at(df, groups, metric, value)
