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
_BAND_INPUT_COLUMNS = ("replica", "recall", "precision")
_BAND_COLUMNS = ("recall", "precision", "low", "high")
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

    Args:
        df: A pandas, Polars, or Spark DataFrame carrying the ``prediction``,
          ``positive``, ``negative``, and ``unlabeled`` columns described in
          the module docstring.
        by: Grouping columns, as one column name or a sequence of names.
          ``None`` builds one table over the whole input. Pass ``replica`` for
          bootstrap output.

    Returns:
        A DataFrame of the same type as ``df``, with the ``by`` columns
        followed by ``threshold``, cumulative ``TP``/``FP``/``UP``, per-score
        ``dTP``/``dFP``/``dUP``, and group totals
        ``positives``/``negatives``/``unlabeled``.

    Raises:
        TypeError: ``df`` is not a pandas, Polars, or Spark DataFrame, or
          ``by`` is not a column name or a sequence of column names.
        ValueError: ``df`` has duplicate column names or is missing a required
          or grouping column, ``by`` repeats a column, or a ``by`` column
          collides with a name this function reads or produces.
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

    Args:
        df: A pandas, Polars, or Spark DataFrame produced by
          :func:`confusion_table`.
        by: Grouping columns, as one column name or a sequence of names, and
          the same grouping used to build ``df``. ``None`` treats the whole
          input as one group.

    Returns:
        A DataFrame of the same type as ``df``, holding the confusion-table
        columns plus ``precision``, ``recall``, and ``average_precision``, with
        thresholds that add no true positives dropped.

    Raises:
        TypeError: ``df`` is not a pandas, Polars, or Spark DataFrame, or
          ``by`` is not a column name or a sequence of column names.
        ValueError: ``df`` has duplicate column names or is missing a
          confusion-table or grouping column, ``by`` repeats a column, or a
          ``by`` column collides with a name this function reads or produces.
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

    Args:
        df: A pandas, Polars, or Spark DataFrame carrying ``threshold`` and the
          targeted metric column, usually the output of :func:`calculate_pr`.
        by: Grouping columns, as one column name or a sequence of names.
          ``None`` treats the whole input as one group. Include ``replica`` to
          get one operating point per replica.
        **kwargs: Exactly one ``metric=value`` target, naming the metric column
          to threshold on and the value it must reach, as in
          ``precision=0.95``.

    Returns:
        A DataFrame of the same type as ``df``, with its columns unchanged,
        holding the single row per group whose ``threshold`` is the lowest one
        satisfying the target. Groups with no qualifying threshold are absent.
        Selecting the lowest qualifying threshold suits a metric that does not
        increase as the threshold falls, which is the case ``at`` is built
        for. Any metric runs; for one with the opposite monotonicity the same
        rule may not pick the intended operating point, as the ``recall``
        example above shows.

    Raises:
        TypeError: ``df`` is not a pandas, Polars, or Spark DataFrame, or
          ``by`` is not a column name or a sequence of column names.
        ValueError: ``kwargs`` does not hold exactly one condition, ``df`` has
          duplicate column names or is missing ``threshold``, the metric
          column, or a grouping column, ``by`` repeats a column, or ``by``
          contains ``threshold``.
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


def pr_band(
    df: FrameT,
    by: ByColumns = None,
    *,
    ci: float = 0.9,
    recall_round: Optional[int] = None,  # noqa: UP045
) -> FrameT:
    """Reduce replicate PR curves to one curve with a confidence band.

    Takes the output of :func:`calculate_pr` for a frame carrying a ``replica``
    column, and returns ``by``, ``recall``, ``precision``, ``low``, ``high``.
    ``precision`` is the original curve (replica ``-1``); ``low`` and ``high``
    are the ``0.5 -/+ ci / 2`` quantiles of the replica curves.  Either side is
    null at a recall value the other does not reach.  Rows sharing a recall
    inside one replica collapse to their highest precision first -- the upper
    envelope is the curve a threshold sweep actually reaches.

    ``recall_round`` rounds recall before the aggregation.  It has to happen
    here rather than in a caller: replica curves rarely land on identical
    recall values, and once the quantiles are taken over exact values the
    sparse band cannot be recovered.  Leave it ``None`` on large data.

    This is the reduction behind :func:`replicas.plotting.plot_pr`, public
    because the numbers are useful without the picture.  It runs natively on
    each backend, so a large curve table is reduced before anything is
    collected.  All three interpolate quantiles linearly, so equivalent input
    gives equal band edges to within floating-point error.  pandas and Polars
    sort by ``by`` then ``recall``; Spark row order is unspecified, as usual.

    Args:
        df: A pandas, Polars, or Spark DataFrame produced by
          :func:`calculate_pr`, carrying ``replica``, ``recall``, and
          ``precision`` columns.
        by: Grouping columns, as one column name or a sequence of names, giving
          one band per group. ``None`` reduces the whole input to one band.
          Do not include ``replica``: the replicas are what the band is taken
          over.
        ci: Width of the band as a share of the replica distribution, so
          ``0.9`` puts ``low`` and ``high`` at the 5th and 95th percentiles.
          Must lie in ``(0, 1]``.
        recall_round: Decimal places to round ``recall`` to before aggregating,
          or ``None`` to keep exact values. Rounding groups nearby recall
          values together, which makes the band less sparse on small data,
          where replica curves rarely land on identical recall values. It also
          changes which replicas contribute at each recall value, so a band
          edge can move in either direction.

    Returns:
        A DataFrame of the same type as ``df``, with the ``by`` columns
        followed by ``recall``, ``precision`` (the original replica ``-1``
        curve), and the ``low``/``high`` band edges. Either edge is null at a
        recall the other does not reach.

    Raises:
        TypeError: ``df`` is not a pandas, Polars, or Spark DataFrame, ``by``
          is not a column name or a sequence of column names, or
          ``recall_round`` is neither an integer nor ``None``.
        ValueError: ``ci`` lies outside ``(0, 1]``, ``df`` has duplicate column
          names or is missing ``replica``, ``recall``, ``precision``, or a
          grouping column, ``by`` repeats a column, or a ``by`` column collides
          with a name this function reads or produces.
    """
    if not 0 < ci <= 1:
        raise ValueError(f"ci must be in the interval (0, 1], got {ci!r}")
    if recall_round is not None and (
        isinstance(recall_round, bool) or not isinstance(recall_round, int)
    ):
        raise TypeError("recall_round must be an integer or None")

    backend = _backend(df)
    groups = _groups(by)
    _validate_columns(df, _BAND_INPUT_COLUMNS, groups)
    reserved = {*_BAND_INPUT_COLUMNS, *_BAND_COLUMNS}
    conflicts = [column for column in groups if column in reserved]
    if conflicts:
        raise ValueError(f"by conflicts with a column pr_band reads or produces: {conflicts}")

    return backend.pr_band(df, groups, 0.5 - ci / 2, 0.5 + ci / 2, recall_round)
