"""Backend-neutral exact bootstrap sampling."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional, Union

from replicas._sampling import (
    normalize_columns,
    run_seed,
    validate_columns,
    validate_fraction,
    validate_n_replicas,
)

# This alias is evaluated at import time on Python 3.9, unlike postponed
# function annotations, so it must use typing's pre-PEP 604 spelling.
ColumnArgument = Optional[Union[str, list[str], tuple[str, ...]]]  # noqa: UP007, UP045


def _backend_name(df: Any) -> str:
    roots = {cls.__module__.partition(".")[0] for cls in type(df).__mro__}
    for root, backend in (("pandas", "pandas"), ("polars", "polars"), ("pyspark", "spark")):
        if root in roots:
            return backend
    raise TypeError(
        "unsupported dataframe type "
        f"{type(df).__module__}.{type(df).__qualname__}; expected pandas, Polars, or Spark"
    )


def _backend(df: Any) -> tuple[str, Any]:
    name = _backend_name(df)
    return name, import_module(f"replicas._backends.{name}")


def sample(
    df: Any,
    by: ColumnArgument = None,
    fraction: float = 1.0,
    *,
    seed: int | None = None,
    order_by: ColumnArgument = None,
) -> Any:
    """Sample with replacement, exactly and independently within each stratum.

    The input may be a pandas, Polars, or Spark DataFrame; the return value has
    the same dataframe type. A supplied seed pins the PCG64 draws. For Spark,
    seeded calls require ``order_by`` to define stable positions within every
    stratum; those columns must uniquely order each stratum.

    Args:
        df: A pandas, Polars, or Spark DataFrame to resample.
        by: Stratification columns, as one column name or a sequence of names.
          ``None`` treats the whole input as a single stratum.
        fraction: How many rows to draw from each stratum, relative to that
          stratum's size. Every stratum yields
          ``round(group_size * fraction)`` rows, matching pandas'
          round-to-nearest ``frac`` semantics: ``1.0`` draws a full-size
          replica, ``0.5`` draws half of each stratum, rounding a stratum of 5
          rows to 2, and a value above ``1.0`` draws more rows than the stratum
          holds. Must be finite and non-negative.
        seed: A non-negative run seed for reproducible draws, or ``None`` to
          take a fresh one from the OS.
        order_by: Columns defining stable positions within each stratum, as one
          name or a sequence of names. Required for seeded Spark calls. For a
          seeded draw to be reproducible these columns must uniquely order
          every stratum: rows tied on them can land in a different order from
          one run to the next. That uniqueness is the caller's responsibility
          and is not verified; only the presence of the columns is checked.

    Returns:
        A DataFrame of the same type and columns as ``df``, holding the drawn
        rows. Row order is unspecified on every backend.

    Raises:
        TypeError: ``df`` is not a pandas, Polars, or Spark DataFrame, or
          ``by``, ``order_by``, ``fraction``, or ``seed`` is of the wrong type.
        ValueError: A column named in ``by`` or ``order_by`` is missing from
          ``df`` or repeated, ``df`` has duplicate column names or already
          contains the reserved ``replica`` column, ``fraction`` is negative or
          not finite, ``seed`` is negative, or a seeded Spark call omits
          ``order_by``.
    """
    by_columns = normalize_columns(by, name="by")
    order_columns = normalize_columns(order_by, name="order_by")
    fraction = validate_fraction(fraction)
    seed_value, seeded = run_seed(seed)
    backend_name, backend = _backend(df)
    validate_columns(
        df.columns,
        by=by_columns,
        order_by=order_columns,
        reject_replica=True,
    )
    if backend_name == "spark" and seeded and not order_columns:
        raise ValueError("seeded Spark sampling requires order_by with unique stratum keys")
    return backend.sample(
        df,
        by=by_columns,
        fraction=fraction,
        run_seed=seed_value,
        order_by=order_columns,
    )


def bootstrap(
    df: Any,
    by: ColumnArgument = None,
    n_replicas: int = 100,
    checkpoint_dir: str | None = None,
    *,
    seed: int | None = None,
    order_by: ColumnArgument = None,
) -> Any:
    """Return the original data and exact bootstrap replicas in long format.

    Replica ``-1`` is the original input and replicas ``0..n_replicas-1`` are
    resampled independently within ``by`` strata. The input and result may be
    pandas, Polars, or Spark DataFrames and always share the same native type.

    With equivalent native strata, a common unique ``order_by``, and a seed,
    source-row multiplicities are identical across backends. NaN values in
    ``by`` or ``order_by`` are outside that guarantee: pandas folds null and
    NaN into one missing-value stratum while Polars and Spark keep them apart.
    See "Backends and reproducibility" in the README.

    Spark results are eagerly checkpointed once; local backends do not accept
    ``checkpoint_dir``. The checkpoint directory is session-wide Spark state,
    so it is set, used, and restored under a lock. One limitation of that
    restore: a directory configured outside this package gains one generated
    path level the first time it happens. See "Core API" in the README.

    Without ``by``, the whole Spark input is one stratum, and an exact draw
    needs it materialized in a single Python worker. Pass ``by`` on any input
    too large for one executor.

    Args:
        df: A pandas, Polars, or Spark DataFrame to resample.
        by: Stratification columns, as one column name or a sequence of names.
          ``None`` treats the whole input as a single stratum.
        n_replicas: How many resampled replicas to draw. Must be a non-negative
          integer; zero returns the original data alone.
        checkpoint_dir: Where the Spark backend writes its one eager
          checkpoint. Spark only; the local backends reject it. ``None`` reuses
          the session's configured directory, or falls back to a per-user
          ``replicas-<user>`` directory under the system temporary directory.
        seed: A non-negative run seed for reproducible draws, or ``None`` to
          take a fresh one from the OS.
        order_by: Columns defining stable positions within each stratum, as one
          name or a sequence of names. Required for seeded Spark calls. For a
          seeded draw to be reproducible these columns must uniquely order
          every stratum: rows tied on them can land in a different order from
          one run to the next. That uniqueness is the caller's responsibility
          and is not verified; only the presence of the columns is checked.

    Returns:
        A DataFrame of the same type as ``df``, in long format: the input
        columns followed by a 32-bit integer ``replica`` column, holding the
        original rows as replica ``-1`` and the resampled rows as replicas
        ``0..n_replicas-1``. Row order is unspecified on every backend.

    Raises:
        TypeError: ``df`` is not a pandas, Polars, or Spark DataFrame, or
          ``by``, ``order_by``, ``n_replicas``, or ``seed`` is of the wrong
          type.
        ValueError: A column named in ``by`` or ``order_by`` is missing from
          ``df`` or repeated, ``df`` has duplicate column names or already
          contains the reserved ``replica`` column, ``n_replicas`` or ``seed``
          is negative, ``checkpoint_dir`` is set on a non-Spark input, or a
          seeded Spark call omits ``order_by``.
    """
    by_columns = normalize_columns(by, name="by")
    order_columns = normalize_columns(order_by, name="order_by")
    n_replicas = validate_n_replicas(n_replicas)
    seed_value, seeded = run_seed(seed)
    backend_name, backend = _backend(df)
    validate_columns(
        df.columns,
        by=by_columns,
        order_by=order_columns,
        reject_replica=True,
    )
    if backend_name == "spark" and seeded and not order_columns:
        raise ValueError("seeded Spark bootstrap requires order_by with unique stratum keys")
    return backend.bootstrap(
        df,
        by=by_columns,
        n_replicas=n_replicas,
        checkpoint_dir=checkpoint_dir,
        run_seed=seed_value,
        order_by=order_columns,
    )
