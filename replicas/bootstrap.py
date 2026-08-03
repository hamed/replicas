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

    With a common unique ``order_by`` and seed, source-row multiplicities are
    identical across backends. Spark results are eagerly checkpointed once;
    local backends do not accept ``checkpoint_dir``.
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
