"""Exact bootstrap sampling for pandas DataFrames."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from replicas._sampling import draw_indices, target_size


def _groups(df: pd.DataFrame, by: tuple[str, ...]) -> Iterator[tuple[Any, pd.DataFrame]]:
    if not by:
        yield (), df
        return

    grouper: str | list[str] = by[0] if len(by) == 1 else list(by)
    yield from df.groupby(grouper, dropna=False, observed=True, sort=False)


def _ordered(df: pd.DataFrame, order_by: tuple[str, ...]) -> pd.DataFrame:
    if not order_by:
        return df
    return df.sort_values(list(order_by), kind="mergesort", na_position="first")


def _group_key(key: Any) -> tuple[Any, ...]:
    """Normalize pandas' single missing-value group to the shared null key."""
    values = key if isinstance(key, tuple) else (key,)
    return tuple(None if bool(pd.isna(value)) else value for value in values)


def _sample_replica(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
    replica: int,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for key, group in _groups(df, by):
        source = _ordered(group, order_by)
        indices = draw_indices(
            len(source),
            run_seed,
            _group_key(key),
            replica,
            n_draws=target_size(len(source), fraction),
        )
        pieces.append(source.take(indices))
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, axis=0)


def sample(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
) -> pd.DataFrame:
    return _sample_replica(
        df,
        by=by,
        fraction=fraction,
        run_seed=run_seed,
        order_by=order_by,
        replica=0,
    )


def bootstrap(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...],
    n_replicas: int,
    checkpoint_dir: str | None,
    run_seed: int,
    order_by: tuple[str, ...],
) -> pd.DataFrame:
    if checkpoint_dir is not None:
        raise ValueError("checkpoint_dir is only supported for Spark DataFrames")

    original = df.copy()
    original["replica"] = -1
    pieces = [original]
    for replica in range(n_replicas):
        sampled = _sample_replica(
            df,
            by=by,
            fraction=1.0,
            run_seed=run_seed,
            order_by=order_by,
            replica=replica,
        ).copy()
        sampled["replica"] = replica
        pieces.append(sampled)
    return pd.concat(pieces, axis=0)
