"""Exact bootstrap sampling for Polars DataFrames."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import polars as pl

from replicas._sampling import draw_indices, target_size


def _groups(df: pl.DataFrame, by: tuple[str, ...]) -> Iterator[tuple[Any, pl.DataFrame]]:
    if not by:
        yield (), df
        return
    yield from df.partition_by(list(by), as_dict=True, maintain_order=True).items()


def _ordered(df: pl.DataFrame, order_by: tuple[str, ...]) -> pl.DataFrame:
    if not order_by:
        return df
    return df.sort(list(order_by), nulls_last=False, maintain_order=True)


def _sample_replica(
    df: pl.DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
    replica: int,
) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []
    for key, group in _groups(df, by):
        source = _ordered(group, order_by)
        indices = draw_indices(
            source.height,
            run_seed,
            key,
            replica,
            n_draws=target_size(source.height, fraction),
        )
        # Polars maps a sequence of integers to positional row access.
        pieces.append(source[indices.tolist()])
    if not pieces:
        return df.clear()
    return pl.concat(pieces, how="vertical", rechunk=True)


def sample(
    df: pl.DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
) -> pl.DataFrame:
    return _sample_replica(
        df,
        by=by,
        fraction=fraction,
        run_seed=run_seed,
        order_by=order_by,
        replica=0,
    )


def bootstrap(
    df: pl.DataFrame,
    *,
    by: tuple[str, ...],
    n_replicas: int,
    checkpoint_dir: str | None,
    run_seed: int,
    order_by: tuple[str, ...],
) -> pl.DataFrame:
    if checkpoint_dir is not None:
        raise ValueError("checkpoint_dir is only supported for Spark DataFrames")

    pieces = [df.with_columns(pl.lit(-1, dtype=pl.Int64).alias("replica"))]
    for replica in range(n_replicas):
        sampled = _sample_replica(
            df,
            by=by,
            fraction=1.0,
            run_seed=run_seed,
            order_by=order_by,
            replica=replica,
        ).with_columns(pl.lit(replica, dtype=pl.Int64).alias("replica"))
        pieces.append(sampled)
    return pl.concat(pieces, how="vertical", rechunk=True)
