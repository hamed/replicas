"""Spark implementation of exact, stratified bootstrap sampling.

The pandas and Arrow engines deliberately implement the same small algorithm:
sort a stratum when reproducibility was requested, draw positional indices,
and take those rows. The pandas fallback processes one replica per group;
Arrow batches keep the plan shallow while yielding one replica at a time.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyspark
from pyspark.sql import DataFrame, GroupedData
from pyspark.sql import functions as F
from pyspark.sql import types as T

from replicas._sampling import draw_indices, target_size

_REPLICAS_PER_BATCH = 10
_LOCAL_CHECKPOINT_DIR = "/tmp/replicas/"


def _version_pair(version: str) -> tuple[int, int]:
    """Return Spark's major/minor pair, including for vendor/dev versions."""
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def _supports_arrow_iterator(version: str | None = None) -> bool:
    """Whether grouped Arrow RecordBatch iterators are available."""
    return _version_pair(version or pyspark.__version__) >= (4, 1) and hasattr(
        GroupedData, "applyInArrow"
    )


def _select_engine() -> str:
    """Select the fastest implementation with the required streaming API."""
    return "arrow" if _supports_arrow_iterator() else "pandas"


def _as_python(value: Any) -> Any:
    if isinstance(value, pa.Scalar):
        return value.as_py()
    if hasattr(value, "item"):
        return value.item()
    return value


def _grouping_columns(df: DataFrame, by: tuple[str, ...]) -> list[Any]:
    # The null flags let pandas distinguish a Spark null in a numeric grouping
    # column (received as NaN) from a real NaN value.
    values = [df[column] for column in by]
    null_flags = [df[column].isNull() for column in by]
    return [*values, *null_flags]


def _group_key(keys: tuple[Any, ...], size: int) -> tuple[Any, ...]:
    values = keys[:size]
    null_flags = keys[size : size * 2]
    return tuple(
        None if bool(_as_python(is_null)) else _as_python(value)
        for value, is_null in zip(values, null_flags)
    )


def _sort_pandas(pdf: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    source = pdf
    if columns:
        source = source.sort_values(list(columns), kind="mergesort", na_position="first")
    return source.reset_index(drop=True)


def _sort_arrow(table: pa.Table, columns: tuple[str, ...]) -> pa.Table:
    if not columns:
        return table
    indices = pc.sort_indices(
        table,
        sort_keys=[(column, "ascending") for column in columns],
        null_placement="at_start",
    )
    return table.take(indices)


def _sample_pandas(
    df: DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
) -> DataFrame:
    columns = tuple(df.columns)
    grouping = _grouping_columns(df, by)

    def resample(keys: tuple[Any, ...], pdf: pd.DataFrame) -> pd.DataFrame:
        source = _sort_pandas(pdf.loc[:, list(columns)], order_by)
        n_draws = target_size(len(source), fraction)
        indices = draw_indices(
            len(source),
            run_seed,
            _group_key(keys, len(by)),
            replica=0,
            n_draws=n_draws,
        )
        return source.take(indices).reset_index(drop=True)

    return df.groupBy(*grouping).applyInPandas(resample, schema=df.schema)


def _sample_arrow(
    df: DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
) -> DataFrame:
    columns = tuple(df.columns)
    grouping = _grouping_columns(df, by)

    def resample(
        keys: tuple[pa.Scalar, ...], batches: Iterator[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        source = _sort_arrow(pa.Table.from_batches(batches).select(columns), order_by)
        n_draws = target_size(source.num_rows, fraction)
        indices = draw_indices(
            source.num_rows,
            run_seed,
            _group_key(keys, len(by)),
            replica=0,
            n_draws=n_draws,
        )
        yield from source.take(pa.array(indices)).to_batches()

    return df.groupBy(*grouping).applyInArrow(resample, schema=df.schema)


def _unique_helper_column(df: DataFrame, stem: str) -> str:
    name = stem
    suffix = 0
    while name in df.columns:
        suffix += 1
        name = f"{stem}_{suffix}"
    return name


def _batched_input(df: DataFrame, n_replicas: int) -> tuple[DataFrame, str]:
    batch_column = _unique_helper_column(df, "__replicas_batch")
    n_batches = math.ceil(n_replicas / _REPLICAS_PER_BATCH)
    batches = df.sparkSession.range(n_batches).select(
        F.col("id").cast(T.IntegerType()).alias(batch_column)
    )
    return df.crossJoin(F.broadcast(batches)), batch_column


def _replicated_input(df: DataFrame, n_replicas: int) -> tuple[DataFrame, str]:
    replica_column = _unique_helper_column(df, "__replicas_replica")
    replicas = df.sparkSession.range(n_replicas).select(
        F.col("id").cast(T.IntegerType()).alias(replica_column)
    )
    return df.crossJoin(F.broadcast(replicas)), replica_column


def _bootstrap_pandas(
    df: DataFrame,
    *,
    by: tuple[str, ...],
    n_replicas: int,
    run_seed: int,
    order_by: tuple[str, ...],
) -> DataFrame:
    columns = tuple(df.columns)
    expanded, replica_column = _replicated_input(df, n_replicas)
    grouping = [*_grouping_columns(expanded, by), expanded[replica_column]]
    output_schema = T.StructType(
        [*df.schema.fields, T.StructField("replica", T.IntegerType(), nullable=False)]
    )

    def resample(keys: tuple[Any, ...], pdf: pd.DataFrame) -> pd.DataFrame:
        source = _sort_pandas(pdf.loc[:, list(columns)], order_by)
        group_key = _group_key(keys, len(by))
        replica = int(_as_python(keys[-1]))
        sampled = source.take(
            draw_indices(len(source), run_seed, group_key, replica=replica)
        ).copy()
        sampled["replica"] = pd.Series(replica, index=sampled.index, dtype="int32")
        return sampled.reset_index(drop=True)

    return expanded.groupBy(*grouping).applyInPandas(resample, schema=output_schema)


def _bootstrap_arrow(
    df: DataFrame,
    *,
    by: tuple[str, ...],
    n_replicas: int,
    run_seed: int,
    order_by: tuple[str, ...],
) -> DataFrame:
    columns = tuple(df.columns)
    expanded, batch_column = _batched_input(df, n_replicas)
    grouping = [*_grouping_columns(expanded, by), expanded[batch_column]]
    output_schema = T.StructType(
        [*df.schema.fields, T.StructField("replica", T.IntegerType(), nullable=False)]
    )

    def resample(
        keys: tuple[pa.Scalar, ...], batches: Iterator[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        source = _sort_arrow(pa.Table.from_batches(batches).select(columns), order_by)
        group_key = _group_key(keys, len(by))
        batch = int(_as_python(keys[-1]))
        first = batch * _REPLICAS_PER_BATCH
        stop = min(first + _REPLICAS_PER_BATCH, n_replicas)
        for replica in range(first, stop):
            sampled = source.take(
                pa.array(draw_indices(source.num_rows, run_seed, group_key, replica=replica))
            )
            for record_batch in sampled.to_batches():
                replica_column = pa.array([replica] * record_batch.num_rows, type=pa.int32())
                yield record_batch.append_column("replica", replica_column)

    return expanded.groupBy(*grouping).applyInArrow(resample, schema=output_schema)


def _checkpoint(df: DataFrame, checkpoint_dir: str | None) -> DataFrame:
    spark_context = df.sparkSession.sparkContext
    if checkpoint_dir is not None:
        spark_context.setCheckpointDir(checkpoint_dir)
    elif spark_context.getCheckpointDir() is None:
        if not spark_context.master.startswith("local"):
            raise ValueError(
                "Spark has no checkpoint directory; configure one or pass checkpoint_dir"
            )
        spark_context.setCheckpointDir(_LOCAL_CHECKPOINT_DIR)
    return df.checkpoint(eager=True)


def sample(
    df: DataFrame,
    *,
    by: tuple[str, ...],
    fraction: float,
    run_seed: int,
    order_by: tuple[str, ...],
) -> DataFrame:
    """Sample with replacement within each Spark stratum."""
    implementation = _sample_arrow if _select_engine() == "arrow" else _sample_pandas
    return implementation(
        df,
        by=by,
        fraction=fraction,
        run_seed=run_seed,
        order_by=order_by,
    )


def bootstrap(
    df: DataFrame,
    *,
    by: tuple[str, ...],
    n_replicas: int,
    checkpoint_dir: str | None,
    run_seed: int,
    order_by: tuple[str, ...],
) -> DataFrame:
    """Generate exact Spark bootstrap replicas and eagerly checkpoint them."""
    original = df.withColumn("replica", F.lit(-1).cast(T.IntegerType()))
    if n_replicas == 0:
        return _checkpoint(original, checkpoint_dir)

    implementation = _bootstrap_arrow if _select_engine() == "arrow" else _bootstrap_pandas
    replicas = implementation(
        df,
        by=by,
        n_replicas=n_replicas,
        run_seed=run_seed,
        order_by=order_by,
    )
    return _checkpoint(original.unionByName(replicas), checkpoint_dir)


__all__ = ["bootstrap", "sample"]
