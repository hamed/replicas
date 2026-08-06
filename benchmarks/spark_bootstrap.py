#!/usr/bin/env python3
"""Compare the legacy, grouped-pandas, and Arrow Spark bootstrap plans.

This is a deliberately small developer benchmark, not a CI performance gate.
It reports elapsed time, output rows, and local checkpoint bytes for balanced
and skewed synthetic inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ENGINES = ("legacy", "pandas", "arrow")
SHAPES = ("balanced", "skewed")


def _legacy_bootstrap(df, by: tuple[str, ...], n_replicas: int, checkpoint_dir: str):
    """The original implementation: one pandas UDF and union per replica."""
    from pyspark.sql import functions as F

    result = df.withColumn("replica", F.lit(-1))
    for replica in range(n_replicas):
        sampled = df.groupBy(list(by)).applyInPandas(
            lambda pdf: pdf.sample(frac=1.0, replace=True),
            schema=df.schema,
        )
        result = result.unionByName(sampled.withColumn("replica", F.lit(replica)))

    df.sparkSession.sparkContext.setCheckpointDir(checkpoint_dir)
    return result.checkpoint(eager=True)


def _current_bootstrap(
    df,
    engine: str,
    by: tuple[str, ...],
    n_replicas: int,
    checkpoint_dir: str,
    seed: int,
):
    """Call a private engine directly so the comparison controls selection."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    from replicas._backends import spark as backend

    original = df.withColumn("replica", F.lit(-1).cast(T.IntegerType()))
    if n_replicas == 0:
        return backend._checkpoint(original, checkpoint_dir)

    implementation = backend._bootstrap_pandas if engine == "pandas" else backend._bootstrap_arrow
    replicas = implementation(
        df,
        by=by,
        n_replicas=n_replicas,
        run_seed=seed,
        order_by=(),
    )
    return backend._checkpoint(original.unionByName(replicas), checkpoint_dir)


def _make_input(spark, *, rows: int, strata: int, shape: str, partitions: int):
    from pyspark.sql import functions as F

    frame = spark.range(rows, numPartitions=partitions)
    row_id = F.col("id")
    if shape == "balanced":
        stratum = F.pmod(row_id, F.lit(strata))
    else:
        cutoff = int(rows * 0.9)
        stratum = F.when(row_id < cutoff, F.lit(0)).otherwise(
            F.lit(1) + F.pmod(row_id - F.lit(cutoff), F.lit(strata - 1))
        )

    result = frame.select(
        row_id.alias("row_id"),
        stratum.cast("int").alias("stratum"),
        (F.pmod(row_id * 17, F.lit(1_000_003)) / 1_000_003.0).alias("value"),
    ).cache()
    if result.count() != rows:
        raise RuntimeError("synthetic input row count is incorrect")
    return result


def _checkpoint_bytes(directory: str) -> int:
    root = Path(directory)
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _run_case(df, *, engine: str, shape: str, repeat: int, args, spark_version: str) -> dict:
    from replicas._backends import spark as backend

    if engine == "arrow" and not backend._supports_arrow_iterator(spark_version):
        return {
            "engine": engine,
            "shape": shape,
            "repeat": repeat,
            "status": "skipped",
            "reason": "Spark 4.1+ is required for the Arrow iterator engine",
        }

    with tempfile.TemporaryDirectory(prefix="replicas-benchmark-") as temporary:
        checkpoint_dir = str(Path(temporary) / "checkpoint")
        started = time.perf_counter()
        if engine == "legacy":
            result = _legacy_bootstrap(df, ("stratum",), args.replicas, checkpoint_dir)
        else:
            result = _current_bootstrap(
                df,
                engine,
                ("stratum",),
                args.replicas,
                checkpoint_dir,
                args.seed,
            )
        wall_seconds = time.perf_counter() - started
        output_rows = result.count()
        checkpoint_bytes = _checkpoint_bytes(checkpoint_dir)

    expected_rows = args.rows * (args.replicas + 1)
    if output_rows != expected_rows:
        raise RuntimeError(f"{engine} produced {output_rows} rows; expected {expected_rows}")

    return {
        "engine": engine,
        "shape": shape,
        "repeat": repeat,
        "status": "ok",
        "rows": args.rows,
        "replicas": args.replicas,
        "wall_seconds": wall_seconds,
        "output_rows": output_rows,
        "checkpoint_bytes": checkpoint_bytes,
        "spark_version": spark_version,
        "python_version": sys.version.split()[0],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", default="-", help="JSONL path; default is stdout")
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.strata < 2:
        parser.error("--strata must be at least 2")
    if args.replicas < 0:
        parser.error("--replicas must be non-negative")
    if args.repeats < 1 or args.partitions < 1 or args.shuffle_partitions < 1:
        parser.error("repeat and partition counts must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(args.master)
        .appName("replicas-bootstrap-benchmark")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    records = []
    try:
        for shape in dict.fromkeys(args.shapes):
            frame = _make_input(
                spark,
                rows=args.rows,
                strata=args.strata,
                shape=shape,
                partitions=args.partitions,
            )
            try:
                for repeat in range(args.repeats):
                    for engine in dict.fromkeys(args.engines):
                        print(
                            f"running shape={shape} repeat={repeat} engine={engine}",
                            file=sys.stderr,
                        )
                        records.append(
                            _run_case(
                                frame,
                                engine=engine,
                                shape=shape,
                                repeat=repeat,
                                args=args,
                                spark_version=spark.version,
                            )
                        )
            finally:
                frame.unpersist()
    finally:
        spark.stop()

    stream = sys.stdout if args.output == "-" else Path(args.output).open("w", encoding="utf-8")
    try:
        for record in records:
            print(json.dumps(record, sort_keys=True), file=stream)
    finally:
        if stream is not sys.stdout:
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
