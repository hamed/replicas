"""Focused conformance tests for the Spark sampling adapter."""

from __future__ import annotations

from collections import Counter

import pytest
from pyspark.sql import functions as F

from replicas._backends import spark as spark_backend
from replicas._sampling import draw_indices


@pytest.fixture
def stratified_rows(spark):
    return spark.createDataFrame(
        [
            (0, "a", 0.1),
            (1, "a", 0.2),
            (2, "a", 0.3),
            (3, None, 0.4),
            (4, None, 0.5),
        ],
        "row_id long, stratum string, value double",
    )


def _fingerprint(df):
    return Counter(
        (row["replica"], row["stratum"], row["row_id"])
        for row in df.select("replica", "stratum", "row_id").collect()
    )


def test_arrow_iterator_feature_detection(monkeypatch):
    # This test isolates the version gate. Spark 3.3's GroupedData correctly
    # lacks applyInArrow, while Spark 4.1+ supplies it.
    monkeypatch.setattr(spark_backend.GroupedData, "applyInArrow", object(), raising=False)

    assert not spark_backend._supports_arrow_iterator("3.5.6")
    assert not spark_backend._supports_arrow_iterator("4.0.1")
    assert spark_backend._supports_arrow_iterator("4.1.0")
    assert spark_backend._supports_arrow_iterator("4.2.0.dev0")


def test_helper_columns_respect_spark_case_insensitive_names(spark):
    frame = spark.createDataFrame(
        [(1, 2, 3)],
        ["__REPLICAS_BATCH", "__replicas_batch_1", "__REPLICAS_REPLICA"],
    )

    assert spark_backend._unique_helper_column(frame, "__replicas_batch") == "__replicas_batch_2"
    assert (
        spark_backend._unique_helper_column(frame, "__replicas_replica") == "__replicas_replica_1"
    )


@pytest.mark.parametrize("engine", ["pandas", "arrow"])
def test_exact_strata_and_null_groups(stratified_rows, tmp_path, monkeypatch, engine):
    if engine == "arrow" and not spark_backend._supports_arrow_iterator():
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: engine)

    result = spark_backend.bootstrap(
        stratified_rows,
        by=("stratum",),
        n_replicas=11,
        checkpoint_dir=str(tmp_path / engine),
        run_seed=41,
        order_by=("row_id",),
    )

    counts = {
        (row["replica"], row["stratum"]): row["count"]
        for row in result.groupBy("replica", "stratum").count().collect()
    }
    assert sorted(result.select("replica").distinct().toPandas()["replica"]) == list(range(-1, 11))
    for replica in range(-1, 11):
        assert counts[replica, "a"] == 3
        assert counts[replica, None] == 2


def test_engines_have_identical_seeded_multiplicities(stratified_rows, tmp_path, monkeypatch):
    if not spark_backend._supports_arrow_iterator():
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")

    monkeypatch.setattr(spark_backend, "_select_engine", lambda: "pandas")
    pandas_result = spark_backend.bootstrap(
        stratified_rows,
        by=("stratum",),
        n_replicas=4,
        checkpoint_dir=str(tmp_path / "pandas-equivalence"),
        run_seed=2026,
        order_by=("row_id",),
    )
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: "arrow")
    arrow_result = spark_backend.bootstrap(
        stratified_rows,
        by=("stratum",),
        n_replicas=4,
        checkpoint_dir=str(tmp_path / "arrow-equivalence"),
        run_seed=2026,
        order_by=("row_id",),
    )

    assert _fingerprint(pandas_result) == _fingerprint(arrow_result)


@pytest.mark.parametrize("engine", ["pandas", "arrow"])
def test_null_and_nan_strata_use_distinct_seed_streams(spark, monkeypatch, engine):
    if engine == "arrow" and not spark_backend._supports_arrow_iterator():
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: engine)
    rows = [
        (0, None),
        (1, None),
        (2, None),
        (3, float("nan")),
        (4, float("nan")),
        (5, float("nan")),
    ]
    frame = spark.createDataFrame(rows, "row_id long, stratum double")
    seed = 0

    sampled = spark_backend.sample(
        frame,
        by=("stratum",),
        fraction=1.0,
        run_seed=seed,
        order_by=("row_id",),
    )

    sampled_ids = [row["row_id"] for row in sampled.select("row_id").collect()]
    # Identify source groups by their disjoint row IDs. The pandas UDF transport
    # itself may represent both Spark null and NaN as pandas NaN on output.
    actual_null = Counter(row_id for row_id in sampled_ids if row_id < 3)
    actual_nan = Counter(row_id - 3 for row_id in sampled_ids if row_id >= 3)
    expected_null = Counter(draw_indices(3, seed, (None,)).tolist())
    expected_nan = Counter(draw_indices(3, seed, (float("nan"),)).tolist())

    assert expected_null != expected_nan
    assert actual_null == expected_null
    assert actual_nan == expected_nan


def test_replica_batch_size_does_not_change_draws(stratified_rows, tmp_path, monkeypatch):
    if not spark_backend._supports_arrow_iterator():
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: "arrow")
    fingerprints = []
    for batch_size in (1, 3, 10):
        monkeypatch.setattr(spark_backend, "_REPLICAS_PER_BATCH", batch_size)
        result = spark_backend.bootstrap(
            stratified_rows,
            by=("stratum",),
            n_replicas=5,
            checkpoint_dir=str(tmp_path / f"batch-{batch_size}"),
            run_seed=2027,
            order_by=("row_id",),
        )
        fingerprints.append(_fingerprint(result))

    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


@pytest.mark.parametrize("engine", ["pandas", "arrow"])
@pytest.mark.parametrize("fraction, expected", [(0.5, {"a": 2, None: 1}), (0.0, {})])
def test_sample_fraction_is_exact(stratified_rows, monkeypatch, engine, fraction, expected):
    if engine == "arrow" and not spark_backend._supports_arrow_iterator():
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: engine)

    sampled = spark_backend.sample(
        stratified_rows,
        by=("stratum",),
        fraction=fraction,
        run_seed=9,
        order_by=("row_id",),
    )

    counts = {row["stratum"]: row["count"] for row in sampled.groupBy("stratum").count().collect()}
    assert counts == expected


@pytest.mark.parametrize("engine", ["pandas", "arrow"])
def test_ungrouped_sampling_and_empty_inputs(stratified_rows, tmp_path, monkeypatch, engine):
    if engine == "arrow" and not spark_backend._supports_arrow_iterator():
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: engine)

    sampled = spark_backend.sample(
        stratified_rows,
        by=(),
        fraction=1.0,
        run_seed=10,
        order_by=("row_id",),
    )
    empty = stratified_rows.limit(0)
    empty_result = spark_backend.bootstrap(
        empty,
        by=(),
        n_replicas=2,
        checkpoint_dir=str(tmp_path / f"empty-{engine}"),
        run_seed=10,
        order_by=("row_id",),
    )

    assert sampled.count() == stratified_rows.count()
    assert empty_result.count() == 0
    assert empty_result.columns == [*stratified_rows.columns, "replica"]


def test_checkpointed_result_is_stable(stratified_rows, tmp_path, monkeypatch):
    monkeypatch.setattr(spark_backend, "_select_engine", lambda: "pandas")
    result = spark_backend.bootstrap(
        stratified_rows,
        by=("stratum",),
        n_replicas=3,
        checkpoint_dir=str(tmp_path / "stable"),
        run_seed=7,
        order_by=(),
    )

    first = result.groupBy("replica").agg(F.sum("row_id").alias("total")).collect()
    second = result.groupBy("replica").agg(F.sum("row_id").alias("total")).collect()
    assert first == second


def test_zero_replicas_returns_only_original(stratified_rows, tmp_path):
    result = spark_backend.bootstrap(
        stratified_rows,
        by=("stratum",),
        n_replicas=0,
        checkpoint_dir=str(tmp_path / "zero"),
        run_seed=8,
        order_by=("row_id",),
    )

    assert result.count() == stratified_rows.count()
    assert [row["replica"] for row in result.select("replica").distinct().collect()] == [-1]


@pytest.mark.parametrize(
    "implementation",
    [spark_backend._bootstrap_pandas, spark_backend._bootstrap_arrow],
)
def test_logical_plan_shape_is_constant_as_replicas_grow(stratified_rows, implementation):
    if (
        implementation is spark_backend._bootstrap_arrow
        and not spark_backend._supports_arrow_iterator()
    ):
        pytest.skip("Spark 4.1+ is required for the Arrow iterator engine")

    def plan_lines(n_replicas):
        result = implementation(
            stratified_rows,
            by=("stratum",),
            n_replicas=n_replicas,
            run_seed=12,
            order_by=("row_id",),
        )
        return result._jdf.queryExecution().logical().treeString().splitlines()

    assert len(plan_lines(1)) == len(plan_lines(100))
