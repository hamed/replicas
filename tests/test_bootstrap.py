"""Tests for the bootstrap primitive.

The most important test in the whole library is `test_replicas_are_stable`:
it verifies that the `checkpoint()` call inside `bootstrap` actually does its
job — that "replica 7" is "replica 7" no matter how many times we query it.

Without `checkpoint()`, Spark would re-roll the random draws on every action,
two `.toPandas()` calls would return different replicas under the same IDs,
and recall would routinely come out greater than 1. The fact that this test
passes is the whole reason the library exists.
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F
from pyspark.sql import types as T

from replicas import bootstrap, calculate_pr, confusion_table


@pytest.fixture
def toy_predictions(spark):
    """A small predictions DataFrame with positives, negatives, unlabeled."""
    schema = T.StructType(
        [
            T.StructField("prediction", T.DoubleType(), True),
            T.StructField("positive", T.IntegerType(), True),
            T.StructField("negative", T.IntegerType(), True),
            T.StructField("unlabeled", T.IntegerType(), True),
            T.StructField("name", T.StringType(), True),
        ]
    )
    rows = [
        (0.95, 1, 0, 0, "m"),
        (0.90, 1, 0, 0, "m"),
        (0.85, 0, 1, 0, "m"),
        (0.80, 1, 0, 0, "m"),
        (0.70, 0, 1, 0, "m"),
        (0.65, 0, 0, 1, "m"),  # unlabeled
        (0.55, 1, 0, 0, "m"),
        (0.45, 0, 1, 0, "m"),
        (0.30, 0, 1, 0, "m"),
        (0.20, 1, 0, 0, "m"),
        (0.10, 0, 1, 0, "m"),
        (0.05, 0, 1, 0, "m"),
    ]
    return spark.createDataFrame(rows, schema)


def test_bootstrap_adds_replica_column(toy_predictions):
    bts = bootstrap(toy_predictions, by=["name", "positive"], n_replicas=5)
    assert "replica" in bts.columns


def test_bootstrap_replica_ids(toy_predictions):
    """Replica -1 is original; 0..n-1 are the resamples."""
    bts = bootstrap(toy_predictions, by=["name", "positive"], n_replicas=10)
    ids = sorted(r["replica"] for r in bts.select("replica").distinct().collect())
    assert ids == list(range(-1, 10))


def test_replicas_are_stable(toy_predictions):
    """The whole point of checkpoint(): a replica's content does not change
    between actions on the result.

    Without checkpoint, the two .toPandas() calls below would yield different
    samples for the same replica id, and the row count check would fail
    sporadically.
    """
    bts = bootstrap(toy_predictions, by=["name", "positive"], n_replicas=20)

    # Take a fingerprint of each replica's content via two separate actions.
    fingerprint_a = (
        bts.groupBy("replica")
        .agg(F.sum("prediction").alias("s"))
        .toPandas()
        .set_index("replica")
        .sort_index()
    )
    fingerprint_b = (
        bts.groupBy("replica")
        .agg(F.sum("prediction").alias("s"))
        .toPandas()
        .set_index("replica")
        .sort_index()
    )

    # If the content were re-rolled, these would differ.
    assert (fingerprint_a == fingerprint_b).all().all()


def test_recall_never_exceeds_one(toy_predictions):
    """The classic symptom of an un-checkpointed bootstrap: recall > 1.

    Across many replicas, recall must stay in [0, 1]. If checkpoint() were
    removed from bootstrap(), this test would fail.
    """
    bts = bootstrap(toy_predictions, by=["name", "positive"], n_replicas=30)
    ct = confusion_table(bts, group_by=["name", "replica"])
    kpi = calculate_pr(ct, group_by=["name", "replica"])

    rows = kpi.select(F.max("recall").alias("max_r"), F.min("recall").alias("min_r")).collect()[0]
    assert rows["max_r"] <= 1.0 + 1e-9
    assert rows["min_r"] >= 0.0 - 1e-9


def test_original_replica_size_matches_input(toy_predictions):
    bts = bootstrap(toy_predictions, by=["name", "positive"], n_replicas=5)
    n_original = bts.filter(F.col("replica") == -1).count()
    assert n_original == toy_predictions.count()


def test_resampled_replicas_preserve_strata_size(toy_predictions):
    """With fraction=1.0, each stratum is resampled to its original size."""
    n_pos = toy_predictions.filter(F.col("positive") == 1).count()
    n_neg = toy_predictions.filter(F.col("positive") == 0).count()

    bts = bootstrap(toy_predictions, by=["name", "positive"], n_replicas=10)
    # Pick one replica and check the per-stratum counts.
    rep0 = bts.filter(F.col("replica") == 0)
    assert rep0.filter(F.col("positive") == 1).count() == n_pos
    assert rep0.filter(F.col("positive") == 0).count() == n_neg


def test_bootstrap_without_strata(toy_predictions):
    """No strata: still works, total size preserved per replica."""
    n = toy_predictions.count()
    bts = bootstrap(toy_predictions, by=[], n_replicas=5)
    rep0 = bts.filter(F.col("replica") == 0).count()
    assert rep0 == n
