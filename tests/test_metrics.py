"""Tests for the precision-recall metric functions."""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F
from pyspark.sql import types as T

from replicas import at, calculate_pr, confusion_table


@pytest.fixture
def perfect_classifier(spark):
    """Predictions where score perfectly separates positives from negatives."""
    schema = T.StructType(
        [
            T.StructField("prediction", T.DoubleType(), True),
            T.StructField("positive", T.IntegerType(), True),
            T.StructField("negative", T.IntegerType(), True),
            T.StructField("unlabeled", T.IntegerType(), True),
        ]
    )
    rows = [
        (0.9, 1, 0, 0),
        (0.8, 1, 0, 0),
        (0.7, 1, 0, 0),
        (0.3, 0, 1, 0),
        (0.2, 0, 1, 0),
        (0.1, 0, 1, 0),
    ]
    return spark.createDataFrame(rows, schema)


def test_confusion_table_basic(perfect_classifier):
    ct = confusion_table(perfect_classifier).orderBy(F.col("threshold").desc())
    rows = ct.collect()

    # Six distinct scores → six rows.
    assert len(rows) == 6

    # Cumulative TP at the lowest threshold equals total positives.
    bottom = ct.orderBy("threshold").first()
    assert bottom["TP"] == 3
    assert bottom["FP"] == 3

    # Totals.
    assert rows[0]["positives"] == 3
    assert rows[0]["negatives"] == 3


def test_perfect_classifier_has_perfect_pr(perfect_classifier):
    ct = confusion_table(perfect_classifier)
    kpi = calculate_pr(ct)
    pr = {r["threshold"]: (r["precision"], r["recall"]) for r in kpi.collect()}

    # At every threshold ≥ 0.7, precision should be 1.0.
    for thr in (0.9, 0.8, 0.7):
        assert pr[thr][0] == pytest.approx(1.0)


def test_unlabeled_counted_as_up_not_fp(spark):
    """Unlabeled rows should contribute to UP, never to FP."""
    schema = T.StructType(
        [
            T.StructField("prediction", T.DoubleType(), True),
            T.StructField("positive", T.IntegerType(), True),
            T.StructField("negative", T.IntegerType(), True),
            T.StructField("unlabeled", T.IntegerType(), True),
        ]
    )
    rows = [
        (0.9, 0, 0, 1),  # unlabeled at top
        (0.8, 1, 0, 0),
        (0.5, 0, 1, 0),
    ]
    df = spark.createDataFrame(rows, schema)
    ct = confusion_table(df).orderBy(F.col("threshold").desc()).collect()

    # Top row: just the unlabeled.
    assert ct[0]["UP"] == 1
    assert ct[0]["TP"] == 0
    assert ct[0]["FP"] == 0

    # After the positive at 0.8: TP=1, UP=1.
    assert ct[1]["TP"] == 1
    assert ct[1]["UP"] == 1
    assert ct[1]["FP"] == 0


def test_at_returns_minimum_threshold(perfect_classifier):
    """`at(precision=1.0)` should return the lowest threshold still meeting it."""
    kpi = calculate_pr(confusion_table(perfect_classifier))
    op = at(kpi, precision=1.0).collect()
    assert len(op) == 1
    # Lowest threshold among the top-3 perfect-precision rows is 0.7.
    assert op[0]["threshold"] == pytest.approx(0.7)


def test_at_requires_one_condition(perfect_classifier):
    kpi = calculate_pr(confusion_table(perfect_classifier))
    with pytest.raises(ValueError):
        at(kpi)
    with pytest.raises(ValueError):
        at(kpi, precision=0.5, recall=0.5)
