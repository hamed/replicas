"""Backend conformance tests for the notebook's precision-recall pipeline."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest
from pyspark.sql import functions as F
from pyspark.sql import types as T

from replicas.metrics import at, calculate_pr, confusion_table

NOTEBOOK_COLUMNS = ["prediction", "negative", "positive", "unlabeled", "name"]
NOTEBOOK_ROWS = [
    (0.98, 0, 1, 0, "model"),
    (0.97, 0, 1, 0, "model"),
    (0.96, 1, 0, 0, "model"),
    (0.95, 0, 1, 0, "model"),
    (0.95, 0, 1, 0, "model"),
    (0.94, 1, 0, 0, "model"),
    (0.93, 0, 1, 0, "model"),
    (0.92, 0, 1, 0, "model"),
    (0.91, 0, 1, 0, "model"),
    (0.90, 0, 1, 0, "model"),
    (0.88, 0, 1, 0, "model"),
    (0.86, 0, 0, 1, "model"),
    (0.75, 1, 0, 0, "model"),
    (0.73, 0, 1, 0, "model"),
    (0.55, 1, 0, 0, "model"),
    (0.53, 1, 0, 0, "model"),
    (0.41, 1, 0, 0, "model"),
    (0.36, 1, 0, 0, "model"),
    (0.36, 1, 0, 0, "model"),
    (0.36, 1, 0, 0, "model"),
    (0.21, 1, 0, 0, "model"),
    (0.16, 1, 0, 0, "model"),
    (0.10, 0, 1, 0, "model"),
    (0.09, 0, 1, 0, "model"),
    (0.06, 1, 0, 0, "model"),
]
CONFUSION_COLUMNS = [
    "name",
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
]


def _backend_params():
    params = ["pandas", "spark"]
    if importlib.util.find_spec("polars") is not None:
        params.append("polars")
    else:
        params.append(pytest.param("polars", marks=pytest.mark.skip(reason="polars unavailable")))
    return params


@pytest.fixture(params=_backend_params())
def backend(request):
    return request.param


def _frame(name, rows, columns, spark):
    if name == "pandas":
        return pd.DataFrame(rows, columns=columns)
    if name == "polars":
        import polars as pl

        return pl.DataFrame(rows, schema=columns, orient="row")
    return spark.createDataFrame(rows, columns)


def _empty_predictions(name, spark):
    if name == "pandas":
        return pd.DataFrame(
            {
                "prediction": pd.Series(dtype="float64"),
                "positive": pd.Series(dtype="int64"),
                "negative": pd.Series(dtype="int64"),
                "unlabeled": pd.Series(dtype="int64"),
                "name": pd.Series(dtype="object"),
            }
        )
    if name == "polars":
        import polars as pl

        return pl.DataFrame(
            schema={
                "prediction": pl.Float64,
                "positive": pl.Int64,
                "negative": pl.Int64,
                "unlabeled": pl.Int64,
                "name": pl.String,
            }
        )
    schema = T.StructType(
        [
            T.StructField("prediction", T.DoubleType()),
            T.StructField("positive", T.LongType()),
            T.StructField("negative", T.LongType()),
            T.StructField("unlabeled", T.LongType()),
            T.StructField("name", T.StringType()),
        ]
    )
    return spark.createDataFrame([], schema)


def _to_pandas(df, order_by=()):
    root = type(df).__module__.partition(".")[0]
    if root == "pandas":
        result = df.copy()
        if order_by:
            columns = [column for column, _ in order_by]
            ascending = [ascending for _, ascending in order_by]
            result = result.sort_values(
                columns, ascending=ascending, kind="mergesort", na_position="last"
            )
        return result.reset_index(drop=True)
    if root == "polars":
        if order_by:
            df = df.sort(
                [column for column, _ in order_by],
                descending=[not ascending for _, ascending in order_by],
                nulls_last=True,
            )
        return df.to_pandas().reset_index(drop=True)

    if order_by:
        expressions = [
            F.col(column).asc_nulls_last() if ascending else F.col(column).desc_nulls_last()
            for column, ascending in order_by
        ]
        df = df.orderBy(*expressions)
    return df.toPandas().reset_index(drop=True)


def test_notebook_golden_values(backend, spark):
    predictions = _frame(backend, NOTEBOOK_ROWS, NOTEBOOK_COLUMNS, spark)
    table = confusion_table(predictions, group_by=["name"])
    assert list(table.columns) == CONFUSION_COLUMNS

    table_pdf = _to_pandas(table, [("name", True), ("threshold", False)])
    assert len(table_pdf) == 22
    tied_positives = table_pdf.loc[table_pdf["threshold"] == 0.95].iloc[0]
    assert tied_positives[["TP", "FP", "dTP"]].tolist() == [4, 1, 2]
    tied_negatives = table_pdf.loc[table_pdf["threshold"] == 0.36].iloc[0]
    assert tied_negatives[["FP", "dFP"]].tolist() == [9, 3]
    assert set(table_pdf["positives"]) == {12}
    assert set(table_pdf["negatives"]) == {12}
    assert set(table_pdf["unlabeled"]) == {1}

    metrics = calculate_pr(table, group_by=["name"])
    assert list(metrics.columns) == [
        *CONFUSION_COLUMNS,
        "precision",
        "recall",
        "average_precision",
    ]
    metrics_pdf = _to_pandas(metrics, [("name", True), ("threshold", False)])
    assert len(metrics_pdf) == 11
    row_095 = metrics_pdf.loc[metrics_pdf["threshold"] == 0.95].iloc[0]
    assert row_095["precision"] == pytest.approx(0.8)
    assert row_095["recall"] == pytest.approx(1 / 3)
    assert row_095["average_precision"] == pytest.approx(0.9)
    final = metrics_pdf.iloc[-1]
    assert final["threshold"] == pytest.approx(0.09)
    assert final["average_precision"] == pytest.approx(0.7709346008259051)

    operating_point = _to_pandas(at(metrics, group_by=["name"], precision=0.81))
    assert len(operating_point) == 1
    assert operating_point.loc[0, "threshold"] == pytest.approx(0.88)
    assert operating_point.loc[0, "precision"] == pytest.approx(9 / 11)
    assert operating_point.loc[0, "recall"] == pytest.approx(0.75)


def test_ties_and_null_group_keys_match_across_backends(spark):
    columns = ["prediction", "positive", "negative", "unlabeled", "name"]
    rows = [
        (0.9, 1, 0, 0, None),
        (0.9, 0, 1, 0, None),
        (0.8, 0, 0, 1, None),
        (0.9, 1, 0, 0, "other"),
        (0.7, 0, 1, 0, "other"),
    ]
    names = ["pandas", "spark"]
    if importlib.util.find_spec("polars") is not None:
        names.append("polars")

    results = []
    for name in names:
        table = confusion_table(_frame(name, rows, columns, spark), group_by=["name"])
        results.append(_to_pandas(table, [("name", True), ("threshold", False)]))

    expected = results[0]
    for actual in results[1:]:
        pd.testing.assert_frame_equal(
            actual.fillna({"name": "<null>"}),
            expected.fillna({"name": "<null>"}),
            check_dtype=False,
        )

    null_group = expected.loc[expected["name"].isna()]
    top = null_group.loc[null_group["threshold"] == 0.9].iloc[0]
    assert top[["TP", "FP", "UP", "dTP", "dFP", "dUP"]].tolist() == [1, 1, 0, 1, 1, 0]
    assert null_group[["positives", "negatives", "unlabeled"]].drop_duplicates().iloc[
        0
    ].tolist() == [
        1,
        1,
        1,
    ]


def test_local_results_have_canonical_order(spark):
    rows = [
        (0.2, 1, 0, 0, "b"),
        (0.1, 1, 0, 0, "a"),
        (0.8, 1, 0, 0, "a"),
        (0.9, 1, 0, 0, "b"),
    ]
    columns = ["prediction", "positive", "negative", "unlabeled", "name"]
    names = ["pandas"]
    if importlib.util.find_spec("polars") is not None:
        names.append("polars")

    for name in names:
        result = confusion_table(_frame(name, rows, columns, spark), group_by=["name"])
        result_pdf = _to_pandas(result)
        actual = result_pdf[["name", "threshold"]].itertuples(index=False, name=None)
        actual = list(actual)
        assert actual == [("a", 0.8), ("a", 0.1), ("b", 0.9), ("b", 0.2)]


def test_ungrouped_pipeline_and_unattainable_target(backend, spark):
    rows = [(0.8, 1, 0, 0), (0.4, 0, 1, 0)]
    columns = ["prediction", "positive", "negative", "unlabeled"]
    table = confusion_table(_frame(backend, rows, columns, spark))
    assert list(table.columns) == CONFUSION_COLUMNS[1:]
    metrics = calculate_pr(table)
    assert len(_to_pandas(metrics)) == 1
    assert _to_pandas(at(metrics, precision=1.1)).empty


def test_multiple_group_columns_are_isolated(backend, spark):
    columns = ["prediction", "positive", "negative", "unlabeled", "name", "replica"]
    rows = [
        (0.9, 1, 0, 0, "a", -1),
        (0.1, 0, 1, 0, "a", -1),
        (0.8, 1, 0, 0, "a", 0),
        (0.2, 0, 1, 0, "a", 0),
        (0.7, 1, 0, 0, "b", -1),
        (0.6, 0, 1, 0, "b", -1),
        (0.5, 1, 0, 0, "b", 0),
        (0.4, 0, 1, 0, "b", 0),
    ]
    table = confusion_table(_frame(backend, rows, columns, spark), group_by=["name", "replica"])
    table_pdf = _to_pandas(table, [("name", True), ("replica", True), ("threshold", False)])
    assert len(table_pdf.groupby(["name", "replica"], dropna=False)) == 4
    assert set(table_pdf["positives"]) == {1}
    assert set(table_pdf["negatives"]) == {1}

    metrics = calculate_pr(table, group_by=["name", "replica"])
    operating_points = _to_pandas(
        at(metrics, group_by=["name", "replica"], precision=1.0),
        [("name", True), ("replica", True)],
    )
    assert operating_points["threshold"].tolist() == [0.9, 0.8, 0.7, 0.5]


def test_empty_input_preserves_native_type_and_output_schema(backend, spark):
    predictions = _empty_predictions(backend, spark)
    table = confusion_table(predictions, group_by=["name"])
    assert type(table) is type(predictions)
    assert list(table.columns) == CONFUSION_COLUMNS
    assert _to_pandas(table).empty

    metrics = calculate_pr(table, group_by=["name"])
    expected_columns = [
        *CONFUSION_COLUMNS,
        "precision",
        "recall",
        "average_precision",
    ]
    assert type(metrics) is type(predictions)
    assert list(metrics.columns) == expected_columns
    assert _to_pandas(metrics).empty

    operating_points = at(metrics, group_by=["name"], precision=0.9)
    assert type(operating_points) is type(predictions)
    assert list(operating_points.columns) == expected_columns
    assert _to_pandas(operating_points).empty


def test_metric_validation_is_backend_neutral(spark):
    df = pd.DataFrame(
        [(0.8, 1, 0, 0)],
        columns=["prediction", "positive", "negative", "unlabeled"],
    )
    with pytest.raises(TypeError, match="not a string"):
        confusion_table(df, group_by="name")
    with pytest.raises(ValueError, match="duplicate"):
        confusion_table(df, group_by=["positive", "positive"])
    with pytest.raises(ValueError, match="conflict"):
        confusion_table(df, group_by=["positive"])
    with pytest.raises(ValueError, match="missing required"):
        confusion_table(df.drop(columns="negative"))
    with pytest.raises(ValueError, match="exactly one"):
        at(df)
    with pytest.raises(ValueError, match="exactly one"):
        at(df, precision=0.9, recall=0.5)
    with pytest.raises(TypeError, match="Unsupported dataframe"):
        confusion_table([{"prediction": 0.8}])
