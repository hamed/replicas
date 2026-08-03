"""Shared sampling contract for the local dataframe backends."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from replicas.bootstrap import bootstrap, sample


@pytest.fixture
def pandas_frame():
    return pd.DataFrame(
        {
            "row_id": list(range(8)),
            "stratum": ["a", "a", "a", "a", "b", "b", None, None],
            "value": [0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 2.0, 2.1],
        }
    )


def test_pandas_sample_is_exact_with_null_strata(pandas_frame):
    result = sample(pandas_frame, by="stratum", fraction=1.0, seed=17, order_by="row_id")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(pandas_frame.columns)
    sizes = result.assign(stratum=result["stratum"].fillna("<null>")).groupby("stratum").size()
    assert sizes.to_dict() == {"<null>": 2, "a": 4, "b": 2}


def test_pandas_bootstrap_contract_and_zero_replicas(pandas_frame):
    result = bootstrap(
        pandas_frame,
        by="stratum",
        n_replicas=3,
        seed=18,
        order_by="row_id",
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [*pandas_frame.columns, "replica"]
    assert sorted(result["replica"].unique()) == [-1, 0, 1, 2]
    sizes = (
        result.assign(stratum=result["stratum"].fillna("<null>"))
        .groupby(["replica", "stratum"])
        .size()
    )
    for replica in (-1, 0, 1, 2):
        assert sizes.loc[(replica, "a")] == 4
        assert sizes.loc[(replica, "b")] == 2
        assert sizes.loc[(replica, "<null>")] == 2

    original_only = bootstrap(pandas_frame, n_replicas=0)
    assert original_only["replica"].eq(-1).all()
    pd.testing.assert_frame_equal(
        original_only.drop(columns="replica").reset_index(drop=True),
        pandas_frame.reset_index(drop=True),
    )


def test_seeded_pandas_results_are_reproducible_and_vary_by_seed():
    frame = pd.DataFrame({"row_id": range(50), "stratum": ["one"] * 50})
    first = bootstrap(frame, by="stratum", n_replicas=4, seed=3, order_by="row_id")
    again = bootstrap(frame, by="stratum", n_replicas=4, seed=3, order_by="row_id")
    different = bootstrap(frame, by="stratum", n_replicas=4, seed=4, order_by="row_id")

    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(different)


def test_empty_pandas_input_preserves_schema():
    frame = pd.DataFrame(
        {
            "row_id": pd.Series(dtype="int64"),
            "stratum": pd.Series(dtype="object"),
        }
    )

    sampled = sample(frame, by="stratum", seed=1, order_by="row_id")
    result = bootstrap(frame, by="stratum", n_replicas=2, seed=1, order_by="row_id")

    assert sampled.empty
    assert sampled.dtypes.to_dict() == frame.dtypes.to_dict()
    assert result.empty
    assert list(result.columns) == ["row_id", "stratum", "replica"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"fraction": -0.1}, ValueError),
        ({"fraction": float("inf")}, ValueError),
        ({"fraction": True}, TypeError),
        ({"by": ["missing"]}, ValueError),
        ({"by": ["stratum", "stratum"]}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 1.5}, TypeError),
    ],
)
def test_sample_validation(pandas_frame, kwargs, error):
    with pytest.raises(error):
        sample(pandas_frame, **kwargs)


@pytest.mark.parametrize("n_replicas", [-1, 1.5, True])
def test_bootstrap_rejects_invalid_replica_counts(pandas_frame, n_replicas):
    with pytest.raises((TypeError, ValueError)):
        bootstrap(pandas_frame, n_replicas=n_replicas)


def test_bootstrap_rejects_reserved_column_and_local_checkpoint(pandas_frame):
    with pytest.raises(ValueError, match="reserved column"):
        bootstrap(pandas_frame.assign(replica=1), n_replicas=1)
    with pytest.raises(ValueError, match="reserved column"):
        sample(pandas_frame.assign(replica=1))
    with pytest.raises(ValueError, match="only supported for Spark"):
        bootstrap(pandas_frame, n_replicas=1, checkpoint_dir="/tmp/unused")


def test_sampling_rejects_duplicate_dataframe_columns():
    frame = pd.DataFrame([[1, 2]], columns=["value", "value"])
    with pytest.raises(ValueError, match="duplicate column"):
        sample(frame)


def test_polars_matches_seeded_pandas_multiplicities(pandas_frame):
    pl = pytest.importorskip("polars")
    polars_frame = pl.from_pandas(pandas_frame)

    pandas_result = bootstrap(
        pandas_frame,
        by="stratum",
        n_replicas=4,
        seed=128,
        order_by="row_id",
    )
    polars_result = bootstrap(
        polars_frame,
        by="stratum",
        n_replicas=4,
        seed=128,
        order_by="row_id",
    )

    assert isinstance(polars_result, pl.DataFrame)
    assert polars_result.columns == [*polars_frame.columns, "replica"]
    pandas_counts = pandas_result.groupby(["replica", "row_id"], dropna=False).size().sort_index()
    polars_counts = (
        polars_result.to_pandas().groupby(["replica", "row_id"], dropna=False).size().sort_index()
    )
    pd.testing.assert_series_equal(pandas_counts, polars_counts)


def test_polars_fraction_empty_and_zero_replica_contract():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"row_id": [0, 1, 2, 3], "stratum": ["a", "a", "b", "b"]})

    sampled = sample(frame, by="stratum", fraction=0.5, seed=7, order_by="row_id")
    original = bootstrap(frame, n_replicas=0)
    empty = sample(frame.clear(), by="stratum", seed=7, order_by="row_id")

    assert sampled.group_by("stratum").len().sort("stratum")["len"].to_list() == [1, 1]
    assert original["replica"].to_list() == [-1, -1, -1, -1]
    assert empty.is_empty()
    assert empty.schema == frame.schema


def test_seeded_multiplicities_match_all_backends(spark, tmp_path):
    pl = pytest.importorskip("polars")
    rows = [
        (0, "a", 0.1),
        (1, "a", 0.2),
        (2, "a", 0.3),
        (3, "b", 1.0),
        (4, "b", 1.1),
        (5, None, 2.0),
        (6, None, 2.1),
    ]
    pandas_frame = pd.DataFrame(rows, columns=["row_id", "stratum", "value"])
    polars_frame = pl.DataFrame(pandas_frame)
    spark_frame = spark.createDataFrame(rows, "row_id long, stratum string, value double")

    results = [
        bootstrap(
            pandas_frame,
            by="stratum",
            n_replicas=3,
            seed=71,
            order_by="row_id",
        ),
        bootstrap(
            polars_frame,
            by="stratum",
            n_replicas=3,
            seed=71,
            order_by="row_id",
        ).to_pandas(),
        bootstrap(
            spark_frame,
            by="stratum",
            n_replicas=3,
            checkpoint_dir=str(tmp_path / "parity"),
            seed=71,
            order_by="row_id",
        ).toPandas(),
    ]

    fingerprints = [Counter(zip(result["replica"], result["row_id"])) for result in results]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_public_seeded_spark_requires_order_by(spark):
    frame = spark.createDataFrame([(0,), (1,)], "row_id long")
    with pytest.raises(ValueError, match="requires order_by"):
        bootstrap(frame, n_replicas=1, seed=7)
