"""Behavioral tests for the optional plotting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

# The minimum-dependencies CI job installs the declared floors and no plotting
# extra, so this module has to skip rather than fail collection there.
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from replicas import plotting  # noqa: E402
from replicas.plotting import box_plot, plot_pr  # noqa: E402


def test_box_plot_preserves_hue_legend():
    rows = []
    for period in ("earlier", "recent"):
        for model, offset in (("baseline", 0.0), ("candidate", 0.1)):
            for replica in range(4):
                rows.append(
                    {
                        "period": period,
                        "model": model,
                        "replica": replica,
                        "threshold": 0.4 + offset + 0.01 * replica,
                    }
                )

    grid = box_plot(
        pd.DataFrame(rows),
        row="period",
        hue="model",
        values=("threshold",),
    )

    assert grid.legend is not None
    assert {text.get_text() for text in grid.legend.texts} == {"baseline", "candidate"}
    plt.close(grid.figure)


def _spark_curve_rows():
    rows = []
    for period in ("earlier", "recent"):
        for segment in ("new", "returning"):
            for model, offset in (("baseline", 0.0), ("candidate", 0.1)):
                for replica in (-1, 0, 1):
                    rows.extend(
                        [
                            (model, period, segment, replica, 0.25, 0.60 + offset),
                            (model, period, segment, replica, 0.75, 0.40 + offset),
                        ]
                    )
    return rows


def test_plot_pr_supports_single_and_faceted_views(spark):
    columns = ["model", "period", "segment", "replica", "recall", "precision"]
    curves = spark.createDataFrame(_spark_curve_rows(), columns)

    single = plot_pr(
        curves.filter("model = 'candidate' AND period = 'recent' AND segment = 'new'"),
        ci=0.9,
        recall_round=2,
    )
    assert single.axes.shape == (1, 1)
    assert single.legend is None
    plt.close(single.figure)

    faceted = plot_pr(
        curves,
        row="period",
        col="segment",
        hue="model",
        ci=0.9,
        recall_round=2,
    )
    assert faceted.axes.shape == (2, 2)
    assert faceted.legend is not None
    assert {text.get_text() for text in faceted.legend.texts} == {"baseline", "candidate"}
    assert {axis.get_title() for axis in faceted.axes.flat} == {
        "earlier | new",
        "earlier | returning",
        "recent | new",
        "recent | returning",
    }
    plt.close(faceted.figure)


def test_box_plot_requires_a_replica_column():
    # Every box is a distribution across replicas, so the column is part of
    # the contract. Without this check pandas raises a melt KeyError that does
    # not say which column is missing or why it is needed.
    frame = pd.DataFrame({"threshold": [0.4, 0.5], "precision": [0.9, 0.8]})

    with pytest.raises(ValueError, match="requires a 'replica' column"):
        box_plot(frame, values=("threshold",))


@pytest.mark.parametrize("ci", [0.0, -0.1, 1.5, 2.0])
def test_plot_pr_rejects_a_ci_outside_the_unit_interval(spark, ci):
    # ci > 1 asks percentile_approx for a negative percentage. Caught here,
    # the caller sees the offending value instead of a Spark analysis error.
    columns = ["model", "period", "segment", "replica", "recall", "precision"]
    curves = spark.createDataFrame(_spark_curve_rows(), columns)

    with pytest.raises(ValueError, match="ci must be in the interval"):
        plot_pr(curves, ci=ci)


def _curve_frame(backend, spark):
    columns = ["model", "period", "segment", "replica", "recall", "precision"]
    rows = _spark_curve_rows()
    if backend == "pandas":
        return pd.DataFrame(rows, columns=columns)
    if backend == "polars":
        pl = pytest.importorskip("polars")
        return pl.DataFrame(rows, schema=columns, orient="row")
    return spark.createDataFrame(rows, columns)


@pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
def test_plot_pr_accepts_every_backend(backend, spark):
    # plot_pr used to require Spark. The band is now computed by pr_band on
    # whichever backend holds the data, and only the reduced curve is
    # collected, so all three draw the same figure.
    grid = plot_pr(
        _curve_frame(backend, spark),
        row="period",
        col="segment",
        hue="model",
        ci=0.9,
        recall_round=2,
    )

    assert grid.axes.shape == (2, 2)
    assert {text.get_text() for text in grid.legend.texts} == {"baseline", "candidate"}
    plt.close(grid.figure)


@pytest.mark.parametrize("backend", ["polars", "spark"])
def test_box_plot_accepts_every_backend(backend, spark):
    columns = ["model", "replica", "threshold"]
    rows = [
        (model, replica, 0.4 + offset + 0.01 * replica)
        for model, offset in (("baseline", 0.0), ("candidate", 0.1))
        for replica in range(4)
    ]
    if backend == "polars":
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame(rows, schema=columns, orient="row")
    else:
        frame = spark.createDataFrame(rows, columns)

    grid = box_plot(frame, hue="model", values=("threshold",))

    assert {text.get_text() for text in grid.legend.texts} == {"baseline", "candidate"}
    plt.close(grid.figure)


def test_polars_is_collected_without_pyarrow(monkeypatch):
    # `replicas[polars,plot]` pulls no PyArrow, and polars' to_pandas goes
    # through Arrow. Both helpers used to raise ModuleNotFoundError there.
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(
        {
            "model": ["a", "b", None],
            "replica": [0, 1, 2],
            "threshold": [0.4, 0.5, None],
        }
    )
    expected = frame.to_pandas()

    monkeypatch.setattr(plotting, "find_spec", lambda name: None)
    collected = plotting._to_pandas(frame)

    assert list(collected.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(collected, expected, check_dtype=False)


def test_box_plot_works_without_pyarrow(monkeypatch):
    pl = pytest.importorskip("polars")
    monkeypatch.setattr(plotting, "find_spec", lambda name: None)
    frame = pl.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "replica": [0, 1, 0, 1],
            "threshold": [0.4, 0.5, 0.6, 0.7],
        }
    )

    grid = box_plot(frame, hue="model", values=("threshold",))

    assert {text.get_text() for text in grid.legend.texts} == {"a", "b"}
    plt.close(grid.figure)
