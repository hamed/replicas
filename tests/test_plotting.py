"""Behavioral tests for the optional plotting helpers."""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

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
