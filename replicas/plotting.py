"""Visualization helpers for bootstrap metrics.

Optional module — requires `matplotlib` and `seaborn`. Install with the
`plot` extra:

    pip install replicas[plot]

For most users, the metrics output is fed into their own plotting code. These
helpers cover the two plots that appear in every bootstrap-CI report: a box
plot of operating-point metrics, and a PR curve with a confidence band.
"""

from __future__ import annotations

from typing import Sequence

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "replicas.plotting requires matplotlib and seaborn. "
        "Install with: pip install 'replicas[plot]'"
    ) from e

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def box_plot(
    df,
    row=None,
    col=None,
    hue=None,
    kind: str = "box",
    values: Sequence[str] = ("threshold", "recall", "precision", "average_precision"),
    **kwargs,
):
    """Distribution of metrics across replicas, as box (or violin) plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Usually the result of `at(...).toPandas()`.
    row, col, hue : str, optional
        Faceting / coloring columns passed through to seaborn.
    kind : str
        Passed to `sns.catplot` — `'box'`, `'violin'`, `'strip'`, etc.
    values : sequence of str
        Which metric columns to show on the x-axis.
    **kwargs
        Forwarded to `sns.catplot`.
    """
    values = list(values)
    id_vars = [v for v in (hue, row, col) if v is not None] + ["replica"]
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=values,
        var_name="metric",
        value_name="value",
    )

    g = sns.catplot(
        data=df_long,
        x="metric",
        y="value",
        hue=hue,
        row=row,
        col=col,
        kind=kind,
        legend=False,
        **kwargs,
    )
    g.set_titles("{row_name} - {col_name}")
    g.add_legend(title="", bbox_to_anchor=(0.0, 1.02), loc="upper left")
    labels = ["AP" if v == "average_precision" else v.capitalize() for v in values]
    g.set_xticklabels(labels)
    g.set_ylabels("")
    g.set_xlabels("")
    return g


def plot_pr(
    df: DataFrame,
    row=None,
    col=None,
    hue=None,
    ci: float = 0.9,
    recall_round: int | None = None,
    **kwargs,
):
    """Precision-recall curve with a bootstrap confidence band.

    Parameters
    ----------
    df : Spark DataFrame
        Output of `calculate_pr` with a `replica` column.
    row, col, hue : str, optional
        Faceting / coloring columns.
    ci : float
        Width of the confidence band (e.g. 0.9 for 5th-95th percentile).
    recall_round : int, optional
        If set, round recall to this many decimals before aggregating across
        replicas. Useful on small datasets where the raw curve is noisy.
        Leave `None` for large datasets to preserve curve resolution.
    **kwargs
        Forwarded to `sns.FacetGrid`.
    """
    low = 0.5 - ci / 2
    high = 0.5 + ci / 2

    group_by = [v for v in (hue, row, col) if v is not None] + ["recall"]

    if recall_round is not None:
        df = df.withColumn("recall", F.round("recall", recall_round))

    df = (
        df.groupBy(*group_by, "replica")
        .agg(F.max("precision").alias("precision"))
    )

    original = (
        df.filter(F.col("replica") == -1)
        .toPandas()
        .set_index(group_by)
        .sort_index()
    )

    bts = (
        df.filter(F.col("replica") >= 0)
        .groupBy(group_by)
        .agg(
            F.percentile_approx("precision", low).alias("low"),
            F.percentile_approx("precision", high).alias("high"),
        )
        .toPandas()
        .set_index(group_by)
        .sort_index()
    )

    combined = original.join(bts, how="outer")

    g = sns.FacetGrid(combined.reset_index(), row=row, col=col, hue=hue, **kwargs)
    g.map_dataframe(plt.fill_between, "recall", "low", "high", alpha=0.1)
    g.map(sns.lineplot, "recall", "low", alpha=0.01)
    g.map(sns.lineplot, "recall", "high", alpha=0.01)
    g.map(sns.lineplot, "recall", "precision", alpha=0.5)

    g.set_titles("{row_name} | {col_name}")
    g.add_legend(title="", bbox_to_anchor=(0.0, 1.1), loc="upper left")
    return g
