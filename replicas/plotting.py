"""Visualization helpers for bootstrap metrics.

Optional module — requires `matplotlib` and `seaborn`. Install with the
`plot` extra:

    pip install replicas[plot]

Both helpers take a pandas, Polars, or Spark DataFrame, like the rest of the
package. Seaborn draws from pandas, so each one reduces the data on its own
backend first and collects only the result. `plot_pr` does that through
`replicas.metrics.pr_band`, which is public if you want the numbers without
the picture.

For most users, the metrics output is fed into their own plotting code. These
helpers cover the two plots that appear in every bootstrap-CI report: a box
plot of operating-point metrics, and a PR curve with a confidence band.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from replicas.metrics import pr_band


def _plot_dependencies():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - exercised in an isolated install
        raise ImportError(
            "replicas.plotting requires matplotlib and seaborn. "
            "Install with: pip install 'replicas[plot]'"
        ) from exc
    return plt, sns


def _to_pandas(df: Any):
    """Collect any supported dataframe to pandas, which is what seaborn reads.

    Polars spells the conversion ``to_pandas`` and Spark spells it
    ``toPandas``; a pandas frame has neither and passes straight through. This
    runs after the backend has already reduced the data to plot size.
    """
    for name in ("toPandas", "to_pandas"):
        collect = getattr(df, name, None)
        if callable(collect):
            return collect()
    return df


def _facet_title(row, col, separator: str) -> str | None:
    if row is not None and col is not None:
        return f"{{row_name}} {separator} {{col_name}}"
    if row is not None:
        return "{row_name}"
    if col is not None:
        return "{col_name}"
    return None


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
    df : pandas, Polars, or Spark DataFrame
        Usually the result of `at(...)`. A `replica` column is required: each
        box is the distribution of one metric across replicas. `at` has already
        reduced the data to one row per group and replica, so this is collected
        to pandas whole.
    row, col, hue : str, optional
        Faceting / coloring columns passed through to seaborn.
    kind : str
        Passed to `sns.catplot` — `'box'`, `'violin'`, `'strip'`, etc.
    values : sequence of str
        Which metric columns to show on the x-axis.
    **kwargs
        Forwarded to `sns.catplot`.
    """
    _, sns = _plot_dependencies()

    df = _to_pandas(df)
    if "replica" not in df.columns:
        raise ValueError(
            "box_plot requires a 'replica' column: every box is a distribution "
            "across bootstrap replicas. Pass the output of at() or calculate_pr() "
            "grouped by 'replica'."
        )

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
        **kwargs,
    )
    title = _facet_title(row, col, "-")
    if title is not None:
        g.set_titles(title)
    labels = ["AP" if v == "average_precision" else v.capitalize() for v in values]
    g.set_xticklabels(labels)
    g.set_ylabels("")
    g.set_xlabels("")
    return g


def plot_pr(
    df: Any,
    row=None,
    col=None,
    hue=None,
    ci: float = 0.9,
    recall_round: Optional[int] = None,  # noqa: UP045
    **kwargs,
):
    """Precision-recall curve with a bootstrap confidence band.

    Parameters
    ----------
    df : pandas, Polars, or Spark DataFrame
        Output of `calculate_pr` with a `replica` column. The band is computed
        on that backend by `replicas.metrics.pr_band`, so only the reduced
        curve is collected to pandas for drawing.
    row, col, hue : str, optional
        Faceting / coloring columns.
    ci : float
        Width of the confidence band (e.g. 0.9 for 5th-95th percentile). Must
        lie in `(0, 1]`.
    recall_round : int, optional
        If set, round recall to this many decimals before aggregating across
        replicas. Useful on small datasets where the raw curve is noisy.
        Leave `None` for large datasets to preserve curve resolution.
    **kwargs
        Forwarded to `sns.FacetGrid`.
    """
    plt, sns = _plot_dependencies()

    by = [v for v in (hue, row, col) if v is not None]
    band = _to_pandas(pr_band(df, by, ci=ci, recall_round=recall_round))
    # Spark row order is unspecified, and a line plot needs the curve in
    # recall order regardless of which backend produced it.
    combined = band.sort_values([*by, "recall"], kind="mergesort").reset_index(drop=True)

    g = sns.FacetGrid(combined, row=row, col=col, hue=hue, **kwargs)
    g.map_dataframe(plt.fill_between, "recall", "low", "high", alpha=0.1)
    g.map(sns.lineplot, "recall", "low", alpha=0.01)
    g.map(sns.lineplot, "recall", "high", alpha=0.01)
    g.map(sns.lineplot, "recall", "precision", alpha=0.5)

    title = _facet_title(row, col, "|")
    if title is not None:
        g.set_titles(title)
    if hue is not None:
        g.add_legend(title="", bbox_to_anchor=(0.0, 1.1), loc="upper left")
    return g
