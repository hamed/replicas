# replicas

Bootstrap confidence intervals for classifier metrics on pandas, Polars, and
Spark.

Most ML evaluation pipelines hand you a single number — precision = 0.873,
recall = 0.612, AUC = 0.94 — and walk away. Those numbers are point estimates.
Run the same model on a slightly different test set and you would get
different numbers. How different? That is the question a single number cannot
answer.

`replicas` answers it by bootstrapping the test set: resampling with
replacement, many times, and computing whatever metric you care about on each
replica. The spread tells you the uncertainty.

## Why this exists

Most bootstrap tools make you choose between a convenient local workflow and
a production-scale distributed one. `replicas` keeps the same small API on
pandas, Polars, and Spark, and returns the same kind of dataframe it receives.
Start locally, then move the same calculation to data already in Spark.

There is a second reason. The obvious PySpark implementation has a bug.
Spark is lazy: a naive chain of `union` calls builds a deferred plan, and
every terminal action re-rolls the random draws. "Replica 7" used to compute
precision is not the same "replica 7" used to compute recall. Recall comes
out greater than 1. Joins between metrics break. The library exists in part
to encode the fix — a `checkpoint()` that materializes the replicas once.

## Install

The base package contains the shared sampler. Install the extra for the
dataframe backend you use:

```bash
pip install 'replicas[pandas]'
pip install 'replicas[polars]'
pip install 'replicas[spark]'
```

If the dataframe library is already installed, plain `pip install replicas`
is sufficient. To install every backend, use `pip install 'replicas[all]'`.

For plotting helpers:

```bash
pip install 'replicas[plot]'
# Spark PR plots need both extras:
pip install 'replicas[spark,plot]'
```

## Quick start

```python
from replicas import bootstrap

# `predictions` may be a pandas, Polars, or Spark DataFrame.
bts = bootstrap(
    predictions,
    by=["name", "positive"],
    n_replicas=100,
    seed=42,
    order_by=["row_id"],
)
```

`bts` has the same dataframe type as `predictions`. It contains the original
data as replica `-1` and resampled replicas `0` through `99`. The
precision-recall helpers preserve that native dataframe type too:

```python
from replicas import at, calculate_pr, confusion_table

ct = confusion_table(bts, group_by=["name", "replica"])
kpi = calculate_pr(ct, group_by=["name", "replica"])

# Operating point: smallest threshold meeting target precision, per replica.
op = at(kpi, group_by=["name", "replica"], precision=0.95)
```

`op` is a distribution of thresholds, not a single number. Summarize it with
the native group-by operations of your dataframe backend.

The bootstrap output is generic. Any statistic grouped by `replica` becomes a
distribution with a CI — AUC, F1, calibration error, or your own domain
metric. PR curves are the demo, not the point.

## Core API

```python
sample(df, by=None, fraction=1.0, *, seed=None, order_by=None)

bootstrap(
    df,
    by=None,
    n_replicas=100,
    checkpoint_dir=None,
    *,
    seed=None,
    order_by=None,
)
```

`by` and `order_by` accept one column name or a sequence. `sample` draws
`round(group_size * fraction)` rows with replacement from every stratum.
`bootstrap` accepts zero replicas, rejects negative counts, and appends
`replica` after the input columns. Both functions reject an existing reserved
`replica` column.
`checkpoint_dir` applies only to Spark; local backends are already eager.

## Backends and reproducibility

`bootstrap` and `sample` choose a backend from the input dataframe. They do
not convert between dataframe libraries. Sampling is exact within each
stratum: every full-size replica contains the same number of rows from each
stratum as the input.

Pass an integer `seed` to repeat a draw. Pandas and Polars use their current
row order when `order_by` is omitted. Spark has no intrinsic row order, so a
seeded Spark bootstrap requires `order_by`; those columns must uniquely order
rows within each stratum. Output row order itself is unspecified on every
backend. With the same seed, strata, and unique ordering, equivalent inputs
produce the same source-row multiplicities across backends.

## Data convention

The metric functions expect three label columns plus a prediction:

| column      | meaning                                              |
|-------------|------------------------------------------------------|
| `prediction`| model score, higher = more likely positive (double)  |
| `positive`  | verified positive (0 or 1)                           |
| `negative`  | verified negative (0 or 1)                           |
| `unlabeled` | row has a prediction but no verified ground truth    |

Exactly one of `positive`, `negative`, `unlabeled` is 1 per row.

The `unlabeled` column is not "negative by default". In fraud detection,
transactions pending investigation are unlabeled; silently treating them as
negative inflates precision. They are tracked separately in the confusion
table (as `UP` — unlabeled positives, the count of unlabeled rows above the
threshold) so you can decide how to handle them at the metric level.

This schema generalizes to multi-class: add one column per class, keep
`unlabeled` for the rows you have not yet verified.

## Why bootstrap, and why Spark too

**Bootstrap.** Works for any statistic. No distributional assumptions. Tells
you what would have happened on a slightly different test set, which is the
question you actually care about when you are deciding whether to ship a
model.

(Caveat: bootstrap underestimates uncertainty. The true CI is usually a bit
wider than the bootstrap CI. Treat the bands as a lower bound on how much you
should worry.)

**Spark.** At production scale, a Python loop over local resamples is too slow
and the data is often already distributed. Comparing 5 models across 20
segments with 100 replicas is 10,000 metric computations that should run in
parallel.

## The `checkpoint()` story

The Spark backend checkpoints its output before returning it. Without that
materialization, Spark's lazy evaluation can re-roll random sampling on every
terminal action. Two `.toPandas()` calls then return different data under the
same replica IDs, and joins between metrics can even produce recall greater
than 1. The local backends are eager and need no checkpoint.

## Reference design

`examples/precision_recall.ipynb` is the notebook this library was ported from,
committed as it was executed — the prose, the figures, and a worked
credit-card-fraud comparison of two models. It is the specification, not a
demo of the package: it defines every function inline so it runs in Colab with
nothing installed. `docs/reference-design.md` records where the library
intentionally differs from it, and which of its odd-looking details are
load-bearing.

## Status

Early days — version 0.1. API may change. Feedback welcome.

## License

MIT.
