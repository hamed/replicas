# replicas

Bootstrap confidence intervals for classifier metrics, on Spark.

Most ML evaluation pipelines hand you a single number — precision = 0.873,
recall = 0.612, AUC = 0.94 — and walk away. Those numbers are point estimates.
Run the same model on a slightly different test set and you would get
different numbers. How different? That is the question a single number cannot
answer.

`replicas` answers it by bootstrapping the test set: resampling with
replacement, many times, and computing whatever metric you care about on each
replica. The spread tells you the uncertainty.

## Why this exists

There are bootstrap libraries for pandas. There are PR-curve libraries for
sklearn. There is no good library for bootstrapped classifier metrics on
**production-scale** data — tens or hundreds of millions of test rows, many
models, stratified by country / segment / time window. A Python loop over
`sklearn.utils.resample` takes hours. The functions here process all
models × all replicas × all groups in one distributed Spark pass.

There is a second reason. The obvious PySpark implementation has a bug.
Spark is lazy: a naive chain of `union` calls builds a deferred plan, and
every terminal action re-rolls the random draws. "Replica 7" used to compute
precision is not the same "replica 7" used to compute recall. Recall comes
out greater than 1. Joins between metrics break. The library exists in part
to encode the fix — a `checkpoint()` that materializes the replicas once.

## Install

```bash
pip install replicas
```

For plotting helpers:

```bash
pip install 'replicas[plot]'
```

## Quick start

```python
from replicas import bootstrap, confusion_table, calculate_pr, at

# `predictions` is a Spark DataFrame with columns:
#   prediction (double), positive (0/1), negative (0/1), unlabeled (0/1), name (str)

bts = bootstrap(predictions, by=['name', 'positive'], n_replicas=100)
ct  = confusion_table(bts, group_by=['name', 'replica'])
kpi = calculate_pr(ct,    group_by=['name', 'replica'])

# Operating point: smallest threshold meeting target precision, per replica.
op = at(kpi, group_by=['name', 'replica'], precision=0.95)

# `op` is a distribution of thresholds, not a single number.
op.toPandas().groupby('name')['threshold'].describe()
```

The bootstrap function is generic. Anything you can compute
`groupBy('replica').agg(...)` becomes a distribution with a CI — AUC, F1,
calibration error, expected calibration error, whatever you like. PR curves
are the demo, not the point.

## Data convention

The metrics functions expect three label columns plus a prediction:

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

## Why bootstrap, and why on Spark

**Bootstrap.** Works for any statistic. No distributional assumptions. Tells
you what would have happened on a slightly different test set, which is the
question you actually care about when you are deciding whether to ship a
model.

(Caveat: bootstrap underestimates uncertainty. The true CI is usually a bit
wider than the bootstrap CI. Treat the bands as a lower bound on how much you
should worry.)

**Spark.** Because at production scale, a Python loop over `resample` is too
slow. Because the data is already in Spark. Because comparing 5 models across
20 segments with 100 replicas is 10,000 metric computations and you would
like them in parallel.

## The `checkpoint()` story

The single most important line in this library is one call to `.checkpoint()`
inside `bootstrap`. Without it, Spark's lazy evaluation re-rolls the random
sampling on every terminal action. Two `.toPandas()` calls return two
different sets of replicas under the same IDs. Joins between metrics produce
recall > 1. The test suite has a regression test for exactly this
(`test_recall_never_exceeds_one`). If you ever feel tempted to remove the
checkpoint to save the I/O, run that test.

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
