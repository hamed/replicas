# Changelog

All notable changes to `replicas` are documented here.

## [Unreleased]

### Added

- `examples/census_income.ipynb`, a worked example against real data. It
  downloads the Census Income set from OpenML, trains four models -- gradient
  boosting, a random forest, a linear SVM, and a logistic regression -- and
  compares them with bands rather than point estimates. Over the full test set
  all four average-precision intervals are disjoint, so the ranking is a
  result and not a hunch. The notebook then re-runs the same pipeline grouped
  by sex and by race, where `plot_pr` draws a row/column grid and the bands
  swallow whole panels: the smallest cell holds 46 people and 3 positives. The
  linear SVM scores through `decision_function` rather than `predict_proba`,
  which shows that a prediction column need only be monotone, not a
  probability. It is committed with its outputs and executed in CI like the
  quickstart, so a change that moves any printed number fails the build. The
  `notebook` extra now declares scikit-learn.

## [0.2.0] - 2026-08-24

### Fixed

- `bootstrap` restores the Spark checkpoint directory the caller had
  configured instead of leaving its own in place for the rest of the session.
  Spark offers no per-call checkpoint path, so the read/set/checkpoint/restore
  sequence runs under a process-wide lock: two concurrent `bootstrap` calls
  can otherwise checkpoint into each other's directory. A restore that fails
  raises a `RuntimeWarning` rather than passing silently. Known limitation:
  Spark reports a checkpoint directory as `<root>/<generated-id>` and setting
  that value back appends another id, so a directory configured outside
  `replicas` ends up one generated level deeper the first time it is restored.
  Later calls reuse the remembered root and the depth holds.
- The local checkpoint fallback is scoped to the current user rather than a
  fixed `/tmp/replicas/` that the first user on a machine takes ownership of.
- `box_plot` explains that it needs a `replica` column instead of raising a
  bare pandas `KeyError`.
- `plot_pr` rejects a `ci` outside `(0, 1]` up front, instead of failing later
  inside `percentile_approx`.

### Changed

- **Breaking.** `box_plot` and `plot_pr` take a pandas, Polars, or Spark
  DataFrame, like the rest of the package. The `plot` extra declares pandas
  and is sufficient on top of any backend: a Polars frame is collected without
  PyArrow when PyArrow is absent, since `replicas[polars,plot]` pulls none and
  `polars.DataFrame.to_pandas` goes through Arrow. `box_plot` was pandas-only and
  `plot_pr` was Spark-only, so the two plot helpers demanded opposite
  backends and a pandas user could not call `plot_pr` at all. Each one now
  reduces the data on its own backend and collects only the result.
- **Breaking.** The `spark` extra requires PySpark 3.5, up from 3.3. `pr_band`
  uses `F.percentile`, added in 3.5. `percentile_approx` returns a different
  order statistic -- on 100 replicas it put the 5th-percentile band edge 2.4%
  off the pandas and Polars value, which is a visible shift in a plotted band,
  not a rounding difference. Spark 3.3 and 3.4 are both past end of life.
- **Breaking.** `confusion_table`, `calculate_pr`, and `at` take `by` instead
  of `group_by`. Every public function that partitions a computation by
  columns now uses the same name, matching `sample` and `bootstrap` and
  pandas' own `groupby(by=...)`. Positional calls are unaffected; a
  `group_by=` keyword call raises `TypeError`. The type alias
  `replicas.metrics.GroupBy` is now `ByColumns`.
- The `replica` column is a 32-bit integer on all three backends. pandas
  previously produced `int64` and Polars `Int64`.
- `at` documents that its "lowest qualifying threshold" rule suits a metric
  that does not increase as the threshold falls, such as `precision`. A
  `recall` target degenerates to the group's minimum threshold.
- `confusion_table` documents that the non-null and mutually-exclusive
  conditions on its indicator columns are the caller's responsibility.
- The metric functions and `bootstrap` document the Spark cost of an empty
  `by`.
- `bootstrap` points at the README for the NaN exception to the cross-backend
  parity guarantee.
- Every public function documents its arguments, its return value, and the
  errors a caller can act on. `replicas.plotting` uses the same docstring
  convention as the rest of the package and names `seaborn.FacetGrid` as what
  both helpers return. `sample` documents the `round(group_size * fraction)`
  draw size, including fractions above 1, and both samplers document that the
  uniqueness `order_by` needs for a reproducible seeded Spark draw is the
  caller's responsibility and is not verified.
- The README defines "replica" once at first use, and states the limitation of
  a percentile-bootstrap band as asymptotic coverage with a finite-sample
  shortfall, instead of claiming the true interval is always the wider one.

### Added

- `pr_band(df, by=..., ci=..., recall_round=...)`, the reduction behind
  `plot_pr`, as a public metric function on all three backends. It returns the
  original curve and a pointwise quantile band across the replicas. A null
  grouping value is one group, as it is everywhere else in the module: the
  three backends reach that through one grouped aggregation rather than a
  join, because they disagree on whether a join matches null keys.
- A `notebook` extra and a CI job that executes `examples/quickstart.ipynb`
  and compares its outputs with the committed ones.

## [0.1.0] - 2026-08-07

First public alpha release.

### Added

- Exact stratified bootstrap sampling for pandas, Polars, and Spark.
- Reproducible seeded draws shared across backends when strata and ordering
  semantics are equivalent.
- Native confusion-table, precision-recall, average-precision, and operating-
  point helpers.
- A constant-depth Spark bootstrap plan with a pandas fallback for Spark
  3.5--4.0 and an Arrow iterator engine for Spark 4.1+.
- Optional backend and plotting dependencies so the base package requires only
  NumPy.
- Notebook conformance tests and an executed reference design.

[Unreleased]: https://github.com/hamed/replicas/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hamed/replicas/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hamed/replicas/releases/tag/v0.1.0
