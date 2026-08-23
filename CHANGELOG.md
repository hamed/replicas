# Changelog

All notable changes to `replicas` are documented here.

## [Unreleased]

### Fixed

- `bootstrap` restores the Spark checkpoint directory the caller had
  configured instead of leaving its own in place for the rest of the session.
  Spark offers no per-call checkpoint path, so the read/set/checkpoint/restore
  sequence runs under a process-wide lock: two concurrent `bootstrap` calls
  can otherwise checkpoint into each other's directory. A restore that fails
  raises a `RuntimeWarning` rather than passing silently.
- The local checkpoint fallback is scoped to the current user rather than a
  fixed `/tmp/replicas/` that the first user on a machine takes ownership of.
- `box_plot` explains that it needs a `replica` column instead of raising a
  bare pandas `KeyError`.
- `plot_pr` rejects a `ci` outside `(0, 1]` up front, instead of failing later
  inside `percentile_approx`.

### Changed

- The `replica` column is a 32-bit integer on all three backends. pandas
  previously produced `int64` and Polars `Int64`.
- `at` documents that its "lowest qualifying threshold" rule suits a metric
  that does not increase as the threshold falls, such as `precision`. A
  `recall` target degenerates to the group's minimum threshold.
- `confusion_table` documents that the non-null and mutually-exclusive
  conditions on its indicator columns are the caller's responsibility.
- The metric functions and `bootstrap` document the Spark cost of an empty
  `group_by` and of an empty `by`.
- `bootstrap` points at the README for the NaN exception to the cross-backend
  parity guarantee.

### Added

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
  3.3--4.0 and an Arrow iterator engine for Spark 4.1+.
- Optional backend and plotting dependencies so the base package requires only
  NumPy.
- Notebook conformance tests and an executed reference design.

[Unreleased]: https://github.com/hamed/replicas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hamed/replicas/releases/tag/v0.1.0
