# Changelog

All notable changes to `replicas` are documented here.

## [Unreleased]

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
