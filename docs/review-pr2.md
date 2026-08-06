# Code review: PR #2 — Add multi-backend bootstrap and metrics

- **Reviewer:** Claude (Fable 5), requested by Hamed
- **Date:** 2026-08-04
- **Scope:** all 4 commits, full diff vs `main` (23 files, +4874/−1367)
- **Verdict:** approve after the two confirmed bugs are addressed. Nothing blocking
  the architecture; the core design (seed derivation, checkpoint invariant, test
  shape) is sound.

Validation performed locally: full suite `66 passed` (matches the PR claim), CI
green (6 checks), notebook JSON validated (49 cells, 13 with outputs — matches
`docs/reference-design.md`), plus targeted repro scripts for every finding
marked *reproduced*.

## Follow-up status (2026-08-06, amended 2026-08-07)

Commit `f36e399` fixed the two confirmed bugs below with cross-backend
regression tests. The Spark null flags were intentionally retained: null and
NaN now derive different streams (`b"n"` vs `b"N"`), so the flags are required
to recover Spark nulls after the pandas UDF path presents numeric null keys as
NaN. Because of that stream split, the pre-fix analysis in sections 2, 3.1,
and 5.5 is superseded — each carries an inline resolution note below.

The same commit also shipped work beyond the two bugs:

- item 4.1 — `group_by` now accepts a plain string (`metrics.py`), tests
  rewritten to use it;
- case-insensitive helper-column collision checks in the Spark sampling
  backend (Spark resolves column names case-insensitively — a hazard the
  review missed);
- Spark metrics `at()` helper renamed from bare `_row` to a collision-proof
  generated name, with tests;
- pandas `calculate_pr` no longer materializes a `_weighted_precision` helper
  column, so a user column of that name survives, with tests.

All other unticked items remain open review notes.

### Verification (2026-08-07)

Fix commit adversarially verified: full suite `91 passed` (was 66), ruff
check/format clean; bug 1.1 repro now raises on pandas and Polars, `at()`
metric-as-group-column deliberately allowed and tested; bug 1.2 fingerprints
identical across pandas float64 / Polars Int64 / Spark long strata, and the
same probe run against the parent commit still fails, proving the fix
load-bearing; null-vs-NaN streams verified distinct end-to-end on Polars and
on Spark under both engines (pyspark 4.2.0, arrow and forced pandas UDF);
signed-zero folding (`-0.0` → `b"i0"`) matches how every backend groups signed
zeros, so no stream collision exists; helper-column casefold and user-column
preservation verified on live Spark bootstraps.

---

## 1. Correctness — confirmed bugs (both reproduced)

### 1.1 `calculate_pr` silently destroys a group column named after an output metric

`replicas/metrics.py:128` checks `group_by` conflicts only against
`_CONFUSION_COLUMNS`, not against the three columns the function *adds*
(`precision`, `recall`, `average_precision`).

Repro:

```python
calculate_pr(ct, group_by=["precision"])  # no error
```

The grouping column is overwritten with computed precision **before** the
grouped cumulative sum runs, so the running average partitions by the
overwritten float values — silently wrong numbers, no exception.

**Fix (one line):** extend the conflict set with
`("precision", "recall", "average_precision")`. The same family applies to
`at()` when the metric kwarg names a group column.

- [x] fix conflict check
- [x] add regression test

### 1.2 Cross-backend parity silently breaks for integer strata containing a null

`pd.DataFrame({"stratum": [1, 1, 2, None]})` coerces to float64, so
`_canonical_value` (`replicas/_sampling.py:121-123`) encodes the keys on the
float path (`b"f"`), while Polars keeps Int64 and encodes ints (`b"i"`).
Same seed + unique `order_by` → different multiplicities, no warning.
Reproduced pandas vs Polars; the same break applies pandas vs Spark long
columns.

This contradicts the README's headline guarantee ("identical source-row
multiplicities across backends") in the most common way a user writes nullable
integer strata.

**Fix options:**
1. (preferred) encode integral floats as integers: `value.is_integer()` →
   `b"i"` path. Within one typed column, int 1 and float 1.0 cannot coexist, so
   no new collision.
2. document the dtype-equivalence requirement explicitly.

- [x] pick option, implement
- [x] add cross-backend parity test with nullable-int strata

---

## 2. Provably inert mechanism — delete

> **Resolution (f36e399): retained intentionally, now load-bearing.** The
> analysis below was correct at review time — under the old encoding both
> null and NaN mapped to `b"n"`, making the flags inert. The fix split the
> streams (`b"n"` vs `b"N"`), and on the pandas-UDF engine (the only engine
> on Spark 3.3–4.0) a numeric null key arrives as NaN, so the flag is now the
> only way to route a null stratum to the null stream. The Arrow engine would
> not need the flags (null keys arrive as None), but the floor engine does.
> Section text below kept as written for the record.

**Spark null-flag grouping machinery** (`replicas/_backends/spark.py:58-72`).
Every grouping key is doubled with an `isNull` flag so `_group_key` can
distinguish Spark null from real NaN. But:

- `_canonical_value` maps both to `b"n"` (`_is_null(NaN)` is true), so the
  reconstructed `None` and a raw NaN produce identical seeds;
- `isNull(col)` is functionally dependent on `col`, so the flags cannot change
  group boundaries either (Spark already separates null from NaN).

Every path with and without the flags yields byte-identical draws. Cost: wider
shuffle keys, the `_group_key` slicing arithmetic, the `keys[-1]` convention.
Removing it collapses `_grouping_columns` to `[df[c] for c in by]`. Existing
parity tests prove behavior unchanged.

- [x] resolved differently: flags retained and made load-bearing by the
      `b"n"`/`b"N"` stream split; deletion no longer applicable

---

## 3. Simplifications (behavior-preserving)

1. ~~**`_is_null` → IEEE definition**~~ **Withdrawn (f36e399):** the fix made
   NaN a distinct stream from null, so `_is_null` must *not* fold `x != x`
   values into the null case anymore. The module-name sniffing now carries
   real semantics (logical null sentinels only); the IEEE one-liner would
   reintroduce the folding the fix removed.
2. **Ungrouped pandas totals** (`replicas/_metrics_backends/pandas_backend.py:58-62`):
   pandas broadcasts scalars — `result[target] = result[source].sum()` replaces
   the hand-built repeated-sum DataFrame.
3. **Polars indexing** (`replicas/_backends/polars.py:46`):
   `source[indices.tolist()]` → `source.gather(indices)`. Verified: `gather`
   accepts the numpy int64 array directly. `.tolist()` materializes one Python
   int per drawn row — the hot loop of the local backend.
4. **`run_seed` dual return**: the `(seed, seeded)` tuple carries a fact the
   caller already knows (`seed is not None`). Return just the seed.
5. **Duplicated constants/dispatch**: `_CONFUSION_COLUMNS` is defined 4 times
   (metrics.py + three backends); the mro-root backend dispatch is written
   twice (`bootstrap.py:21-34`, `metrics.py:52-60`). Unify; the shared
   dispatch is also the right place to reject pandas `Series` / polars
   `LazyFrame` with a clean `TypeError` instead of a downstream
   `AttributeError`.
6. **Speculative type support** in `_canonical_value`
   (`replicas/_sampling.py:124-135`): `Decimal`, `time`, `bytes` strata are
   YAGNI; unknown types already raise a clean `TypeError`.
7. **Duplicate test**: `tests/test_import_isolation.py:28-29` parametrizes the
   same import with the names swapped; order cannot matter.
8. **Benchmark scope** (`benchmarks/`): 267 lines whose main job is comparing
   against `_legacy_bootstrap` — a reimplementation of code this PR deletes —
   with no committed numbers. Either commit one result table in
   `benchmarks/README.md` or drop the legacy arm. Related: the Arrow engine is
   the single biggest complexity driver in `spark.py` and is currently
   justified only by this unrun benchmark; keep it, but put one measured
   number in the tree.

---

## 4. API intuitiveness

1. **`group_by` rejects the plain string `by` accepts** — **done in
   f36e399**: `GroupBy` now accepts `str`, `_groups` normalizes it, bytes get
   an explicit `TypeError`, tests rewritten to the string form.
2. **`by` vs `group_by`** — two names for one concept across a 5-function API.
   Pick one (pandas precedent: `by`), alias the other. Cheapest now, at 0.1.
3. **`order_by` reads as output ordering** — SQL instinct; actually it defines
   row identity for reproducible draws, and output order is explicitly
   unspecified. Rename (`row_key` / `id_by`) or make the docstring lead with
   "does not sort the output".
4. **`sample` defaults surprise pandas users** — pandas `df.sample()` is
   without replacement; `replicas.sample(df)` is always with replacement and
   returns a full-size resample with duplicates. Consider `resample`, or state
   "with replacement, same size by default" in the first docstring line.
5. **`checkpoint_dir` is positional but Spark-only** —
   `bootstrap(df, ["a"], 100, "/tmp/ckpt")` is legal and raises on local
   backends. Move behind the `*`.
6. **Magic `-1`** — export `ORIGINAL = -1` so call sites read
   `replica == replicas.ORIGINAL`.

Deliberately unchanged: `at(kpi, precision=0.95)` kwargs form (best call in
the API), the three-step metrics chain, the invisible same-type-in/out
dispatch.

---

## 5. Hygiene / CI

1. **Declared floors are never tested.** CI resolves pyspark 4.0.x/4.2.x; the
   `pyspark>=3.3` floor never runs, and the code sits exactly on it
   (`withColumns` was added *in* 3.3). Same for pandas 1.3 / polars 1.0 /
   numpy 1.21. Add one matrix leg with lowest-bound pins
   (e.g. `uv pip install --resolution lowest-direct`).
2. **Version string duplicated** — `replicas/__init__.py:10` and
   `pyproject.toml` both hardcode `0.1.0`; will drift on the first bump. Use
   `importlib.metadata` or setuptools `dynamic = ["version"]`.
3. **Spark Connect unsupported** — `_checkpoint`
   (`replicas/_backends/spark.py:236`) touches `sparkSession.sparkContext`
   unconditionally, which raises on Connect sessions even when
   `checkpoint_dir` is passed. Fine for a classic-3.3 floor; worth one
   documented limitation line.
4. **Head commit `73b0632` has no message body** — thin for a 4.9k-line
   change, given how carefully the three docs commits are written.
5. Edge notes, doc-line severity — **partially superseded by f36e399**: the
   shared-stream claim is now false (null and NaN strata derive distinct
   streams, `b"n"` vs `b"N"`), and the README gained a parity-caveat
   paragraph covering NaN in `by`/`order_by` — exactly what this item asked
   for. Still standing: float `order_by` columns containing NaN sort
   differently across backends (pandas: missing-first; polars/arrow: value
   ordering); `order_by` uniqueness is a documented but unverified contract —
   violation on Spark is silent nondeterminism; local `replica` dtype is
   int64 vs Spark's int32.

---

## 6. Strengths (keep doing this)

- The `checkpoint()` invariant survives the rewrite and its original
  regression tests pass untouched.
- Test design: golden notebook values across all three backends, pandas/Arrow
  engine equivalence, batch-size invariance, constant logical-plan depth,
  subprocess-based import isolation. Right coverage shape for the parity
  claims made.
- Extras split + backend-free base import is verified by tests, not just
  claimed.
- `docs/reference-design.md` recording load-bearing "wrong-looking" details is
  unusually good practice; its factual claims about the notebook check out
  against the committed file.
