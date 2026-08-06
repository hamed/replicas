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

## Follow-up status (2026-08-06)

The two confirmed bugs below were fixed with cross-backend regression tests.
The Spark null flags were intentionally retained: null and NaN now derive
different streams, so the flags are required to recover Spark nulls after the
pandas UDF path presents numeric null keys as NaN. The remaining suggestions
are still review notes rather than completed work.

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

- [ ] delete flags, simplify `_group_key`

---

## 3. Simplifications (behavior-preserving)

1. **`_is_null` → IEEE definition** (`replicas/_sampling.py:93-105`): replace
   pandas module-name sniffing with `value is None or value != value` guarded
   by `except TypeError: return True` (covers `pd.NA`, whose bool coercion
   raises; `NaT != NaT` is already true). Shorter and strictly more general.
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

1. **`group_by` rejects the plain string `by` accepts** —
   `sample(df, by="stratum")` works, `confusion_table(df, group_by="name")`
   raises (`metrics.py:66-67`). First thing a user hits between step 1 and
   step 2 of the quick start. Accept a single string. Non-breaking.
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
5. Edge notes, doc-line severity: NaN and null strata share one PCG stream
   (both encode `b"n"`), and pandas merges None+NaN into one group where
   Polars keeps two; float `order_by` columns containing NaN sort differently
   across backends (pandas: missing-first; polars/arrow: value ordering);
   `order_by` uniqueness is a documented but unverified contract — violation
   on Spark is silent nondeterminism; local `replica` dtype is int64 vs
   Spark's int32.

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
