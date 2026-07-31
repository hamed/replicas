# Reference design

`examples/precision_recall.ipynb` is the reference design for this library, not
merely an example of it. It is the original Colab notebook, committed as it was
executed: 49 cells, 13 of them with outputs, 6 figures. The library in
`replicas/` is a port of it.

When the two disagree, the notebook is the specification and the library is the
thing that needs explaining.

## Why it stays self-contained

The notebook defines `sample`, `bootstrap`, `confusion_table`, `calculate_pr`,
`at`, `box_plot`, and `plot_pr` inline, duplicating the library. That is
deliberate. A reader can open it in Colab and run it top to bottom with no
install, and every function appears next to the prose explaining why it is
written that way. Rewriting the cells as `from replicas import ...` would delete
the explanation along with the code.

The cost is drift: two copies that can diverge silently. The intended control is
a conformance test that pins the notebook's published numbers (see
*Derived work* below), not an edit to the notebook.

## What must not be "cleaned"

Several details read as redundant, inconsistent, or wrong. Each is load-bearing.
Cell numbers below are 0-indexed as stored in the `.ipynb`; ruff reports the
same cells 1-indexed.

**The setup cell re-imports `numpy` and `pyplot` (cell 8).** The cell above it
(cell 2, the bootstrap explainer plots) opens with `# skip this cell, only for
demonstration.`, so setup has to stand on its own — `plot_pr` calls
`plt.fill_between`. Ruff sees a redefinition of an import from the cell above and
offers to delete it. It has done so once already; a reader who skipped cell 2 as
instructed then hit `NameError: name 'plt' is not defined` at the first
`plot_pr` call. `pyproject.toml` ignores F811 for notebooks to prevent a repeat.

**The two `alpha=0.01` line plots in `plot_pr`** (`replicas/plotting.py:137-138`).
They are invisible on purpose. `map_dataframe(plt.fill_between, ...)` draws the
confidence band without registering its data with the `FacetGrid`, so the axes
do not autoscale to it and the hue levels do not reach the legend. These two
near-transparent `map` calls are what fix both. They look like dead code.

**`F.max('precision')` per (recall, replica)** (`replicas/plotting.py:117`).
Many thresholds map to a single recall value. This takes the upper envelope of
the curve, which is the correct PR curve; averaging or taking the first row
would draw a different, wrong curve.

**`positives` / `negatives` / `unlabeled` totals joined onto every row**
(`replicas/metrics.py:64-79`). This looks recomputable from the window function.
It is not: `calculate_pr` filters `dTP > 0` first
(`replicas/metrics.py:132`), and the recall denominator has to survive that
filter as a materialized column.

**Aggregating by score before the cumulative window**
(`replicas/metrics.py:58-62`). Resampling with replacement makes the same row —
and therefore the same score — appear several times. Collapsing scores first is
what makes `dTP` a count rather than a 0/1 flag. The toy data in cell 23 has
deliberate ties (`0.95` twice, `0.36` three times) to exercise exactly this;
cell 25's committed output shows `threshold 0.95, dTP 2`.

**`precision=0.81` in cells 27 and 31.** An odd-looking target, chosen to sit in
a specific gap. Cell 26's output gives precision `0.800` at threshold `0.90` and
`0.818` at threshold `0.88`: precision is not monotonic in threshold, so `at`
returns the *lower* threshold with the *higher* precision. Rounding the target
to `0.8` or `0.9` makes the demo stop demonstrating that.

**No seed in `sample`** (`replicas/bootstrap.py:47`). The function is called once
per replica. A fixed seed would make every replica identical. Reproducibility
belongs at the SparkContext level.

**`checkpoint()` in `bootstrap`** (`replicas/bootstrap.py:106`). The single
load-bearing line in the library. Without it Spark re-rolls the random draws on
every terminal action, "replica 7" differs between two metrics computed from the
same DataFrame, and recall comes out greater than 1.
`tests/test_bootstrap.py::test_replicas_are_stable` and
`::test_recall_never_exceeds_one` are the regression tests.

**`unlabeled` is not a negative.** It never enters the precision denominator; it
is reported separately as `UP`. Treating pending-investigation rows as negative
inflates precision, which is the failure this schema exists to prevent.

**`average_precision` is a running mean**, equal to the standard area under the
PR curve only at the last row of each group. Intentional — it makes the column
readable at any threshold.

**`groupBy(group_by)` next to `groupBy(*group_by, ...)`** in the same function
(`replicas/plotting.py:117` and `:123`). Inconsistent style, identical
semantics. Not worth a diff.

## Where the library intentionally differs

Four hardening changes. Everything else is a faithful port; treat any other
divergence as a bug in the library.

| Behavior | Notebook | Library |
|---|---|---|
| `confusion_table` with empty `group_by` | `join(totals, on=[])` fails | branches to `crossJoin` (`metrics.py:78-81`) |
| `at` with no `group_by` | `orderBy(*[])` raises | guarded (`metrics.py:181`) |
| `at` with 0 or 2+ conditions | silently uses the first | raises `ValueError` (`metrics.py:168`) |
| checkpoint directory | hardcoded `/tmp/bootstraps/` | `checkpoint_dir` parameter, default `/tmp/replicas/` (`bootstrap.py:56`) |

The library also takes the `SparkSession` from
`SparkSession.builder.getOrCreate()` rather than the notebook's module-level
`spark` global (`bootstrap.py:104`).

## Present in the notebook, absent from the library

- **`plot_steps` / `plot_ci`** (cell 2) — the two figures that explain what
  bootstrap is before any Spark appears. No equivalent in `replicas.plotting`.
- **Two-model comparison** faceted by `hue='name'` (cells 42-43) — the payoff of
  the whole notebook, and untested in the library, whose fixtures use a single
  model.

## Deliberately out of scope

**The AUC section (cells 45-47) is not a missing feature.** It exists to show
that the bootstrap output is generic — that any metric you can express as
`groupBy('replica').agg(...)` becomes a distribution, using whatever tool you
like, with no support from this library. That is the notebook's thesis, and the
demonstration only works if the AUC is computed with something outside the
package. Shipping `replicas.auc()` would argue the opposite: that metrics need
to be blessed by the library first.

The `toPandas()` loop in cell 46 is therefore illustrative, not a placeholder.
Cell 47's aside about a Spark-native trapezoidal AUC over the confusion table is
advice to the reader with large data, not a TODO for this package. Cell 46 has
no committed output; it was never executed.

The same reasoning applies to any future "add metric X" proposal. The library's
job is `bootstrap`; PR curves ship with it because they are the worked example
and their Spark implementation is subtle, not because metrics belong here.

## Rules for changing the notebook

1. Do not reformat it. `pyproject.toml` excludes it from `ruff format` and
   ignores E402, I001, and F811 for it, each for a reason recorded there.
2. Outputs are part of the artifact. A source edit that invalidates a committed
   output means re-executing the notebook, not stripping the outputs.
3. Re-executing requires Colab or an equivalent: cell 35 is a `!pip install`,
   and cells 36-37 pull the credit-card-fraud dataset via `kagglehub` and train
   two CatBoost models. This cannot run in CI, which is why the notebook is
   committed with outputs rather than executed by the test job.

## Derived work

Read off the notebook, in the order worth doing:

1. **Golden-value tests from the committed outputs.** Cell 23's toy data with its
   ties, cell 25's `dTP 2` at threshold `0.95`, and cell 27's operating point
   (threshold `0.88` for `precision=0.81`) are published numbers the library
   should be pinned to. Current fixtures in `tests/` use all-distinct scores and
   one model, so tie collapsing and multi-model grouping have no coverage.
2. **Single-pass `bootstrap`.** The notebook's loop over `union` is 101 scans of
   the input and 100 shuffles at `n_replicas=100`. A `crossJoin` against replica
   ids followed by one `applyInPandas` has constant plan depth. The output
   contract — `replica` ids `-1..n-1`, per-stratum size preserved, checkpointed —
   is unchanged, so this is an optimization of the reference rather than a
   divergence from it.
