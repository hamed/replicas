# Spark bootstrap benchmark

This small, non-CI benchmark compares:

- `legacy`: one `applyInPandas` and union branch per replica;
- `pandas`: the constant-depth Spark 3.3+ grouped-pandas plan;
- `arrow`: the batched `RecordBatch` iterator used on Spark 4.1+.

It runs balanced and 90%-hot-stratum inputs and writes one JSON line per case
with wall time, output rows, and local checkpoint bytes.

```bash
python benchmarks/spark_bootstrap.py \
  --rows 100000 \
  --replicas 20 \
  --repeats 3 \
  --output benchmark.jsonl
```

For a quick smoke test:

```bash
python benchmarks/spark_bootstrap.py \
  --engines pandas arrow \
  --shapes skewed \
  --rows 100 \
  --replicas 2 \
  --partitions 2 \
  --shuffle-partitions 2
```

The numbers are exploratory, not release gates. Use the Spark UI when shuffle,
spill, stage, or executor-memory diagnostics are needed; keeping that monitoring
logic outside this script makes the comparison easy to read and modify.
