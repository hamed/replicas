"""replicas — bootstrap confidence intervals for classifier metrics on Spark.

The library gives you a long-format DataFrame of bootstrap replicas. What you
do with them — precision-recall curves, AUC, F1, calibration — is up to you.

Quick start
-----------
    from replicas import bootstrap, confusion_table, calculate_pr, at

    bts = bootstrap(predictions, by=['name', 'positive'], n_replicas=100)
    ct  = confusion_table(bts, group_by=['name', 'replica'])
    kpi = calculate_pr(ct,   group_by=['name', 'replica'])
    op  = at(kpi, group_by=['name', 'replica'], precision=0.95)
"""

from replicas.bootstrap import bootstrap, sample
from replicas.metrics import at, calculate_pr, confusion_table

__version__ = "0.1.0"

__all__ = [
    "bootstrap",
    "sample",
    "confusion_table",
    "calculate_pr",
    "at",
]
