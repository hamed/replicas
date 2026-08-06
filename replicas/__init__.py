"""Bootstrap confidence intervals for pandas, Polars, and Spark.

The public API comes from lightweight dispatch modules. Importing
:mod:`replicas` does not import any optional dataframe or plotting framework.
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
