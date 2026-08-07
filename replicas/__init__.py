"""Bootstrap confidence intervals for pandas, Polars, and Spark.

The public API comes from lightweight dispatch modules. Importing
:mod:`replicas` does not import any optional dataframe or plotting framework.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from replicas.bootstrap import bootstrap, sample
from replicas.metrics import at, calculate_pr, confusion_table

try:
    __version__ = _distribution_version("replicas")
except PackageNotFoundError:
    __version__ = "0.0.0+source"

__all__ = [
    "__version__",
    "bootstrap",
    "sample",
    "confusion_table",
    "calculate_pr",
    "at",
]
