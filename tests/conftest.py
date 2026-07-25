"""Shared test configuration.

Spark launches its Python workers as separate processes. It picks the
interpreter from `PYSPARK_PYTHON`, falling back to whatever `python3` resolves
to on `PATH` — which is *not* the virtualenv interpreter running pytest. The
driver then imports pyarrow fine while every worker dies with
`ModuleNotFoundError: No module named 'pyarrow'`, and any test touching
`applyInPandas` (i.e. all of `bootstrap`) fails.

Pinning both to `sys.executable` makes the suite run against the same
environment pytest was launched from, with no shell setup required.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
