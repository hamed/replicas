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

import pytest  # noqa: E402  (must follow the env setup above)
from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    """One SparkSession for the whole test session.

    This lives here rather than in each test module on purpose. `getOrCreate`
    returns the *same* JVM session to every caller, so two session-scoped
    fixtures do not produce two sessions — they produce one session with two
    teardowns, and whichever finishes first calls `.stop()` while the other
    module may still need it. A single fixture makes the lifetime unambiguous.
    """
    session = (
        SparkSession.builder.master("local[2]")
        .appName("replicas-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()
