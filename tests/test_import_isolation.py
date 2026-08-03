"""Import-time dependency boundaries for the optional backends."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_MODULES = {
    "matplotlib",
    "pandas",
    "polars",
    "pyarrow",
    "pyspark",
    "seaborn",
}


@pytest.mark.parametrize(
    "statement",
    [
        "import replicas",
        "import replicas.plotting",
        "from replicas import bootstrap, sample; assert callable(bootstrap) and callable(sample)",
        "from replicas import sample, bootstrap; assert callable(sample) and callable(bootstrap)",
    ],
)
def test_optional_frameworks_are_not_imported_eagerly(statement):
    script = textwrap.dedent(
        f"""
        import sys

        {statement}

        optional = {OPTIONAL_MODULES!r}
        loaded = {{name.partition('.')[0] for name in sys.modules}}
        unexpected = sorted(optional & loaded)
        if unexpected:
            raise AssertionError(f"optional modules imported eagerly: {{unexpected}}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
