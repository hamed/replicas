"""Distribution version contract."""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import replicas


def test_runtime_version_matches_package_metadata():
    assert replicas.__version__ == version("replicas")


def test_source_checkout_imports_without_package_metadata():
    code = """
import importlib.metadata

def missing_version(_name):
    raise importlib.metadata.PackageNotFoundError

importlib.metadata.version = missing_version
import replicas
assert replicas.__version__ == "0.0.0+source"
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        check=True,
    )
