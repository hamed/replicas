"""Distribution version contract."""

from importlib.metadata import version

import replicas


def test_runtime_version_matches_package_metadata():
    assert replicas.__version__ == version("replicas")
