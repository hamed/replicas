"""Small, dataframe-independent building blocks for exact bootstrap draws."""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import numpy as np

REPLICA_COLUMN = "replica"


def normalize_columns(value: str | Sequence[str] | None, *, name: str) -> tuple[str, ...]:
    """Normalize a public column argument without importing a dataframe library."""
    if value is None:
        columns: tuple[str, ...] = ()
    elif isinstance(value, str):
        columns = (value,)
    elif isinstance(value, Sequence):
        columns = tuple(value)
    else:
        raise TypeError(f"{name} must be a column name, a sequence of names, or None")

    if not all(isinstance(column, str) for column in columns):
        raise TypeError(f"every {name} column must be a string")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} contains duplicate columns")
    return columns


def validate_columns(
    available: Sequence[str],
    *,
    by: tuple[str, ...],
    order_by: tuple[str, ...],
    reject_replica: bool,
) -> None:
    """Validate inexpensive schema conditions shared by all backends."""
    if len(available) != len(set(available)):
        raise ValueError("dataframe contains duplicate column names")
    names = set(available)
    missing = [column for column in (*by, *order_by) if column not in names]
    if missing:
        raise ValueError(f"columns not found in dataframe: {missing}")
    if reject_replica and REPLICA_COLUMN in names:
        raise ValueError(f"input dataframe already contains reserved column {REPLICA_COLUMN!r}")


def validate_fraction(fraction: float) -> float:
    """Return a finite, non-negative sampling fraction."""
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise TypeError("fraction must be a real number")
    result = float(fraction)
    if not math.isfinite(result) or result < 0:
        raise ValueError("fraction must be finite and non-negative")
    return result


def validate_n_replicas(n_replicas: int) -> int:
    """Return a non-negative Python integer replica count."""
    if isinstance(n_replicas, bool) or not isinstance(n_replicas, Integral):
        raise TypeError("n_replicas must be an integer")
    result = int(n_replicas)
    if result < 0:
        raise ValueError("n_replicas must be non-negative")
    return result


def run_seed(seed: int | None) -> tuple[int, bool]:
    """Resolve one driver-side seed and record whether the caller supplied it."""
    if seed is None:
        return secrets.randbits(128), False
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer or None")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be non-negative")
    return result, True


def target_size(size: int, fraction: float) -> int:
    """Match pandas' round-to-nearest group-size semantics for ``frac``."""
    return round(size * fraction)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    value_type = type(value)
    if value_type.__module__.startswith("pandas.") and value_type.__name__ in {
        "NAType",
        "NaTType",
    }:
        return True
    try:
        return isinstance(value, Real) and math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _canonical_value(value: Any) -> bytes:
    """Encode common scalar group keys identically across dataframe libraries."""
    if _is_null(value):
        return b"n"

    # NumPy scalars expose ``item``; converting them is what makes pandas,
    # Polars, and Spark keys share one representation.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        value = value.item()

    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, Integral):
        return b"i" + str(int(value)).encode("ascii")
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, Decimal):
        return b"d" + str(value.normalize()).encode("ascii")
    if isinstance(value, datetime):
        return b"t" + value.isoformat().encode("utf-8")
    if isinstance(value, date):
        return b"D" + value.isoformat().encode("ascii")
    if isinstance(value, time):
        return b"T" + value.isoformat().encode("utf-8")
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"y" + value
    raise TypeError(
        "seeded sampling does not support group-key values of type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_group_key(group_key: Any) -> bytes:
    """Return an unambiguous byte representation of a stratum key."""
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    encoded = bytearray()
    for value in values:
        item = _canonical_value(value)
        encoded.extend(len(item).to_bytes(8, "big"))
        encoded.extend(item)
    return bytes(encoded)


def draw_indices(
    size: int,
    run_seed: int,
    group_key: Any,
    replica: int = 0,
    *,
    n_draws: int | None = None,
) -> np.ndarray:
    """Draw exact positional indices using a stable PCG64 stream."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if n_draws is None:
        n_draws = size
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    if size == 0:
        if n_draws:
            raise ValueError("cannot draw rows from an empty stratum")
        return np.empty(0, dtype=np.int64)

    digest = hashlib.blake2b(digest_size=16, person=b"replicas-v1")
    digest.update(str(run_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(replica).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_group_key(group_key))
    derived_seed = int.from_bytes(digest.digest(), "big")
    generator = np.random.Generator(np.random.PCG64(derived_seed))
    return generator.integers(0, size, size=n_draws, dtype=np.int64)
