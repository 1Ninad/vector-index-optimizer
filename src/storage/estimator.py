"""Storage estimation helpers."""

from __future__ import annotations

from src.data.models import Configuration


def estimate_storage(configuration: Configuration, theta_storage: int) -> tuple[int, bool]:
    """Return index count and whether it satisfies the storage threshold."""
    count = len(configuration.indexes)
    valid = count <= theta_storage
    return count, valid
