"""Fail-closed V2X-Graph preprocessing parity evidence tools."""

from .parity import (
    CONTRACT_ID,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    ParityContractError,
    compare_candidate,
)

__all__ = [
    "CONTRACT_ID",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "ParityContractError",
    "compare_candidate",
]
