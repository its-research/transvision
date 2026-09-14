"""Auditable RTP-V2X method components.

The modules exported here are dependency-light reference implementations of
three protocol-critical operations: causal message visibility, reliability-
aware association, and deadline-aware communication selection.  They are not
an end-to-end trained model and do not authorize scientific claims by
themselves.
"""

from .association import (
    AssociationCostConfig,
    AssociationError,
    AssociationResult,
    compose_association_cost,
    solve_with_unmatched,
)
from .causal_memory import (
    CausalMemoryError,
    CausalMessageMemory,
    MessageRecord,
)
from .communication_selection import (
    CommunicationCandidate,
    CommunicationSelection,
    SelectionError,
    select_budgeted_messages,
)

__all__ = (
    "AssociationCostConfig",
    "AssociationError",
    "AssociationResult",
    "CausalMemoryError",
    "CausalMessageMemory",
    "CommunicationCandidate",
    "CommunicationSelection",
    "MessageRecord",
    "SelectionError",
    "compose_association_cost",
    "select_budgeted_messages",
    "solve_with_unmatched",
)
