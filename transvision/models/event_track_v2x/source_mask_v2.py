"""Immutable source-availability views over a sealed DetectionCacheV2 frame.

The mask is experimental control, not part of the detector payload. Callers must
bind it separately in their frozen plan and control receipts. In particular,
``available_at`` reports effective availability (sensor time AND enabled source),
not sensor timing alone. No arrays, metadata, selection rule, or digest change.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from .detection_cache_v2 import DetectionCacheV2


@dataclass(frozen=True, slots=True, eq=False)
class SourceMaskedCacheV2(DetectionCacheV2):
    """A source mask of 1 (vehicle), 2 (infrastructure), or 3 (both)."""

    active_agent_mask: int = 3

    def __post_init__(self):
        if type(self.active_agent_mask) is not int or self.active_agent_mask not in {1, 2, 3}:
            raise ValueError('active agent mask must be an integer in {1, 2, 3}')
        # Explicit base calls avoid dataclass(slots=True)'s zero-argument super
        # class-replacement trap and keep the sealed cache validation unchanged.
        DetectionCacheV2.__post_init__(self)

    @classmethod
    def from_frame(cls, frame: DetectionCacheV2, agent_mask: int) -> SourceMaskedCacheV2:
        """Wrap an original frame; nested/foreign views cannot silently remask it."""
        if type(frame) is not DetectionCacheV2:
            raise TypeError('source mask requires an original DetectionCacheV2 frame')
        return cls(**{field.name: getattr(frame, field.name) for field in fields(DetectionCacheV2)},
                   active_agent_mask=agent_mask)

    def available_at(self, decision_timestamp_us):
        return (DetectionCacheV2.available_at(self, decision_timestamp_us)
                and bool(self.metadata['agent_mask'] & self.active_agent_mask))
