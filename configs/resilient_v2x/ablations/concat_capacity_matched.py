"""Deployment baseline: parameter-capacity-matched concat fusion.

The paper requires this comparison but does not specify its internal concat
topology.  The selected topology is therefore an explicit implementation
choice: masked modality-expert outputs are concatenated and passed through the
synergy projection while the DER gate parameters remain dormant.
"""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(routing_mode="concat")
experiment = dict(
    name="dair_complexity_concat_capacity_matched",
    stage="complexity_baseline",
    implementation_choice=(
        "masked modality-expert concat through the synergy projection; "
        "DER gate retained but not executed for exact parameter matching"
    ),
)
