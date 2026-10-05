"""Fixed component order with the same exclusive frontier and search resources.

This ablation replaces only inter-component allocation order. Each component's
search operations, legal support, residual regions, decoder and continuous
states are inherited unchanged. It is not a fixed internal prefix-order test.
"""
from .exclusive_completion_tracking import ExclusiveCompletionTracker


class ExclusiveCompletionFixed(ExclusiveCompletionTracker):
    SCHEMA = 'persistent_exclusive_completion_fixed_component_allocation_v1'

    def _select_allocation(self, candidates, **state):
        ordered = sorted(candidates)
        if not ordered:
            raise ValueError('fixed allocation requires an eligible component')
        component = ordered[0]
        return component, dict(fixed_component_allocation=dict(
            recipe='ascending_current_support_component_identifier_v1',
            eligible_components=ordered,
            selected_component=component,
            quality_or_priority_features_used=False))

    def _allocation_summary(self):
        return dict(super()._allocation_summary(),
                    allocation_policy='fixed_ascending_support_component_identifier')

    def _audit_properties(self):
        return dict(super()._audit_properties(), kind=self.SCHEMA,
                    fixed_component_order=True,
                    internal_prefix_order_changed=False,
                    learned_allocation_policy=False)
