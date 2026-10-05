"""Explicit residual-prefix regions with retained terminal-class exclusions.

The subtraction uses a padded gross upper and a padded lower retained mass.
These are conservative float64 operations, not a formal interval certificate.
"""
from __future__ import annotations

import math

from .hypothesis_bank import logsumexp

EPS = 2.220446049250313e-16
RECIPE = 'prefix_region_minus_retained_terminal_classes_v1'


def residual_log_upper(gross_upper, excluded_log_weights, *, nodes):
    """Upper for a prefix region after removing known retained classes.

No exclusions returns the original gross bound. With exclusions, lowering the
subtracted mass avoids treating approximate retained arithmetic as an exact
amount to subtract. Padding is explicit and separately auditable.
"""
    excluded = tuple(float(v) for v in excluded_log_weights)
    if type(nodes) is not int or nodes < 0 or not math.isfinite(gross_upper):
        raise ValueError('finite gross bound and nonnegative integer depth required')
    if any(not math.isfinite(w) for w in excluded):
        raise ValueError('finite excluded class weights required')
    if not excluded:
        return dict(log_upper=float(gross_upper), log_excluded_mass=None,
                    gross_padding=0., excluded_lower_padding=0., output_padding=0.)
    known = logsumexp(excluded)
    scale = max(1., abs(gross_upper), abs(known), nodes, len(excluded))
    padding = 64*EPS*scale
    gross = math.nextafter(gross_upper+padding, math.inf)
    lower = math.nextafter(known-padding, -math.inf)
    if lower >= gross:
        raise ValueError('excluded mass exceeds gross prefix upper; do not clamp away inconsistent support')
    value = gross+math.log(-math.expm1(lower-gross))
    output_padding = 32*EPS*max(1., abs(value), len(excluded))
    value = math.nextafter(value+output_padding, math.inf)
    if not math.isfinite(value):
        raise ValueError('residual mass log arithmetic exceeds float64 range')
    return dict(log_upper=value, log_excluded_mass=known,
                gross_padding=padding, excluded_lower_padding=padding,
                output_padding=output_padding)
