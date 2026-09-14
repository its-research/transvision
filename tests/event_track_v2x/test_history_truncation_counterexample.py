import math

import pytest

from tools.event_track_v2x.diagnose_history_truncation import star_model, diagnose_case
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from test_persistent_forest import brute


@pytest.mark.parametrize('bits', [0, 1, 2, 4, 8])
def test_connected_example_partition_marginal_and_regret_against_full_enumeration(tmp_path, bits):
    raw, rows = star_model(bits)
    p, roots, z = brute(ForestFactors(tuple(o.node for o in raw), rows))
    assert len(p) == 2**(bits+1)
    report = diagnose_case(tmp_path, bits)
    marginal = math.fsum(probability for history, probability in p.items() if roots[history][-1] == 0)
    assert math.log(z) == pytest.approx(report['log_partition_exact'])
    assert marginal == pytest.approx(report['current_match_probability'])
    top = sorted(p.values(), reverse=True)[:report['retained_histories']]
    assert 1-math.fsum(top) == pytest.approx(report['exact_omitted_history_mass'])
    assert report['candidate_exact_model_regret'] == 0.
    assert report['threshold_selected_model_regret'] <= report['threshold_selected_regret_upper']+1e-12


def test_exact_partition_bound_is_uninformative_for_current_action_with_many_past_bits(tmp_path):
    report = diagnose_case(tmp_path, 24)
    assert report['exact_omitted_history_mass'] > .999999
    assert report['candidate_reported_regret_upper'] > .999999
    assert report['candidate_exact_model_regret'] == 0.
    assert report['threshold_selected_model_regret'] > .99
    assert report['log_partition_upper'] == pytest.approx(report['log_partition_exact'])
    assert not math.isnan(report['threshold_selected_regret_upper'])
