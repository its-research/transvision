from dataclasses import replace

import pytest
import torch

from transvision.models.event_track_v2x.forest_potentials import neural_parent_logits
from transvision.models.event_track_v2x.forest_row_context import ForestRowContext, build_row_contexts
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.forest_training import batched_row_logits
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from test_forest_tracking import observation


@pytest.mark.parametrize('seed', [1337, 2027, 3407])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_batched_eval_logits_equal_scalar_inference_with_mixed_lengths_sources_and_late_times(seed, dtype):
    torch.manual_seed(seed)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=.1).to(dtype=dtype).eval()
    raw = tuple(observation(str(i), float(i), source=(i+1) % 2, state_us=1_000_000+i*100,
                            arrival_us=1_001_000) for i in range(5))
    contexts = [ForestRowContext(tuple(range(n)), raw[:n], 1_100_000+n) for n in (5, 1, 3, 2)]
    batched = batched_row_logits(model, contexts)
    for result, context in zip(batched, contexts):
        expected = neural_parent_logits(model, context.observations, context.support, context.decision_us)[-1]
        torch.testing.assert_close(result, expected, rtol=2e-6, atol=2e-6 if dtype == torch.float32 else 1e-12)


def test_batched_and_scalar_gradients_agree_without_dropout():
    torch.manual_seed(1)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).double()
    raw = (observation('0', source=1), observation('1'), observation('2', source=1, state_us=1_010_000))
    contexts = [ForestRowContext(tuple(range(n)), raw[:n], 1_100_000) for n in (1, 3, 2)]
    sum(r.square().sum() for r in batched_row_logits(model, contexts)).backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    sum(neural_parent_logits(model, c.observations, c.support, c.decision_us)[-1].square().sum() for c in contexts).backward()
    for key, parameter in model.named_parameters():
        torch.testing.assert_close(gradients[key], parameter.grad, rtol=1e-8, atol=1e-9)


def test_shared_context_checks_old_arrival_before_forward_and_preserves_ties():
    raw = (observation('old0'), observation('old1', source=1), observation('new', state_us=1_020_000))
    contexts = build_row_contexts(raw[2:], old_count=2, older_candidates=lambda lo, hi: enumerate(raw[:2]),
                                 config=ForestTrackingConfig(parent_limit=1), decision_us=1_100_000, sequence_id='0003')
    assert contexts[0].indices == (0, 2)
    future = replace(raw[0], node=replace(raw[0].node, arrival_us=1_200_000))
    with pytest.raises(ValueError, match='future'):
        build_row_contexts(raw[2:], old_count=2, older_candidates=lambda lo, hi: [(0, future)],
                           config=ForestTrackingConfig(), decision_us=1_100_000, sequence_id='0003')
