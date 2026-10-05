from copy import deepcopy
import pytest

from transvision.models.event_track_v2x.allocation_training import training_binding, validate_backend_binding
from transvision.models.event_track_v2x.exclusive_completion_tracking import PersistentExclusiveCompletionConfig
from transvision.models.event_track_v2x.covered_completion_tracking import PersistentCoveredCompletionConfig


def test_exclusive_priority_binds_actual_solver_transitive_sources():
    bound = training_binding(PersistentExclusiveCompletionConfig(), 'scorer', 'cache')
    sources = bound['backend_implementation_sha256']
    prefix = 'transvision/models/event_track_v2x/'
    for name in ('exclusive_completion_tracking.py', 'residual_frontier.py',
                 'persistent_forest.py', 'paper_decision.py', 'fusion.py', 'exclusive_paper_runtime.py'):
        assert prefix+name in sources
    validate_backend_binding(bound, plan_sources=sources)
    for corrupt in ('missing', 'changed'):
        bad = deepcopy(bound)
        if corrupt == 'missing':
            del bad['backend_implementation_sha256']
        else:
            bad['backend_implementation_sha256'][prefix+'residual_frontier.py'] = '0'*64
        with pytest.raises(ValueError, match='source binding'):
            validate_backend_binding(bad)
    plan = dict(sources)
    plan[prefix+'exclusive_completion_tracking.py'] = '0'*64
    with pytest.raises(ValueError, match='teacher exclusive'):
        validate_backend_binding(bound, plan_sources=plan)


def test_historical_binding_is_not_silently_upgraded():
    bound = training_binding(PersistentCoveredCompletionConfig(), 'scorer', 'cache')
    assert 'backend_implementation_sha256' not in bound
    validate_backend_binding(bound)
    bound['backend_implementation_sha256'] = {}
    with pytest.raises(ValueError, match='legacy'):
        validate_backend_binding(bound)
