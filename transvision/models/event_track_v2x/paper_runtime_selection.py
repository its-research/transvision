"""Explicit runtime selection shared by replay and priority checkpoint binding.

Missing backend retains the historical covered configuration. Unknown explicit
backends fail instead of silently running a different scientific method.
"""

EXCLUSIVE = 'exclusive_root_partition_regions_v1'
RECOVERY_OFF = 'exclusive_event_boundary_recovery_off_v1'


def runtime(configuration):
    backend = configuration.get('backend')
    if backend is None:
        from . import paper_runtime
        return paper_runtime
    if backend == EXCLUSIVE:
        if configuration['method'] != 'rbf':
            raise ValueError('exclusive backend requires the RBF method')
        from . import exclusive_paper_runtime
        return exclusive_paper_runtime
    if backend == RECOVERY_OFF:
        if configuration['method'] != 'rbf' or configuration['allocation'] not in {'bound', 'teacher', 'learned'}:
            raise ValueError('recovery-off requires bound, teacher or learned allocation')
        from . import recovery_off_paper_runtime
        return recovery_off_paper_runtime
    raise ValueError('unknown explicit paper backend: ' + str(backend))


def allocation_configuration(configuration):
    from .forest_tracking import load_tracking_config
    runtime(configuration)
    if configuration.get('backend') == RECOVERY_OFF:
        from .recovery_off_tracking import RecoveryOffConfig as Config
    elif configuration.get('backend') == EXCLUSIVE:
        from .exclusive_completion_tracking import PersistentExclusiveCompletionConfig as Config
    else:
        from .covered_completion_tracking import PersistentCoveredCompletionConfig as Config
    return Config(state=load_tracking_config(configuration['state']), **configuration['limits'])
