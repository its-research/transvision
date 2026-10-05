"""Fixed finite conditional-Bayes action outside retained roots: CPU interface.

The live tracker receives raw observations and potentials only. Offline target
actions are absent from this producer. Two immutable output commits expose the
chosen branch's own state for an independent readback oracle.
"""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from transvision.models.event_track_v2x.exclusive_completion_tracking import ExclusiveCompletionTracker, PersistentExclusiveCompletionConfig
from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import IdentityNode

ROWS = (((-1, -0.83543419800285),),
        ((-1, 0.29849794381405287), (0, -0.9438906640875988)),
        ((-1, -2.61036708408103), (0, 0.016887610689038574), (1, 2.223981195651915)),
        ((-1, 1.7335093303534301), (0, -3.2879711429522462), (1, 2.451484271147476), (2, 1.361942605584113)),
        ((-1, -1.3670961684321727), (0, 1.1291475277153917), (1, -0.9236459010074382), (2, 0.025647750044303894), (3, 1.1971864978978768)),
        ((-1, 2.188460372045248), (0, -0.3217336751105322), (1, -0.5867312160485939), (2, -0.2162372882167674), (3, 0.5209413443416285), (4, 2.293136795273082)))


def run(root, progress=lambda v: None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    raw = []
    for i in range(6):
        clock = 1_000_000 + i * 100_000
        feature = np.zeros(203)
        feature[138] = 1.
        feature[200] = feature[201] = .8
        raw.append(RawIdentityDetection('fixture', IdentityNode(str(i), i % 2, clock, clock, str(i)),
                   0, clock, [i * .3, 0., 1., 4., 2., 1.5, 0., .1, .0], np.eye(9) * .2,
                   .8, feature, 'a' * 64))
    config = PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(
        active_limit=4, expansion_budget=512, max_frontier=4096,
        decision_mode='all-legal-hamming', paper_action_budget=4096,
        paper_action_frontier=8192, max_model_regret=1.))
    path = root / 'nonretained-action.sqlite'
    tracker = ExclusiveCompletionTracker(path, sequence_id='fixture', config=config)
    started = time.monotonic()
    snapshots = []
    try:
        for event, clock in enumerate((1_700_000, 1_900_000)):
            commit = tracker.step(raw if event == 0 else (), ROWS if event == 0 else (),
                                  frame_id=str(event), event_id=str(event), reference_us=clock, decision_us=clock)
            snapshots.append(dict(audit=commit.audit, prediction=commit.prediction))
            progress(dict(stage='exclusive_nonretained_action_cpu', completed_events=event+1, total_events=2,
                          completed_cases=int(event == 1), total_cases=1,
                          ETA_seconds=(time.monotonic()-started)*(1-event)/(event+1),
                          ETA_scope='one finite CPU production fixture only; independent acceptance excluded'))
        sealed = tracker.close()
        tracker = None
        return dict(kind='rbf_exclusive_nonretained_action_finite_CPU_output_v1',
                    raw=[asdict(r) for r in raw], rows=ROWS, snapshots=snapshots,
                    configuration=asdict(config), database=dict(path=path.name, sha256=sealed, bytes=path.stat().st_size),
                    completed_cases=1, completed_events=2, case_count=1, device='cpu', dataset_read=False,
                    offline_target_not_passed_to_tracker=True, trained_priority_policy=False,
                    full_stage_two_complete=False, complete_online_method_accepted=False,
                    physical_identity_or_tracking_metric_claim=False, same_resource_performance_comparison=False,
                    paper_performance_complete=False, elapsed_seconds=time.monotonic()-started)
    finally:
        if tracker is not None:
            tracker.close()
