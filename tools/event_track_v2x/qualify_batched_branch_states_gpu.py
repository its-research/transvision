"""Measured CUDA candidate against an admitted complete CPU reference.

Run only inside an exclusively assigned GPU worker. This entry point does not
dispatch, reserve hardware, change forest limits, or promote its result.
"""
import argparse
import json
from pathlib import Path
import sys

from qualify_batched_branch_states import main as qualify, sha
from rbf_gpu_replay_measurement import measure_replay


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--events', type=int, required=True)
    parser.add_argument('--recorded-reference-acceptance', type=Path, required=True)
    parser.add_argument('--reference-acceptance-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import re
    if not re.fullmatch(r'cuda:[0-9]+', args.device):
        parser.error('explicit cuda:N is required; CPU fallback is forbidden')
    if args.events < 1 or not 1 <= args.max_batch <= 4096:
        parser.error('positive events and max-batch in 1..4096 required')
    if args.output.exists():
        parser.error('new output directory required; previous failures are retained')
    return args


def main(argv=None):
    args = arguments(argv)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The inner qualifier verifies all reference hashes, full-event coverage,
    # search/work identity, and every selected and unselected branch state.
    # Recorded reference mode avoids rerunning an already accepted CPU replay.
    command = [str(Path(__file__).with_name('qualify_batched_branch_states.py'))]
    for name in ('replay', 'receipt_sha256', 'sequence', 'events',
                 'recorded_reference_acceptance', 'reference_acceptance_sha256',
                 'device', 'max_batch', 'output'):
        command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    command.append('--no-profile')

    def replay(*, output):
        original_argv = sys.argv
        try:
            sys.argv = command
            qualify()
        finally:
            sys.argv = original_argv
        result = json.loads((output/'candidate-check.json').read_bytes())
        if (result['device']['GPU_executed'] is not True
                or result['complete_reference_sequence_checked'] is not True
                or result['events'] != args.events):
            raise ValueError('complete CUDA candidate acceptance required')
        return dict(completed_events=result['events'])

    try:
        measure_replay(replay, device=args.device, output=args.output)
    finally:
        if args.output.is_dir():
            sources = {name: sha(Path(__file__).with_name(name)) for name in (
                Path(__file__).name, 'qualify_batched_branch_states.py',
                'rbf_gpu_replay_measurement.py')}
            evidence = dict(kind='rbf_measured_branch_state_CUDA_execution_binding_v1',
                command=command, execution_sources_sha256=sources,
                CPU_reference_reuse_requested=True, CPU_reference_acceptance_sha256=args.reference_acceptance_sha256,
                GPU_memory_target_percent=[75, 80], memory_target_independently_accepted=False,
                measurement_scope='candidate replay plus complete numerical comparison; includes CPU work',
                full_forest_on_GPU=False, production_promotion_allowed=False)
            with (args.output/'GPU-launch-binding.json').open('x') as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
                stream.write('\n')


if __name__ == '__main__':
    main()
