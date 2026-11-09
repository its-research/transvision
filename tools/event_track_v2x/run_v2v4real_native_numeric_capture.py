#!/usr/bin/env python3
"""Original official-test runtime for read-only numeric traces, never raw re-export.

All 3986 original frames and accepted raw payloads are required. This uses the
qualified original source/extension/point-reader setup, then the trace driver.
Actual runtime, capacity, traces and independent numeric acceptance remain
separate gates. No network downloads, labels, fitting or task dispatch occur.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

# Direct rank execution uses a private package so the original lazy point-reader
# can bind its own tools/transvision namespaces without accepting ambient code.
if not __package__:
    import types
    if any(n in ('tools','transvision') or n.startswith(('tools.','transvision.')) for n in sys.modules):
        raise ValueError('fresh isolated numeric rank interpreter required')
    package_name='rbf_numeric_rank_runtime'
    package=types.ModuleType(package_name)
    package.__path__=[str(Path(__file__).resolve().parent)]
    sys.modules[package_name]=package
    __package__=package_name

import numpy as np

from . import run_v2v4real_official_test_raw_features as original
from .capture_v2v4real_native_numeric_rows import capture_rows


def run(*, source_archive, inputs_root, admission, checkpoint, seed, output,
        accepted_raw_index, accepted_raw_sha, accepted_raw_root, max_trace_bytes,
        device='cuda:0',trace_layout='compact-cells-v1',rank=0,world_size=1):
    if seed not in original.CHECKPOINTS or not str(device).startswith('cuda'):
        raise ValueError('registered native seed and CUDA device required')
    if type(max_trace_bytes) is not int or max_trace_bytes <= 0:
        raise ValueError('explicit positive full trace byte budget required')
    if type(world_size) is not int or world_size not in (1,4,8) or type(rank) is not int or not 0<=rank<world_size:
        raise ValueError('qualified rank/card assignment required')
    producer_sha = original.sha(__file__)
    source_archive = original.asset(source_archive, original.SOURCE_ARCHIVE_SHA)
    checkpoint = original.asset(checkpoint, original.CHECKPOINTS[seed])
    inputs_root = Path(inputs_root).absolute()
    projection_path = original.asset(inputs_root/'manifest.json', original.PROJECTION_SHA)
    frames = original.asset(inputs_root/'frames.jsonl', original.FRAME_SHA)
    admission = original.asset(admission, original.ADMISSION_SHA)
    raw_index = original.asset(accepted_raw_index, accepted_raw_sha)
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('fresh output directory required')
    import torch
    import yaml
    out.mkdir(parents=True)
    files = original.extract_source(source_archive, out/'frozen-runtime')
    binding = original.bind_frozen_point_reader(out/'frozen-runtime', files)
    core = original.frozen_core(out/'frozen-runtime')
    extension_path = out/'extension-build-receipt.json'
    extensions = original.build_frozen_extensions(out/'frozen-runtime', files, extension_path)
    extension_sha = original.sha(extension_path)
    loader = importlib.import_module(original.PACKAGE+'.v2v4real_inputs')
    projection, records = loader.load_prepared_frames(inputs_root,
        expected_manifest_sha256=original.PROJECTION_SHA)
    original.validate_scope(projection, json.loads(admission.read_bytes()), records)
    if any(n == 'opencood' or n.startswith('opencood.') for n in sys.modules):
        raise ValueError('ambient detector namespace forbidden')
    source = out/'frozen-runtime/official'
    sys.path.insert(0, str(source))
    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
    from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
    from opencood.models.point_pillar import PointPillar
    from opencood.utils.pcd_utils import mask_ego_points, mask_points_by_range
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = load_point_pillar_params(yaml.safe_load(
        (source/'opencood/hypes_yaml/point_pillar_late_fusion.yaml').read_text()))
    anchors = np.asarray(build_postprocessor(config['postprocess'], train=False).generate_anchor_box())
    if anchors.ndim != 4 or anchors.shape[-1] != 7 or not np.isfinite(anchors).all():
        raise ValueError('invalid original anchors')
    detector = core.FrozenPointPillar(PointPillar(config['model']['args']),
        SpVoxelPreprocessor(config['preprocess'], train=False), anchors,
        checkpoint=checkpoint, checkpoint_sha256=original.CHECKPOINTS[seed], device=device,
        formal_seed=seed, formal_protocol=original.PROTOCOL_ID,
        official_commit='5a821e13753bafc611f95c47bc1a306acdcb0f7c')
    pcd = importlib.import_module(original.PACKAGE+'.v2v4real_pcd')
    result = capture_rows(records, inputs_root, detector, core,
        config['preprocess']['cav_lidar_range'], pcd.read_native_pcd,
        mask_ego_points, mask_points_by_range, seed=seed, source_archive=source_archive,
        accepted_raw_index=raw_index, accepted_raw_sha=accepted_raw_sha,
        accepted_raw_root=accepted_raw_root, output=out/'trace', max_trace_bytes=max_trace_bytes,
        trace_layout=trace_layout,
        rank=rank,world_size=world_size,
        progress=lambda v: print(json.dumps(v), flush=True))
    original.verify_runtime_sources(out/'frozen-runtime', files)
    original.verify_built_extensions(out/'frozen-runtime', extensions)
    for path, sha in ((source_archive, original.SOURCE_ARCHIVE_SHA),
                      (checkpoint, original.CHECKPOINTS[seed]), (projection_path, original.PROJECTION_SHA),
                      (frames, original.FRAME_SHA), (admission, original.ADMISSION_SHA),
                      (raw_index, accepted_raw_sha), (extension_path, extension_sha),
                      (__file__, producer_sha)):
        original.asset(path, sha)
    from . import v2v4real_official_test_runtime as runtime_policy
    rank_runtime=runtime_policy.inspect_runtime();runtime_policy.require_runtime(rank_runtime)
    result.update(kind='v2v4real_native_numeric_trace_producer_v1', seed=seed,
        rank_runtime=rank_runtime, TF32_enabled=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        producer_sha256=producer_sha, point_reader_binding=binding,
        actual_device=original.device_identity(torch, device), gt_read=False,
        projection_frames_sha256=original.FRAME_SHA, extension_build_sha256=extension_sha,
        max_uncompressed_trace_bytes=max_trace_bytes)
    with (out/'numeric-capture-receipt.json').open('x') as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive', 'inputs-root', 'admission', 'checkpoint', 'output',
                 'accepted-raw-index', 'accepted-raw-sha', 'accepted-raw-root'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, choices=sorted(original.CHECKPOINTS), required=True)
    parser.add_argument('--max-trace-bytes', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,choices=[1,4,8],default=1)
    parser.add_argument('--trace-layout',choices=['full','compact-cells-v1'],default='compact-cells-v1')
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
    main()
