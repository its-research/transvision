#!/usr/bin/env python3
"""Safe, sequential author-weight audit; no synthetic or native predictions."""
import argparse
import ast
import collections
import copy
import datetime
import gc
import hashlib
import json
import math
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def selected_ast(path, kind, name):
    nodes = [n for n in ast.parse(path.read_bytes()).body
             if isinstance(n, kind) and n.name == name]
    assert len(nodes) == 1
    node = nodes[0]
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    return node, digest


def write(path, value):
    path.open('x').write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def tensor_record(tensor):
    assert tensor.layout == torch.strided and tensor.device.type == 'cpu'
    assert not tensor.is_complex(), 'Unexpected complex author weight'
    assert bool(torch.isfinite(tensor).all()), 'Non-finite author tensor'
    value = tensor.detach().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy()
    return dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(),
                bytes=raw.nbytes, sha256=hashlib.sha256(memoryview(raw)).hexdigest())


def all_tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from all_tensors(child, prefix + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from all_tensors(child, prefix + '/' + str(index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['auditor_sha256']
    for source in plan['sources'].values():
        assert sha(source['path']) == source['sha256']
    output = Path(plan['output'])
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_path = Path(plan['sources']['dms_model']['path'])
    main_path = Path(plan['sources']['dms_main']['path'])
    config_node, config_sha = selected_ast(main_path, ast.FunctionDef, 'get_config')
    class_node, class_sha = selected_ast(model_path, ast.ClassDef, 'ObservationCovarianceNet')
    namespace = dict(torch=torch, nn=nn, np=np, math=math, os=os, copy=copy, time=time)
    exec(compile(ast.Module(body=[config_node, class_node], type_ignores=[]),
                 '<unmodified-pinned-DMSTrack-AST>', 'exec'), namespace)
    cfg = types.SimpleNamespace(det_name='multi_sensor_differentiable_kalman_filter',
                                dataset='v2v4real')
    author_config = namespace['get_config'](cfg, 'Car', 'train', 'val', 'all', False)
    dms_config = author_config[6]
    assert author_config[5] == ['ego', '1']
    assert author_config[1]['seq_eval'] == [f'{i:04d}' for i in range(9)]
    inference_doc = Path(plan['sources']['dms_inference']['path']).read_text()
    assert '--feature fusion' in inference_doc and '--use_multiple_nets' in inference_doc
    checkpoints = {r['name']: Path(r['path']) for r in plan['checkpoints']}
    ego = checkpoints['model_ego_epoch_3.pth']
    remote = checkpoints['model_1_epoch_3.pth']
    assert Path(str(ego).replace('ego', '1')) == remote and not ego.samefile(remote)
    receipts = []
    for spec in plan['checkpoints']:
        path = Path(spec['path'])
        proof = Path(spec['byte_receipt'])
        assert sha(proof) == spec['byte_receipt_sha256']
        prior = json.loads(proof.read_bytes())
        assert sha(path) == spec['sha256'] == prior['sha256']
        assert path.stat().st_size == spec['bytes'] == prior['bytes']
        obj = torch.load(path, map_location='cpu', weights_only=True)
        state = obj['state_dict'] if spec['method'] == 'CoopTrack' else obj
        assert isinstance(state, dict) and state and all(isinstance(v, torch.Tensor) for v in state.values())
        tensors = {name: tensor_record(tensor) for name, tensor in all_tensors(obj)}
        state_records = {name: tensor_record(tensor) for name, tensor in state.items()}
        strict = None
        if spec['method'] == 'DMSTrack':
            model = namespace['ObservationCovarianceNet'](dms_config, 'fusion').cpu()
            incompatible = model.load_state_dict(state, strict=True)
            assert not incompatible.missing_keys and not incompatible.unexpected_keys
            loaded = model.state_dict()
            assert set(loaded) == set(state)
            assert all(torch.equal(loaded[k], state[k]) for k in state)
            strict = dict(author_class='ObservationCovarianceNet', feature='fusion',
                          config=dms_config, class_ast_sha256=class_sha,
                          config_function_ast_sha256=config_sha,
                          missing_keys=[], unexpected_keys=[],
                          every_loaded_tensor_exactly_equal=True)
            del loaded, model
        result = dict(kind='rbf_author_native_checkpoint_safe_tensor_and_class_audit',
                      name=path.name, method=spec['method'], checkpoint_sha256=spec['sha256'],
                      byte_receipt_sha256=spec['byte_receipt_sha256'],
                      auditor_sha256=sha(__file__), plan_sha256=sha(args.plan),
                      source_commit=prior['source']['author_source_commit'],
                      safe_weights_only=True, all_checkpoint_tensors_finite=True,
                      checkpoint_tensor_count=len(tensors), state_tensor_count=len(state),
                      state_numel=sum(v['numel'] for v in state_records.values()),
                      tensors=tensors, state_tensors=state_records, native_class_strict_load=strict,
                      CoopTrack_native_class_load_complete=False,
                      native_features_qualified=False, native_predictions_produced=False,
                      native_metrics_independently_verified=False, native_method_accepted=False,
                      checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        receipt = output / (path.name + '.tensor-audit.json')
        write(receipt, result)
        receipts.append(dict(name=path.name, method=spec['method'], path=str(receipt),
                             sha256=sha(receipt), state_tensor_count=len(state),
                             state_numel=result['state_numel'],
                             native_class_strict_load_complete=strict is not None))
        print(json.dumps(receipts[-1]), flush=True)
        del obj, state, tensors, state_records
        gc.collect()
    write(output / 'qualification-summary.json', dict(
        kind='rbf_five_author_weight_safe_tensor_qualification_summary',
        receipts=receipts, plan_sha256=sha(args.plan), sources=plan['sources'],
        checkpoint_bytes=sum(r['bytes'] for r in plan['checkpoints']),
        all_five_safe_tensor_audits_passed=True,
        DMSTrack_both_CAV_native_class_strict_load_complete=True,
        DMSTrack_native_ego_to_remote_path_binding_verified=True,
        author_AST_unmodified=True, full_native_runtime_imported=False,
        native_feature_forward_run=False, native_predictions_produced=False,
        native_method_accepted=False, same_resource_measured=False, paper_complete=False,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
