"""Execute reviewed numerical definitions from SHA-pinned local DMSTrack files.

Reference source is downloaded separately for local academic research, never
vendored or uploaded by this tool. Only the listed unchanged AST definitions
are executed; no package import hooks, datasets, detector or filesystem code.
This isolates dependency loading, not a security sandbox for arbitrary code.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import sys
import types

import numpy as np
import torch

COMMIT = 'd3b9949499c8e68ea33060873bd1cb95b6d4d323'
SOURCE_HASHES = {
    'box_utils.py': '961914a1291f57e86d9a7416e7318a3e911f0de8fb98a0c4492e2a9d5785de86',
    'transformation_utils.py': '09a5b0f3ce37a95d9d084de2edb5ea2c72263da0cbe965344416e0cb04a89741',
    'common_utils.py': '12d89510b0e8f2c29def3b2cf1963ae790e226fa056d8d7cc0b0f70cd9d5be21',
    'base_postprocessor.py': 'a284b213699355294895aba1bbbe29624d3bf7ebf7249aaf226c22f2cdb3fcde',
    'datasets_init.py': '56e02cf9227ac6f4f4b6d4a08501dbb947e696722997a9ea723110e8d375c85e',
}


def verified_sources(root):
    result = {}
    for name, digest in SOURCE_HASHES.items():
        p = Path(root).absolute() / name
        if any(x.is_symlink() for x in (p, *p.parents)) or not p.is_file() or p.stat().st_size > 100_000:
            raise ValueError('ordinary bounded oracle source required')
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('oracle source differs from reviewed pinned commit: ' + name)
        result[name] = ast.parse(data, filename=str(p))
    return result


def definitions(tree, names, namespace, filename):
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('reviewed numerical definitions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)


@contextmanager
def native_oracle(root, *, vehicle=False):
    if type(vehicle) is not bool:
        raise ValueError('explicit boolean native vehicle selection required')
    sources = verified_sources(root)
    modules = {n: types.ModuleType(n) for n in ('opencood', 'opencood.utils',
        'opencood.data_utils', 'opencood.data_utils.datasets')}
    if any(n in sys.modules for n in modules):
        raise ValueError('oracle requires a process without an imported OpenCOOD package')
    values = [ast.literal_eval(n.value) for n in sources['datasets_init.py'].body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GT_RANGE' for t in n.targets)]
    if values != [[-100, -40, -5, 100, 40, 3]]:
        raise ValueError('reference GT range changed')
    modules['opencood.data_utils.datasets'].GT_RANGE = values[0]
    numeric = dict(np=np, torch=torch)
    definitions(sources['transformation_utils.py'], ('x_to_world', 'x1_to_x2'), numeric, 'reference/transformation_utils.py')
    common = dict(np=np, torch=torch)
    definitions(sources['common_utils.py'], ('check_numpy_to_torch', 'rotate_points_along_z'), common, 'reference/common_utils.py')
    box = dict(np=np, torch=torch, sys=sys, x1_to_x2=numeric['x1_to_x2'], common_utils=types.SimpleNamespace(**common))
    definitions(sources['box_utils.py'], ('corner_to_center', 'boxes_to_corners_3d', 'project_box3d',
        'get_mask_for_boxes_within_range_torch', 'mask_boxes_outside_range_numpy', 'create_bbx', 'project_world_objects'),
        box, 'reference/box_utils.py')
    base = dict(np=np, torch=torch, box_utils=types.SimpleNamespace(**box))
    definitions(sources['base_postprocessor.py'], ('BasePostprocessor',), base, 'reference/base_postprocessor.py')
    processor = base['BasePostprocessor'](dict(order='lwh', max_num=100), train=False)
    def reference(metadata_by_cav, ego_cav):
        data = {}
        for cav in [ego_cav, *sorted(set(metadata_by_cav)-{ego_cav})]:
            metadata = copy.deepcopy(metadata_by_cav[cav])
            if vehicle:
                # The unchanged pinned project_world_objects applies its own
                # exact Pedestrian exclusion. No producer class helper is used.
                if any(not isinstance(v.get('obj_type'), str) or not v['obj_type'] for v in metadata['vehicles'].values()):
                    raise ValueError('explicit native obj_type required')
            else:
                metadata['vehicles'] = {k: v for k, v in metadata['vehicles'].items() if v['obj_type'] == 'Car'}
            boxes, mask, ids = processor.generate_object_center([dict(params=metadata, cav_id=int(cav))], np.eye(4))
            transform = numeric['x1_to_x2'](metadata['lidar_pose'], metadata_by_cav[ego_cav]['lidar_pose'])
            data[cav] = dict(object_bbx_center=torch.from_numpy(boxes), object_bbx_mask=torch.from_numpy(mask),
                object_ids=ids, gt_transformation_matrix=torch.from_numpy(transform).float())
        corners, ids = processor.generate_gt_bbx(data)
        return {int(i): p for i, p in zip(ids.tolist(), corners.numpy())}
    sys.modules.update(modules)
    try:
        yield reference
        verified_sources(root)
    finally:
        for name, module in modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
