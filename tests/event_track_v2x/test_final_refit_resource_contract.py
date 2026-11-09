"""Pure metadata admission tests; no tensors, fitting, or tracking execution."""
import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT/'transvision/models/event_track_v2x/final_refit_resource_contract.py'
spec = importlib.util.spec_from_file_location('metadata_contract', MODULE)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bundle(root, seed=1337, mutate=None):
    root.mkdir(parents=True)
    checkpoint = dict(kind='persistent_forest_row_identity_checkpoint_v1', seed=seed,
        data_split='train', fixture=False, labels_in_model_inputs=False,
        validation_or_test_selection=False, weights={'sha256': 'a'*64}, model_sha256='b'*64,
        selected_epoch=3, frozen_cache_identity={'calibration': seed})
    counts = {str(i).zfill(4): 1 for i in range(46)}
    recipe = dict(seed=seed, fit_config={'epochs': 3}, frozen_selection_receipt_sha256='c'*64)
    plan = dict(kind='rbf_nested_selected_epoch_all_class_full_train_refit_candidate_v1', recipe=recipe)
    train = dict(kind='rbf_new_final_refit_six_artifacts_independent_byte_metadata_admission_v1', seed=seed,
        all_six_registered_artifact_bytes_independently_read=True,
        frozen_nested_epoch_and_full_train_recipe_verified=True,
        two_head_training_and_producer_CPU_reference_gradient_report_bound=True,
        weights_sha256='a'*64, model_sha256='b'*64, task_id='training', epochs=3,
        recipe_sha256=digest(recipe), artifacts={})
    forward = dict(kind='rbf_new_final_refit_all_row_independent_byte_coverage_admission_v1', seed=seed,
        all_registered_output_bytes_verified=True, all_46_train_sequences_row_coverage_verified=True,
        weights_sha256='a'*64, sequence_counts=counts, rows=46)
    numeric = dict(kind='rbf_new_final_refit_full_independent_float64_joint_reference_v1', seed=seed,
        full_independent_numeric_pass=True, all_rows_selected_without_GT=True, numeric_failed_rows=0,
        weights_sha256='a'*64, atol=1e-4, rtol=1e-4, rows=46,
        sequences=[dict(sequence_id=k, rows=v, failures=0) for k, v in counts.items()])
    if mutate:
        mutate(checkpoint, train, forward, numeric, plan)
    cp_path = root/'checkpoint.json'; cp_path.write_text(json.dumps(checkpoint))
    plan_path = root/'plan.json'; plan_path.write_text(json.dumps(plan))
    train['artifacts'] = dict(checkpoint=dict(sha256=reference(cp_path)['sha256'], task='training'),
                              plan=dict(sha256=reference(plan_path)['sha256']))
    paths = {'plan': plan_path}
    for name, data in [('training', train), ('forward', forward), ('numeric', numeric)]:
        if name == 'forward': data['train_byte_admission_sha256'] = reference(paths['training'])['sha256']
        if name == 'numeric': data['prediction_proof_sha256'] = reference(paths['forward'])['sha256']
        paths[name] = root/(name+'.json'); paths[name].write_text(json.dumps(data))
    evidence = {k: reference(p) for k, p in paths.items()}
    return checkpoint, reference(cp_path), evidence


class MetadataAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
    def tearDown(self):
        self.tmp.cleanup()
    def validate(self, mutate=None):
        cp, ref, evidence = bundle(self.root/'input', mutate=mutate)
        return contract.validate_final_refit(cp, ref['sha256'], evidence)
    def test_qualified_current_refit_needs_no_checkpoint_relabel(self):
        result = self.validate(); self.assertEqual(result['rows'], 46)
        self.assertFalse(result['new_runtime_numerically_qualified'])
    def test_incomplete_or_mismatched_evidence_rejected(self):
        changes = [
            lambda c,t,f,n,p: t.update(all_six_registered_artifact_bytes_independently_read=False),
            lambda c,t,f,n,p: f.update(all_46_train_sequences_row_coverage_verified=False),
            lambda c,t,f,n,p: n.update(full_independent_numeric_pass=False),
            lambda c,t,f,n,p: n.update(seed=3407),
            lambda c,t,f,n,p: n.update(weights_sha256='d'*64),
            lambda c,t,f,n,p: n['sequences'].pop(),
            lambda c,t,f,n,p: n['sequences'][0].update(failures=1),
            lambda c,t,f,n,p: n.update(atol=1e-3),
            lambda c,t,f,n,p: c.update(validation_or_test_selection=True),
            lambda c,t,f,n,p: c.update(fixture=True),
            lambda c,t,f,n,p: c.update(selected_epoch=9),
            lambda c,t,f,n,p: p['recipe'].update(seed=3407),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                cp, ref, ev = bundle(self.root/str(index), mutate=change)
                with self.assertRaises(ValueError): contract.validate_final_refit(cp, ref['sha256'], ev)
    def test_reference_tampering_and_missing_proof_rejected(self):
        cp, ref, ev = bundle(self.root/'input')
        with self.assertRaises(ValueError): contract.validate_final_refit(cp, ref['sha256'], {k:v for k,v in ev.items() if k!='numeric'})
        Path(ev['numeric']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes differ'): contract.validate_final_refit(cp, ref['sha256'], ev)
    def test_cross_seed_checkpoint_rejected(self):
        cp, ref, ev = bundle(self.root/'input'); cp['seed'] = 3407
        with self.assertRaisesRegex(ValueError, 'seed differs'): contract.validate_final_refit(cp, ref['sha256'], ev)


@dataclass
class Config:
    state: str = 'test-config'


class SeedInputRouting(unittest.TestCase):
    setUp = MetadataAdmission.setUp
    tearDown = MetadataAdmission.tearDown
    def namespace(self):
        module = types.ModuleType('resource_metadata_checks.final_refit_resource_contract')
        module.validate_final_refit = contract.validate_final_refit
        self.modules = {'resource_metadata_checks': types.ModuleType('resource_metadata_checks'),
                        'resource_metadata_checks.final_refit_resource_contract': module,
                        'tools.event_track_v2x.run_tracking_v2': types.ModuleType('tools.event_track_v2x.run_tracking_v2')}
        self.modules['tools.event_track_v2x.run_tracking_v2'].schedule_rows = lambda path, sha: json.loads(path.read_text())
        tree = ast.parse((ROOT/'transvision/models/event_track_v2x/resource_sweep.py').read_text())
        names = {'_reference', 'validate_current_refit_spec', 'validate_spec'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        def cache(path, sha):
            data = json.loads((Path(path)/'manifest.json').read_text())
            return types.SimpleNamespace(manifest_json=json.dumps(data), frozen=data['frozen'])
        n = dict(__package__='resource_metadata_checks', sys=sys, KIND='rbf_cpu_resource_sweep_v1', Path=Path, json=json, math=__import__('math'),
            SEEDS=contract.SEEDS, CURRENT_REFIT_KIND='rbf_current_final_refit_cpu_resource_sweep_v2',
            require_l40_cpu_worker=lambda: None, contained_file=lambda p,f: Path(p)/f,
            sha_file=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            VerifiedForestCache=cache, frozen_cache_identity=lambda c:c.frozen,
            configuration=lambda j:Config(), asdict=asdict,
            load_identity_checkpoint=lambda root,sha=None,manifest_sha256=None,**kw:(types.SimpleNamespace(signature='s'), json.loads((Path(root)/'checkpoint.json').read_text())),
            LEARNED_COMPONENT_KINDS={'learned_component'}, _digest=digest, random=random,
            source_snapshot=lambda:{'mock': 'software-only'}, THREAD_ENV={})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<resource-metadata-check>', 'exec'), n)
        return n
    def resource_spec(self):
        inputs, jobs = {}, []
        for seed in contract.SEEDS:
            root = self.root/str(seed); cp, cr, evidence = bundle(root, seed=seed)
            cache = root/'cache'; cache.mkdir(); manifest = cache/'manifest.json'
            manifest.write_text(json.dumps(dict(split='val', frame_count=7189, sequences=['seq'], frozen=cp['frozen_cache_identity'])))
            schedule = root/'schedule.json'; schedule.write_text(json.dumps([{'sequence_id': 'seq'}]))
            inputs[str(seed)] = dict(cache=dict(reference(manifest), path=str(cache)), schedule=reference(schedule),
                checkpoint=dict(cr, path=str(root)), refit_evidence=evidence)
            jobs.extend(dict(id=f'{backend}-{seed}', backend=backend, configuration={}, seed=seed)
                        for backend in ('jpda_ci', 'pkf'))
        return dict(kind='rbf_current_final_refit_cpu_resource_sweep_v2', inputs=inputs, jobs=jobs,
                    repetitions=3, order_seed=1337, timeout_seconds=900)
    def test_each_method_gets_its_own_seed_inputs(self):
        value = self.resource_spec(); n = self.namespace()
        with patch.dict(sys.modules, self.modules): plan = n['validate_current_refit_spec'](value)
        self.assertEqual(len(plan['execution_order']), 18)
        for job in plan['jobs']:
            self.assertEqual(job['cache'], value['inputs'][str(job['seed'])]['cache'])
        self.assertFalse(plan['new_runtime_numerically_qualified'])
    def test_cross_seed_cache_and_missing_seed_and_priority_rejected(self):
        value = self.resource_spec(); n = self.namespace()
        changes = [lambda v:v['inputs']['3407'].update(cache=v['inputs']['1337']['cache']),
                   lambda v:v['jobs'].pop(), lambda v:v.update(repetitions=2),
                   lambda v:v['jobs'][0].update(backend='learned_component'),
                   lambda v:v['inputs']['3407'].update(checkpoint=v['inputs']['1337']['checkpoint'])]
        with patch.dict(sys.modules, self.modules):
            for change in changes:
                bad = copy.deepcopy(value); change(bad)
                with self.assertRaises(ValueError): n['validate_current_refit_spec'](bad)

    def test_legacy_single_cache_schema_preserves_original_flag_gate(self):
        current = self.resource_spec(); n = self.namespace(); common = current['inputs']['1337']
        jobs = []
        for seed in contract.SEEDS:
            item = current['inputs'][str(seed)]
            cp_path = Path(item['checkpoint']['path'])/'checkpoint.json'
            cp = json.loads(cp_path.read_text()); cp['full_official_train'] = True
            cp['frozen_cache_identity'] = {'calibration':1337}
            cp_path.write_text(json.dumps(cp))
            jobs.append(dict(id=f'legacy-{seed}', backend='pkf', configuration={},
                checkpoint=dict(reference(cp_path), path=str(cp_path.parent))))
        old = dict(kind='rbf_cpu_resource_sweep_v1', cache=common['cache'], schedule=common['schedule'],
                   jobs=jobs, repetitions=3, order_seed=1337, timeout_seconds=900)
        with patch.dict(sys.modules, self.modules):
            plan = n['validate_spec'](old)
            self.assertEqual(len(plan['execution_order']), 9)
            cp_path = Path(jobs[0]['checkpoint']['path'])/'checkpoint.json'
            cp = json.loads(cp_path.read_text()); cp.pop('full_official_train')
            cp_path.write_text(json.dumps(cp)); jobs[0]['checkpoint']['sha256'] = reference(cp_path)['sha256']
            with self.assertRaisesRegex(ValueError, 'full-train three-seed'):
                n['validate_spec'](old)


class L40Placement(unittest.TestCase):
    def test_designated_worker_and_cuda_disabled_required(self):
        tree = ast.parse((ROOT/'transvision/models/event_track_v2x/resource_sweep.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_l40_cpu_worker')
        clearml = types.ModuleType('clearml')
        task = types.SimpleNamespace(data=types.SimpleNamespace(last_worker='10.100.35.121-L40S:gpu3,4,5'), reload=lambda:None)
        clearml.Task = types.SimpleNamespace(get_task=lambda **kw:task)
        env = {'CUDA_VISIBLE_DEVICES': '', 'CLEARML_TASK_ID': 'task'}
        n = dict(sys=types.SimpleNamespace(platform='linux'), os=types.SimpleNamespace(environ=env))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<L40-routing>', 'exec'), n)
        with patch.dict(sys.modules, {'clearml':clearml}):
            self.assertEqual(n['require_l40_cpu_worker'](), task.data.last_worker)
            for bad in ('10.100.35.112:gpu0', '10.100.34.18-A100:gpu0,1,2,3', None):
                task.data.last_worker = bad
                with self.assertRaises(ValueError): n['require_l40_cpu_worker']()
            task.data.last_worker = '10.100.35.121-L40S:gpu3,4,5'
            env['CUDA_VISIBLE_DEVICES'] = '0'
            with self.assertRaises(ValueError): n['require_l40_cpu_worker']()
            env['CUDA_VISIBLE_DEVICES'] = ''; env.pop('CLEARML_TASK_ID')
            with self.assertRaises(ValueError): n['require_l40_cpu_worker']()
            env['CLEARML_TASK_ID'] = 'task'; n['sys'].platform = 'darwin'
            with self.assertRaises(ValueError): n['require_l40_cpu_worker']()


if __name__ == '__main__': unittest.main()
