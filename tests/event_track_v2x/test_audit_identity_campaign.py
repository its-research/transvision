"""Campaign aggregation tests; per-run numerical checks have a separate suite."""
import json

import pytest

from tools.event_track_v2x import audit_identity_campaign as campaign
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file


@pytest.fixture
def campaign_files(tmp_path, monkeypatch):
    data = tmp_path/'manifest.json'
    data.write_text('{}')
    calls = []
    runs = []
    for seed in campaign.SEEDS:
        root = tmp_path/str(seed)
        model = root/f'seed-{seed}'
        model.mkdir(parents=True)
        (model/'weights.pt').write_bytes(str(seed).encode())
        (model/'checkpoint.json').write_text(json.dumps({'weights': {'path': 'weights.pt'}}))
        (model/'epochs.jsonl').write_text('{}\n')
        plan = dict(seeds=[seed], required_campaign_seeds=list(campaign.SEEDS),
                    rank_runtime=[{'host': str(seed)}], source_sha256={'model': 'a'},
                    fit_config={'epochs': 10}, selection='fixed_final_epoch_no_validation_search')
        (root/'plan.json').write_text(json.dumps(plan))
        receipt = dict(plan_sha256=sha_file(root/'plan.json'),
            required_campaign_seeds=list(campaign.SEEDS), elapsed_seconds=1.,
            rank_runtime=plan['rank_runtime'], seeds=[dict(seed=seed,
                checkpoint_manifest=f'seed-{seed}/checkpoint.json',
                epochs_sha256=sha_file(model/'epochs.jsonl'))])
        (root/'receipt.json').write_text(json.dumps(receipt))
        runs.append((seed, root, sha_file(root/'receipt.json')))

    def verify(root, receipt_sha256, data_manifest, data_sha256, *, seed, require_full_train):
        calls.append((seed, require_full_train))
        assert sha_file(root/'receipt.json') == receipt_sha256
        model = root/f'seed-{seed}'
        return dict(seed=seed, verified=True, model_sha256=str(seed),
                    checkpoint_sha256=sha_file(model/'checkpoint.json'),
                    weights_sha256=sha_file(model/'weights.pt'))

    monkeypatch.setattr(campaign, 'verify', verify)
    return runs, data, sha_file(data), calls


def test_aggregate_checks_each_run_keeps_original_receipts_and_no_online_claim(campaign_files):
    runs, data, digest, calls = campaign_files
    report = campaign.verify_campaign(reversed(runs), data, digest)
    assert calls == [(seed, True) for seed in campaign.SEEDS]
    assert report['complete_three_seed_campaign'] and report['full_official_train']
    assert report['seeds'] == list(campaign.SEEDS)
    assert not report['remote_live_status_checked']
    assert not report['multi_machine_concurrency_verified']
    assert not report['paper_eligible'] and not report['tracking_validation_performed']
    assert all(sha_file(root/'receipt.json') == receipt_sha for _, root, receipt_sha in runs)


@pytest.mark.parametrize('indices', [(0, 1), (0, 1, 1), (0, 1, 2, 2)])
def test_missing_duplicate_or_extra_seed_rejected(campaign_files, indices):
    runs, data, digest, calls = campaign_files
    with pytest.raises(ValueError, match='exactly one run'):
        campaign.verify_campaign([runs[i] for i in indices], data, digest)
    assert not calls


def test_same_resolved_directory_is_not_independent(campaign_files):
    runs, data, digest, calls = campaign_files
    runs[1] = (runs[1][0], runs[0][1]/'.', runs[0][2])
    with pytest.raises(ValueError, match='independent single-seed'):
        campaign.verify_campaign(runs, data, digest)
    assert not calls


@pytest.mark.parametrize('field,value', [
    ('fit_config', {'epochs': 9}), ('source_sha256', {'model': 'b'}),
    ('selection', 'validation_best'), ('new_protocol_field', True),
    ('seeds', [2027, 3407]), ('required_campaign_seeds', [1337, 2027]),
])
def test_different_plans_or_non_single_seed_contract_rejected(campaign_files, field, value):
    runs, data, digest, _ = campaign_files
    seed, root, _ = runs[1]
    path = root/'plan.json'
    plan = json.loads(path.read_text())
    plan[field] = value
    path.write_text(json.dumps(plan))
    path = root/'receipt.json'
    receipt = json.loads(path.read_text())
    receipt['plan_sha256'] = sha_file(root/'plan.json')
    path.write_text(json.dumps(receipt))
    runs[1] = seed, root, sha_file(path)
    with pytest.raises(ValueError, match='plans differ|single-seed run'):
        campaign.verify_campaign(runs, data, digest)


def test_duplicate_saved_models_rejected(campaign_files, monkeypatch):
    runs, data, digest, _ = campaign_files
    original = campaign.verify

    def reused_model(*args, **kwargs):
        return dict(original(*args, **kwargs), model_sha256='same')

    monkeypatch.setattr(campaign, 'verify', reused_model)
    with pytest.raises(ValueError, match='identical saved model'):
        campaign.verify_campaign(runs, data, digest)


def test_artifact_changed_after_one_run_was_verified_rejected(campaign_files, monkeypatch):
    runs, data, digest, _ = campaign_files
    original = campaign.verify

    def changing_file(*args, **kwargs):
        report = original(*args, **kwargs)
        if kwargs['seed'] == 3407:
            (runs[0][1]/'seed-1337/weights.pt').write_bytes(b'changed')
        return report

    monkeypatch.setattr(campaign, 'verify', changing_file)
    with pytest.raises(ValueError, match='changed during verification'):
        campaign.verify_campaign(runs, data, digest)


def test_fixture_cannot_claim_full_official_train(campaign_files):
    runs, data, digest, calls = campaign_files
    result = campaign.verify_campaign(runs, data, digest, require_full_train=False)
    assert result['local_fixture_only'] and not result['full_official_train']
    assert calls == [(seed, False) for seed in campaign.SEEDS]
