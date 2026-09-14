from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tools.event_track_v2x.train_forest_identity import FitConfig
from tools.event_track_v2x.train_forest_identity_ddp import RowLoss, fit_distributed, rank_rows
from tools.event_track_v2x.audit_forest_identity_ddp import verify
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import TrainingShard
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from test_forest_training_data import prepared_rows


@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 63, 64, 65])
def test_rank_partition_covers_each_row_once_without_padding(size):
    rows = np.arange(size)
    groups = [rank_rows(rows, rank, 4) for rank in range(4)]
    assert sorted(int(i) for group in groups for i in group) == list(range(size))
    assert max(map(len, groups))-min(map(len, groups)) <= 1


def _distributed_fixture(rank, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        data = Path(root)/'train-rows'
        manifest = json.loads((data/'manifest.json').read_bytes())
        config = FitConfig(epochs=2, batch_size=4, hidden=8, heads=2, dropout=0.)
        record = max(manifest['shards'], key=lambda r: r['supervised_rows'])
        shard = TrainingShard(data/record['path'], record, 8)
        examples = [shard.example(int(i)) for i in shard.valid_indices[:3]]
        assert len(examples) < 4  # Explicit empty-rank backward, not duplicated padding.
        torch.manual_seed(812)
        model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).double()
        full = RowLoss(model, config, manifest['row_protocol'])
        full(examples, len(examples), 1).backward()
        reference = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        parallel = DistributedDataParallel(full)
        parallel(examples[rank::4], len(examples), 4).backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, reference[name], rtol=1e-7, atol=1e-9)
        del parallel, full, model
        receipt = fit_distributed(data, sha_file(data/'manifest.json'), Path(root)/'ddp-fit',
            config=config, device='cpu', require_full_train=False, seeds=(1337,))
        assert receipt['world_size'] == 4 and not receipt['paper_eligible']
        assert not receipt['complete_three_seed_campaign']
        with pytest.raises(ValueError, match='CUDA/NCCL'):
            fit_distributed(data, sha_file(data/'manifest.json'), Path(root)/'rejected', device='cpu')
    finally:
        dist.destroy_process_group()


def test_four_process_gradient_parity_uneven_tail_and_real_rank_receipt(prepared_rows, tmp_path):
    data, manifest, *_ = prepared_rows
    # Spawn distinct processes; a sequential rank emulation cannot verify DDP.
    mp.spawn(_distributed_fixture, args=(str(tmp_path), str(tmp_path/'rendezvous')), nprocs=4, join=True)
    output = tmp_path/'ddp-fit'
    receipt = json.loads((output/'receipt.json').read_bytes())
    checkpoint = receipt['seeds'][0]
    _, metadata = load_identity_checkpoint(output/'seed-1337', checkpoint['checkpoint_sha256'], config=ForestTrackingConfig())
    assert metadata['distributed_world_size'] == 4 and metadata['model_sha256'] != metadata['initial_model_sha256']
    epochs = [json.loads(line) for line in (output/'seed-1337/epochs.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in epochs] == [1, 2]
    assert all(sum(x['supervised_rows'] for x in r['rank_progress']) == sum(s['supervised_rows'] for s in manifest['shards']) for r in epochs)
    assert all(len({x['model_sha256'] for x in r['rank_progress']}) == 1 for r in epochs)
    assert not (tmp_path/'rejected').exists()
    report=verify(output,sha_file(output/'receipt.json'),data/'manifest.json',sha_file(data/'manifest.json'),
                  seed=1337,require_full_train=False)
    assert report['verified'] and report['world_size']==4 and not report['remote_live_status_checked']
    assert not report['paper_eligible'] and not report['complete_three_seed_campaign']
    path=output/'seed-1337/epochs.jsonl'
    original=path.read_bytes()
    path.write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='epoch-log identity'):
        verify(output,sha_file(output/'receipt.json'),data/'manifest.json',sha_file(data/'manifest.json'),
               seed=1337,require_full_train=False)
    path.write_bytes(original)
    receipt_path=output/'receipt.json'
    receipt_original=receipt_path.read_bytes()
    receipt['complete_three_seed_campaign']=True
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='three-seed campaign completion'):
        verify(output,sha_file(receipt_path),data/'manifest.json',sha_file(data/'manifest.json'),
               seed=1337,require_full_train=False)
    receipt_path.write_bytes(receipt_original)


def test_requires_real_initialized_process_group(tmp_path):
    with pytest.raises(ValueError, match='initialized distributed'):
        fit_distributed(tmp_path, '0'*64, tmp_path/'out', device='cpu', require_full_train=False)
