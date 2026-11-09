"""CPU/Gloo integration only; it does not qualify GPU inputs or training."""
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tools.event_track_v2x.train_forest_identity import FitConfig
from tools.event_track_v2x.train_v2v4real_joint_identity import JointHeadLoss
from transvision.models.event_track_v2x.forest_training import batched_row_logits
from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
from transvision.models.event_track_v2x.forest_supervision import RowSupervision
from transvision.models.event_track_v2x.paper_calibration import separate_identity_losses
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from test_forest_tracking import observation


def examples():
    data=[]
    for sources in [(0,0),(1,0),(0,1,0)]:
        raw=tuple(observation(str(i),i*.2,source=s) for i,s in enumerate(sources))
        c=ForestRowContext(tuple(range(len(raw))),raw,1_100_000)
        data.append((c,RowSupervision((True,)+(False,)*(len(raw)-1),(True,)*len(raw),'birth')))
    return data


def worker(rank,world,rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=world,timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(7)
        model=RecoverableIdentityModel(hidden=8,heads=2,dropout=0.).double().eval()
        direct=RecoverableIdentityModel(hidden=8,heads=2,dropout=0.).double().eval()
        direct.load_state_dict(model.state_dict())
        config=FitConfig(hidden=8,heads=2,dropout=0.)
        ddp=DistributedDataParallel(JointHeadLoss(model,config,{'process_noise':.1}))
        batch=examples()
        value=ddp(batch);value['loss'].backward()
        contexts,targets=zip(*batch)
        full=separate_identity_losses(batched_row_logits(direct,contexts),contexts,targets)
        full['loss'].backward()
        assert value['local_rows']==(1 if rank<3 else 0)
        assert value['global_counts']==full['counts']==dict(cross_source=2,temporal=2)
        for h in ('cross_source','temporal'):
            torch.testing.assert_close(torch.tensor(value['global_sums'][h],dtype=torch.float64),
                full[h].detach()*full['counts'][h],atol=1e-12,rtol=1e-10)
        for (name,p),(other,q) in zip(model.named_parameters(),direct.named_parameters()):
            assert name==other and p.grad is not None
            expected=q.grad if q.grad is not None else torch.zeros_like(q)
            torch.testing.assert_close(p.grad,expected,atol=1e-11,rtol=1e-8)
    finally:
        dist.destroy_process_group()


def test_real_ddp_collectives_match_full_joint_model_gradient_with_empty_rank(tmp_path):
    torch.multiprocessing.spawn(worker,args=(4,'file://'+str(tmp_path/'rendezvous')),nprocs=4,join=True)
