"""DDP row partitioning with a separate GLOBAL denominator for each loss
head."""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tools.event_track_v2x.train_forest_identity_ddp import _rank_zero, rank_rows
from .forest_training import batched_row_logits
from .paper_calibration import separate_identity_losses


class _RowLoss(torch.nn.Module):

    def __init__(self, model, config, protocol):
        super().__init__()
        self.model = model
        self.config = config
        self.protocol = protocol

    def forward(self, examples):
        selected = rank_rows(examples, dist.get_rank(), dist.get_world_size())
        zero = sum(p.sum() * 0 for p in self.model.parameters())
        if selected:
            contexts, targets = zip(*selected)
            logits = batched_row_logits(
                self.model,
                contexts,
                max_nodes=self.protocol['parent_limit'] + 1,
                max_batch=self.config.batch_size,
                geometry_weight=self.config.geometry_weight,
                process_noise=self.protocol['process_noise'])
            result = separate_identity_losses(logits, contexts, targets)
        else:
            result = dict(cross_source=zero, temporal=zero, counts=dict(cross_source=0, temporal=0))
        counts = torch.tensor([result['counts'][k] for k in ('cross_source', 'temporal')], device=zero.device, dtype=torch.long)
        dist.all_reduce(counts)
        global_counts = dict(zip(('cross_source', 'temporal'), counts.cpu().tolist()))
        loss = zero
        for key in global_counts:
            loss = loss + result[key] * (dist.get_world_size() * result['counts'][key] / max(1, global_counts[key]))
        return dict(loss=loss, counts=global_counts)


class PaperDDP:

    def __init__(self, device, *, fixture):
        if not dist.is_initialized():
            raise ValueError('initialized torchrun process group required')
        self.world_size = dist.get_world_size()
        self.device = torch.device(device)
        if not fixture and (self.world_size < 4 or self.device.type != 'cuda' or dist.get_backend() != 'nccl'):
            raise ValueError('real paper DDP requires at least four CUDA/NCCL ranks')
        if self.device.type == 'cuda' and 'A100' not in torch.cuda.get_device_name(self.device):
            raise ValueError('paper training hardware contract requires A100')

    def write(self, operation):
        return _rank_zero(operation)

    def bind(self, model, config, protocol):
        return DistributedDataParallel(_RowLoss(model, config, protocol), device_ids=[self.device.index] if self.device.type == 'cuda' else None)

    def reported_loss(self, loss):
        result = loss.detach().clone()
        dist.all_reduce(result)
        return float(result / self.world_size)
