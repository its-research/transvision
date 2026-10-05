"""Contract controls only; synthetic inventory is not physical GPU acceptance."""
from copy import deepcopy
import unittest

from tools.event_track_v2x.priority_gpu_runtime import validate_runtime


def inventory(name='Tesla V100', world=4):
    return [dict(rank=i,world_size=world,host='worker',backend='nccl',device=f'cuda:{i}',
        gpu_uuid=f'uuid-{i}',gpu_name=name,capability=[7,0],native_architectures=['sm_70'],
        total_memory_bytes=32*1024**3,TF32_matmul=False,TF32_cudnn=False) for i in range(world)]


class RuntimeTests(unittest.TestCase):
    def test_authorized_device_name_is_not_a_whitelist(self):
        for name in ('Tesla V100','NVIDIA A100','RTX 5090','RTX 3090','RTX 2080 Ti','supported other GPU'):
            with self.subTest(name=name):
                validate_runtime(inventory(name),require_full_train=True)
        validate_runtime(inventory(world=8),require_full_train=True)

    def test_actual_rank_and_runtime_fail_closed(self):
        changes = (
            ('world_size',8),('rank',1),('host','another'),('backend','gloo'),('device','cuda:1'),
            ('gpu_uuid','uuid-1'),('gpu_uuid','unavailable'),('gpu_name','NVIDIA L40S'),
            ('gpu_name',''),('capability',[8,0]),('native_architectures',['compute_70']),
            ('total_memory_bytes',0),('TF32_matmul',True),('TF32_cudnn',True))
        for key,value in changes:
            with self.subTest(key=key,value=value):
                rows=deepcopy(inventory());rows[0][key]=value
                with self.assertRaises(ValueError):
                    validate_runtime(rows,require_full_train=True)
        with self.assertRaises(ValueError):
            validate_runtime(inventory(world=2),require_full_train=True)

    def test_low_level_CPU_fixture_does_not_acquire_full_train_permission(self):
        rows=[dict(rank=0,world_size=1,host='worker',backend='gloo',device='cpu')]
        validate_runtime(rows,require_full_train=False)
        with self.assertRaises(ValueError):
            validate_runtime(rows,require_full_train=True)


if __name__=='__main__':
    unittest.main()
