"""GPU4/GPU8 overlap, stale heartbeat and CPU-only hardware admission."""
import datetime
import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[2]/'tools/event_track_v2x/submit_rbf_final_identity.py'
SPEC = importlib.util.spec_from_file_location('final_refit_dispatch',MODULE)
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.datetime(2026,10,3,17,tzinfo=datetime.timezone.utc)
        self.queues = [dict(id='q4',name='GPU4-V100',entries=[]),dict(id='q8',name='GPU8-V100',entries=[])]
        self.workers = [self.worker('0,1,2,3','q4'),self.worker('4,5,6,7','q4'),self.worker('0,1,2,3,4,5,6,7','q8')]

    def worker(self,cards,queue,**changes):
        value = dict(id='host:gpu'+cards,ip='10.0.0.1',queues=[dict(id=queue)],
                     task={},last_activity_time=self.now.isoformat())
        value.update(changes)
        return value

    def choices(self,reserved=()):
        return dispatch.available(self.workers,self.queues,self.now,reserved)

    def test_idle_v100_eight_cards_prioritized(self):
        self.assertEqual([x['world_size'] for x in self.choices()],[8,4])

    def test_busy_eight_card_worker_blocks_both_four_card_workers(self):
        self.workers[-1]['task']={'id':'other-task'}
        self.assertEqual(self.choices(),[])

    def test_busy_four_card_worker_blocks_overlapping_eight_card_queue(self):
        self.workers[0]['task']={'id':'other-task'}
        self.assertEqual([x['queue_name'] for x in self.choices()],['GPU4-V100'])
        self.assertEqual(self.choices()[0]['bindings'],[['10.0.0.1',[4,5,6,7]]])

    def test_pending_eight_card_queue_blocks_four_card_workers(self):
        self.queues[1]['entries']=[dict(task='queued-task')]
        self.assertEqual(self.choices(),[])

    def test_pending_four_card_queue_blocks_eight_card_worker(self):
        self.queues[0]['entries']=[dict(task='queued-task')]
        self.assertEqual(self.choices(),[])

    def test_reserved_physical_cards_block_overlap(self):
        self.assertEqual(self.choices([('10.0.0.1',set(range(8)))]),[])

    def test_stale_worker_rejected(self):
        for worker in self.workers:
            worker['last_activity_time']=(self.now-datetime.timedelta(seconds=91)).isoformat()
        self.assertEqual(self.choices(),[])

    def test_future_worker_timestamp_rejected(self):
        for worker in self.workers:
            worker['last_activity_time']=(self.now+datetime.timedelta(seconds=1)).isoformat()
        self.assertEqual(self.choices(),[])

    def test_l40_remains_cpu_only(self):
        for queue in self.queues:
            queue['name']=queue['name'].replace('V100','L40S')
        self.assertEqual(self.choices(),[])

    def test_disabled_queue_rejected(self):
        for queue in self.queues:
            queue['tags']=['force_workers:off']
        self.assertEqual(self.choices(),[])

    def test_same_host_different_worker_aliases_still_collide(self):
        self.workers[-1]['id']='alias:gpu0,1,2,3,4,5,6,7'
        self.workers[-1]['task']={'id':'other-task'}
        self.assertEqual(self.choices(),[])

    def test_other_host_does_not_collide(self):
        self.workers[-1]['ip']='10.0.0.2'
        self.workers[-1]['task']={'id':'other-task'}
        self.assertEqual([x['queue_name'] for x in self.choices()],['GPU4-V100'])


if __name__=='__main__':
    unittest.main()
