"""Managed-worker reporting gaps must not permit physical card collisions."""
import copy
import datetime
import unittest

from tools.event_track_v2x.gpu_worker_task_admission import choices_with_active_tasks, query_fleet


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.datetime.now(datetime.timezone.utc)
        self.queues = [dict(id='q4', name='GPU4-A100', entries=[]),
                       dict(id='q8', name='GPU8-A100', entries=[])]
        self.workers = [dict(id='host-A100:gpu'+cards, ip='host',
                             queues=[dict(id=queue)], last_activity_time=self.now.isoformat())
                        for cards, queue in [('0,1,2,3', 'q4'), ('4,5,6,7', 'q4'),
                                             ('0,1,2,3,4,5,6,7', 'q8')]]

    def running(self, index=0):
        return dict(id='running-'+str(index), status='in_progress',
                    last_worker=self.workers[index]['id'], execution=dict(queue='q4'))

    def choose(self, tasks=(), **kwargs):
        return choices_with_active_tasks(self.workers, self.queues, self.now, tasks,
                                         complete=kwargs.get('complete', True),
                                         reserved=kwargs.get('reserved', ()))

    def test_backend_running_binding_blocks_missing_worker_task_and_overlapping_eight(self):
        before = copy.deepcopy(self.workers)
        choices, report = self.choose([self.running()])
        self.assertEqual([c['world_size'] for c in choices], [4])
        self.assertEqual(choices[0]['bindings'], [['host', [4, 5, 6, 7]]])
        self.assertEqual(report['task_binding_reservations'][0]['cards'], [0, 1, 2, 3])
        self.assertEqual(self.workers, before)  # Derived occupancy never alters raw evidence.

    def test_null_worker_task_is_also_guarded(self):
        self.workers[0]['task'] = None
        self.assertEqual(self.choose([self.running()])[0][0]['bindings'], [['host', [4, 5, 6, 7]]])

    def test_both_four_card_running_bindings_block_entire_host(self):
        self.assertEqual(self.choose([self.running(0), self.running(1)])[0], [])

    def test_running_eight_blocks_both_four_card_workers(self):
        self.assertEqual(self.choose([self.running(2)])[0], [])

    def test_no_running_bindings_preserves_eight_card_recipe(self):
        self.assertEqual(self.choose()[0][0]['world_size'], 8)

    def test_queued_task_blocks_pool_before_queue_entry_is_reported(self):
        pending = dict(id='queued', status='queued', execution=dict(queue='q4'))
        self.assertEqual(self.choose([pending])[0], [])

    def test_queued_object_queue_identity_is_supported(self):
        pending = dict(id='queued', status='queued', execution=dict(queue=dict(id='q8')))
        self.assertEqual(self.choose([pending])[0], [])

    def test_unresolved_running_binding_is_not_idle(self):
        task = self.running()
        task['last_worker'] = 'unreported-A100:gpu0,1,2,3'
        choices, report = self.choose([task])
        self.assertEqual(choices, [])
        self.assertEqual(report['unresolved_active_task_ids'], [task['id']])

    def test_incomplete_task_coverage_is_not_admission(self):
        self.assertEqual(self.choose(complete=False)[0], [])

    def test_missing_active_status_is_not_silently_dropped(self):
        task = self.running()
        del task['status']
        self.assertEqual(self.choose([task])[0], [])

    def test_terminal_task_does_not_permanently_reserve_cards(self):
        task = self.running()
        task['status'] = 'completed'
        self.assertEqual(self.choose([task])[0][0]['world_size'], 8)

    def test_original_reservations_are_retained(self):
        self.assertEqual(self.choose(reserved=[('host', set(range(8)))])[0], [])

    def test_bounded_query_requires_final_partial_page(self):
        class Row:
            def __init__(self, value): self.value = value
            def to_dict(self): return self.value
        class Listing:
            def __init__(self, rows): self.rows = rows
            def get_all(self, **kwargs): return [Row(v) for v in self.rows]
        class API: pass
        api = API()
        api.workers = Listing(self.workers)
        api.queues = Listing(self.queues)
        api.tasks = Listing([self.running()])
        choices, snapshot = query_fleet(api, page_size=1, max_pages=1)
        self.assertEqual(choices, [])
        self.assertFalse(snapshot['active_task_metadata_complete'])
        choices, snapshot = query_fleet(api, page_size=100)
        self.assertEqual(choices[0]['bindings'], [['host', [4, 5, 6, 7]]])
        self.assertTrue(snapshot['active_task_metadata_complete'])


if __name__ == '__main__':
    unittest.main()
