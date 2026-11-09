"""Cross-check managed worker occupancy against independently reported tasks."""
import datetime
import time

from tools.event_track_v2x.submit_rbf_final_identity import available, binding


def choices_with_active_tasks(workers, queues, now, tasks, *, complete, reserved=()):
    """Keep running-task cards reserved even during a worker-reporting gap."""
    task_reservations = []
    unresolved = []
    task_pending = {}
    running_by_worker = {}
    by_worker = {w['id']: w for w in workers}
    known_queues = {q['id'] for q in queues}
    for task in tasks:
        if task.get('status') in ('completed', 'failed', 'stopped', 'closed', 'published'):
            continue
        if not task.get('id') or task.get('status') not in ('in_progress', 'queued'):
            unresolved.append(task.get('id') or '<missing-task-id>')
            continue
        queue = (task.get('execution') or {}).get('queue')
        if isinstance(queue, dict):
            queue = queue.get('id')
        if task['status'] == 'queued' and queue in known_queues:
            task_pending.setdefault(queue, []).append({'task': task['id']})
        worker = by_worker.get(task.get('last_worker'))
        cards = binding(worker) if worker is not None else None
        if task['status'] == 'in_progress':
            if cards is None:
                unresolved.append(task['id'])
            else:
                task_reservations.append((task['id'], cards))
                running_by_worker.setdefault(worker['id'], []).append(task['id'])
        elif queue not in known_queues:
            unresolved.append(task['id'])
    report = dict(active_task_metadata_complete=complete,
                  unresolved_active_task_ids=unresolved,
                  task_binding_reservations=[dict(task_id=t, host=b[0], cards=sorted(b[1]))
                                             for t, b in task_reservations],
                  active_task_queue_reservations=task_pending,
                  task_derived_worker_occupancy=running_by_worker,
                  worker_task_field_absence_not_idle_proof=True)
    if not complete or unresolved:
        return [], report
    guarded_queues = [dict(q, entries=list(q.get('entries') or []) +
                          task_pending.get(q['id'], [])) for q in queues]
    guarded_workers = [dict(w, task=w.get('task') or
                            ({'id': running_by_worker[w['id']][0]}
                             if w['id'] in running_by_worker else None)) for w in workers]
    choices = available(guarded_workers, guarded_queues, now,
                        tuple(reserved) + tuple(b for _, b in task_reservations))
    return choices, report


def query_fleet(api, *, reserved=(), page_size=100, max_pages=10, seconds_cap=30):
    started = time.monotonic()
    workers = [w.to_dict() for w in api.workers.get_all()]
    queues = [q.to_dict() for q in api.queues.get_all()]
    tasks = []
    counts = []
    complete = False
    for page in range(max_pages):
        if time.monotonic() - started >= seconds_cap:
            break
        rows = api.tasks.get_all(status=['in_progress', 'queued'],
                                 only_fields=['id', 'status', 'last_worker', 'execution.queue'],
                                 page_size=page_size, page=page)
        counts.append(len(rows))
        tasks.extend(t.to_dict() for t in rows)
        if len(rows) < page_size:
            complete = True
            break
    complete = complete and time.monotonic() - started < seconds_cap
    now = datetime.datetime.now(datetime.timezone.utc)
    choices, evidence = choices_with_active_tasks(workers, queues, now, tasks,
                                                  complete=complete, reserved=reserved)
    snapshot = dict(checked_at_utc=now.isoformat(),
                    workers=[dict(id=w['id'], ip=w.get('ip'),
                                  task_id=(w.get('task') or {}).get('id'),
                                  task_field_present='task' in w,
                                  last_activity_time=str(w['last_activity_time'])) for w in workers],
                    queues=[dict(id=q['id'], name=q['name'],
                                 pending=[e['task'] for e in q.get('entries') or []]) for q in queues],
                    active_tasks=tasks, active_task_page_counts=counts,
                    observation_caps=dict(page_size=page_size, max_pages=max_pages,
                                          seconds=seconds_cap), **evidence)
    return choices, snapshot
