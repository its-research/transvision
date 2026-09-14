from collections import Counter, OrderedDict
import random

import pytest

from transvision.models.event_track_v2x.persistent_component_tracking import _CacheView, _GlobalLRU


def check_index(cache, reference):
    assert tuple(cache.items()) == tuple(reference.items())
    assert cache.counts == dict(Counter(key[0] for key in reference))
    assert set(cache.namespace_order) == set(cache.counts)
    for namespace, order in cache.namespace_order.items():
        assert tuple(order) == tuple(key for key in reference if key[0] == namespace)
    assert sum(map(len, cache.namespace_order.values())) == len(cache)


@pytest.mark.parametrize('seed', range(3))
def test_namespace_index_matches_global_order_under_eviction_restore_and_recency(seed):
    rng = random.Random(seed)
    cache, reference = _GlobalLRU(), OrderedDict()
    for _ in range(1000):
        operation = rng.choice(('set', 'move', 'delete', 'global_pop', 'namespace_pop', 'restore'))
        key = (rng.randrange(5), rng.randrange(12))
        last = bool(rng.randrange(2))
        if operation == 'set':
            value = rng.random()
            cache[key] = reference[key] = value
        elif operation == 'move' and reference:
            key = rng.choice(tuple(reference))
            cache.move_to_end(key, last=last); reference.move_to_end(key, last=last)
        elif operation == 'delete' and reference:
            key = rng.choice(tuple(reference))
            del cache[key]; del reference[key]
        elif operation == 'global_pop' and reference:
            assert cache.popitem(last=last) == reference.popitem(last=last)
        elif operation == 'namespace_pop':
            available = [k for k in reference if k[0] == key[0]]
            if available:
                selected = available[-1 if last else 0]
                assert cache.pop_namespace(key[0], last=last) == (selected, reference.pop(selected))
            else:
                with pytest.raises(KeyError):
                    cache.pop_namespace(key[0], last=last)
        elif operation == 'restore':
            # Offline teacher currently snapshots and restores in global order.
            snapshot = tuple(cache.items())
            cache.clear()
            for k,v in snapshot:
                cache[k] = v
        check_index(cache, reference)


def test_namespace_eviction_and_clear_never_iterate_unrelated_global_entries(monkeypatch):
    cache = _GlobalLRU()
    other, target = _CacheView(cache, 'other', 8192), _CacheView(cache, 'target', 8192)
    for i in range(6000):
        other[i] = i
    for i in range(4):
        target[i] = i
    def forbidden(*args):
        raise AssertionError('namespace operation scanned the global cache')
    with monkeypatch.context() as patch:
        patch.setattr(_GlobalLRU, '__iter__', forbidden)
        patch.setattr(_GlobalLRU, '__reversed__', forbidden)
        assert target.popitem(last=False) == (0,0)
        target.move_to_end(1)
        assert target.popitem(last=True) == (1,1)
        target.clear()
    assert len(target) == 0 and len(other) == 6000
    assert 'target' not in cache.namespace_order
    check_index(cache, OrderedDict((('other',i),i) for i in range(6000)))


def test_global_cap_evicts_from_both_value_store_and_namespace_index():
    cache = _GlobalLRU()
    a, b = _CacheView(cache,'a',3), _CacheView(cache,'b',3)
    a[1]=1; b[1]=2; a[2]=3; a.move_to_end(1); b[2]=4
    assert tuple(cache) == (('a',2),('a',1),('b',2))
    check_index(cache,OrderedDict(cache.items()))
    b.clear()
    check_index(cache,OrderedDict(((('a',2),3),(('a',1),1))))
