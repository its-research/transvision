"""Isolated lookup overhead optimization; preserve shared LRU and SQL semantics."""
from collections import OrderedDict
from functools import lru_cache

def install(namespace_class, tracker_class, cache_view_class):
    old_sql=namespace_class._sql
    # Bound translated statements globally; parameters are never cached.
    @lru_cache(maxsize=1024)
    def translated(prefix,sql):
        obj=object.__new__(namespace_class)
        obj.prefix=prefix
        return old_sql(obj,sql)
    def sql(self,statement):
        return translated(self.prefix,statement)
    namespace_class._sql=sql
    old_prefix=tracker_class._prefix
    def prefix(self,handle):
        cache=self.cache
        if type(cache) is cache_view_class:
            shared=cache.shared
            pair=(cache.namespace,handle)
            try:value=shared[pair]
            except KeyError:return old_prefix(self,handle)
            # Same two recency indexes as _GlobalLRU.move_to_end, same order.
            OrderedDict.move_to_end(shared,pair)
            shared.namespace_order[cache.namespace].move_to_end(pair)
            return value
        return old_prefix(self,handle)
    tracker_class._prefix=prefix
