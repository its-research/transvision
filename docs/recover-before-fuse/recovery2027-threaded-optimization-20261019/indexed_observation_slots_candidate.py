"""Equivalent lookup through existing indexes; isolated namespace SQL adapter."""
ORIGINAL_SQL='SELECT i FROM observations WHERE source=? AND frame=? AND i<?'
INDEXED_SQL='''SELECT m.local_i FROM observations AS o INDEXED BY observation_slots
CROSS JOIN component_members AS m
WHERE o.source=? AND o.frame=? AND m.component=? AND m.global_i=o.i
AND m.local_i<? ORDER BY m.local_i'''

def install(namespace_class):
    original=namespace_class.execute
    def execute(self,sql,parameters=()):
        if sql==ORIGINAL_SQL:
            source,frame,depth=parameters
            component=int(self.prefix[2:-1])
            return self.connection.execute(INDEXED_SQL,(source,frame,component,depth))
        return original(self,sql,parameters)
    namespace_class.execute=execute
