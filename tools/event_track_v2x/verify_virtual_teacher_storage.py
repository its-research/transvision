"""Independent stored-member readback, separate from virtual compaction.

Imports no producer codec or compaction module. Checks physical compressed and
decoded bytes, every typed table and reconstructed witness against the bound
storage result. This never establishes teacher target or cohort acceptance.
"""
import hashlib
import json
from pathlib import Path

from compact_teacher_reader import decode,open_compressed
from verify_compact_teacher_collection import table_digest,read_pinned


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def pinned(path,sha):
    path=Path(path).absolute()
    if (not path.is_file() or path.stat().st_size>1024**2 or
            any(p.is_symlink() for p in (path,*path.parents))):
        raise ValueError('ordinary bounded metadata required')
    return read_pinned(path,sha)


def verify(root,*,qualification_sha256,binding_path,binding_file_sha256,
           scratch_parent,max_scratch_bytes,min_free_bytes=8*1024**3,
           progress=lambda r:None):
    root=Path(root).absolute()
    if (type(max_scratch_bytes) is not int or not 0<max_scratch_bytes<=8*1024**3 or
            type(min_free_bytes) is not int or min_free_bytes<8*1024**3):
        raise ValueError('original bounded scratch and reserve required')
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('ordinary qualification directory required')
    proof=pinned(root/'storage-qualification.json',qualification_sha256)
    binding=pinned(binding_path,binding_file_sha256)
    if binding.get('status')!='completed':raise ValueError('complete original binding required')
    if (proof['kind']!='single_original_teacher_virtual_storage_qualification_candidate_v1' or
            proof['full_teacher_target_acceptance'] is not False or
            proof['full_cohort_qualified'] is not False or proof['scientific_acceptance'] is not False):
        raise ValueError('single-member storage scope required')
    if hashlib.sha256(canonical(binding)).hexdigest()!=proof['source_binding_canonical_sha256']:
        raise ValueError('original binding differs')
    members=[m for m in binding['files'] if m['name']==proof['member']['name']]
    if len(members)!=1 or members[0]!=proof['member']:
        raise ValueError('exact original member extent/identity required')
    if (binding['task_id']!=proof['task_id'] or
            binding['block_manifest']['sha256']!=proof['archive_sha256']):
        raise ValueError('original producer/archive differs')
    compact=proof['compaction'];encoding=proof['encoding'];member=proof['member']
    if ((compact['source_bytes'],compact['source_sha256'])!=(member['bytes'],member['sha256']) or
            compact['all_witness_raw_bytes_equal'] is not True or
            compact['source_binding_canonical_sha256']!=proof['source_binding_canonical_sha256'] or
            compact['source_member_name']!=member['name'] or
            encoding['encoding']!='gzip' or
            compact['output_sha256']!=encoding['decoded_sha256'] or
            compact['output_bytes']!=encoding['decoded_bytes']):
        raise ValueError('original and compact storage link differs')
    with open_compressed(root/'compact.sqlite.gz',stored_sha256=encoding['stored_sha256'],
            decoded_sha256=encoding['decoded_sha256'],decoded_bytes=encoding['decoded_bytes'],
            scratch_parent=scratch_parent,max_scratch_bytes=max_scratch_bytes,
            min_free_bytes=min_free_bytes) as db:
        if db.execute('PRAGMA main.integrity_check').fetchone()[0]!='ok':
            raise ValueError('independent SQLite integrity failure')
        names={r[0] for r in db.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if names-{'priority_probe_witnesses','priority_witness_chunks'}!=set(compact['tables']):
            raise ValueError('independent table coverage differs')
        for name,expected in compact['tables'].items():
            if table_digest(db,name)!=expected:raise ValueError('independent typed table differs: '+name)
        count=0
        for key,size in db.execute('SELECT sha,raw_bytes FROM main.priority_probe_witnesses'):
            raw=decode(db,key)
            if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=key:
                raise ValueError('independent witness bytes differ')
            count+=1
            if count%500==0:progress({'stage':'independent_virtual_member_readback','witnesses':count})
        if count!=compact['witnesses'] or count!=proof['independent_witnesses']:
            raise ValueError('independent witness coverage differs')
    # Re-pin metadata after full byte/row traversal before granting this scope.
    if pinned(root/'storage-qualification.json',qualification_sha256)!=proof:
        raise ValueError('qualification changed during readback')
    if pinned(binding_path,binding_file_sha256)!=binding:
        raise ValueError('binding changed during readback')
    return {'kind':'independent_single_virtual_teacher_storage_readback_v1',
        'qualification_sha256':qualification_sha256,'binding_file_sha256':binding_file_sha256,
        'task_id':proof['task_id'],'original_member':member,'witnesses':count,
        'tables':len(compact['tables']),'stored_sha256':encoding['stored_sha256'],
        'decoded_sha256':encoding['decoded_sha256'],'own_successful_scratch_released':True,
        'full_cohort_qualified':False,'full_teacher_target_acceptance':False,
        'scientific_acceptance':False}
