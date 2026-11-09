"""Independent full storage readback for a collected registered teacher archive.

No producer codec is imported. Original archive identity, native receipt hashes,
all table digests and every reconstructed witness are checked. This establishes
storage fidelity only; it never grants numerical teacher-target acceptance.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct

from compact_teacher_reader import decode, open_compressed


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(root, name):
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name, 'unsafe stored path')
    path = Path(root).absolute().joinpath(*rel.parts)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'ordinary stored file required')
    return path


def hashed(stream):
    digest = hashlib.sha256(); size = 0
    for block in iter(lambda: stream.read(1024**2), b''):
        size += len(block); digest.update(block)
    return size, digest.hexdigest()


def read_pinned(path, expected_sha):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha, 'metadata SHA differs')
    return json.loads(data)


def table_digest(db, table):
    quote = lambda s: '"' + s.replace('"', '""') + '"'
    schema = db.execute('SELECT sql FROM main.sqlite_master WHERE type=? AND name=?', ('table', table)).fetchone()
    require(schema is not None, 'missing table')
    if 'WITHOUT ROWID' in schema[0].upper():
        columns = db.execute('PRAGMA main.table_info(' + quote(table) + ')').fetchall()
        order = ','.join(quote(r[1]) for r in sorted(columns, key=lambda r:r[5]) if r[5])
    else:
        order = 'rowid'
    digest = hashlib.sha256(); count = 0
    for row in db.execute('SELECT * FROM main.' + quote(table) + ' ORDER BY ' + order):
        count += 1; digest.update(len(row).to_bytes(4, 'big'))
        for value in row:
            if value is None: tag, data = b'n', b''
            elif isinstance(value, bytes): tag, data = b'b', value
            elif isinstance(value, str): tag, data = b's', value.encode()
            elif isinstance(value, int): tag, data = b'i', str(value).encode()
            elif isinstance(value, float): tag, data = b'f', struct.pack('>d', value)
            else: raise TypeError(type(value))
            digest.update(tag + len(data).to_bytes(8, 'big') + data)
    return {'rows':count, 'sha256':digest.hexdigest()}


@contextmanager
def decoded_stream(root, entry):
    path = safe(root, entry['stored_name'])
    require(entry['encoding'] in ('identity', 'gzip'), 'not an ordinary member')
    with (gzip.open(path, 'rb') if entry['encoding'] == 'gzip' else path.open('rb')) as stream:
        yield stream


def verify(root, *, manifest_sha256, inventory, scratch_parent, max_scratch_bytes,
           min_free_bytes=8*1024**3, progress=lambda value:None):
    root = Path(root).absolute()
    proof = read_pinned(safe(root, 'compact-storage.json'), manifest_sha256)
    require(proof['kind'] == 'registered_teacher_compact_storage_v2', 'collection kind differs')
    require(proof['archive_sha256'] == inventory['archive_sha256'] and
            proof['archive_bytes'] == inventory['archive_bytes'], 'registered archive binding differs')
    expected = {x['name']:x['bytes'] for x in inventory['files']}
    entries = {x['original_name']:x for x in proof['files']}
    require(len(expected) == len(inventory['files']) and len(entries) == len(proof['files']), 'duplicate archive member')
    require(set(expected) == set(entries), 'archive member coverage differs')
    stored_names = [x['stored_name'] for x in entries.values()]
    require(len(set(stored_names)) == len(stored_names), 'stored member alias')
    results = []; witness_total = 0
    for name, entry in entries.items():
        require(entry['original_bytes'] == expected[name], 'original member size differs')
        path = safe(root, entry['stored_name'])
        require(path.stat().st_size == entry['stored_bytes'], 'stored member size differs')
        if entry['encoding'] != 'shared_witness_sqlite_v2+gzip':
            with decoded_stream(root, entry) as stream:
                size, digest = hashed(stream)
            require((size, digest) == (entry['original_bytes'], entry['original_sha256']), 'original member readback differs')
            continue
        report_path = safe(root, entry['storage_receipt'])
        report_bytes = report_path.read_bytes(); report = json.loads(report_bytes)
        require(report['source_sha256'] == entry['original_sha256'] and
                report['source_bytes'] == entry['original_bytes'], 'source database binding differs')
        require(report['output_sha256'] == entry['decoded_sha256'] and
                report['output_bytes'] == entry['decoded_bytes'], 'compact database binding differs')
        require(report.get('all_witness_raw_bytes_equal') is True, 'original witness comparison missing')
        with open_compressed(path, stored_sha256=entry['stored_sha256'],
                decoded_sha256=entry['decoded_sha256'], decoded_bytes=entry['decoded_bytes'],
                scratch_parent=scratch_parent, max_scratch_bytes=max_scratch_bytes,
                min_free_bytes=min_free_bytes) as db:
            require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'SQLite integrity failure')
            tables = {r[0] for r in db.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            require(tables - {'priority_probe_witnesses','priority_witness_chunks'} == set(report['tables']), 'table coverage differs')
            for table, bound in report['tables'].items():
                require(table_digest(db, table) == bound, 'table digest differs: ' + table)
            count = 0
            for key, size in db.execute('SELECT sha,raw_bytes FROM main.priority_probe_witnesses'):
                raw = decode(db, key)
                require(len(raw) == size, 'witness length differs')
                count += 1
                if count % 500 == 0: progress({'stage':'independent_witness_readback','member':name,'rows':count})
            require(count == report['witnesses'], 'witness coverage differs')
            witness_total += count
        results.append({'member':name,'witnesses':count,'tables':len(report['tables']),
                        'storage_receipt_sha256':hashlib.sha256(report_bytes).hexdigest()})
        progress({'stage':'independent_database_readback','member':name,'witnesses':count})
    # Native receipt links must point at the ORIGINAL member identities, not the
    # new compressed representation. Require every native file and DB binding.
    receipts = [name for name in entries if PurePosixPath(name).name == 'receipt.json']
    require(receipts, 'native receipts missing')
    covered = set()
    for name in receipts:
        parent = PurePosixPath(name).parent
        with decoded_stream(root, entries[name]) as stream: receipt = json.load(stream)
        require(receipt['kind'] == 'rbf_paper_replay_receipt_v1' and
                receipt['status'] == 'software_replay_completed', 'native completion missing')
        covered.add(name)
        for leaf, digest in receipt['files'].items():
            require(not PurePosixPath(leaf).is_absolute() and '..' not in PurePosixPath(leaf).parts, 'unsafe receipt member')
            original = (parent/leaf).as_posix()
            require(original in entries and entries[original]['original_sha256'] == digest, 'native file receipt binding differs')
            covered.add(original)
        for info in receipt['databases'].values():
            leaf = PurePosixPath(info['path'])
            require(not leaf.is_absolute() and '..' not in leaf.parts, 'unsafe database receipt path')
            original = (parent/leaf).as_posix()
            require(original in entries and entries[original]['original_sha256'] == info['sha256'], 'native database receipt binding differs')
            covered.add(original)
    require(covered == set(entries), 'unbound archive members')
    require(read_pinned(root/'compact-storage.json', manifest_sha256) == proof, 'manifest changed')
    return {'kind':'teacher_compact_collection_independent_storage_readback_v1',
            'archive_sha256':proof['archive_sha256'],'manifest_sha256':manifest_sha256,
            'members':len(entries),'databases':results,'witnesses':witness_total,
            'all_original_member_bindings_verified':True,'all_witnesses_independently_decoded':True,
            'scientific_acceptance':False,'full_teacher_target_acceptance':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','inventory','scratch-parent','output'): parser.add_argument('--'+key,type=Path,required=True)
    for key in ('manifest-sha256','inventory-sha256'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--max-scratch-bytes',type=int,required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'prior result preserved')
    inventory = read_pinned(args.inventory, args.inventory_sha256)
    result = verify(args.root, manifest_sha256=args.manifest_sha256, inventory=inventory,
                    scratch_parent=args.scratch_parent,max_scratch_bytes=args.max_scratch_bytes,
                    progress=lambda value:print(json.dumps(value),flush=True))
    with args.output.open('x') as out: json.dump(result,out,indent=2)


if __name__ == '__main__': main()
