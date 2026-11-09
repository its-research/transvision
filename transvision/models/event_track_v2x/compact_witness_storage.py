"""Lossless, content-addressed storage for offline teacher snapshots.

Only the representation changes. Snapshot SHA256 always names the original
canonical JSON. Chunks and manifests participate in the caller's transaction;
this module never commits, prunes evidence, or grants scientific acceptance.
"""
import hashlib
import json
import zlib

MAGIC = b'RBFW2\x00'
CHUNK_ROWS = 64
FORMAT = 'canonical_json_shared_list_chunks_v2'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inflate(payload, expected_bytes):
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError('invalid recorded witness length')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, expected_bytes + 1)
    if (len(raw) != expected_bytes or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError('witness compressed length or stream integrity differs')
    return raw


def create_tables(db):
    db.execute('CREATE TABLE IF NOT EXISTS priority_probe_witnesses('
               'sha TEXT PRIMARY KEY, raw_bytes INTEGER NOT NULL, payload BLOB NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS priority_witness_chunks('
               'sha TEXT PRIMARY KEY, raw_bytes INTEGER NOT NULL, payload BLOB NOT NULL) WITHOUT ROWID')


def store(db, value):
    raw = canonical(value)
    sha = digest(raw)
    prior = db.execute('SELECT raw_bytes,payload FROM priority_probe_witnesses WHERE sha=?', (sha,)).fetchone()
    if prior is not None:
        if read(db, sha) != raw:
            raise ValueError('witness hash collision or corrupt existing witness')
        return sha
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError('snapshot must be an object with string keys')
    fields, lists = {}, {}
    for key, item in sorted(value.items()):
        if not isinstance(item, list):
            fields[key] = item
            continue
        refs = []
        for i in range(0, len(item), CHUNK_ROWS):
            part = canonical(item[i:i + CHUNK_ROWS])
            part_sha = digest(part)
            old = db.execute('SELECT raw_bytes,payload FROM priority_witness_chunks WHERE sha=?', (part_sha,)).fetchone()
            if old is None:
                db.execute('INSERT INTO priority_witness_chunks VALUES(?,?,?)',
                           (part_sha, len(part), zlib.compress(part, 6)))
            elif inflate(old[1], old[0]) != part:
                raise ValueError('chunk hash collision or corrupt existing chunk')
            refs.append(part_sha)
        lists[key] = {'rows': len(item), 'chunks': refs}
    manifest = canonical({'format': FORMAT, 'fields': fields, 'lists': lists})
    # Decompression is bounded by each recorded length and checked against its
    # content SHA; no additional per-witness workload limit is introduced.
    payload = MAGIC + len(manifest).to_bytes(4, 'big') + zlib.compress(manifest, 6)
    db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)', (sha, len(raw), payload))
    return sha


def read(db, sha):
    row = db.execute('SELECT raw_bytes,payload FROM priority_probe_witnesses WHERE sha=?', (sha,)).fetchone()
    if row is None:
        raise ValueError('missing witness')
    expected, payload = row
    if type(expected) is not int or expected < 0:
        raise ValueError('invalid recorded witness length')
    if not payload.startswith(MAGIC):
        raw = inflate(payload, expected)
    else:
        n = len(MAGIC)
        if len(payload) < n + 4:
            raise ValueError('truncated manifest header')
        manifest = json.loads(inflate(payload[n + 4:], int.from_bytes(payload[n:n + 4], 'big')))
        if set(manifest) != {'format', 'fields', 'lists'} or manifest['format'] != FORMAT:
            raise ValueError('unknown witness storage format')
        value = dict(manifest['fields'])
        total = 0
        # Combining k nonempty JSON arrays removes k-1 bytes of brackets/
        # separators. Thus all chunk bytes fit within snapshot bytes plus the
        # total number of chunks, independent of the absolute snapshot size.
        chunks_count = sum(len(spec['chunks']) for spec in manifest['lists'].values())
        for key, spec in manifest['lists'].items():
            if key in value or set(spec) != {'rows', 'chunks'} or type(spec['rows']) is not int or spec['rows'] < 0:
                raise ValueError('invalid chunk manifest')
            rows = []
            for chunk_sha in spec['chunks']:
                chunk = db.execute('SELECT raw_bytes,payload FROM priority_witness_chunks WHERE sha=?', (chunk_sha,)).fetchone()
                if chunk is None:
                    raise ValueError('missing witness chunk')
                total += chunk[0]
                if total > expected + chunks_count:
                    raise ValueError('chunk lengths exceed recorded snapshot size')
                part = inflate(chunk[1], chunk[0])
                if digest(part) != chunk_sha:
                    raise ValueError('witness chunk SHA mismatch')
                decoded = json.loads(part)
                if not isinstance(decoded, list) or not 1 <= len(decoded) <= CHUNK_ROWS:
                    raise ValueError('invalid chunk rows')
                rows.extend(decoded)
                if len(rows) > spec['rows']:
                    raise ValueError('excess chunk rows')
            if len(rows) != spec['rows']:
                raise ValueError('missing chunk rows')
            value[key] = rows
        raw = canonical(value)
    if len(raw) != expected or digest(raw) != sha:
        raise ValueError('reconstructed witness size/SHA mismatch')
    return raw
