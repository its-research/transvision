#!/usr/bin/env python3
"""Read-only, hash-pinned prefix-bound comparison; NOT an online replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.ranked_class_bound import SingleClassBoundModel


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): result.update(block)
    return result.hexdigest()


def closed_identity(path,expected):
    if not path.is_file() or path.is_symlink(): raise ValueError('ordinary closed database required')
    if any(Path(str(path)+suffix).exists() for suffix in ('-wal','-journal')):
        raise ValueError('database has an active or unresolved transaction sidecar')
    if sha(path)!=expected: raise ValueError('closed database identity differs')


def audit(database,database_sha256,*,largest_components=3):
    path=Path(database).absolute()
    if type(largest_components) is not int or not 1<=largest_components<=32:
        raise ValueError('bounded component selection required')
    closed_identity(path,database_sha256)
    component_rows=[]
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        selected=db.execute('SELECT m.component,count(*) n FROM component_members m '
            'JOIN component_catalog c ON c.component=m.component WHERE c.live=1 '
            'GROUP BY m.component ORDER BY n DESC,m.component LIMIT ?',(largest_components,)).fetchall()
        if not selected: raise ValueError('no live committed components')
        for component,n in selected:
            if type(component) is not int or component<0 or not 1<=n<=10_000:
                raise ValueError('bounded component catalog required')
            prefix=f'pc{component}_'
            state=json.loads(db.execute(f"SELECT v FROM {prefix}meta WHERE k='state'").fetchone()[0])
            if state['n']!=n or not state['active'] or len(state['active'])>1024:
                raise ValueError('complete bounded retained classes required')
            rows=[[] for _ in range(n)]
            count=0
            for i,p,w in db.execute(f'SELECT i,p,w FROM {prefix}potentials ORDER BY i,p'):
                count+=1
                if type(i) is not int or not 0<=i<n or count>200_000:
                    raise ValueError('bounded causal local row catalog required')
                rows[i].append((p,w))
            observations=db.execute(f'SELECT i,source,frame FROM {prefix}observations ORDER BY i').fetchall()
            if [r[0] for r in observations]!=list(range(n)): raise ValueError('observation coverage differs')
            model=SingleClassBoundModel(rows,slots=[(s,f) for _,s,f in observations])
            samples=[]
            depths=sorted({0,n//2,max(0,n-32),max(0,n-16),n})
            for active in state['active']:
                handle=active;roots=[None]*n;expected_depth=n
                while expected_depth:
                    row=db.execute(f'SELECT parent,depth,root FROM {prefix}prefixes WHERE h=?',(handle,)).fetchone()
                    if row is None or row[1]!=expected_depth:
                        raise ValueError('retained prefix ancestry differs')
                    handle,depth,root=row;roots[depth-1]=root;expected_depth-=1
                if handle!=0: raise ValueError('retained prefix does not reach the root')
                for depth in depths:
                    bound=model.for_prefix(roots[:depth])
                    samples.append(dict(active_handle=active,**asdict(bound),
                        log_bound_reduction=bound.log_row_partition_upper-bound.log_single_class_upper))
            component_rows.append(dict(component=component,node_count=n,factor_terms=count,
                retained_classes=len(state['active']),decision_us=state['decision_us'],
                selected_prefix_depths=depths,samples=samples))
    closed_identity(path,database_sha256)
    return dict(kind='closed_state_single_class_bound_diagnostic_v1',database=str(path),
        database_sha256=database_sha256,selection='largest live components; node count descending, id ascending',
        largest_components=largest_components,components=component_rows,
        diagnostic_source_sha256=sha(__file__),bound_source_sha256=sha(ROOT/'tools/event_track_v2x/ranked_class_bound.py'),
        online_replay=False,failed_event_reconstructed=False,top_k_certified=False,
        partition_mass_bound=False,formal_numeric_certificate=False,
        ground_truth_used=False,parameter_training=False,paper_eligible=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True,type=Path)
    parser.add_argument('--database-sha256',required=True)
    parser.add_argument('--largest-components',type=int,default=3)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists() or any(p.is_symlink() for p in (args.output,*args.output.parents)):
        raise ValueError('new nonsymlink output required')
    result=audit(args.database,args.database_sha256,largest_components=args.largest_components)
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),components=len(result['components']))))


if __name__=='__main__': main()
