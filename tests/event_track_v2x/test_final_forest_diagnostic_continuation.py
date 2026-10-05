"""One-shot recovery controls; all mutations and child stubs are temporary."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as N

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
import continue_final_forest_seed2027_after_diagnostic as d


def fixture():
    start=json.loads((d.DIAG/'started.json').read_bytes())
    failed=json.loads(Path(start['failure_receipt']['path']).read_bytes())
    finished=dict(copy.deepcopy(start),failures={},all_selected_bytes_match=True,original_partial_preserved=True,
        canonical_cache_promoted=False,full_byte_event_factor_admission=False,full_forest_CPU_admission=False,paper_performance_complete=False,
        independently_read_artifacts=copy.deepcopy(start['selected']))
    return start,finished,failed


def test_complete_diagnostic_shape_preserves_both_old_failures():
    d.validate_finished(*fixture())


@pytest.mark.parametrize('change',['identity','failure','complete','partial','promoted','byte-pass','CPU-pass','paper',
    'rank-missing','rank-hash','rank7-extra','inventory','inventory-selected','prior-task','prior-seed','prior-recipe',
    'prior-failure','prior-success','prior-changed','prior-six','prior-snapshot','bad-equals-good'])
def test_incomplete_mixed_or_inflated_diagnostic_cannot_promote(change):
    start,end,failed=fixture()
    edits={'identity':('recipe_sha256','other'),'failure':('failures',{'replay-rank5':'AssertionError'}),
        'complete':('all_selected_bytes_match',False),'partial':('original_partial_preserved',False),'promoted':('canonical_cache_promoted',True),
        'byte-pass':('full_byte_event_factor_admission',True),'CPU-pass':('full_forest_CPU_admission',True),'paper':('paper_performance_complete',True)}
    if change in edits:
        key,value=edits[change];end[key]=value
    elif change=='rank-missing':end['independently_read_artifacts'].pop('replay-rank7')
    elif change=='rank-hash':end['independently_read_artifacts']['replay-rank5']['sha256']='f'*64
    elif change=='rank7-extra':end['independently_read_artifacts']['replay-rank0']={}
    elif change=='inventory':start['all_terminal_registered_artifacts'].pop('receipt');end['all_terminal_registered_artifacts']=copy.deepcopy(start['all_terminal_registered_artifacts'])
    elif change=='inventory-selected':start['all_terminal_registered_artifacts']['replay-rank7']['sha256']='f'*64;end['all_terminal_registered_artifacts']=copy.deepcopy(start['all_terminal_registered_artifacts'])
    elif change=='prior-task':failed['task_id']='other'
    elif change=='prior-seed':failed['seed']=1337
    elif change=='prior-recipe':failed['recipe_sha256']='other'
    elif change=='prior-failure':failed['failures']={}
    elif change=='prior-success':failed['all_snapshot_bytes_match']=True
    elif change=='prior-changed':failed['registered_snapshot_unchanged_at_end']=False
    elif change=='prior-six':failed['independently_read_artifacts'].pop('replay-rank6')
    elif change=='prior-snapshot':failed['registered_snapshot']['replay-rank5']['sha256']='f'*64
    else:
        start['original_partial']['sha256']=start['selected']['replay-rank5']['sha256'];end['original_partial']=copy.deepcopy(start['original_partial'])
    with pytest.raises(AssertionError):d.validate_finished(start,end,failed)


def test_unknown_observation_never_triggers_exit_or_restart():
    values=iter([('unknown',None),('live','same-process'),('unknown',None),('absent',None)])
    slept=[];observed=[]
    def observe(pid):observed.append(pid);return next(values)
    d.wait_diagnostic(42,'same-process',observe=observe,sleep=slept.append,clock=lambda:0.)
    assert observed==[42]*4 and slept==[30]*3


def test_pid_reuse_refuses_downstream_work():
    with pytest.raises(AssertionError):d.wait_diagnostic(42,'old',observe=lambda _:('live','new'),sleep=lambda _:pytest.fail('unexpected sleep'))


def test_timeout_preserves_reader_without_treating_it_as_terminal():
    clock=iter([0.,172801.])
    with pytest.raises(TimeoutError):d.wait_diagnostic(42,'same',observe=lambda _:('unknown',None),sleep=lambda _:pytest.fail('unexpected sleep'),clock=lambda:next(clock))


def test_registered_live_diagnostic_identity_uses_started_registry_kind(monkeypatch):
    start,_,_=fixture();base=d.base_module()
    identity='Sun Oct  4 21:58:38 2026 '+base.READER_PYTHON+' '+str(d.DIAG_SOURCE/'read_final_forest_rank5_and_rank7_diagnostic.py')+' --execute'
    monkeypatch.setattr(base,'observe',lambda pid:('live',identity))
    got,status,actual=d.bind_diagnostic(start['pid'])
    assert got==start and status=='live' and actual==identity


@pytest.mark.parametrize('mutation',['wrong-command','unknown','absent-without-receipt'])
def test_initial_diagnostic_observation_is_required(monkeypatch,mutation):
    start,_,_=fixture();base=d.base_module()
    status,value={'wrong-command':('live','Sun Oct  4 21:58:38 2026 python unrelated.py'),
                  'unknown':('unknown',None),'absent-without-receipt':('absent',None)}[mutation]
    monkeypatch.setattr(base,'observe',lambda pid:(status,value))
    if mutation=='absent-without-receipt':
        original=Path.is_file
        monkeypatch.setattr(Path,'is_file',lambda p:False if p==d.DIAG/'finished.json' else original(p))
    with pytest.raises(AssertionError):d.bind_diagnostic(start['pid'])


def test_exclusive_copy_keeps_bad_partial_and_source(tmp_path):
    src=tmp_path/'diagnostic.tar.gz';src.write_bytes(b'correct bytes')
    dst=tmp_path/'canonical.tar.gz';bad=tmp_path/'canonical.tar.gz.partial';bad.write_bytes(b'corrupt full partial')
    item=dict(bytes=src.stat().st_size,sha256=d.sha(src));bad_sha=d.sha(bad)
    result=d.copy_verified(src,dst,item)
    assert d.sha(dst)==d.sha(src)==item['sha256'] and d.sha(bad)==bad_sha
    assert result['original_diagnostic_bytes_preserved'] is True
    with pytest.raises(AssertionError):d.copy_verified(src,dst,item)


@pytest.mark.parametrize('mutation',['wrong-hash','wrong-size','source-symlink','destination-symlink'])
def test_copy_does_not_accept_unverified_or_indirect_bytes(tmp_path,mutation):
    src=tmp_path/'source';src.write_bytes(b'correct')
    dst=tmp_path/'destination';item=dict(bytes=src.stat().st_size,sha256=d.sha(src))
    if mutation=='wrong-hash':item['sha256']='f'*64
    elif mutation=='wrong-size':item['bytes']+=1
    elif mutation=='source-symlink':src.rename(tmp_path/'original');src.symlink_to(tmp_path/'original')
    else:dst.symlink_to(tmp_path/'does-not-exist')
    with pytest.raises(AssertionError):d.copy_verified(src,dst,item)
    assert not dst.exists()


def test_interrupted_copy_preserves_partial_evidence_and_cannot_repeat(tmp_path,monkeypatch):
    src=tmp_path/'source';src.write_bytes(b'correct');dst=tmp_path/'destination'
    item=dict(bytes=src.stat().st_size,sha256=d.sha(src))
    def fail_copy(a,b,*args):b.write(b'part');raise OSError('deliberate control')
    monkeypatch.setattr(d.shutil,'copyfileobj',fail_copy)
    with pytest.raises(OSError):d.copy_verified(src,dst,item)
    assert dst.read_bytes()==b'part' and src.read_bytes()==b'correct'
    with pytest.raises(AssertionError):d.copy_verified(src,dst,item)


@pytest.mark.parametrize('mutation',['none','status','script','recipe','plan','artifact','missing'])
def test_live_terminal_task_identity_and_artifacts(monkeypatch,mutation):
    start,_,_=fixture();base=d.base_module()
    job=next(j for j in json.loads(base.JOURNAL.read_bytes())['jobs'] if j['task_id']==d.TASK)
    plan=job['plan'];script='fixture source'
    # Script bytes must hash to the submitted plan, using a temporary plan only.
    plan=copy.deepcopy(plan);plan['bootstrap_sha256']=hashlib.sha256(script.encode()).hexdigest()
    recipe=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    job=copy.deepcopy(job);job['plan']=plan;job['recipe_sha256']=recipe;start['recipe_sha256']=recipe
    class FakeJournal:
        def read_bytes(self):return json.dumps({'jobs':[job]}).encode()
    monkeypatch.setattr(base,'JOURNAL',FakeJournal())
    params={'General/plan':json.dumps(plan),'General/recipe_sha256':recipe}
    task=N(status='completed',data=N(script=N(diff=script)),artifacts={k:N(hash=x['sha256'],size=x['bytes']) for k,x in start['all_terminal_registered_artifacts'].items()},get_parameters=lambda:params)
    if mutation=='status':task.status='in_progress'
    if mutation=='script':task.data.script.diff+='changed'
    if mutation=='recipe':params['General/recipe_sha256']='other'
    if mutation=='plan':params['General/plan']='{}'
    if mutation=='artifact':task.artifacts['replay-rank7'].hash='other'
    if mutation=='missing':task.artifacts.pop('replay-rank0')
    FakeTask=N(get_task=lambda **kw:task)
    if mutation=='none':assert d.qualify_live(start,FakeTask)==job
    else:
        with pytest.raises(AssertionError):d.qualify_live(start,FakeTask)


def mocked_chain(tmp_path,monkeypatch):
    base0=d.base_module()
    root=tmp_path/'artifacts/controller';cache=tmp_path/'artifacts/cache';cache.mkdir(parents=True)
    diag=tmp_path/'artifacts/diagnostic';diag.mkdir()
    monkeypatch.setattr(d,'ROOT',root);monkeypatch.setattr(d,'CACHE',cache);monkeypatch.setattr(d,'DIAG',diag)
    gate=cache/'byte-proof.json';monkeypatch.setattr(d,'GATE',gate)
    bad=cache/'replay-rank5.tar.gz.partial';bad.write_bytes(b'bad old bytes');monkeypatch.setattr(d,'BAD_SHA',d.sha(bad))
    selected={}
    for rank in (5,7):
        path=diag/f'replay-rank{rank}.tar.gz';path.write_bytes(f'recovered {rank}'.encode());selected[f'replay-rank{rank}']=dict(sha256=d.sha(path),bytes=path.stat().st_size)
    start=dict(selected=selected,original_partial=dict(path=str(bad)),failure_receipt={'old':'prefetch'},controller_failure={'old':'controller'})
    d.new(diag/'started.json',start);d.new(diag/'finished.json',dict(fixture_only=True))
    sources={'fixture':'source'};job={'recipe_sha256':'recipe'};calls=[]
    base=N(READER_PYTHON=base0.READER_PYTHON,READER=base0.READER,PYTHON=base0.PYTHON,DRIVER=base0.DRIVER,
        observe=lambda _:('absent',None),reject_existing=lambda:None,
        reject_competing_processes=lambda:None,reject_prior_CPU=lambda:None)
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'model')
    def qualify_bytes(actual):
        assert actual==job and gate.is_file();calls.append('bytes-qualified');return checkpoint
    base.qualify_bytes=qualify_bytes
    monkeypatch.setattr(d,'base_module',lambda:base)
    monkeypatch.setattr(d,'source_gate',lambda:sources)
    monkeypatch.setattr(d,'bind_diagnostic',lambda pid:(start,'absent',None))
    monkeypatch.setattr(d,'qualify_finished',lambda actual:dict(fixture_only=True))
    monkeypatch.setattr(d,'qualify_live',lambda actual:job)
    monkeypatch.setattr(d,'register',lambda path,kind:calls.append(('registered',path.name)))
    def child(name,command):
        calls.append((name,command))
        if name=='terminal-readback':
            assert all(d.sha(cache/(key+'.tar.gz'))==item['sha256'] for key,item in selected.items())
            assert d.sha(bad)==d.BAD_SHA;d.new(gate,dict(fixture_only=True))
        else:
            assert 'bytes-qualified' in calls
            output=root/'independent-acceptance';output.mkdir();d.new(output/'acceptance.json',dict(fixture_only=True))
    monkeypatch.setattr(d,'run_child',child)
    return root,calls,base


def test_default_does_not_copy_or_start_children(tmp_path,monkeypatch):
    root,calls,_=mocked_chain(tmp_path,monkeypatch)
    d.run(123,execute=False)
    assert not root.exists() and calls==[]


def test_complete_one_shot_chain_requires_bytes_before_CPU(tmp_path,monkeypatch):
    root,calls,base=mocked_chain(tmp_path,monkeypatch)
    d.run(123,execute=True)
    children=[x for x in calls if isinstance(x,tuple) and x[0]!='registered']
    assert children[0]==('terminal-readback',[base.READER_PYTHON,str(base.READER),'--seed','2027','--method','rbf'])
    assert children[1][0]=='full-CPU-admission' and children[1][1][:2]==[base.PYTHON,str(base.DRIVER)]
    assert len(children)==2 and (root/'recovered-byte-binding.json').is_file()
    final=json.loads((root/'process-exit.json').read_bytes())
    assert final['requires_independent_receipt_review'] is True and final['experiment_accepted'] is False
    with pytest.raises(AssertionError):d.run(123,execute=True)


@pytest.mark.parametrize('stage',['diagnostic','live-task','readback','CPU'])
def test_chain_failure_retains_evidence_and_never_starts_later_stage(tmp_path,monkeypatch,stage):
    root,calls,_=mocked_chain(tmp_path,monkeypatch)
    def fail(*a,**kw):raise AssertionError('deliberate control')
    if stage=='diagnostic':monkeypatch.setattr(d,'qualify_finished',fail)
    elif stage=='live-task':monkeypatch.setattr(d,'qualify_live',fail)
    else:
        original=d.run_child
        def child(name,command):
            if name==('terminal-readback' if stage=='readback' else 'full-CPU-admission'):
                calls.append(('failed-child',name));fail()
            original(name,command)
        monkeypatch.setattr(d,'run_child',child)
    with pytest.raises(AssertionError):d.run(123,execute=True)
    failure=json.loads((root/'controller-failure.json').read_bytes())
    assert failure['automatic_retry'] is failure['experiment_accepted'] is False
    assert failure['original_failed_attempts_preserved'] is True
    assert not (root/'process-exit.json').exists()
    if stage!='CPU':assert not (root/'independent-acceptance').exists()
    if stage in ('diagnostic','live-task'):assert not (root/'recovered-byte-binding.json').exists()
