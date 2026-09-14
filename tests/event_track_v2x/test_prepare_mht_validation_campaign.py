from pathlib import Path
import pytest

from tools.event_track_v2x import prepare_mht_validation_campaign as tool


def checkpoints(tmp_path):
    return [dict(seed=s,path=str(tmp_path/str(s)),sha256=h) for s,h in tool.contract.CHECKPOINTS.items()]


def test_jobs_cover_fixed_seeds_and_shared_k4_budgets(tmp_path):
    jobs = tool.make_jobs(tmp_path/'cache',tmp_path/'schedule.json',checkpoints(tmp_path),tmp_path/'output')
    assert [j['seed'] for j in jobs] == [1337,2027,3407]
    assert len({j['output'] for j in jobs}) == 3
    for job in jobs:
        a=job['arguments']
        assert a['width']==4 and a['device']=='cpu' and a['max_assignment_solves']==10000
        assert a['max_assignment_frontier']==10000 and a['max_generated_candidates']==100000
        assert not Path(job['output']).exists()


@pytest.mark.parametrize('bad',['missing','duplicate','unapproved','existing'])
def test_invalid_or_duplicate_campaign_jobs_rejected(tmp_path,bad):
    refs=checkpoints(tmp_path)
    if bad=='missing':refs.pop()
    elif bad=='duplicate':refs[1]=dict(refs[0])
    elif bad=='unapproved':refs[0]['sha256']='a'*64
    else:(tmp_path/'seed-1337-mht-k4-v1').mkdir()
    with pytest.raises(ValueError):tool.make_jobs(tmp_path/'cache',tmp_path/'schedule.json',refs,tmp_path)


def test_wrong_dataset_rejected_without_loading_payloads(tmp_path,monkeypatch):
    monkeypatch.setattr(tool,'VerifiedForestCache',lambda *a:pytest.fail('wrong dataset reached payload loader'))
    with pytest.raises(ValueError,match='frozen full official'):
        tool.prepare(tmp_path/'cache','bad',tmp_path/'schedule','bad',[],tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_missing_seed_rejected_before_detector_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(tool,'VerifiedForestCache',lambda *a:pytest.fail('missing seed reached payload loader'))
    with pytest.raises(ValueError,match='three pinned'):
        tool.prepare(tmp_path/'cache',tool.contract.CACHE_SHA,tmp_path/'schedule',tool.contract.SCHEDULE_SHA,
                     [],tmp_path/'output')
    assert not (tmp_path/'output').exists()
