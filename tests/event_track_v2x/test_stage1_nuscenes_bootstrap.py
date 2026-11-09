import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('stage1_nuscenes_bootstrap', Path(__file__).parents[2] / 'tools/event_track_v2x/bootstrap_stage1_nuscenes.py')
bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)


def test_undefined_draw_is_not_silently_removed_from_confidence_interval():
    result = bootstrap.interval(np.array([1., np.nan, 3.]))
    assert result['available'] is False
    assert result['finite_replicates'] == 2 and result['total_replicates'] == 3
    assert 'low' not in result and 'high' not in result


def test_missing_full_author_cache_admission_rejects_before_creating_bootstrap_output(tmp_path):
    root = tmp_path / 'cache'; root.mkdir(); proof = root / 'campaign-receipt.json'
    proof.write_text(json.dumps(dict(all13_complete_original_cohort_score_subset_caches_admitted=False, results=[])))
    output = tmp_path / 'new-bootstrap'
    args = argparse.Namespace(cache_campaign=str(root), output=str(output), cache_campaign_sha256=bootstrap.sha(proof))
    with pytest.raises(ValueError, match='all13 actual author matching'):
        bootstrap.evaluate(args)
    assert not output.exists()


def test_linked_or_modified_cache_proof_is_not_admitted(tmp_path):
    original = tmp_path / 'original.json'; original.write_text('{"admitted":true}')
    link = tmp_path / 'linked.json'; link.symlink_to(original)
    digest = bootstrap.sha(original)
    with pytest.raises(ValueError, match='ordinary admitted input'):
        bootstrap.read(link, digest)
    original.write_text('{"admitted":false}')
    with pytest.raises(ValueError, match='exact SHA'):
        bootstrap.read(original, digest)
