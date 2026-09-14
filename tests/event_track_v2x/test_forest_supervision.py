from dataclasses import replace
import math

import pytest
import torch

from transvision.models.event_track_v2x.forest_supervision import (
    AnnotationIdentity, AnnotationIdentityIndex, DetectionIdentityTarget,
    RowSupervision, make_row_supervision, relation, set_valued_parent_loss,
)
from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
from test_forest_tracking import observation


def ann(side, frame, track, token):
    return AnnotationIdentity('0003', side, frame, track, token)


def target(identity, source=0, **kwargs):
    return DetectionIdentityTarget('0003', source, identity, **kwargs)


def test_exact_links_temporal_ids_and_unlinked_cross_source_unknown():
    a, b, c, d = ann(0, 'a', '1', 'aa'), ann(1, 'a', '4', 'bb'), ann(0, 'b', '1', 'cc'), ann(1, 'b', '8', 'dd')
    index = AnnotationIdentityIndex([a, b, c, d], [(a.reference, b.reference)])
    ta, tb, tc, td = [index.targets[x.reference] for x in (a, b, c, d)]
    assert relation(ta, tb) is True and relation(ta, tc) is True
    assert relation(ta, td) is None and relation(tb, td) is False


def test_duplicate_source_id_poison_propagates_through_joined_identity():
    a, b, c = ann(0, 'a', '1', 'aa'), ann(0, 'a', '1', 'cc'), ann(1, 'a', '4', 'bb')
    index = AnnotationIdentityIndex([a, b, c], [(a.reference, c.reference)])
    assert {t.status for t in index.targets.values()} == {'ambiguous'}
    assert index.audit['ambiguous_components'] == 1


def test_joined_same_source_fragments_allowed_only_without_coexistence():
    a, b, c, d = ann(0, 'a', '1', 'aa'), ann(1, 'a', '4', 'bb'), ann(0, 'b', '2', 'cc'), ann(1, 'b', '4', 'dd')
    index = AnnotationIdentityIndex([a, b, c, d], [(a.reference, b.reference), (c.reference, d.reference)])
    assert len({t.identity for t in index.targets.values()}) == 1
    assert {t.status for t in index.targets.values()} == {'matched'}
    e = ann(0, 'b', '1', 'ee')
    index = AnnotationIdentityIndex([a, b, c, d, e], [(a.reference, b.reference), (c.reference, d.reference)])
    assert {t.status for t in index.targets.values()} == {'ambiguous'}


@pytest.mark.parametrize('bad', ['missing', 'same_source', 'reuse', 'duplicate_token'])
def test_bad_exact_annotations_rejected(bad):
    a, b = ann(0, 'a', '1', 'aa'), ann(1, 'a', '4', 'bb')
    annotations, links = [a, b], [(a.reference, b.reference)]
    if bad == 'missing':
        links = [(a.reference, ann(1, 'a', '5', 'bb').reference)]
    elif bad == 'same_source':
        links = [(a.reference, a.reference)]
    elif bad == 'reuse':
        links *= 2
    else:
        annotations.append(replace(a, track_id='2'))
    with pytest.raises(ValueError):
        AnnotationIdentityIndex(annotations, links)


def test_multiple_positive_parents_unknown_not_negative_and_no_false_birth():
    raw = (observation('0'), observation('1', source=1), observation('2', source=1), observation('3', state_us=1_100_000))
    context = ForestRowContext((0, 1, 2, 3), raw, 1_200_000)
    labels = [target('a'), target('a', 1), target('unlinked', 1), target('a')]
    result = make_row_supervision(context, labels, set(labels[:-1]))
    assert result == RowSupervision((False, True, True, False), (True, True, True, False), 'parent')
    narrower = ForestRowContext((2, 3), raw[2:], 1_200_000)
    assert make_row_supervision(narrower, labels, labels[:-1]).reason == 'out_of_support'
    assert make_row_supervision(narrower, labels, [labels[2]]).reason == 'uncertain_birth'
    first = ForestRowContext((3,), raw[-1:], 1_200_000)
    assert make_row_supervision(first, labels, []).reason == 'birth'


def test_partial_set_loss_excludes_unknown_gradient_and_counts_unusable():
    logits = torch.tensor([0., 1., 2., 100.], dtype=torch.double, requires_grad=True)
    target_row = RowSupervision((False, True, True, False), (True, True, True, False), 'parent')
    skipped = RowSupervision((False,), (False,), 'out_of_support')
    result = set_valued_parent_loss([logits, torch.zeros(1)], [target_row, skipped])
    assert result['loss'].item() == pytest.approx(math.log(1+math.e+math.e**2)-math.log(math.e+math.e**2))
    result['loss'].backward()
    assert logits.grad[-1] == 0 and result['supervised_rows'] == 1
    assert result['reason_counts']['out_of_support'] == 1
    with pytest.raises(ValueError, match='no supervised'):
        set_valued_parent_loss([torch.zeros(1)], [skipped])
