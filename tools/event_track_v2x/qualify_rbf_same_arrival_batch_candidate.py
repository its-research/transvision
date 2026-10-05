"""CPU parity on real fixed causal prefixes, not a GPU or full-forest result."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import tarfile
import time
import traceback
import types

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert root.is_relative_to(R/'artifacts')
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir()
    archive=R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    assert sha(archive)=='038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
    with tarfile.open(archive) as stream:
        members=stream.getmembers()
        assert len({m.name for m in members})==len(members)
        assert all((m.isfile() or m.isdir()) and not m.issym() and not m.islnk()
            and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        stream.extractall(source,filter='data')
    candidate=Path(__file__).resolve().parents[2]/'transvision/models/event_track_v2x/batched_row_context_scoring.py'
    destination=source/'transvision/models/event_track_v2x'/candidate.name
    with destination.open('xb') as stream:stream.write(candidate.read_bytes())
    source_hashes={str(p.relative_to(source)):sha(p) for p in (source/'transvision/models/event_track_v2x').glob('*.py')}
    new(root/'source-binding.json',dict(original_archive_sha256=sha(archive),source_files=source_hashes,
        qualifier_sha256=sha(__file__),candidate_sha256=sha(candidate),production_runtime_modified=False))
    for name in ('transvision','transvision.models','transvision.models.event_track_v2x'):
        module=types.ModuleType(name);module.__path__=[str(source.joinpath(*name.split('.')))];sys.modules[name]=module
    try:
        import torch
        import numpy as np
        from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
        from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
        from transvision.models.event_track_v2x.batched_row_context_scoring import BatchedLearnedRowScorer
        from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
        from transvision.models.event_track_v2x.forest_training_data import TrainingShard
        from transvision.models.event_track_v2x.recoverable_identity import model_digest
        torch.set_num_threads(1)
        assert torch.__version__=='2.6.0' and np.__version__=='1.26.4' and not torch.cuda.is_available()
        train=R/'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed2027'
        proof=json.loads((train/'acceptance.json').read_bytes());checkpoint=json.loads((train/'checkpoint').read_bytes())
        assert sha(train/'checkpoint')==proof['artifacts']['checkpoint']['sha256']
        weights=train/'archive-unpack/training/seed-2027/weights.pt'
        assert sha(weights)==checkpoint['weights']['sha256']==proof['weights_sha256']
        model=RecoverableIdentityModel(**checkpoint['architecture'])
        model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        model.eval().requires_grad_(False);assert model_digest(model)==checkpoint['model_sha256']
        original=LearnedForestScorer(model,max_nodes=9,max_pairs=81,geometry_weight=1.,process_noise=.1)
        batch=BatchedLearnedRowScorer(original,max_batch=64)
        rows_root=R/'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed2027'
        manifest=json.loads((rows_root/'manifest').read_bytes())
        assert sha(rows_root/'manifest')==checkpoint['dataset_sha256']
        assert manifest['row_protocol']['candidate_protocol']=='rbf-all-class-top64-v1'
        schedule_path=R/'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed2027/events.json'
        assert sha(schedule_path)=='3440542d0fb6b7c52a6b97a886a888ab5acca73ab1b73a37139e1c49b37c6c25'
        schedule=json.loads(schedule_path.read_bytes());sequences=sorted(schedule['origin_us_by_sequence'])[:4]
        assert len(schedule['events'])==7445 and len(schedule['origin_us_by_sequence'])==46
        numeric_root=R/'artifacts/rbf-final-refit-all-row-independent-numeric-v1-20261004/seed2027'
        numeric_proof=json.loads((numeric_root/'independent-byte-coverage-receipt.json').read_bytes())
        references={sequence:{} for sequence in sequences}
        for key,record in numeric_proof['artifacts'].items():
            if not key.startswith('predictions-rank'):continue
            path=numeric_root/(key+'.jsonl');assert sha(path)==record['sha256']
            for line in path.read_text().splitlines():
                value=json.loads(line)
                if value['sequence_id'] in references:references[value['sequence_id']][value['row']]=value
        outcomes=[];all_contexts=[];began=time.monotonic();max_serial=max_saved=0.;total_rows=0
        for sequence in sequences:
            events=[e for e in schedule['events'] if e['sequence_id']==sequence]
            assert len({e['decision_us'] for e in events})==len(events), 'cannot identify real events by an ambiguous time'
            selected=events[:32];assert events[32]['decision_us']>selected[-1]['decision_us']
            rec=next(rec for rec in manifest['shards'] if rec['sequence_id']==sequence)
            hits=list((rows_root/'rows-unpack').rglob(rec['path']));assert len(hits)==1 and sha(hits[0])==rec['sha256']
            shard=TrainingShard(hits[0],rec,manifest['row_protocol']['parent_limit']);arrays=shard.arrays
            outcomes_events=[]
            for event in selected:
                ids=np.flatnonzero(arrays['decision_us']==event['decision_us']).tolist()
                contexts=[]
                for i in ids:
                    n=int(arrays['lengths'][i]);indices=tuple(map(int,arrays['contexts'][i,:n]));assert indices[-1]==i
                    contexts.append(ForestRowContext(indices,tuple(shard._observation(k) for k in indices),event['decision_us']))
                # Every row at this real event participates; no valid_indices,
                # targets or GT labels select queries or reach the scorer.
                started=time.monotonic();candidate_factors=batch.score_contexts(contexts);batch_seconds=time.monotonic()-started
                started=time.monotonic()
                serial_factors=[original(c.observations,c.support,c.decision_us) for c in contexts]
                serial_seconds=time.monotonic()-started
                for i,context,candidate_factor,serial_factor in zip(ids,contexts,candidate_factors,serial_factors,strict=True):
                    assert candidate_factor.nodes==serial_factor.nodes==tuple(o.node for o in context.observations)
                    assert tuple(tuple(p for p,_ in row) for row in candidate_factor.rows)==context.support
                    for got,want in zip(candidate_factor.rows,serial_factor.rows,strict=True):
                        assert [p for p,_ in got]==[p for p,_ in want]
                        g=np.asarray([w for _,w in got]);w=np.asarray([w for _,w in want])
                        assert np.allclose(g,w,atol=1e-4,rtol=1e-4)
                        max_serial=max(max_serial,float(np.max(np.abs(g-w))))
                    ref=references[sequence][i]
                    assert ref['context_indices']==list(context.indices) and ref['decision_us']==event['decision_us']
                    values=np.asarray(ref['logits'],dtype=np.float64);maximum=values.max()
                    expected=values-(maximum+np.log(np.exp(values-maximum).sum()))
                    actual=np.asarray([w for _,w in candidate_factor.rows[-1]])
                    assert np.allclose(actual,expected,atol=1e-4,rtol=1e-4)
                    max_saved=max(max_saved,float(np.max(np.abs(actual-expected))))
                total_rows+=len(contexts)
                outcomes_events.append(dict(event_id=event['event_id'],decision_us=event['decision_us'],rows=len(contexts),
                    CPU_candidate_wall_seconds=batch_seconds,CPU_serial_wall_seconds=serial_seconds,
                    timings_not_isolated_or_GPU_or_paper_latency=True))
                if contexts and len(all_contexts)<2:all_contexts.append(contexts[0])
            outcome=dict(sequence_id=sequence,shard_sha256=sha(hits[0]),events=outcomes_events)
            new(root/f'sequence-{sequence}.json',outcome);outcomes.append(outcome)
            print(json.dumps(dict(stage='CPU_same_arrival_batch_parity',completed_sequences=len(outcomes),total_sequences=4,
                checked_rows=total_rows,ETA_seconds=(time.monotonic()-began)*(4-len(outcomes))/len(outcomes),
                ETA_scope='this finite CPU candidate qualification only')),flush=True)
        refusals=[]
        def refuse(label,call):
            try:call()
            except (TypeError,ValueError):refusals.append(label)
            else:raise AssertionError('candidate failed to refuse '+label)
        refuse('batch_above_admitted_64',lambda:BatchedLearnedRowScorer(original,max_batch=65))
        refuse('boolean_batch_size',lambda:BatchedLearnedRowScorer(original,max_batch=True))
        assert len(all_contexts)==2 and all_contexts[0].decision_us!=all_contexts[1].decision_us
        refuse('mixing_actual_arrival_events',lambda:batch.score_contexts(all_contexts))
        model.train();refuse('training_mode',lambda:batch.score_contexts(all_contexts[:1]));model.eval()
        parameter=next(model.parameters());saved=parameter.detach().clone()
        with torch.no_grad():parameter.add_(.01)
        refuse('changed_checkpoint_weights',lambda:batch.score_contexts(all_contexts[:1]))
        with torch.no_grad():parameter.copy_(saved)
        assert model_digest(model)==checkpoint['model_sha256']
        for name,digest in source_hashes.items():assert sha(source/name)==digest
        acceptance=root/'CPU-prefix-parity.json'
        new(acceptance,dict(kind='rbf_real_same_arrival_batched_scorer_CPU_prefix_parity_candidate_v1',
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=2027,sequences=sequences,
            actual_schedule_prefix_events=128,rows=total_rows,max_absolute_original_serial_difference=max_serial,
            max_absolute_saved_admitted_GPU_factor_difference=max_saved,atol=1e-4,rtol=1e-4,
            checkpoint_sha256=sha(train/'checkpoint'),weights_sha256=sha(weights),model_sha256=checkpoint['model_sha256'],
            source_binding_sha256=sha(root/'source-binding.json'),candidate_execution_recipe=batch.execution_recipe,
            unchanged_primitive_source_sha256=source_hashes['transvision/models/event_track_v2x/forest_training.py'],
            results=[dict(path=str(root/f'sequence-{s}.json'),sha256=sha(root/f'sequence-{s}.json')) for s in sequences],
            refusal_controls=refusals,runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=1,device='CPU'),
            GPU_execution_or_memory_target_admitted=False,full_forest_replay_or_predictions_admitted=False,
            full_three_seed_or_Stage2_or_paper_cost_admitted=False,production_integration_enabled=False))
        register(acceptance,'rbf-real-same-arrival-batched-scorer-CPU-prefix-parity')
        print(json.dumps(dict(receipt=str(acceptance),rows=total_rows,max_serial_error=max_serial,max_saved_error=max_saved,
            GPU_or_full_forest_accepted=False)),flush=True)
    except BaseException as error:
        new(root/'failure.json',dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            actual_GPU_experiment_failed=False,production_runtime_modified=False))
        register(root/'failure.json','rbf-same-arrival-batch-CPU-candidate-qualification-failure')
        raise


if __name__=='__main__':main()
