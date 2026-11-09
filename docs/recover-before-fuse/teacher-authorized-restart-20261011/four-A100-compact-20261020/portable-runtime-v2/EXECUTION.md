# teacher3407 independent target candidate

This candidate verifies the completed outputs of task
`2642a57055da45d19d817b131d84fac1`. It does not submit or restart that task.
It is not yet admitted on a remote execution host.

## Evidence and scope

- Reference reader comparison covers all 46 sequences and 223,355 model-reference rows.
- Reference files are byte-preserving copies, mapped from their historical names.
- Numeric target and trajectory checks retain `atol = rtol = 1e-8`.
- Model-factor checks retain `atol = rtol = 1e-4`.
- The tree and legal-action dependencies retain their original bytes and SHA.
- Local isolated imports and synthetic report joins are software checks, not teacher acceptance.

## Before execution

1. Wait for the exact producer task to complete successfully. Collect every registered
   archive and member with the original `collect_sequences.py` into the Mac external
   evidence disk. Require the full 46-sequence/7,445-event collection.
2. Keep the collection manifest, source manifest, receipt, events, per-sequence
   storage records, byte-readback records and `collection.json` together. The reader
   rechecks full-cohort metadata even when verifying a single selected sequence.
3. Admit a concrete CPU execution host. Check the actual archive and unpacked
   sequence sizes, all retained prior outputs, reference/runtime files, SQLite
   temporary storage, free-space reserve and RAM. Reference size alone is not a
   storage or memory admission. The current reader loads all model-reference rows.
4. Verify the runtime file hashes against `dependency-isolation-check.json`, and
   the reference bundle against the pinned inventory. Retain failed attempts.
   A changed source, model, schedule, tolerance or resource contract requires
   its applicable qualification; this document grants no new authorization.

## Single-sequence interface

In a fresh admitted CPU process, with this runtime directory on `sys.path`:

```python
from verify_portable_sequence_targets_candidate import verify_collected_sequence

report = verify_collected_sequence(
    collection=collection_directory,
    sequence=sequence_id,
    output=new_output_directory,
    reference_bundle=reference_bundle_directory,
)
```

The output directory must not exist. The function verifies metadata, materializes
only the selected archive, checks native streams, targets, search trajectory and
original model factors, then writes `independent-targets.json`. It preserves the
materialized evidence on success and failure. Do not assume automatic space
reclamation or copy all archives to a constrained host without accounting for them.

## Full-cohort join

After all 46 real sequence reports exist, use
`join_portable_sequence_targets_candidate.join_files` with their exact paths,
validated collection indexes, submitted plan, all frozen events and the collection
snapshot. It rejects missing or duplicate sequences, wrong source/plan/database
bindings, incomplete coverage, and relaxed tolerances.

The joined report deliberately retains `priority_training_admitted = False` and
`full_teacher_scientific_acceptance = False`. Qualified dataset/cache/model and
full causal trajectory evidence must be reconciled separately before priority
training. No overall A–K completion claim follows from this candidate.
