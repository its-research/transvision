# Trajectory parallelism boundary: source evidence, no implementation admission

112 verified the exact frozen trajectory sourceSHA37ddf0a7437064d41f6df36a0879eb67b19a430e306c2fb1c8d6e48827c894a9 against its original source-freeze. No module/model/numerical oracle was run by this audit. Current0033 and3090 jobs were not changed.

## Whole events cannot simply be divided among workers

The original verify_database maintains previous/catalog/memberships/owners across events; enforces ordered ordinals, monotone reference/arrival clocks and a cumulative factor hash; derives each component partition from previous state; and commits the chosen operation before the next selection. Final checks require exact prefix/membership catalog and complete witness-reference coverage. Thus dividing the events into independent chunks without independently reconstructing and verifying their preceding history would remove original checks. Reconstructing the entire earlier history for every chunk may add work; no speedup is assumed.

## Smaller potential boundary, still unqualified

Within a single allocation selection, the original code iterates candidate records and calls the frozen numeric.verify_candidate separately, then applies the chosen result and advances remaining/prefix/frontier state. The numeric source is independently pinnedSHA2aecc66e8c6667b8cf3dc1b6b70eda596049d9eab5cfc2a63ab29e9d5298eef5; it reconstructs RootTree objects and retains Decimal90 residual bounds. Python-level serial execution and high-precision work are source facts, not a live-stack hotspot measurement.

A candidate-only process boundary could be investigated while preserving:

- Parent-owned serial events, partitions, factor hash, selection decisions and original ordered reductions.
- Every candidate before/after witness, full independent root/decision/18feature/target/charged-step recomputation; no production or numerical-answer cache.
- Exact union of all worker-loaded witness references for the original orphan/unverified-witness final assertion, plus original parent before-state and committed-prefix equality checks.
- Original source and numerical environment, Decimal90, input/model, search/action budgets and1e-8/1e-4; complete original outputs and exceptions.
- Real complete original/candidate result comparison and end-to-end measurement before any rollout. IPC, repeated parsing, worker memory and process initialization may outweigh parallel work. No claimed speedup or model qualification from this audit.

No adapter has been qualified or deployed; no actual control attempt, extra resource cap, GPU disabling, task restart or code publication occurred. A later concrete real comparison remains subject to the existing specific execution/publication gates. Do not replace this audit with a toy benchmark or call the full A–K objective complete.
