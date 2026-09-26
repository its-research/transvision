# SPD teacher event timing: unresolved real-input admission

On 2026-09-18, authenticated readback of ClearML schedule task `7a0cb79d3ff144d689b565560d1b844b` reproduced SHA-256 `66197cec03671866a0f452b5e4ea21273aadf4d5015d2d9e77169347ee4dc29f`: 7,445 pairs across 46 sequences. Each side has 7,445 distinct `(sequence_id, frame_id)` keys, with no repeated source frame. This establishes pair uniqueness, not causal availability.

The current isolated `spd_teacher_schedule_coverage.check_events` requires exactly both official sources for each event, but only requires `decision_us >= reference_us`; its cache validation checks information time <= arrival <= decision. Those checks do not establish the paper's fixed deadline or the completeness of a realistic arrival stream. Passing these software checks alone is insufficient to freeze a formal teacher replay.

The paper's `experiments.md` specifies the existing 100 ms deadline for fixed-input mechanism comparisons and forbids removing lateness constraints to gain extra observations. Its README reports late vehicle frames in the historical **val** cohort. That historical count must not be extrapolated to train; actual train metadata still needs inspection.

Before constructing the real teacher events:

1. Read hash-bound train cache metadata and the applicable frozen scheduling contract, not GT labels, to determine information timestamps and declared arrival semantics.
2. Audit official train pairs against the fixed deadline, including per-sequence ordering, late observations, first receipt, and possible later use.
3. Separate complete reference-event coverage from source availability. If either source is legitimately unavailable, the runtime/export coverage rule must represent that case instead of forcing two inputs, extending deadlines, or inventing early arrival times.
4. Verify resulting saved events, actual cache payloads, prediction alignment, and ingestion audits before admitting a full official teacher trace.

No teacher event file was generated or admitted by this inspection. Existing frozen training jobs and completed evidence were not modified.

## Real train metadata audit

Authenticated streaming readback of seed 1337 cache task `5a76cc027970456f8e3b025dbc6d0e9b` inspected all 16,338 metadata files. Each file's byte count and SHA-256 matched the separately downloaded manifest (SHA-256 `4882abe1c88c48559ba188afad45ef2d3be630dd5e79f90e265ea0bc6be39f7a`). The archive was streamed without extracting payload arrays; this audit does not newly attest array contents or the compressed archive digest.

For each official train pair, computed `max(box_reference_timestamp_us, source_image_timestamp_us) - schedule_reference_us`, using the pinned schedule above. Information later than `reference + 100000` cannot be available by that deadline even with zero added transport delay.

| Source | Pairs | Information after 100 ms deadline | Minimum offset, us | Maximum offset, us |
| --- | ---: | ---: | ---: | ---: |
| vehicle-side | 7445 | 3693 | 58186 | 155793 |
| infrastructure-side | 7445 | 0 | -67793 | 90632 |

This is an actual train-cohort timing result, not the historical val count. It proves that the current requirement of two delivered sources at every official event is incompatible with the fixed deadline for this cohort. It does not measure communication latency or infer a transport schedule. Formal teacher admission must first support causal source availability and explicit handling of late/missing observations while retaining all reference events. Do not extend deadlines or redate information to pass the existing coverage checker.

## Deadline guard implemented; availability work remains

The isolated coverage checker now requires integer `decision_us == reference_us + 100000`. Tests reject early/extended deadlines, boolean/float decision times, and extending the deadline to accommodate a late source; arrival exactly at the deadline remains legal. Coverage, teacher receipt and end-to-end regression: **31 passed**, zero skips. This does not resolve the two-source requirement or establish a complete real teacher replay.

Completed ClearML software evidence task `fcaba97e7c6448c8b44512be747f4912` contains three independently byte-for-byte readback-verified artifacts: coverage source (4683 bytes, SHA-256 `6710b22785fbe857bf8dab3399b157f533a15a83de7d50e31fef4b93068b56a9`), tests (2789 bytes, `b4f16acbab605f1c8efe8ef419c1bbfd37c211e93b9b9781157c3c22f62c63f3`), and JUnit regression (5045 bytes, `91f47c28712b6e2c7e9128794e9f53f38c86769afc6a7ead87fdc2d86c809da1`). Existing immutable training bundles were not changed.

## Explicit-arrival event construction and verification

Added isolated `spd_causal_teacher_events.py`: every official reference event is retained, first arrivals are consumed at their first eligible fixed deadline, and arrivals beyond the last event remain in an explicit tail. Complete per-source arrival records are required; the builder never infers timestamps or treats missing arrival records as dropped packets. Its verifier rejects omitted/withheld/repeated observations, substituted frames, reordered events, changed timestamp types and concealed tail entries. An independent test oracle locates each delivery's first eligible event and checks sequence isolation.

Combined causal scheduling, coverage and end-to-end tests: **37 passed**, zero skips. Completed ClearML task `1195463096764c3aa8cee4575e778b8f` has byte-for-byte readback-verified source (3960 bytes, `852c1600647909b1cadea08eecc26fc623b9df9c8723dbd62fda7f894a65e2d6`), tests (3874 bytes, `9f1076dc4f2769360c9e5fe232ea980da011ce73cfccae083d0789fc362b3178`), and regression XML (5960 bytes, `50a7df20a849e39d019431ec4ef747f37b7edcfc925c47d6c048603b7b386ce7`).

This remains software preparation, not formal admission: actual arrival-protocol binding, cache verification including late-tail metadata, replacement of the exporter's legacy two-source-per-event coverage check, and a real full replay are still required. The builder's receipt explicitly sets `formal_teacher_admitted=false`.

The next local revision adds `verify_arrival_cache`, reusing the real verified-cache loader to check every explicitly scheduled source, including the unconsumed tail, against producer identity, manifest, frame/payload hashes and information-time <= arrival. Its audit uses each arrival as a validation bound only; no tracking output or historical deadline is changed. A real synthetic-cache test rejects early arrival, wrong frame/manifest hashes and corrupted tail array bytes. Causal scheduling plus coverage tests: **35 passed**, zero skips. This latest revision has not yet been published as a replacement frozen source artifact; the earlier ClearML source hash above remains historical. Transport protocol validation and formal exporter integration still remain.

Subsequent publication of that revision is now complete: ClearML task `d25dde3809ec4f19986fe8702966e1f5`, with all three artifacts independently read back byte-for-byte. Source: 4746 bytes, SHA-256 `ded0576f263509729a43f8703b29dae6b620f139f560cbcb17bc331bd0b6965a`; tests: 6105 bytes, `74b3de57817996a8fc1576614ca938c805a39325b41028c78370683ac85c413c`; combined causal, coverage and three-dataset end-to-end regression: **38 passed**, zero skips, XML 6120 bytes, `0beab68dad9c6a20b50d979c89cd4a257ab9483736e9711aff9cea84b223e201`. This is a partial software source publication, not a full runtime freeze or formal teacher experiment.

An additional local integration test now feeds generated events into the actual `PersistentForestCacheStream` with a verified two-frame synthetic V2 cohort and the all-class paper configuration. It verifies 2 then 4 new observations, an explicit one-frame late tail, exactly three durable frame receipts, and byte-identical first prediction/audit when replayed after the second commit. Causal tests plus V2-cache regression: **47 passed**, zero skips. Two initial fixture-construction failures (raw sharding mismatch, then use of the legacy rather than paper configuration class) were corrected in the synthetic test; no production validation was weakened. This latest test addition is not yet included in the preceding published test artifact. Actual complete teacher export remains pending.

## Real information-availability scheduling diagnostic

Generated a diagnostic schedule using the pinned official map and seed 1337 cache. All 16,338 metadata payloads were checked against their manifest hashes/sizes. For this diagnostic only, explicit arrivals equal `max(source_image_timestamp_us, box_reference_timestamp_us)`; no measured network latency is claimed. The generated schedule passed exact first-arrival coverage verification.

Results: 7,445 reference events, 14,865 consumed source frames and 25 frames after the last deadline. Delivery-count histogram: 0 -> 216 events; 1 -> 3095; 2 -> 835; 3 -> 3096; 4 -> 203. There are 3,067 deliveries whose official pair belongs to a later reference event. This is not itself future-information leakage (their information timestamps passed the current deadline), but it proves the full-stream schedule is not identical to the historical paired-input mechanism schedule. They must not be silently substituted or pooled in the fixed-input table. No formal teacher admission or prediction run occurred.

Completed diagnostic ClearML task `16887bff8e5142ffa970ac950e2847bd`; all artifacts were independently read back byte-for-byte:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| arrivals | 2687647 | `059567b897b95e07442167d64611ceb2178779e0d4d14decdafe47988ec308a0` |
| events | 3837413 | `8a8d503a53b2af5870eb4585b7f5e37ebffa5c7100129a33f80869fa3ee4bcb7` |
| report | 369 | `1d3a235eaad724ff1bde1e745a604e4cc38f54f19cbba49c219b3dae70a9ced2` |
| schedule-receipt | 4639 | `1c28f0dac62f9d1d59a6e30fe997c1e651b785df028dd51208f3d69605b7d2fb` |

Both `formal_teacher_admitted` and `measured_communication` remain false. The next integration must explicitly distinguish full-stream system evaluation from fixed paired-input mechanism comparisons and bind whichever schedule is used before replay.

## Formal exporter now requires an explicit arrival schedule

The isolated priority exporter now treats the official reference map, actual cache root and complete explicit arrival schedule as one indivisible formal input. Supplying only one or two is rejected. It reconstructs first-eligible delivery assignment, verifies the late tail and every scheduled cache payload, then records the arrival-schedule SHA-256, protocol ID and measured-communication flag in coverage. Reference-event coverage no longer assumes exactly two paired deliveries; the historical strict paired checker remains available for its narrower diagnostic contract.

Combined receipt, schedule, causal-cache and three-dataset end-to-end tests: **50 passed**, zero skips. Completed ClearML software evidence task `20aceffb7f4840cb99f31daa14e002e9`; all artifacts were independently read back byte-for-byte:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| exporter | 13331 | `63f5fc9e3e5d6ad1b0a9bbf24f0c10435387df43f75948aede0df39702e67a7b` |
| coverage | 6019 | `27faad5eb22f162111068e633a4269d66c47ce5926359236736d807fb49fb895` |
| scheduler | 4882 | `57bb77e70e38e7417375bd8697c27c99cc96322c49a5a82f9db5381af0e718c3` |
| end-to-end tests | 22183 | `8f3de0669b5e1d757ed80ea4f90d55fd318902bc11641a55c27c7b365924ec45` |
| scheduler tests | 9704 | `78b11915836199a36849584a13a098a2a755eaa1dad3b448c48a4f86b1b60e00` |
| regression XML | 7921 | `ed9fe1db9316bea58d6439fd919dc9003533556703f213909232086d6ea8ac50` |

This is still a partial source/test publication, not a full immutable runtime bundle and not an actual teacher replay. The real arrival envelope must be protocol-approved and bound before dispatch.

## Zero-added-transport arrival protocol frozen

The SPD train teacher/system schedule now uses protocol ID `spd-zero-added-transport-information-availability-v1`: `arrival_us = max(box_reference_timestamp_us, source_image_timestamp_us)`, added transport delay 0 us, fixed deadline 100,000 us. It is deterministic simulated availability, not measured communication latency. Fixed paired-input mechanism comparisons remain a separate protocol and must not consume later-pair frames merely because their information timestamp is already available.

The formal arrival envelope binds its official schedule SHA, seed-specific cache manifest SHA, arrival rule, deadline and complete 14,890-source list. The exporter rejects incomplete envelopes and re-verifies all cache payloads. Sealer/exporter regression: **51 passed**, zero skips. Software evidence task `f84c88b0330b4811b95adcd0269a13f2` completed with byte-for-byte readback:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| sealer | 3062 | `9a0939d415611ceace34e642dbb8f33cdf050b10eddeba7590c11dcad89d8dfe` |
| exporter | 13967 | `936908e89418f9d48f2d836c7f6c23bf6a6d7af734a0a971af638309bd870d69` |
| sealer tests | 1635 | `a5a8c37f54d28a834a2249e33b0158b6a2c4d6ebeb10c268c515ac42c28d1951` |
| end-to-end tests | 22629 | `c9c47383fb7cb03ff8854c26cd319a055d8c14385b1f26b922f9a29ef2d2458a` |
| regression XML | 8069 | `d585192c65a3eb0a5931ad844a0393b09258a65e455d0178abfd174ad98a4a78` |

Seed-specific arrival assets, both fully read back from ClearML:

| Seed/cache | Task | Envelope SHA-256 | Metadata payloads checked | Formal replay run |
| --- | --- | --- | ---: | --- |
| 1337 / `4882abe1...` | `b07247aa34ab4ea5bf7ac65cf5dab78c` | `8916c4634c1b063b8620f12cfa602a8d6723fd883c5e8b58697b2852116d59b4` | 16338 | not yet |
| 2027 / `2c0616ca...` | `974b4d93f4db492d81c74c79ffeaf620` | `b4d01954a9cf503d5c0aaec32788a2412f9661d0d9b0ad53d77b68097c4f9997` | 16338 | not yet |

Both arrival receipts retain `formal_teacher_admitted=false`: publication of an input does not prove a completed replay. The replay waits for the matching trained identity checkpoint and a full frozen runtime bundle.
