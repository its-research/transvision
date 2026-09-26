# SPD nested controller parameter casting

Live ClearML inspection confirmed that `Task.get_parameters()` defaults to `cast=False`: completed V2 seed and nested-calibration values were strings `1337` and `True`. With `cast=True`, these were integer 1337 and boolean true. Typed producer admission otherwise rejects the correct real producer.

Task `2486a74c9d534749b8d5834b5148ec51` was dequeued before execution at 2026-09-18 09:14:17 UTC, and observed as `created`. Its historical dispatch record is retained. Running seed 2027 V2 and seed 3407 detector jobs were not modified.

The isolated paper worktree now requests typed parameters at both row and identity producer-validation boundaries. Identity controller mocks now require explicit casting. Existing frozen bundles remain unchanged and MUST NOT be used to redispatch this task. Next action: publish corrected row freeze, update the downstream identity source binding and publish its new freeze, then update and enqueue the existing row task without creating a duplicate.

Local checks initially passed 62 with 20 dependency skips. The full manager environment exposed four mocks lacking the new keyword (78 passed, 4 failed); these mocks were corrected to enforce the real ClearML contract. Final regression receipt is `parameter-casting-regression.xml` under `/home/lbin/Desktop/rbf-paper-identity-tests-20260918.BJzcKc`. No actual training or paper-performance completion is claimed by these software checks.

## Corrected freezes and redispatch

Both freezes below are completed publication tasks, with every uploaded artifact independently fetched and SHA-256/size checked. Original freezes remain preserved.

- Row freeze: `8a482b15670e44449d98c5f22fc5afbd`; code 848338 bytes, SHA-256 `8eb9f8730a46bf167b2d7b07b3cdf8d18420de2ed30082ec8f919d43d108c6e6`. Bootstrap unchanged (`b0d6f2bb4c956a10d4b0714d8bf849356f273ca4a22b662064797adf0f653bf8`). Regression 82 passed, zero skips; XML `640e8d5f739653344c32b6605127f314b1cb4619b728e0f147ec25f997328dfe`.
- Existing row task `2486a74c9d534749b8d5834b5148ec51` was updated only to select that new source task/hash and re-enqueued on GPU4. Observed status: queued. No duplicate experiment was created.
- Identity freeze: `7a7586375f1c467b91956abf3a681d21`; code 1207564 bytes, SHA-256 `f3649c4232ac29379b3a0eb39b4965bd61d284a7a0e365e40292001da8c1547f`. Bootstrap unchanged (`bd2f914881fe3a09c59b1f02a4c538449d9e0cbd67811c648251fcde7c410c2a`). This bundle includes both typed-reading fixes and explicitly pins the new row freeze. Following that binding update, 82 tests passed with zero skips; XML `69f6a761c0f025f303ddc9d3cea0dc6235fcb843d22c98fc0466c3776a2f8854`. Updated controller tests are separately published, SHA-256 `482de057e48852de462a6f483d574ba39543107d667535559c1a2cae3e8f2354`.

All subsequent nested row/identity dispatches must use these corrected freezes. Identity training still requires completed, verified row artifacts; its software freeze is not a completed training result.

## Live producer admission check

At the next queue inspection (2026-09-18 around 09:19 UTC), the repaired row admission function was executed directly against the actual completed seed 1337 V2 producer and the actual queued row task parameters. The default uncast parameters were rejected as expected; `get_parameters(cast=True)` passed the complete producer check. The queued task source task/hash and bootstrap SHA also matched the corrected freeze. This checks the real ClearML serialization boundary, beyond the mocked software regression. It does not assert that row preparation has run.

At that inspection seed 3407 detector training remained live at vehicle iteration 4370/8040. Seed 2027 cache generation remained live and had logged both vehicle shard totals (4256 and 4248 frames) plus infrastructure shard 1 (3673 frames); no completed cache publication was yet present. The row task remained queued behind that cache job. No job was restarted.
