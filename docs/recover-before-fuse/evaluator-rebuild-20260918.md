# Independent evaluator reconstruction

The historical temporary evaluator and research Python paths no longer exist locally. Manager conda environments list only base, py37 and py38; its base research test environment lacks independent evaluation dependencies.

Created `/private/tmp/rbf-evaluation-20260918.F5p07w` using `/opt/homebrew/bin/python3.10 -m venv`, then installed the unchanged `environments/event_track_v2x/requirements-evaluation-macos-arm64-py310.lock` with `pip install --require-hashes`. `pip check` passed. Training environments were not changed.

Using this independent interpreter, `tests/event_track_v2x/test_paper_evaluation.py`: **12 passed, zero skips**, 60.67 seconds. Receipt: `/private/tmp/rbf-evaluation-20260918.F5p07w/paper-evaluation.xml`. These are synthetic metric/evaluation tests, not real dataset performance.

An attempted resource-scan test invocation in that pure evaluator stopped during collection because its fixture helper imports Torch. Failure receipt is preserved as `paper-resource-scan.xml` in the same directory; it is not a successful resource scan. Both XML files were copied to `/home/lbin/Desktop/rbf-priority-seeds-tests-20260918.CKpe6p` for subsequent ClearML archival. Remaining work: provide a separate compatible research interpreter to run the resource scan while pointing `RBF_EVALUATOR_PYTHON` at the evaluator; do not add research Torch to the independent evaluator merely to bypass this separation.

## Research/evaluator separation restored

Created `/private/tmp/rbf-research-20260918.S9JLFw` with `/Users/lbin/.local/bin/python3.12 -m venv` and installed the version-pinned `requirements-research-tests.txt` (its transitive dependencies are not a hash lock). Running `test_paper_resource_scan.py` through this research interpreter with `RBF_EVALUATOR_PYTHON=/private/tmp/rbf-evaluation-20260918.F5p07w/bin/python` produced **5 passed, zero skips**, 28.91 s, including both previously skipped fresh-process cases. Success XML resides under the research environment and was copied separately as `paper-resource-scan-success.xml` on the manager; the earlier failure is retained.

The initial evaluator success and resource collection failure are archived in completed ClearML task `f80cad20506d48698f81331c574cc6d0`, both fully read back. Evaluation XML SHA `b79f14a70428f2012391dcacca088142dee6966eb0ecd7aa9bb797a0909428d0`; failure XML SHA `6b8396a249180c85dc7c3fbb8af978f35ad8939b1a4a1eb6c75abb2b88b04259`. These receipts and the successful resource scan are synthetic software evidence, not real resource/performance experiments.

The successful five-test resource regression is archived in completed ClearML task `de9f0813b22b4356888e6743e230b753`. Independent authenticated readback on 2026-09-18 verified 1032 bytes and SHA-256 `aaca71823bd08218da05b02e73dea0932768cb235cdd37c56cd565abcaec4bf1`.

## Sealed engine conformance rechecked

Ran `tools/event_track_v2x/verify_evaluation_runtime_v2.py --output /private/tmp/rbf-evaluation-20260918.F5p07w/sealed-selftest` using the independent evaluator Python, `PYTHONDONTWRITEBYTECODE=1`, and `MPLBACKEND=Agg`. The sealed source/version checks, three original golden cases and four extended analytic cases passed. No real dataset or test payload was read. The optional upstream reference directory was not supplied: installed fingerprints were checked, but upstream reference bytes were not independently re-fetched or rechecked.

All five JSON outputs were uploaded to completed ClearML task `5884cef390384eccbb15332a810f414f` and independently read back byte-for-byte:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| plan | 542 | `245874ea2b191d46dba3c2c4acc68bdc3034b6abe31197a9c2d72013fa6baedb` |
| runtime | 27207 | `ab10f1998f0a143ec353e6078021a7cd2a87d9159a3d42589f4aeb7e240db410` |
| golden-cases | 3252 | `d5a2faeec6c72efc8aa161cca3f750eb100b47e2b3a4bac018d5215facb858b1` |
| extended-cases | 5277 | `49725ce73cdab58c5437a4f54f0de67ea78c7ea10af12d2a9cb17b775aa63f5a` |
| receipt | 666 | `ba1377848cd6ad4bd44cc9a5d6dc4ee87ee4a5579d35e56484b3291b20088a4d` |

This is evaluation-engine software conformance only; `paper_eligible=false` and `real_tracking_method_result=false` remain explicit.
