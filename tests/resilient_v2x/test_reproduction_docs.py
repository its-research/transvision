from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROOT_README = ROOT / "README.md"
REPRODUCTION = ROOT / "docs" / "resilient_v2x" / "reproduction.md"
COVERAGE = ROOT / "docs" / "resilient_v2x" / "paper-coverage.md"
CORE_API = ROOT / "docs" / "resilient_v2x" / "core-api.md"
CONFIG_README = ROOT / "configs" / "resilient_v2x" / "README.md"

MAIN_CONDITION_CONFIGS = (
    "conditions/global_delay_000_full.py",
    "conditions/global_delay_100_full.py",
    "conditions/global_delay_200_full.py",
    "conditions/global_delay_300_full.py",
    "conditions/causal_delay_000_l_fail.py",
    "conditions/causal_delay_000_c_fail.py",
    "conditions/causal_delay_100_l_fail.py",
    "conditions/causal_delay_100_c_fail.py",
    "conditions/causal_delay_200_l_fail.py",
    "conditions/causal_delay_200_c_fail.py",
    "conditions/causal_delay_300_l_fail.py",
    "conditions/causal_delay_300_c_fail.py",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reproduction_uses_real_cli_entrypoints_and_required_protocol_flags() -> None:
    document = _read(REPRODUCTION)
    entrypoints = (
        "python tools/resilient_v2x/check_environment.py",
        "python tools/resilient_v2x/capture_environment.py",
        "python tools/resilient_v2x/prepare_data.py",
        "python tools/resilient_v2x/build_overlays.py cohort",
        "python tools/resilient_v2x/build_overlays.py train",
        "python tools/resilient_v2x/build_overlays.py evaluation",
        "python tools/train.py",
        "python tools/test.py",
        "python tools/resilient_v2x/profile.py",
        "python tools/resilient_v2x/build_evidence.py",
    )
    for entrypoint in entrypoints:
        assert entrypoint in document

    required_flags = (
        "--mode controlled",
        "--classification controlled",
        "--expected-split-sha256",
        "--protocol-scope controlled",
        "--delta-t-ms 100",
        "--history-limit 3",
        "--max-capture-skew-ms 75",
        "--split val",
        "--max-delay-ms 300",
        "--max-duration 1",
        "--max-duration 4",
        "--protocol-seed 20250218",
        "--p-lidar 0.2",
        "--p-camera 0.2",
        "--delays 0 100 200 300",
        "--conditions Full L-Fail C-Fail",
        "--agents E+R E-only R-only",
        "--run-id",
        "--metrics",
        "--profile",
        "--predictions",
        "--conditions",
        "--artifact",
    )
    for flag in required_flags:
        assert flag in document
    assert "validation_cohort.json" in document
    assert "validation_duration_cohort.json" in document
    assert "test_cohort.json" not in document
    assert (
        'export RESILIENT_V2X_TEST_TRANSPORT_DELAY_300_OVERLAY="$DAIR_ARTIFACT_ROOT/val_transport_delay_300.jsonl.zst"'
        in document
    )
    assert (
        'export RESILIENT_V2X_TEST_CAUSAL_DELAY_300_L_FAIL_OVERLAY="$DAIR_ARTIFACT_ROOT/val_causal_delay_300_l_fail.jsonl.zst"'
        in document
    )
    assert "$DAIR_ARTIFACT_ROOT/test_transport_delay_300.jsonl.zst" not in document
    assert "$DAIR_ARTIFACT_ROOT/test_causal_delay_300_l_fail.jsonl.zst" not in document


def test_reproduction_separates_main_and_continuous_fault_cohorts() -> None:
    document = _read(REPRODUCTION)
    assert (
        "--max-delay-ms 300 \\\n  --max-duration 1 \\\n"
        '  --out "$DAIR_ARTIFACT_ROOT/validation_cohort.json"'
    ) in document
    assert (
        "--max-delay-ms 0 \\\n  --max-duration 4 \\\n"
        '  --out "$DAIR_ARTIFACT_ROOT/validation_duration_cohort.json"'
    ) in document

    continuous = re.search(
        r"for duration in 2 3 4; do\n(?P<body>.*?)\ndone",
        document,
        flags=re.DOTALL,
    )
    assert continuous is not None
    body = continuous.group("body")
    assert '--cohort "$DAIR_ARTIFACT_ROOT/validation_duration_cohort.json"' in body
    assert "--delays 0 \\" in body
    assert "--delays 0 100 200 300" not in body
    assert "论文没有规定持续故障与非零时延的笛卡尔积" in document
    assert "max_delay_ms / delta_t_ms + duration - 1 <= history_limit" in document


def test_reproduction_exports_repository_root_before_training() -> None:
    document = _read(REPRODUCTION)
    root_export = 'export RESILIENT_V2X_REPO_ROOT="$(pwd)"'
    pythonpath_export = (
        'export PYTHONPATH="$RESILIENT_V2X_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"'
    )
    assert root_export in document
    assert pythonpath_export in document
    assert document.index(root_export) < document.index("python tools/train.py")
    assert document.index(pythonpath_export) < document.index("python tools/train.py")


def test_multiline_bash_options_have_command_continuations() -> None:
    document = _read(REPRODUCTION)
    bash_blocks = re.findall(r"```bash\n(.*?)\n```", document, flags=re.DOTALL)
    assert bash_blocks
    for block in bash_blocks:
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if line.lstrip().startswith("--"):
                assert index > 0
                assert lines[index - 1].rstrip().endswith("\\"), (
                    f"missing continuation before: {line}"
                )


def test_reproduction_lists_exactly_the_twelve_main_condition_configs() -> None:
    document = _read(REPRODUCTION)
    for config in MAIN_CONDITION_CONFIGS:
        assert document.count(f"`{config}`") == 1
        assert (ROOT / "configs" / "resilient_v2x" / config).is_file()


def test_paper_coverage_references_only_existing_repository_paths() -> None:
    document = _read(COVERAGE)
    referenced_paths = set(
        re.findall(
            r"`((?:configs|docs|environments|tests|tools|transvision)/"
            r"[^`\n]+)`",
            document,
        )
    )
    assert referenced_paths
    missing = sorted(path for path in referenced_paths if not (ROOT / path).exists())
    assert missing == []


def test_paper_coverage_states_choices_dataset_boundary_and_no_result_claim() -> None:
    document = _read(COVERAGE)
    required_boundaries = (
        "论文未给出的 implementation choices",
        "Pair 离线序列适配=C、Standard source-trace 导出=C",
        "Standard 方法参评状态=N/A",
        "不能声称已复现任何 V2XSet 数值",
        "当前不能声明",
        "论文任一 AP、PDR、FLOPs、显存、时延或消融差值已复现",
        "当前工作区没有可声明的训练 checkpoint",
        "至少三个已声明 seed",
        "endpoint-relative observed/propagated availability",
        "只有真正当前 Ego reliability 为 1",
    )
    for boundary in required_boundaries:
        assert boundary in document

    for config in MAIN_CONDITION_CONFIGS:
        full_path = f"configs/resilient_v2x/{config}"
        assert document.count(f"`{full_path}`") == 1


def test_core_api_uses_endpoint_relative_flags_and_interval_units() -> None:
    document = _read(CORE_API)
    assert "endpoint_tick: int | None" in document
    assert "observed delayed RSU endpoint" in document
    assert "Only a truly\ncurrent Ego source has reliability one" in document
    assert "branch source ages `h=(t-s*)/Delta t` in sampling intervals" in document
    assert "normalized branch ages" not in document


def test_docs_do_not_use_placeholder_digest_or_fake_result_language() -> None:
    combined = _read(REPRODUCTION) + "\n" + _read(COVERAGE)
    assert re.search(r"\b0{64}\b", combined) is None
    assert "fixture 结果可作为论文结果" not in combined
    assert "开发环境输出可作为正式证据" not in combined
    assert "未训练权重可作为论文结果" not in combined


def test_config_readme_links_both_long_form_documents() -> None:
    readme = _read(CONFIG_README)
    assert "../../docs/resilient_v2x/reproduction.md" in readme
    assert "../../docs/resilient_v2x/paper-coverage.md" in readme


def test_root_readme_lists_all_paper_result_tables_without_claiming_results() -> None:
    readme = _read(ROOT_README)
    table_labels = (
        "tab:normal_comparison",
        "tab:latency",
        "tab:modality_missing",
        "tab:joint_degradation",
        "tab:ablation",
        "tab:sensitivity_p",
        "tab:duration_complexity",
    )
    for label in table_labels:
        assert readme.count(f"`{label}`") == 1

    evidence_boundaries = (
        "没有任何满足提交条件的受控实测结果",
        "—（待测）",
        "数值†",
        "数值‡",
        "至少 3 个随机种子",
        "受控主故障协议中的 L-Fail 和 C-Fail 均为 `E+R`",
        "E-only / R-only agent-scope 诊断",
        "尚未创建这 16 个诊断 ID",
        "没有注册任何 V2XSet result ID",
        "提交门禁补充结果",
        "69.33‡",
        "同协议方法差值",
        "V2XSet-Standard",
        "V2XSet-Pair",
        "旧失败检查点及其 AP 均不满足这些门槛",
    )
    for boundary in evidence_boundaries:
        assert boundary in readme

    assert (
        "| V2XSet-Standard | 原生 multi-agent、LiDAR-only | "
        "—（待测：需 adapter） | —（补充待测：需 adapter） | "
        "—（待测：需 adapter） | —（待测：需 adapter） | N/A | N/A |"
    ) in readme


def test_root_readme_separates_published_non_comparable_baselines() -> None:
    readme = _read(ROOT_README)
    required = (
        "公开论文的外部对照（不可回填受控主表）",
        "How2comm / DAIR-V2X 默认噪声",
        "100 ms",
        "0.2 m /\n0.2°",
        "1 MB",
        "| V2X-ViT | LiDAR | How2comm / DAIR-V2X 默认噪声 | 51.68 | 39.97 |",
        "| CoBEVT | LiDAR | How2comm / DAIR-V2X 默认噪声 | 56.08 | 41.45 |",
        "mAP/NDS = 35.56/41.21；64.68/69.28；68.52/71.38",
        "这里的两个 BEVFusion 是不同论文",
        "固定引用包含完整 `0/200/300 ms` 序列的表 1",
    )
    for statement in required:
        assert statement in readme


def test_root_readme_registers_controlled_baseline_implementations() -> None:
    readme = _read(ROOT_README)
    required = (
        "受控适配实现状态",
        "ResilientTemporalDataset",
        "模型不包含本文方法的 PTF、repair、DER、teacher 或 distillation 路径",
        "configs/resilient_v2x/baselines/v2x_vit.py",
        "configs/resilient_v2x/baselines/cobevt.py",
        "configs/resilient_v2x/baselines/coformernet.py",
        "configs/resilient_v2x/baselines/bevfusion.py",
        "configs/resilient_v2x/baselines/ffnet.py",
        "transvision/models/resilient_v2x/baseline_inputs.py",
        "transvision/models/detectors/controlled_v2x_baseline.py",
        "bit-exact reproduction",
        "| V2X-ViT-style | 本仓受控适配已实现",
        "| CoBEVT-style | 本仓受控适配已实现",
        "| CoFormerNet-style | 本仓受控适配已实现",
        "| MIT-HAN BEVFusion-style | 本仓受控适配已实现",
        "| FFNet-style | 本仓受控适配已实现",
    )
    for statement in required:
        assert statement in readme
