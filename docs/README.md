# 文档导航

整理日期为 2026-11-09。本页按项目和用途组织现有文档。

## 项目入口

| 项目 | 入口 | 阅读内容 |
| --- | --- | --- |
| Recover Before Fuse / EventTrackV2X | [实验文档导航](recover-before-fuse/README.md) | A–K 计划、任务交接、独立验收和论文证据 |
| Resilient V2X | [复现说明](resilient_v2x/reproduction.md) | 既有检测项目的复现流程 |
| Resilient V2X | [论文覆盖](resilient_v2x/paper-coverage.md)、[核心 API](resilient_v2x/core-api.md) | 该项目的论文与实现映射 |

仓库根 README 介绍 Resilient V2X。Recover Before Fuse 的完成状态与证据从其专用入口查询。

## 通用使用资料

| 用途 | 文档 |
| --- | --- |
| 开始使用 | [get_started.md](get_started.md) |
| 数据准备 | [dataset_prepare.md](dataset_prepare.md) |
| 数据说明 | [中文](dataset_info_zh-CN.md)、[英文](dataset_info.md) |
| 坐标约定 | [coord.md](coord.md) |
| 环境兼容 | [compatibility.md](compatibility.md) |
| ClearML | [clearml-local.md](clearml-local.md) |
| 模型与基准 | [BEVFusion](bevfusion.md)、[benchmark.md](benchmark.md)、[BEVFormer 资料](bevformer_docs/) |
| 已知问题 | [issues.md](issues.md) |

## 维护方式

- 项目说明与跨任务约定放在项目入口附近，具体尝试的日志、收据、脚本和失败证据留在原尝试目录。
- 新的阶段性材料从项目入口链接，优先引用原文件，避免复制出多份执行计划。
- 文件名中的日期表示记录所属时间。`current` 和 `latest` 文件也须检查内部时间与来源。
- 带 SHA、绝对路径或任务身份绑定的证据文件保持原路径。移动前须核对所有读取方及运行任务依赖。
