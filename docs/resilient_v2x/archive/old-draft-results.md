# ResilientV2X Old-Draft Results Archive

| 字段 | 值 |
| :--- | :--- |
| 状态 | Archived / unverified |
| 首次 Git 记录时间 | 2026-07-31 13:13:54 UTC+8 |
| 首次记录提交 | `82981b7a512736108de4d6ab15b168018874a371` |
| 实际实验时间 | 未知 |
| 原始证据 | 无对应 ClearML task、checkpoint、日志或原始预测文件 |

## 正常输入

| 方法 | 模态 | BEV AP@0.5 | BEV AP@0.7 |
| :--- | :---: | ---: | ---: |
| Resilient V2X | LiDAR + Camera | 71.5 | 58.1 |

## RSU 时延

| 方法 | 0 ms | 200 ms | 300 ms | PDR |
| :--- | ---: | ---: | ---: | ---: |
| Resilient V2X | 58.1 | 57.7 | 57.3 | 1.4 |

## 单模态故障

结果格式：`BEV AP@0.5 / BEV AP@0.7（基于 AP@0.7 的 PDR）`。

| Normal | L-Fail | C-Fail |
| :---: | :---: | :---: |
| 71.5 / 58.1 | 45.8 / 28.6（↓50.8） | 68.7 / 54.8（↓5.7） |

## 模态故障与时延联合退化

| 条件 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | ---: | ---: | ---: | ---: |
| Full | 58.1 | — | 57.7 | 57.3 |
| L-Fail | 28.6 | — | — | — |
| C-Fail | 54.8 | — | — | — |

## 模块消融

| Exp. | 配置 | Full | L-Fail | 300 ms |
| :---: | :--- | ---: | ---: | ---: |
| A | Concat fusion；无 PTF、无蒸馏 | 53.0 | 10.1 | 32.5 |
| B | DER；无 PTF、无蒸馏 | 54.7 | 24.3 | 33.2 |
| C | DER + 线性传播；无蒸馏 | 56.3 | 25.9 | 50.0 |
| D | DER + trajectory-field propagation；无蒸馏 | 57.0 | 27.5 | 55.0 |
| E | DER + trajectory-field propagation + consistency distillation | 58.1 | 28.6 | 57.3 |

## 训练故障概率敏感性

| `p_L = p_C` | Full | L-Fail | 300 ms |
| ---: | ---: | ---: | ---: |
| 0.0 | 57.4 | 24.6 | 53.1 |
| 0.1 | 57.7 | 26.5 | 54.8 |
| 0.3 | 58.1 | 28.6 | 57.3 |
| 0.5 | 56.9 | 28.8 | 56.9 |
