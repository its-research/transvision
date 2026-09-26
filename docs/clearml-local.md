# 本机 ClearML 直连环境

2026-09-20 已在 `Lis-Work-MacBook-Pro.local` 配置并验证。后续在这台 Mac 上查询任务、读取 Artifact 或运行提交脚本，使用 `clearml-local` 入口；连接直接到 ClearML 服务，112 不参与客户端操作。

## 使用

```bash
# 只读检查连接、认证、任务、队列和 Worker
clearml-local check

# 在独立 SDK 环境运行脚本，当前工作目录保持不变
clearml-local python path/to/script.py [参数]

# 在当前终端启用相同环境
source /Users/lbin/.local/share/clearml/env.sh
```

`clearml-local python` 支持 Python 原有的 `-c`、`-m` 和标准输入方式。项目脚本如需额外的数据处理或训练依赖，按对应项目环境安装；本环境提供 ClearML 管理与提交客户端，GPU 训练由 ClearML Worker 执行。

## 本机配置

| 项目 | 值 |
| --- | --- |
| Python | 3.12.12 |
| ClearML SDK | 2.1.3 |
| 独立环境 | `/Users/lbin/.local/share/clearml/venv` |
| 启动入口 | `/Users/lbin/.local/bin/clearml-local` |
| API | `http://10.100.35.118:8008` |
| Web | `http://10.100.35.118:8080` |
| 文件服务 | `http://10.100.35.118:8081` |
| 凭据配置 | `/Users/lbin/clearml.conf`，权限 `600` |
| 下载缓存 | `/Users/lbin/.cache/clearml` |
| 已安装依赖锁定记录 | `/Users/lbin/.local/share/clearml/requirements.lock` |
| 只读检查脚本 | `/Users/lbin/.local/share/clearml/check_connection.py` |
| 本次验证回执 | `/Users/lbin/.local/share/clearml/verification-20260920.json` |

启动入口显式设置 `CLEARML_CONFIG_FILE`、三个服务地址及缓存目录，并把 `10.100.35.118` 加入 `NO_PROXY` 和 `no_proxy`。配置使用现有内网服务地址，凭据只存放在用户目录，不写入项目或 OneDrive。

连接配置遵循 [ClearML 官方 SDK 配置流程](https://clear.ml/docs/latest/docs/clearml_sdk/clearml_sdk_setup/)。

## 验证结果

- 本机地址 `10.100.16.24` 与 `10.100.35.118` 的 8008、8080、8081 端口直接建立连接。
- SDK 认证、任务查询、队列查询及 Worker 查询均通过。
- 读取已有任务 `98d72fb5afc94b418aa0836f927fa7d4` 的 `catalog` Artifact，带认证 HTTP 读取和 SDK 原生下载均通过。
- 文件大小 232,968 bytes，SHA256 为 `f6d8fe076a0224b7a738f0aa8b5f86034304c86d20543a5bdcdc77dd4f30aaea`，与已有归档记录一致。
- 独立环境 23 个包的依赖一致性检查通过，启动脚本语法检查通过。

复验指定 Artifact：

```bash
clearml-local check \
  --task 98d72fb5afc94b418aa0836f927fa7d4 \
  --artifact catalog \
  --sha256 f6d8fe076a0224b7a738f0aa8b5f86034304c86d20543a5bdcdc77dd4f30aaea
```

本次验证读取现有实验及小型 Artifact；未创建、入队或修改训练任务，上传与任务提交未做写入测试。检查脚本将单个验证 Artifact 限制在 5 MiB，并只向配置中的文件服务发送认证信息。
