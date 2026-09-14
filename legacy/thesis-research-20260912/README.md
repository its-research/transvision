# 从 thesis 迁入的研究代码

2026-09-12 从 `/Users/lbin/Desktop/Codes/thesis` 迁入 58 个研究源码和测试文件，
共 1,194,848 字节。原始字节、相对路径和文件权限保存在本目录；
`migration-manifest.json` 逐项记录 SHA256，`copies-verified.json` 绑定清单。
原路径中的这 58 个文件已移除，可按清单从本目录恢复。没有改写 Git 历史。

本目录是旧 RTP-V2X、适配器及 ClearML 工具的恢复副本，不是 Recover Before Fuse
的新方法实现。新代码在 `transvision/models/event_track_v2x/`，入口在
`tools/event_track_v2x/`，测试在 `tests/event_track_v2x/`。

论文正文、协议、历史证据、构建工具及构建依赖继续保留在 thesis。
不要在恢复副本中直接修改源码，也不要将这里的旧运行脚本直接当作已就绪的真实实验。

## 运行迁移回归

在 transvision 根目录、具备 NumPy 的 Python 环境中运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/event_track_v2x/test_migrated_thesis.py \
  --thesis-root /Users/lbin/Desktop/Codes/thesis
```

工具先校验所有迁入文件，再在 `work_dirs/legacy-thesis-tests/run-*/context/` 中生成
可执行副本。副本包含迁入代码、原论文的协议 JSON、4 个保留的共享构建依赖，以及
1 份历史合成 canary 的 4 个小型测试载荷。每次运行记录输入哈希、测试日志和结果。
它不恢复文件到 thesis，不覆盖本归档，不创建或发布 ClearML 任务。

六组测试分别在独立进程中运行；发现 0 项测试也会失败。迁移首轮实际通过 361 项，
没有跳过项。这个结果只证明旧代码在显式依赖副本中的回归，不证明真实数据集性能。

## 恢复范围

按 `migration-manifest.json` 的 `files` 逐项选择文件，先验证归档 SHA256，
只恢复到不存在的原相对路径。若原路径已有文件，应先核对差异，不覆盖现有文件。
Makefile、latexmkrc、构建脚本、历史证据脚本和生成缓存未迁走，不需要从本归档恢复。
