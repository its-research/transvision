# 历史来源暂存与来源准入已通过

北京时间2026-11-02 16:36:51，既有单次暂存入口exit0。13份来源文件逐文件完整SHA与保存后读回通过；原归档285,917,529字节完整SHA 94d2f042d2550814136522e983a74861380b2de81f9522d0fe980f558713a2f4，gzip CRC/ISIZE及精确EOF通过；数据库1,667,805,184字节SHA 439eb4dc5e38dded28512bc986d9633e9c58274af625a6889315d0e0f4b27dc4，解码和保存后SHA通过。数据库在112 /dev/shm/rbf-unit6-historical-source-20261101/replay/sequence-0.sqlite，不删除来源。

用户要求通过后继续来源准入，已于16:40:18在112冻结e5b249d9镜像执行原qualified_query_cache_linux_entry.py --preflight-only，exit0、stderr为空：complete_original_control_and_source_gate_passed=true、Linux_runtime_passed=true、numeric_attempt_consumed=false。原route SHA8abfa79cf6b8e09d76acb32312c6c09ca71ea7fcd54b62296666d2603a10535a保持不变。原历史检查读取实际数据库只读挂载，不使用元数据占位替代字节。未新增CPU/RAM/线程限制，未禁用GPU。

远端来源收据SHA 68d4a86631e75f0037aabd41bf1e4890066eff19f53fdc05bf7e1efe6f6f636f；远端准入报告 /home/lbin/rbf-approved-V2V-unit6-control-preparation-20261101/historical-source-admission-once-20261102/terminal.json，小报告原样读回同目录historical-source-admission-terminal-on112-20261102.json。

该结果仅关闭历史来源缺口及此入口来源/运行时检查，不增加V2V 5/81科学完成计数，不代表unit6完整容量/物理GPU/独立运行全流程准入，不复用已过期探针或启动模型。

16:40主森林原归档进程39338/39347仍存活，墙钟07:28:33，保存28,968,157,184/35,727,575,040字节，仍为.part。未全SHA或234成员终验，不启动依赖它的重跑。Mac仅运维与字节传输，所有SHA/AST在112执行，不停止或重启任何现有任务。
