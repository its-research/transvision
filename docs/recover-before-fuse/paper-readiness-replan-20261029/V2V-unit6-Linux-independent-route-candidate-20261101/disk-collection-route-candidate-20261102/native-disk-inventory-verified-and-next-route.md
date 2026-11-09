# ubuntu26-2原生磁盘实测与后续路线

只读Task0d16e98cd819465bb6c2e76e8e5fdcf3完成。实际采样2026-11-02 22:53:34–22:53:35北京时间；平台启动22:47:22，前约6分钟为容器和依赖准备，并非数值计算。报告3945字节，SHA bbcf91cb878feeb48cda79ff0e28539c1eed6b0c9175532cf5ae84207f27e756，112完整HTTP读回和Mac保存副本反向SHA均通过。

容器原生ext4挂载/clearml_agent_cache，设备252:1、挂载root /.clearml/cache，空闲5624844779520字节；mount table前后稳定、os.access可写。未实际写测试文件。另系统盘各缓存仅约68.42GB空闲，不能作为大数据库完整收集空间。RAM MemAvailable57826258944字节，cgroup memory.max=max，cpu.max=max 100000，affinity0–63；不代表64核已使用。

前四卡UUID：cb25f779-7739-80fd-839f-518b96d9fdce、9223f2c3-17fd-8061-278c-abafe4c8a1de、1642eeac-6fb2-2254-4570-96e99452c91c、d19154bd-5c8e-8bbb-f9e3-4692c8fbe550；采样时均RTX2080Ti11264MiB，无GPU计算进程。

下一道完整来源准入候选已在112准备：新Task入口7249字节，SHA c019d283b9106e0918f61c5f5b6064b3ffd88298b6911a137e7d382f02fd8b3a。原生缓存盘独立目录拟为宿主/root/.clearml/cache/rbf-v2v-unit6-source-admission-20261102，绑定到容器原/Volumes/Data/test/recover-before-fuse。保留原镜像和其余容器参数。先验证实际绑定原生磁盘与456649250158字节空闲下限，再下载已经发布37,130,240字节63成员包及原285,917,529字节历史归档；逐字节完整SHA、仅解出原1,667,805,184字节数据库及gzip全CRC，原完整来源/运行时/磁盘入口通过后封存报告。所有资料和失败保留，不修改现有任务，不执行模型、事件或六项数值，不消耗单元6生产/独立次数。

API新Task源码发布及数据库暂存的一次具体授权已向用户提出，尚未执行。不要用此前只读库存许可（明确不含数据库下载）代替。实际产物C/E仍须生产终态完整核定，当前不写full_route_qualified=true，不放宽原容量公式。
