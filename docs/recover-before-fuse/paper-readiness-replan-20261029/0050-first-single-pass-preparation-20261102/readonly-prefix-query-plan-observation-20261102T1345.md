# 0050只读查询计划诊断

2026-11-02北京时间13:43–13:45，通过112读取活进程挂载中的冻结源码及工作库schema，并仅执行EXPLAIN QUERY PLAN。没有执行查询正文、读取观测数据、创建索引或修改数据库；读取前后大小、mtime_ns、ctime_ns一致。

原冻结trajectory.py先读取组件前缀；fresh oracle.py按handle查询前缀。工作库三个原查询形状的计划如下：

- `SELECT h,parent,depth,choice,root,previous_root FROM pc1_prefixes WHERE h=?`：`SEARCH pc1_prefixes USING INTEGER PRIMARY KEY (rowid=?)`。
- `SELECT raw_bytes,payload FROM priority_probe_witnesses WHERE sha=?`：`SEARCH priority_probe_witnesses USING INDEX sqlite_autoindex_priority_probe_witnesses_1 (sha=?)`。
- `SELECT global_i FROM component_members WHERE component=? ORDER BY local_i`：`SEARCH component_members USING INDEX sqlite_autoindex_component_members_1 (component=?)`。

这些计划不支持“以上三类查询因缺少索引而退化为全表扫描”的解释，不据此给活库添加索引。仅检查pc1查询形状，不证明所有组件、所有查询和实际适配器路径都已测量；未取得当前调用栈，不是逐调用性能剖析。查询频次、重复树遍历、见证解码或数值重算仍需实际证据区分。

0050保持原入口运行。既有0087单次完整入口资格批准仍以0050终态及完整留证、重新资源准入为前提；该批准不自动授权发布新完整队列、停止旧任务或放宽数值标准。Mac只编辑本记录，未计算实验或SHA。
