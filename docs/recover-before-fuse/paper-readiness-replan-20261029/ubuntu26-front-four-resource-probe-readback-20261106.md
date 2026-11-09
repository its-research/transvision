# ubuntu26-2前四卡资源探针完整读回

Task f88274e7f6e64b69872dd467f380a6e0于14:06:04接取，实际只读检查14:12:26.752672至14:12:27.032150 +08，随后completed。前四卡UUID已记录；四张2080Ti各11264MiB，实测各1MiB占用、0%利用率，无计算进程。MemAvailable=47,257,890,816字节，原生ext4缓存盘空闲5,594,208,509,952字节，cgroup cpu.max为max 100000，memory.max为max；后四卡仍有其他SAM3任务，不能扩展八卡占用。

14:15:36在112通过SDK认证和专用可写缓存完整读回3945字节注册报告，SHA a1b5184bf0d4d77e99bd02df90a6f17ae813a3d2abc85a7dec153285fc58458f与平台一致、注册前后稳定。直接未认证HTTP401和旧共享SDK缓存PermissionError均保留；未修改共享权限或重跑探针。

112证据目录/home/lbin/rbf-clearml-idle-dispatch-20261106。资源样本仅为该时刻观察；原生可写性仅os.access，未实际写测试。full_storage_route_qualified仍false，不等于V2V单元7完整输出容量、模型运行或SparseCoop四卡资格通过。Mac不计算，现有任务不改动。
