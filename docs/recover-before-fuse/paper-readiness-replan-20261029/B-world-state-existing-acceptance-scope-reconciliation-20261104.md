# B项：已有world-state证据范围与当前来源连接

本轮只读原报告和冻结验收源码，没有执行SHA、数组、模型或数值重算。以下是原证据声明及字面来源连接，不是本轮新增数值资格。

## 原数值范围已覆盖的内容

`v2v-full-replay-inputs-20261011/world-state-full-numeric-readback-20261012.json`声明六组合通过：train/official_test × 1337/2027/3407。其terminal-binding记录exit0、source_hashes_unchanged、54588帧、3060073行，读回报告SHA79e621d5aa6d9228a4477779c35b5d2dfe7fc032103e9c2e24492729d7aece56，来源绑定SHAee751f5932b0f3aa60a55afe5a0462be823a944812c01240ef867c32f987fa5c。

直接查看原`audit_world_state_20261012.py`确认：遍历各manifest全部帧，逐份metadata和array核对注册SHA，读取完整states/covariances；独立列向量变换XYZ、长宽顺序、heading、速度及9×9协方差雅可比，再对delta=0/0.2秒与生产physical_world比较。空行不跳过帧来源核验，计数最终对manifest总行数。它不执行原始检测器，不证明物理模型正确性，也不覆盖任意未来传播时长、回放分支/动作/输出。

原该项assert容差为ATOL/RTOL=1e-4，报告最大状态误差约9.1e-13、协方差约2.85e-14。不能将小误差数值重标为一次按1e-8新执行的验收；后续语义1e-8与因子1e-4要求保持原范围。

## 当前official_test来源的字面连接

原raw-seven-state全字节报告与10月27日current-input-model-applicability-readback的三个official_test manifest标识相同：

| seed | manifest SHA256 |
|---|---|
|1337|901ce50208291bfc3b5753484a775890e65e8fed02ee2d87bd30a5da8e3479ec|
|2027|88c4d428083adf316e62634412a5e7b608b045f3f546f7584a92249e243eefd4|
|3407|32c56e9abdad47a7bd699558ed40a574a73a8d957aa465a0aafa08494a249d80|

原world-state来源绑定与10月27日当前打包来源记录中的tracking_v2.py（c89214cf…）和prediction_features.py（770b35f0…）相同。10月19日world-state-evidence-chain-revalidation明确引用更晚数值终态，并说明不要因为更早motion requirement“本轮未查前七列”重复执行六组合比较。

因此不可把整个前七列保留或完整world-state数值比较继续当作未做；也不可仅凭旧bridge/admission的false标志要求重拟合。各旧false字段仍原样保留，其历史作用域不能静默覆盖较晚验收。当前打包大载荷、运行时代码是否仍精确匹配，仍须用已有当前全字节收据与实际生产来源逐项连接；本轮没有重新全读大载荷。

## 下一步与仍未关闭的门槛

1. 汇合原完整motion/causal/array/world-state终态、当前full-cache字节收据与实际unit6输入身份，形成可独立读回的统一来源收据；严格保留每项原容差、代码和范围。
2. 只有字节或函数实际不匹配、或原验证未覆盖当前路径时，补对应缺口；不因旧字段false重跑完整拟合/六组合变换。
3. V2V unit6四卡的来源、原环境、实际产物保存容量、新包发布授权及实际生产/六项数值验收仍独立待满足。已有5/81不能扩展成81/81。
4. 不据这些文件完成B整体、C完整科学资格、G正式指标或全A–K声明。population velocity prior并非实测速度、nominal10Hz和官方split内容重叠披露继续保留。

## 严格汇合入口准备

join_B_world_state_existing_evidence.py已准备，112源码SHA5272bc3071b0dfeb3d53d49c70ea0e86d48b4edbb468cae4deebd15034fddaa6。入口固定五份原记录的完整SHA，检查六组合唯一覆盖、原总数/终态/变换来源/1e-4范围，并将当前三种子manifest及变换源码接回原证据；输出明确保留B整体、完整81单元、论文指标和A–K均未关闭，不覆盖旧bridge字段。

112合成协议1正7拒绝通过（错误当前manifest、重复/缺失组合、变换来源变化、失败终态、原容差变化、当前全字节覆盖不足），未读取真实历史记录或缓存、未执行数值。仅在两个既有112根目录的有界查找未发现这五份原记录，不宣称全机不存在。真实五文件完整SHA与汇合尚未执行；若需新增跨机暂存，仍遵守具体内部来源传输授权门槛。
