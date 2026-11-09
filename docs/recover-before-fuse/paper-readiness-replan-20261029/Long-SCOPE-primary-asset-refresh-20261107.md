# Long-SCOPE公开资产与协议核查：2026-11-07

本轮远端网页检索，不在Mac执行实验计算。主来源：[arXiv摘要](https://arxiv.org/abs/2604.09206)、[作者论文v1正文](https://arxiv.org/html/2604.09206v1)、[CVF论文入口](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Long-SCOPE_Fully_Sparse_Long-Range_Cooperative_3D_Perception_CVPR_2026_paper.html)。CVF搜索索引可见，直接读取返回403；不得描述为完整读取CVF页面或补充材料。

作者正文4.1将V2X-Seq范围从标准50米扩至150米，Griffin至100米，主要报告car的AP/AMOTA，并另报通信与吞吐。这与RBF当前SPD范围不能直接混排；拿到资产后仍须冻结类别、split、频率、ROI、特征及原生指标协议。正文4.2描述900基础queries、最多40额外动态queries、CAA四层。此为配置定位，不是实现或权重资格证明。

本轮未在所读作者正文找到可用实现/权重链接；GitHub限定标题检索也未确认作者仓库。该有限检索不能证明资产不存在。仅定位到arXiv v1，不声明已检查所有版本、作者主页或完整补充材料。未联系作者、未下载模型、未启动任务；J项仍未完成，不能填入文献数字冒充复现结果。

下一步：若获得可核实作者仓库、发布权重及原生评测入口，逐项来源与协议准入后再安排原生复现；当前不把相似实现或generic embedding替代Long-SCOPE。保留完整A–K目标与已知公开基线缺口。

已核对references.bib：原条目已是CVPR 2026，无需改动。主稿main.tex第255行补充150米原生ROI及必须匹配评价范围才能进入shared-input比较的限制，读回确认；全部六行aligned comparison仍pending，未填任何文献数值。仅文本编辑，未编译/计算/提交。
