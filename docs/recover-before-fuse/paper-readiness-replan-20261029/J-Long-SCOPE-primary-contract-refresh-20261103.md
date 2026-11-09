# Long-SCOPE原生复现：本轮一手来源核对

来源：[作者arXiv v1](https://arxiv.org/html/2604.09206v1)、[摘要与版本](https://arxiv.org/abs/2604.09206)、[CVF登记页](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Long-SCOPE_Fully_Sparse_Long-Range_Cooperative_3D_Perception_CVPR_2026_paper.html)。CVF搜索索引可见，但本轮直接打开返回403；正文合同取作者arXiv，未把第三方排名页作为技术证据。

作者正文4.1–4.2给出car类别、V2X-Seq挑战协议，扩大到150米；ResNet50，静态900查询加最多40动态查询，CAA四层。两阶段48+24 epoch，batch16、AdamW、weight decay0.01、初始LR2e-4及cosine；V2X图像960×540，实验使用四张3090。没有将论文表格指标填入本项目实测结果。

## 对当前执行计划的影响

四卡本身不是该论文公开硬件合同的矛盾项，但论文文字不构成可运行资产。本轮针对准确标题、GitHub、code、checkpoint的有界检索和论文链接检查，仍未取得可绑定的作者代码commit、实际权重、运行环境锁及完整配置。arXiv网页的GitHub链接属于渲染问题反馈，不是作者实现。不能据此宣称全网无代码，也不联系作者或发送消息。

不能用这篇论文中重实现SparseCoop的batch16、48+24 epoch替换本项目已冻结的原SparseCoop batch48、800epoch合同；这是不同复现协议。也不能用SparseCoop/LT3D替代Long-SCOPE行。

后续准入清单：作者关联仓库与许可；冻结commit及完整源码SHA；真实checkpoint/训练来源；原配置、两阶段预算与类别/ROI/split；原生预测保存及独立评价；另列统一输入/同资源对照。上述资产未齐前维持J未完成，不占卡启动猜测实现、不填文献数字。

本轮仅网页核对和外置盘文本记录，无Mac数值/SHA计算，无文件发布或新Task，无活任务修改。
