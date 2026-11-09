# Long-SCOPE作者资产有界复核（2026-11-05）

本次仅公开网页阅读和文本记录，没有下载权重、运行模型、联系作者或在Mac计算。0056原来源流保持同一session31230。

- [作者论文arXiv摘要](https://arxiv.org/abs/2604.09206)及[原文HTML v1](https://arxiv.org/html/2604.09206v1)可访问。当前返回的正文没有检索到作者实现/权重链接；页面中GitHub链接为arXiv渲染问题反馈等通用链接，不能当模型仓库。
- [CVF原论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Long-SCOPE_Fully_Sparse_Long-Range_Cooperative_3D_Perception_CVPR_2026_paper.html)此次读取403，不能用读取失败判断其资产内容。
- [既有作者关联账号](https://github.com/wang-jh18-SVM)仍有Griffin入口；[仓库列表](https://github.com/wang-jh18-SVM?tab=repositories)含SparseCoop和Griffin，但返回列表带加载错误且搜索缓存较旧。GitHub API用户仓库/精确搜索接口本次工具不可达，未完成全量56仓库的实时清单复核；不声称作者没有公开Long-SCOPE仓库。
- 对Long-SCOPE精确标题和GitHub的有界检索没有找到可核验的作者实现、checkpoint或匹配配置。第三方论文汇总及同名无关项目不计为作者资产；[SparseCoop原仓库](https://github.com/wang-jh18-SVM/SparseCoop)不替代Long-SCOPE。

原文4.1–4.2说明原生实验扩展V2X-Seq评价距离至150m、Griffin至100m，并使用car类和四张3090。未来资产准入必须冻结原评价范围与训练配置；不能直接把其文献结果塞进本项目50m或统一输入表，也不能由此恢复用户已停用的3090。此处仅记录论文协议，未产生任何复现指标。

当前门槛仍是作者可核验实现/版本、匹配权重与配置、预处理/split/频率/ROI、原生评价入口及实际环境资格。既有作者资产请求草稿可复用，但本次没有发送消息。资产到位前J项Long-SCOPE仍明确未完成；本次检索不证明资产不存在，也不撤销其他已验方法。

## PDF与补充入口核对

[arXiv原PDF](https://arxiv.org/pdf/2604.09206)可读，共11页；本轮文本检索未找到GitHub、Appendix或Supplement指向，code命中为论文讨论基线代码，不是Long-SCOPE下载入口。CVF原PDF返回403；按惯例构造的supplemental路径本次不可访问，未取得官方页面链接佐证，不能将该候选路径当作存在的补充材料或由其不可访问推断不存在。没有下载到Mac、没有读回原生权重或运行任何基线。资产门槛保持未关闭。

## GitHub只读连接器补查

GitHub连接器的公开repository search/fetch可用，因此网页接口失败不再是本轮唯一检索依据。查询`"Long-SCOPE"`、`user:wang-jh18-SVM Long`、`"Fully Sparse" "Cooperative"`及`2604.09206 in:readme`仍未给出可核验的本方法作者实现/匹配权重。

直接读取原仓库元数据确认：[mrconter1/LongScope](https://github.com/mrconter1/LongScope)为大语言模型长上下文基准，[tsaoo/LongScope](https://github.com/tsaoo/LongScope)为示波器长期测量工具；不能据名字将其引入合作感知复现。作者关联查询命中[wang-jh18-SVM/LT3D](https://github.com/wang-jh18-SVM/LT3D)，其原README标题及作者对应另一篇长尾3D检测论文，也不能替代Long-SCOPE。SparseCoop原公开releases接口返回空列表，仅说明该endpoint无GitHub release，不证明其README/其他渠道没有权重。

arXiv编号的README检索命中论文自动汇总仓库，本轮没有将这些聚合页当作者代码。作者完整所有仓库/所有分支及其他外部托管并未穷尽；结论保持“本轮未找到可准入资产”，不宣称作者资产不存在。没有下载或执行上述仓库、没有发消息或修改任何远程仓库。
