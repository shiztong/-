# 22 逐模型证据审计：MF、YoutubeDNN、SDM、MIND、ComiRec、SIM、EGES、SE、SSL、X-Rec

核验日：2026-11-03。本章对用户给定清单逐行纠错，优先使用作者源码、作者演示材料及明确标注的框架复现；不能取得原文数值的地方留空，不补“行业经验提升”。来源固定提交和 SHA-256 见 [model-audit-sources.json](evidence/model-audit-sources.json)。本轮没有训练这些模型或复现线上实验。

## 1. 先给结论：原清单不能直接作为选型表

- **ComiRec 不等于 Attention 蒸馏**：有 DR 和 SA 两种兴趣抽取器，作者公开实现使用 sampled softmax，不以教师蒸馏为其定义。
- **SIM 不应作为通用多向量 ANN 召回主干列入同一排名**：其对应研究是面向 CTR 的超长历史兴趣建模，内部历史检索与全库内容召回是不同层级。
- **MIND 不应一律写固定有效 K**：物理输出上限与每用户有效兴趣数要区分，已读实现存在动态 K/mask。
- **QPS×K 是错误或至少歧义很大的成本表达**：增加的是每请求 query 向量数与检索工作量，不是服务吞吐自动乘 K。
- **X-Rec 需要唯一标识**：已找到的 HKUDS/XRec 是可解释推荐，不是“动态生成兴趣数量”的召回模型。
- **SE 与 SSL 不应和召回主干混成同一代际**：一个是可插拔结构组件，一个是训练方法家族；必须指定依附的主干与具体实现。

## 2. 纠正后的模型对照表

证据标记：**A**=本轮读到作者/组织实现或作者材料；**B**=读到具名框架复现；**P**=原论文定位明确但本轮未取得原文/原厂代码，性能数值待核验；**U**=名称无法唯一确认。A 不表示作者公开了生产全部代码，B 不代表原论文精确复现。

| 具体模型 | 正确角色与输出 | 真实机制 | 应纠正的描述 | 证据与收益状态 |
|---|---|---|---|---|
| 纯 MF | 潜在因子协同基线；用户/内容各一个向量 | `puᵀqi`，损失可用评分回归、隐式加权或 BPR 等不同版本 | 纯 ID-MF 不使用属性；“所有 MF 都不能用特征”错误，不能混同特征化扩展 | 数学基线；未指定数据、损失和实现，不填统一收益 |
| DeepMatch `YoutubeDNN` | 单用户向量 + 内容 ID embedding 的候选生成复现 | 用户特征/历史聚合→MLP，内容 lookup，sampled softmax | 不是 Google 生产代码；不是必然对称的两个深 MLP；“稳定易上线”需要实测 | B，[S1]；代码可定位，无可搬用的统一收益 |
| SDM | 单向量序列 matching | 短期 LSTM/多头注意力、长期注意力、门控融合 | 不能把“多头”当“多向量检索”；公开 JD 代码≠正式线上版本 | A+B，[S2][S3]；作者 slides 有离线与线上报告，见第 5 节 |
| MIND | 有上限的多兴趣向量 | Capsule routing + label-aware readout；实现可有动态有效兴趣数 | 路由本身不保证顺序建模；训练目标感知不等于线上知道待推荐内容 | A+B，[S4][S5]；未取得可信性能原表，不填收益 |
| ComiRec-DR | 固定配置 K 个兴趣向量 | Dynamic Routing；目标相关 hard readout；sampled softmax | 不是 Attention 蒸馏；不保证普遍更稳定 | A，[S6]；作者代码与评估风险可核验 |
| ComiRec-SA | 固定配置 K 个兴趣向量 | 多组 attention pooling；公开实现支持位置 embedding | 不是任意 Transformer，也不是定义上的蒸馏模型 | A，[S6]；可控多样性还依赖聚合阶段，不只依赖 SA |
| SIM | 长历史 CTR/精排兴趣模块；不是固定定义的全库 ANN 召回塔 | General Search 筛历史，Exact Search 做候选相关交互 | “单/多向量召回主干”“低活专项通道”没有由此自动成立的证据 | P；原文链接 [P1]，未取得可信原表，第三方实现不充当原厂证据 |
| EGES | 物品图 embedding，可服务 item-to-item 通道 | 行为构图/游走上下文，融合物品与侧信息 embedding | 用户会话顺序可能进入构图；不能写成完全无时序信号；不能未经实验断言新节点一定差 | P；原文 [P2]，本轮未取得原文数值，不用第三方改版数据替代 |
| SE feature reweighting | 字段/通道重加权组件，通常保持输入形状 | squeeze→小型门控网络→字段缩放 | 不构成独立召回主干；“任意塔、开销极低、一定有效”均需限定 | A/项目实现，[S7]；以 FiBiNET 特征重权为具体例子，不引用未读收益 |
| SGL | 具体的图协同 + SSL 模型；导出用户/物品向量 | 图视图增强 + 推荐损失 + 对比损失 | “SSL”本身不是具体模型；不代表所有底座同样获益 | A，[S8]；公开配置可核验，未重跑效果 |
| `X-Rec`（用户所指） | 未确认 | 未确认 | 暂停“动态生成数量、推理极高”等断言 | U；等待具体论文标题/链接 |
| HKUDS `XRec` | LLM 可解释推荐框架 | 协同关系与 LLM 语义，经 MoE adapter 生成解释 | 不是上一行所述动态兴趣召回 | A，[S9]；不能移植其解释评价作为召回收益 |

这里将纯 MF、框架复现、原论文模型和泛化范式分层列出，避免拿不同层次对象做一个性能排序。

## 3. 模型细节：可直接对照代码的差异

### 3.1 YoutubeDNN：单向量不等于用户只能有一种兴趣

[S1] 的用户输入可包含离散、连续和历史字段，经用户 DNN 得到一个检索向量。内容侧代码限制为一个 item 字段，查 embedding 表并归一化；最后使用 SampledSoftmaxLayer。

单向量是检索表达瓶颈，不是模型只读取一个兴趣。它可以把多种偏好压缩进同一个空间，但所有候选最终通过一个方向打分。均值池化是常见历史聚合选择，具体由特征配置决定，不能拿一个极简 `mean(history)` 与完整 SDM 对比后把全部增益归于时序模型。

### 3.2 SDM：长短期分支最终融合成一个向量

作者 slides 的短期序列使用 `session id + time`，长期示例窗口是过去一周；这只是该研究配置，不是迁移到视频流的强制定义。短期经循环网络和多头 Self-Attention，长期按侧信息做注意力，再融合：

\[
g=\sigma(W[h_S;h_L;e_u]+b),\qquad h_u=g\odot h_S+(1-g)\odot h_L.
\]

多头帮助捕捉短会话内多种倾向，但最终 `hu` 仍是一个向量。线上由单向量作近邻检索，不能将 SDM 的头数直接计入检索 query 数。

DeepMatch 复现默认 `units=64,rnn_layers=2,num_head=4`，并声明该实现不支持 dense feature、内容侧只支持一个字段。这是**该复现的接口限制**，不能推广成论文或所有 SDM 都不能使用连续特征。作者材料也讨论 `d=64` 时四头的实验结果，但不能当作所有场景最优参数。

### 3.3 MIND：路由与有效兴趣数是两件事

历史 embedding 经胶囊路由产生多兴趣。在训练中使用目标内容进行 label-aware readout；在线不能用未知正例提前选择最佳兴趣，应导出有效兴趣并检索。

DeepMatch 的具体实现具有 `dynamic_k` 开关：

\[
K_u=\left\lfloor\max(1,\min(K_{max},\log_2(1+L_u)))\right\rfloor.
\]

物理张量仍可为 `[B,Kmax,d]`，通过 mask 区分有效头。该公式来自所读复现，不能当作所有版本必须采用的原厂公式。Alibaba EasyRec 的实现也暴露 `num_high_capsules` 和 `user_emb_num`，因此“所有用户严格固定 K 个有效兴趣”不准确。

不加入位置、时间或专门序列模块时，动态路由主要对行为集合聚合，不应直接称为捕捉了完整行为顺序。

### 3.4 ComiRec：两种抽取器 + 聚合控制，不是蒸馏

DR 用动态路由得到 `H:[B,K,d]`；SA 的公开实现：

\[
A=softmax_L(W_2\tanh(W_1(E+P))),\qquad H=AE.
\]

注意力权重使用可选位置表征 `P`，加权的 value 仍是历史 item embedding `E`。代码中没有把“知识蒸馏”定义为这一模型的必要步骤。

作者实现对正例计算各兴趣与正例的相似度，取最佳兴趣作为 sampled-softmax 输入；服务时输出全部兴趣。默认 `embedding_dim=64, num_interest=4, topN=50`。这是具体代码默认值，不是效果最优结论。

“可控多样性”还包括候选聚合：代码支持在候选质量与已选类目数量之间调整系数。只换 SA 不配置聚合策略，不能自动宣称实现了同样的多样性控制。

### 3.5 SE 与 SSL：要说明插在哪里、训练什么

SE 示例输入为 `F` 个同维字段 embedding `e_f∈R^d`。先在维度内 squeeze，门控产生每字段缩放权重，再缩放原 embedding。输出仍为 `F×d`，但字段交叉、不同字段维度和归一化方式影响迁移。

若门控是 `F→F/r→F`，仅两层矩阵参数约为 `2F²/r`，另有偏置与缩放计算。它可能很小，但仍需在具体塔和硬件上测量，不能用“开销极低”替代数据。

SSL 应指定具体增强与损失。例如 SGL 在图上构造视图，对齐同一用户/内容在不同视图的表示：`L=Lrecommendation+lambda*Lcontrastive`。增强可能破坏真实兴趣，温度和权重也会影响结果；保留导出维度不等于训练开销不变。

## 4. “检索 QPS×K”的正确拆解

设业务到达率为 `R` 请求/秒，每请求有效兴趣数 `Ku`。索引面对的 query 向量速率约为 `R*E[Ku]`，不是业务可承载 QPS 提升为 `R*K`。

ComiRec 公开代码把 `[B,K,d]` reshape 成 `[B*K,d]`，一次批量 `index.search`；每兴趣取 `N` 个，再合并去重到最终 `N`。因此：

- 逻辑 query 数由 `B` 变成 `B*K`；RPC 或 Python 调用次数不一定变 K 倍；
- 合并前最多 `K*N` 个结果，不等于最终候选也扩大 K 倍；
- 索引访问、计算、批量效率、重叠率决定实际成本，P99 不必线性增加；
- 固定硬件下可承载的业务请求吞吐通常受新增工作量约束，必须实测。

可验证的代码事实是 `K=4,topN=50` 时每用户最多产生 200 个待合并位置，合并目标仍是 50；这不是任何线上 QPS benchmark。

## 5. 有来源的数据：SDM 作者材料原表

来源为作者仓库的 **23 页演示 slides**，文件名 `SDM_CIKM2019.pdf`，不是论文全文。已将第 18 页渲染后人工核对表格；不能把 slides 与论文版次混称。以下保留原模型行名，不把 `DNN` 擅自改名成 YoutubeDNN。

| 原表模型 | Taobao HitRate@100 | Taobao Recall@100 | JD HitRate@20 | JD Recall@20 |
|---|---:|---:|---:|---:|
| Item-based CF | 60.27% | 3.24% | 67.50% | 9.08% |
| DNN | 60.88% | 2.85% | 68.43% | 8.93% |
| GRU4REC | 65.60% | 3.66% | 69.44% | 9.33% |
| PSDMMAL | 70.72% | 3.86% | 73.25% | 9.47% |
| PSDMMAL-N | 73.13% | 3.83% | 74.33% | 9.68% |

由原表计算，PSDMMAL-N 对 DNN：

- Taobao HitRate@100：绝对 `+12.25` 个百分点，相对约 `+20.12%`；
- Taobao Recall@100：绝对 `+0.98` 个百分点，相对约 `+34.39%`；
- JD HitRate@20：绝对 `+5.90` 个百分点，相对约 `+8.62%`。

这只说明作者该设置下对应模型行的结果，不是 SDM 对所有双塔的提升。还应看到：PSDMMAL-N 的 Taobao Recall `3.83%` 低于 PSDMMAL 的 `3.86%`，不能说它在所有指标上最好。不同 Top-K 和数据集不跨列直接比优劣。

第 21 页原文为“merged with item similarity based CF, +10% CTR”，并提到自 2018 年 12 月全量部署。这里应引用为**作者报告的混合系统 CTR 增益**，不能改写成纯 SDM 单通道、短视频观看时长或本项目可获得 +10%。该页未给出完整流量、方差、置信区间和可独立重跑的生产数据。

机器可读原表摘录：[sdm-author-slides-results.csv](evidence/sdm-author-slides-results.csv)。本项目未复现这些实验。

## 6. ComiRec：为什么不直接抄 NDCG 作为证据

作者 README 明确披露：旧版本 NDCG 实现不符合论文定义，建议优先使用 Recall 和 HitRate。更重要的是，不能只听分支名：本轮固定提交 `a576eed` 的 `src/train.py` 中仍能看到 IDCG 循环按实际命中数 `recall` 累计。

在常见的二值、多正例用户级 NDCG@N 定义下，IDCG 应由 `min(N,相关内容数)` 决定，而不是实际命中数。示例：真实相关集合 `{a,b,c}`，前三位仅命中首位 `a`。这种按命中数的计算得到 `1.0`，按标准理想排序归一化得到约 `0.46928`。这是数学核对示例，不是新跑出的模型指标。

若任务每用户只有一个正例，该差异可能不出现；若使用论文特定定义，需另行对照。跨库比较前必须固定评价定义，不能因为 README 称“已修正”就停止审计代码。

本轮未取得可完整核验的 ComiRec 原论文效果表，因此不填写其“比 MIND 提升 X%”。作者代码可以用于设计受控复现实验，不能替代实测结果。

## 7. 实际选型：只基于当前证据能说什么

1. 使用纯 MF 和具名 YoutubeDNN 复现作为不同层次的基线；固定损失、特征、负例、切分与评价集合。
2. 单向量会话路线比较 SDM/序列编码，多兴趣路线分别比较 MIND、ComiRec-DR、ComiRec-SA，不能将 K 与总候选预算同时放大后声称结构收益。
3. SIM 放到精排长历史工作包；EGES 放到图/item-to-item 工作包，不强制排成同一条替代链。
4. SE 与 SSL 作为正交消融：每次说明底座、插入位置、损失和额外成本，不给独立“主干排名”。
5. X-Rec 未唯一定位前不进入选型清单；已发现的 XRec 不支持原描述。
6. 本表不宣称任一历史论文是当前信息流/短视频的行业最优，也不根据其他业务离线结果承诺本项目收益。

## 8. 可直接检查的来源

- [S1：DeepMatch YoutubeDNN 代码，复现](https://github.com/shenweichen/DeepMatch/blob/fd74417629b19ba4518add93d05cafa551ac7c09/deepmatch/models/youtubednn.py)。
- [S2：SDM 作者 slides，第 18/21 页](https://github.com/alicogintel/SDM/blob/ad898dd471d448ee2745ecc48c1a46b2af38e516/SDM_CIKM2019.pdf)；[作者公开版本免责声明](https://github.com/alicogintel/SDM/blob/ad898dd471d448ee2745ecc48c1a46b2af38e516/README.md#disclaimer)。
- [S3：DeepMatch SDM 复现](https://github.com/shenweichen/DeepMatch/blob/fd74417629b19ba4518add93d05cafa551ac7c09/deepmatch/models/sdm.py)。
- [S4：Alibaba EasyRec MIND 实现](https://github.com/alibaba/EasyRec/blob/cbb4355c0de5a4a5d08c627bc1fa0549ee3fcecc/easy_rec/python/model/mind.py)。
- [S5：DeepMatch MIND 的 dynamic_k 与有效兴趣数](https://github.com/shenweichen/DeepMatch/blob/fd74417629b19ba4518add93d05cafa551ac7c09/deepmatch/models/mind.py#L32-L58)。
- [S6：ComiRec 作者模型实现](https://github.com/THUDM/ComiRec/blob/a576eed8b605a531f2971136ce6ae87739d47693/src/model.py#L199-L248)；[候选聚合与评价](https://github.com/THUDM/ComiRec/blob/a576eed8b605a531f2971136ce6ae87739d47693/src/train.py)；[NDCG 说明](https://github.com/THUDM/ComiRec/blob/a576eed8b605a531f2971136ce6ae87739d47693/README.md#common-issues)。
- [S7：FiBiNet 项目，区分 v1 与 ++](https://github.com/recommendation-algorithm/FiBiNet/tree/af5558bfe3915c266f233ae0c83770712ea96cbd)。
- [S8：SGL 作者代码与数据集配置](https://github.com/wujcan/SGL/tree/58fe93149ca6fb733639f556343c41f12206ba55)。
- [S9：HKUDS/XRec](https://github.com/HKUDS/XRec/tree/8b73f3f04f6dde71cd1339654b087bac0e2d80d5)；[另一个同名 XRec 项目](https://github.com/HCDM/XRec/tree/c0e9edd7a306350b09665055140e336cc750d89f)。
- [P1：SIM 原论文入口](https://arxiv.org/abs/2006.05639)、[P2：EGES 原论文入口](https://arxiv.org/abs/1803.02349)。本轮 arXiv 请求被代理拒绝；此两项保留论文入口，但不声称已读全文或核验其中收益。第三方改写、教学复现不升级为原厂证据。
