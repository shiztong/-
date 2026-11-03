# 13 已核验的业界证据与可迁移边界

核验日期：2026-11-03。以下读取了公开源码/配置/README，未在本机重跑上游训练、硬件基准或线上 A/B。完整提交、路径、字节数和 SHA-256 见 [来源清单](evidence/source-manifest.json)。核验内容与原 08 的未逐条核验文献表分开维护。

证据等级：**A** 官方公开实现和配置可定位；**B** 官方报告的离线/性能结果，但本项目未复现；**C** 本文迁移假设，需本项目实验。A 不等于生产就绪，B 不等于本业务收益。

## 1. Meta Generative Recommenders：模型与系统协同

固定提交：[b8e6ff0](https://github.com/facebookresearch/generative-recommenders/tree/b8e6ff07c5d4c3354b0807ead76b74d771557fd7)。

已读实现：`generative_recommenders/research/modeling/sequential/hstu.py`；配置：`configs/ml-1m/*final.gin`；系统扩展：`generative_recommenders/modules/hstu_ultra.py`。

### 可确认的机制（A）

HSTU 不只是将 DIN 换成标准 Transformer。研究实现中先对归一化输入联合投影 U/V/Q/K；`rel_bias` 分支使用带相对偏置的 QK 交互，经 SiLU 与长度缩放，再与 V 聚合，并经过门控和输出投影。源码也有 softmax 变体，必须记录所用 normalization，不能混称同一算子。

当前 Ultra 模块提供 512 维、8 层半局部与 12 层 target-aware 两种栈，支持 FP8 投影权重缓存。这证明公开代码存在相应接口，不证明本业务硬件能得到相同性能，也不表示之前 09 的普通 Transformer 已实现这些机制。

### 数字及其口径（B）

官方 README 的 ML-1M HR@10：SASRec `0.2853`、HSTU `0.3097`、HSTU-large `0.3294`。因此相对变化分别约 `+8.55%`、`+15.46%`；HSTU-large 的绝对差为 `+0.0441`，不是线上点击增长 4.41 个百分点。

large 配置为 8 blocks、2 heads、item embedding 50；所读 SASRec 配置为 2 blocks、1 head、item embedding 50。二者均为长度 200、batch 128、128 个负例、温度 0.05，但架构和容量不同。large 相比 SASRec 的数字不是严格等容量的纯算子收益。

官方对照中的 SASRec 使用 sampled softmax，不能误写成原始 BCE 训练的 SASRec。公开 MovieLens 电影评分序列与自动播放短视频具有显著分布差异。

README 当前报告完整半局部 8 层栈、长度 512、H100 96GB 条件下 BF16 `6.036 ms`、FP8 `5.325 ms`，约 `1.13x`。同页 attention-only benchmark 的约 `2x` 加速不包含投影，不能当作整栈或端到端加速。README 还明确该集成路径尚未包含 cached incremental decoding 和 scripted inference export。

### 迁移建议（C）

先建立相同样本与预算的 SASRec/HSTU 对照，再扩长序列与 target-aware 候选；先测完整栈和真实候选数，再决定自定义 kernel。不要把论文标题中的万亿参数规模当作本业务立刻应达到的稠密模型规模。

## 2. ByteDance Monolith：真正的实时稀疏学习是系统问题

固定提交：[135c491](https://github.com/bytedance/monolith/tree/135c491a52b151772b976af989d8bc938c44d210)。

官方 README 明确支持 collisionless embedding、batch/real-time training 与 serving，构建基于 TensorFlow，示例依赖 Bazel 3.1.0、Linux。这是框架能力说明（A），不是本项目已经验证的现代环境兼容性或线上收益。

可迁移的是无碰撞 ID 管理、稀疏参数生命周期、训练与服务更新一致性这些系统思想。不能在当前标准库服务中增加一个实时任务，就宣称实现了 Monolith。若主栈采用 PyTorch，应先评估现有参数服务与团队维护能力，再决定复用或借鉴，避免仅为模型实验迁移全套平台。

本轮未读取其线上收益原始表格，因此不引用具体收益百分比。

## 3. KuaiRand：短视频、多界面与随机曝光

固定提交：[f8dbf66](https://github.com/chongminggao/KuaiRand/tree/f8dbf6678b3c9594050e3e813aeff0c942260ec4)。

官方说明包含来自快手两个 UI 的反馈，12 类行为信号；随机替换约占交互 `0.37%`，从限定内容池进行随机展示。其 1K 版本标准策略日志有 1,000 用户、约 1,171 万交互，内容约 437 万；不能因为“1K”就当作轻量玩具数据。

可用于多场景反馈、时长与偏差研究，但需要按数据字典验证 UI 与业务场景映射，不能只按“有两个 UI”就声称与我们的列表页/视频流完全一致。随机曝光仅覆盖指定池和用户，不能泛化为全库随机化，更不能推导全策略无偏。

统计特征文件可能不符合逐曝光时点可用性，需审计时间口径；用户序列研究不要无差别采用序列信息不完整的裁剪版本。数据不能提供的播放器状态、用户标识跨系统归因或删失事件不得伪造。

数据下载在 Zenodo；本环境当前允许的包管理网络不包含该站点。本轮只核验 GitHub 上的说明，未下载该数据或绕过网络策略。真正复现实验需通过环境设置增加相应目的域，并检查数据许可和校验值。

## 4. TensorFlow Recommenders：可复现双塔基线

固定提交：[a8a90ae](https://github.com/tensorflow/recommenders/tree/a8a90aee3f0b470cda96d47e26de99eedb7fc730)。

官方示例包括用户/内容独立编码、Retrieval loss 与 FactorizedTopK 指标，是清晰的双塔实现入口（A）。README 示例随机切分是教学范例，不能直接用于本项目的线上时序验证。

本项目若选择 PyTorch，不需要为该示例换框架；重点复现的是分离编码、全候选评价和服务一致性的契约，不是机械复制依赖。

## 5. 尚未核验的方向如何处理

本轮尝试的 `Kuaishou-OneRec/OneRec` main/master README 返回 404，不能用猜测链接证明官方实现存在。生成式推荐可继续作为研究候选，但本轮不写 OneRec 的数字收益、上线状态或代码复现承诺；404 也不证明相关研究不存在。

Google Research 的 YouTube 论文页面被当前网络代理拒绝，因此此次不能声称重新读过其全文。08 中的论文链接仍是文献入口；需要全文收益结论时应另外取得并核验原文。

截至核验日，本文只描述读取到的公开实现状态，不声称穷尽 2026 年最新进展，也不把第三方复现仓库当成原厂生产代码。
