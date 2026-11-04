# 25 从研究结论到可执行实验：比较合同、证据分级与上线决策

状态更新（2026-11-04）：本章原先只交付解析探针；现在已新增真实 PyTorch 训练链路、六模型三种子合成实验、索引导出与查询。见 [运行与源码入口](../recall/README.md)。下文“尚无深度训练器/本业务实验”的历史表述需分别理解：参考训练器已经有了，真实业务和原论文效果复现仍未完成。

本章补足“模型公式正确，但不知道如何证明值得上线”的缺口。当前没有本业务日志、硬件和线上基线，因此交付实验设计与解析检查，不伪造训练完成、模型领先或上线收益。

## 1. 四类证据分开存放

| 类别 | 本轮具体内容 | 能证明 | 不能证明 |
|---|---|---|---|
| 作者报告 | 22 中 SDM slides 原表与混合系统 CTR | 作者在对应设置报告了哪些结果 | 本项目可复制同等收益 |
| 固定代码 | ComiRec、DeepMatch、DIN、TFRS、PaddleRec | 特定版本的接口与计算表达式 | 当前公司完整生产架构 |
| 解析/数值探针 | 参数量、排列不变性、gate 有限差分、PLE 索引 | 公式性质、静态索引问题和数值计算 | 召回效果、P99、真实峰值显存 |
| 本业务实验 | 尚无 | 待采集、训练与受控实验 | 当前不得填写成功结果 |

新增源码见 [deep-study-sources.json](evidence/deep-study-sources.json)，原召回与作者 slides 来源见 [model-audit-sources.json](evidence/model-audit-sources.json)。

## 2. 实验输入合同

一个有效运行至少记录以下内容，缺少其中关键项不能只凭一行 Recall 宣称可比：

```yaml
identity:
  experiment_id: required
  data_snapshot_hash: required
  code_commit: required
  framework_and_hardware: required
samples:
  scene_and_exposure_definition: required
  train_valid_test_time_boundaries: required
  label_window_and_maturity_rule: required
  available_history_cutoff_and_ingest_rule: required
  positive_definition: required
  negative_sampler_and_probability_rule: required
features:
  field_list_and_transform_versions: required
  id_vocabulary_and_oov: required
  item_encoder_version: required
evaluation:
  candidate_universe_and_exclusions: required
  exact_search_or_ann_and_parameters: required
  multi_positive_metric_definition: required
  final_candidate_budget: required
  history_length_and_active_interest_distribution: required
resources:
  training_budget_and_seed_list: required
  request_batch_candidate_count_and_concurrency: required
  p50_p95_p99_and_queueing_definition: required
```

这是合同模板，不是已经可交给现有训练器执行的配置。当前仓库尚无深度模型训练器，需依 14 的工作包实现后才能运行。

## 3. 召回实验矩阵：每次只回答一个主要问题

| ID | 基线 → 候选 | 固定项 | 主要变量 | 必须报告 |
|---|---|---|---|---|
| RET-A | 池化 MLP → 衰减池化 MLP | 特征/负例/内容塔 | 时间加权 | 兴趣切换、整体 Recall、成本 |
| RET-B | 衰减池化 → SDM | 可获得历史与新鲜度 | 长短期序列结构 | 长短分支消融、空会话、延迟 |
| RET-C | 单向量 → MIND | 字段、损失和最终 N | 多兴趣表示 | 头级贡献、query 数与总检索量 |
| RET-D | MIND → DR/SA | K、历史、负例与评价库 | 抽取机制 | 参数/预算曲线、位置消融、种子 |
| RET-E | SA 无位置 → 有位置 | 其他全部 | 顺序位置 | 历史置换敏感、时间迁移切片 |
| RET-F | K=1/2/4/8 对照 | 一组固定结果位置数，另一组固定最终 N | 头数与预算 | 去重、后过滤不足、精确/ANN损失 |
| RET-G | 当前内容塔 → 语义增强 | 用户底座与探索策略 | 内容表征 | 零/低曝光切片和供给贡献 |

网格 `K=1/2/4/8` 为建议实验，不是最优结论。不是每项都要跑满；先在可承担的预算上筛选，再对进入线上候选的设置做重复种子与性能验证。

### 3.1 防止指标定义导致伪提升

固定 Recall 的分母、HitRate 的用户平均方式、多正例 NDCG 的 IDCG、已看过滤和曝光机会。不要把随机负例候选评价与全库评价放同一张胜负表；不要跨 Top-K 直接计算相对提升。

ComiRec 源码中发现的 IDCG 风险说明：评价器也要接受代码审查。用人工构造的单正例、多正例、零命中、全命中和重复候选验证，且要与当前任务定义一致。

## 4. 精排实验矩阵

| ID | 研究问题 | 第一对照 | 必须防止的混淆 |
|---|---|---|---|
| RANK-A | 是否需要候选相关兴趣 | 同历史池化 vs DIN | 给 DIN 额外更长历史 |
| RANK-B | 是否需要历史量幅度信息 | 归一化 vs 非归一化局部激活 | BN、长度特征同时变更 |
| RANK-C | 显式交叉的额外价值 | 匹配参数 DNN vs DCN，低秩/全秩 | 只与明显更小 DNN 比 |
| RANK-D | 任务是否需要专家共享 | Shared Bottom vs MMoE | loss mask/目标同时变化 |
| RANK-E | 专属专家是否缓解冲突 | 参数/成本匹配 MMoE vs PLE | 错索引、更多总算力 |
| RANK-F | 列表页是否需要漏斗约束 | 同标签独立任务 vs 事件乘积 | 事件关系根本不成立 |
| RANK-G | 视频流是否需要分布时长 | 同特征回归 vs 分桶，再有条件试 hazard | 用未记录的删失事件补标签 |

评价既包含各任务损失、校准、排序，也包含融合后效用和服务成本。新模型应在真实候选分布上评估；全曝光、点击后条件与有效起播不是同一个测试集合。

## 5. 把错误分析变成下一条路标

| 失败现象 | 先诊断 | 再决定的演进 |
|---|---|---|
| 多兴趣只增加热门 | 头塌缩、负例、共享特征和候选重叠 | 改抽取器/负例/聚合，不先盲目加 K |
| 序列离线好线上差 | 状态延迟、位置编码、缓存/索引版本 | 先修实时链路，不直接扩长历史 |
| DIN 权重集中但收益无增量 | 历史同质、logit尺度、重复行为 | 归一化/时间/历史检索实验 |
| PLE 不如小模型 | 任务mask、索引、专家数据量和预算 | 减专属专家或回退共享结构 |
| 时长高但满意度低 | 复播、播放质量、长内容偏置、效用定义 | 改标签与决策，而非只换回归头 |
| 重模型超时更多 | 用户/候选共享、排队、特征RPC | 蒸馏/压缩/候选预算，再谈规模 |

这才构成有证据的发展路线：不是模型名单按年代排列，而是每个升级都对应上一阶段可观察的失效模式。

## 6. 线上接入与经济验收

先做 A/A 与日志核对，再 shadow 验证服务和数值；shadow 不提供真实业务提升。正式 A/B 使用稳定用户分桶，按场景与跨场景总价值分别统计，处理重复观测和延迟标签。

放量需同时通过：已确定的主指标统计标准、负反馈/质量/留存护栏、端到端延迟、增量成本和回滚。经济门槛见 16，不再给任意模型预设统一“提升 3%–5%”。

若公开报告显示较大收益，而本业务只能得到小收益，也可能是原基线、特征、训练规模和场景不同，不能据此认定实现错误；需要逐项对照合同。

## 7. 本轮可执行部分与验证结果

```bash
python3 tools/model_mechanism_probes.py
python3 -m unittest discover -s tests -p 'test_model_mechanism_probes.py' -v
```

该脚本仅用 Python 标准库，对固定尺寸参数计算、简化注意力性质、gate 导数、PLE 索引和效用期望做解析/数值检查。它不加载 TensorFlow/Paddle，不训练 DIN/ComiRec，不提供任何业务 benchmark。

运行产物记录在 [mechanism-probe-results.json](evidence/mechanism-probe-results.json)。这些产物的作用是让关键机制可检查；深度模型效果和服务吞吐仍为未执行项。它们不能被写成“新模型测试已通过、可以上线”。
