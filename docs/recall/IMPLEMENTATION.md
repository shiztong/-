# 六模型实现手册与复现边界

## 1. 代码的完成范围

实现为 CPU PyTorch 参考设计，不是直接搬运旧 TF1 仓库。所有模型有实际参数、前向、loss、反向、训练、评估、导出与加载。没有实现大规模 ANN 或生产 RPC；当前标准库 HTTP 教学服务仍独立运行，不会自动切换到这些深度模型。

| CLI 名称 | 本地类 | 实现的关键机制 | 与原论文/作者代码的显式差异 |
|---|---|---|---|
| `mf` | `MatrixFactorization` | 用户/内容 ID 向量、归一化、全库 CE | 归一化 softmax 基线，不是评分平方误差或原始 BPR-MF |
| `pooled` | `PooledTwoTower` | 历史平均 + user ID → MLP，item lookup | YoutubeDNN 风格基线，非 Google 生产模型；非深层多模态内容塔 |
| `session_gate` | `SessionGateRetriever` | 近期 LSTM、Self-Attention、用户条件长短聚合与门控 | 固定最近 8 条作为短窗口，无真实 session 切分、侧信息分组长期注意力、时间间隔编码 |
| `mind_ref` | `MINDReference` | 共享投影、三次路由、动态有效头、hard readout | 固定种子的 routing 初始 buffer，而非每调用重采样；统一 user 融合与归一化；非精确复现 |
| `comirec_sa_ref` | `ComiRecSAReference` | `d→4d→K` 多头池化、可选位置、hard readout | 统一 user 融合/归一化；全库 loss；无原可控多样性聚合 |
| `comirec_dr_ref` | `ComiRecDRReference` | `[L,K,d,d]` 投影、三次 routing、停止中间路由梯度 | 使用 einsum 避免字面高维广播；统一 user 融合/归一化；全库 loss |

默认所有六模型都是 ID-only，避免将类别信息增量伪装成结构收益；`--side-features` 可为非 MF 塔加入类别字段，必须作为单独实验运行，不支持 `all` 一次混入。图模型、SE、SSL、HSTU 和生成式模块目前没有在这套本地训练器中实现。

## 2. 安装与一键实验

建议 Python 3.12、CPU。依赖是官方 CPU wheel 的 PyTorch 2.5.1 与 NumPy 1.26.4，安装文件 [requirements-retrieval-cpu.txt](../../requirements-retrieval-cpu.txt)。模型代码本身不调用 NumPy，但固定该依赖避免 PyTorch 初始化警告。

在仓库根目录执行：

```bash
python3 -m venv /workspace/recsys-venv
/workspace/recsys-venv/bin/python -m pip install -r requirements-retrieval-cpu.txt
/workspace/recsys-venv/bin/python -m tools.run_recall_suite \
  --config configs/recall/cpu-smoke.json --output artifacts/recall-fresh
```

该配置为 96 用户、240 内容、32 事件/用户、6 模型、3 个训练种子、12 epochs、32 维、K=4。固定同一数据 seed=17，训练 seed=17/18/19；不是三份独立测试集。输出必须为新目录，重复执行换一个目录即可，不覆盖旧实验。

## 3. 单模型调试

```bash
/workspace/recsys-venv/bin/python -m recommender.retrieval.experiment generate \
  --output artifacts/single/data.json
/workspace/recsys-venv/bin/python -m recommender.retrieval.experiment train \
  --data artifacts/single/data.json --output artifacts/single/mind-k4 \
  --model mind_ref --interests 4 --dim 32 --epochs 12
```

`--no-position` 只改变 SA 中的显式位置表，不会删除 DR 本身的位置相关投影。`--max-history` 是最近历史截断；历史按真实顺序右 padding，空历史走用户向量回退。维度须为 4 的正倍数以匹配 attention heads。

## 4. 自有数据格式与时间切分

输入 JSON：

```json
{
  "users": 2,
  "items": 6,
  "categories": 2,
  "item_categories": [0, 1, 2, 1, 2, 1, 2],
  "synthetic": false,
  "interactions": [
    [1, 1, 100], [1, 2, 200], [1, 3, 300], [1, 4, 400],
    [2, 2, 100], [2, 3, 200], [2, 4, 300], [2, 5, 400]
  ]
}
```

用户和内容必须映射为从 1 开始的整数，0 为 padding；timestamp 为有限数字，同一用户不能时间并列；每个观测用户至少 4 事件。内容目录在样本时点应已可推荐，示例生成器假设全部内容起始时就存在；自有数据必须补充准入/发布时间过滤后再评估。

当前切分是每用户最后一条测试、倒数第二条验证、之前训练。为防跨用户未来泄漏，代码强制 `max(train_time)<min(valid_time)` 且 `max(valid_time)<min(test_time)`；自有不规则时间数据通常不满足，程序会拒绝，须先实现显式全局窗口切分，不能删除校验凑出分数。

训练样本为每个历史前缀预测下一物品；测试可以用验证事件作为已经发生的历史，不能用测试标签选 checkpoint。类别表是静态元数据，不得回填未来行为统计。

## 5. 训练目标与限制

对单向量使用该用户向量；多兴趣在训练中依据正例选获胜有效兴趣，再计算全目录 softmax。索引 0 被排除。该目标简单可检查，但开销随目录大小增长：程序限制目录不超过 20,000，不能用它宣称支持工业全库训练。

扩大目录前需新增 sampled-softmax/对比采样、重复正例与假负例处理、采样校正等实现，并重新核对评价。不得无声把全库 CE 结果与原论文 sampled-softmax 结果当成严格复现。

每轮验证，选最高验证 Recall；并列选最早 epoch。测试只在选定 checkpoint 后评估。optimizer 为 Adam、lr=0.003、batch=64、梯度裁剪=5；CPU 单线程和 deterministic algorithms 开启。无系统调参或早停统计最优保证。

## 6. 导出、查询与版本

每模型生成：

```text
seed17/model_name/
  model.pt       # 配置、state_dict、类别表、数据hash、selected_epoch
  index.pt       # 全部 item 向量、checkpoint hash
  metrics.json   # 参数量、逐轮训练loss/验证Recall、最终测试指标
```

查询示例：

```bash
/workspace/recsys-venv/bin/python -m recommender.retrieval.experiment query \
  --checkpoint artifacts/recall-fresh/seed17/comirec_sa_ref/model.pt \
  --index artifacts/recall-fresh/seed17/comirec_sa_ref/index.pt \
  --user-id 1 --history 1,2,3 --top-k 10
```

服务阶段不使用真实目标选兴趣；对所有有效兴趣打分取 max，过滤输入的全部已看内容。加载时检查 checkpoint SHA-256，且重算内容向量与索引比对；混用两个版本会报错。SHA 检查防误配，不是外部文件真实性签名；只加载自己生成的可信文件，代码使用 `weights_only=True`。

`index.pt` 是精确向量表，不是 Faiss/HNSW 索引。查询时做全库点积，不提供工业 QPS 或 P99 保证。未知 user ID 暂不支持，需要显式 OOV/新用户通道；不能把缺失 user ID 偷换为别人的向量。

## 7. 评价定义

每用户一个未来目标，因此本协议的 Recall@K=HitRate@K，NDCG 为命中位置的 `1/log2(rank+1)`。它与多正例集合 Recall 不是同一统计量。

默认评估新颖内容：目标若已在历史中出现，计入 `skipped_repeated_targets`，不通过泄漏目标来修改已看 mask。若所有目标均重复则报错。真实视频复播任务应另建重复消费评价，不能静默丢弃后继续声称覆盖全部用户消费。

当前元数据还记录测试 query 数、推荐内容覆盖、平均 Top-K 类目数、训练时间和整批精确评价时间；小合成集的这些耗时不是在线服务 benchmark。

## 8. 测试、差异和下一步

```bash
/workspace/recsys-venv/bin/python -m unittest discover -s tests -p 'test_retrieval_models.py' -v
```

测试覆盖六模型空历史与反向梯度、padding 不变性、SA 无位置的排列性质、MIND 有效头数、全局未来泄漏拒绝、训练—导出—查询、已看过滤与版本错误。未安装 PyTorch 的标准库环境会显式跳过这些可选测试；完整验证必须使用上述虚拟环境，不能把跳过算通过。

工业接入下一步依次是：真实全局时间窗和标签 → sampled loss 与可推荐目录 → ANN 与增量更新 → 生产特征/服务与频控 → 压测与随机实验。后续工作与收益判据见 [总路标](README.md)。
