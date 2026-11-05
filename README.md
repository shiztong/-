# 推荐系统

基于 Python 标准库的推荐系统起步项目，无第三方运行依赖。支持用户协同过滤、热门推荐、冷启动回退及 HTTP JSON API。适合学习和原型开发；内置 HTTP 服务不用于生产部署。

## 召回实操入口

[召回路线与数据流图册](docs/recall/diagrams/README.md) 使用 beautiful-mermaid 生成六张 SVG/PNG，提供离线 HTML 缩放浏览和节点源码跳转，明确区分当前实现与生产目标。

深入专项：[SDM 与序列优化](docs/recall/SDM_SEQUENCE.md) 讲解会话、时间、多行为、长期记忆和缓存；[位置消偏](docs/recall/POSITION_DEBIAS.md) 区分列表展示、视频流到达、PBM/IPS 与策略评估，并提供可执行权重/loss 组件和独立模拟。

[召回完整路标、预期收益与核心代码](docs/recall/README.md) 将每条演进路径映射到具体源码、运行状态、收益依据与验收条件。

新增独立的可选 PyTorch 召回实验包，已实现并实际训练 MF、池化双塔、SDM 风格长短期融合、MIND 及 ComiRec-SA/DR 参考模型。使用 [实现手册](docs/recall/IMPLEMENTATION.md) 运行训练、评估、索引导出和查询；[三种子运行报告](docs/recall/BENCHMARK.md) 保留合成数据实测结果。该包需要单独安装依赖，不改变标准库 HTTP 服务的运行依赖，也不等同于原论文复现或生产服务。

## 算法设计文档

模型深入研究入口：[召回结构、梯度与检索预算](docs/recommendation/23-retrieval-code-and-gradients.md)、[精排源码、专家路由与目标](docs/recommendation/24-ranking-code-and-objectives.md)、[对照实验与数值检查](docs/recommendation/25-falsifiable-experiment-plan.md)。

完整发展路线见 [技术演进路标树](docs/recommendation/17-evolution-roadmap-tree.md)，包含召回、粗排、精排、重排、特征和训练评估的历史脉络、并行分支及未来方向；落到业务项目见 [分场景演进计划](docs/recommendation/21-scene-adoption-roadmap.md)。

[信息流与短视频推荐：分层模型设计](docs/recommendation/README.md) 分别讲解两种场景的召回、粗排、精排、重排，以及特征、标签、损失函数、训练和服务。召回部分已有上述六个可运行参考实现；完整多模态、排序与生产架构仍属于设计与后续工作。

方案评审从 [落地审查](docs/recommendation/12-readiness-review.md) 开始；[业界证据](docs/recommendation/13-industry-evidence.md) 记录本次核验的官方源码，[执行计划](docs/recommendation/14-production-execution.md)、[前沿规划](docs/recommendation/15-frontier-portfolio.md) 和 [收益评估](docs/recommendation/16-value-and-experiments.md) 说明实施依赖与验收方法。收益测算工具使用假设输入，不代表已获得实际业务增益。

## 启动

需要 Python 3.10 或更新版本，在仓库根目录运行：

```bash
python3 -m recommender.server
```

在另一个终端请求：

```bash
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/recommendations?user_id=u1&limit=3'
```

未知用户自动返回热门商品，已交互商品不会再次推荐。`limit` 取值为 1–100；候选不足时返回较少结果。缺失用户 ID 或非法 limit 返回 HTTP 400。

## 数据与算法

`data/interactions.csv` 包含 `user_id,item_id,title,rating`，评分范围为 (0, 5]。同一用户与商品的重复记录以最后一条评分为准。替换数据后重启服务，或指定 `--data /path/to/interactions.csv`。

算法先计算用户评分向量的余弦相似度，再对相似用户评分进行加权平均；候选不足时按累计评分补充热门商品。结果中的 `source` 标识来源；协同过滤与热门分数的量纲不同，不能直接比较。当前数据保存在内存，计算随用户数量增长，适用于小规模数据。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

代码结构：`recommender/engine.py` 为算法，`recommender/server.py` 为服务，`tests/` 为算法与 API 测试。自定义监听地址或端口可使用 `--host`、`--port`。
