# 推荐系统

基于 Python 标准库的推荐系统起步项目，无第三方运行依赖。支持用户协同过滤、热门推荐、冷启动回退及 HTTP JSON API。适合学习和原型开发；内置 HTTP 服务不用于生产部署。

## 算法设计文档

[信息流与短视频推荐：分层模型设计](docs/recommendation/README.md) 分别讲解两种场景的召回、粗排、精排、重排，以及特征、标签、损失函数、训练和服务。深度模型属于设计文档内容，当前教学代码尚未实现。

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
