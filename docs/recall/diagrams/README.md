# 召回演进与数据流图册

由 **beautiful-mermaid 1.1.3** 渲染 Mermaid 源码为 SVG，再由 **@resvg/resvg-js 2.6.2** 导出 PNG。Node 依赖与锁文件保存在 [tools/diagrams](../../../tools/diagrams/package.json)，独立于 Python 推荐服务。

[离线交互图册 index.html](index.html) 支持四图切换、缩放、适应宽度、下载 SVG/PNG，以及点击节点跳转源码或建设任务。GitHub 不执行仓库中的 HTML；下载此目录后本地打开 `index.html`，或直接浏览以下 SVG/PNG。无需启动本地服务器或加载 CDN。

## 四张图

| 图 | 说明 | SVG | PNG | 可编辑源码 |
|---|---|---|---|---|
| 01 召回演进路线 | 协同/MF/双塔 → 会话和多兴趣 → 语义/图 → 蒸馏/HSTU/生成式增量；不是必然替换顺序 | [矢量图](01-evolution.svg) | [高清图](01-evolution.png) | [Mermaid](01-evolution.mmd) |
| 02 当前代码数据流 | 数据校验、时间留出、六模型训练、验证选模、向量导出、加载校验、精确查询与测试评价 | [矢量图](02-current-dataflow.svg) | [高清图](02-current-dataflow.png) | [Mermaid](02-current-dataflow.mmd) |
| 03 生产目标数据流 | 日志与实时状态、样本/训练、内容编码、ANN、分发和实验反馈；全图为目标架构 | [矢量图](03-production-target.svg) | [高清图](03-production-target.png) | [Mermaid](03-production-target.mmd) |
| 04 单向量/多兴趣 | 用户表征、训练 hard readout 与 loss、检索时对有效兴趣取 max、未来 ANN 分头合并 | [矢量图](04-multi-interest-flow.svg) | [高清图](04-multi-interest-flow.png) | [Mermaid](04-multi-interest-flow.mmd) |

## 状态图例与阅读边界

- **绿色 L**：当前仓库有本地可运行参考实现。六模型已做合成数据实验，不代表原厂精确复现或生产上线。
- **蓝色 E**：外部源码已核验，未在当前环境完成对应训练复现。
- **橙色 P**：规划或待实现能力。生产目标图全部为 P。
- **紫色 X**：所读代码/证据不足以支撑当前声称，例如 EGES 候选的侧信息训练链。

节点同时保留字母，避免只靠颜色识别。箭头表示能力演进关系或数据流，不表示后者一定优于前者；路线图虚线表示尚未落地的扩展关系，生产目标图的实线也不表示已实现。

图内不新增收益百分比。公开结果、合成数据运行结果、立项目标和业务预期仍分别以 [召回总表](../README.md)、[运行报告](../BENCHMARK.md) 为准。多兴趣增加 query 向量数，不意味着业务 QPS 自动乘 K。

## 安装、重新生成与校验

在仓库根目录执行，建议 Node.js 20 或更新版本：

```bash
npm ci --prefix tools/diagrams
npm --prefix tools/diagrams run render
npm --prefix tools/diagrams run check
```

当前环境已安装依赖并生成四组 SVG/PNG。修改 `.mmd` 后运行 `render`；标题、说明及节点代码链接在 [render.mjs](../../../tools/diagrams/render.mjs) 中维护。生成器解析节点状态、检查代码符号、计算实际源码行号，不手工硬编码类定义的行数。

中文导出使用 Noto Sans CJK SC。默认读取系统的 `NotoSansCJK-Regular.ttc`、`NotoSansCJK-Bold.ttc`；无对应字体时应先安装字体，也可用 `RECALL_DIAGRAM_FONTS` 指定字体文件列表（按操作系统的路径分隔符分开）。当前字体文件哈希记录在 [manifest.json](manifest.json)。跨平台字体版本变化会改变 PNG，需要重新生成再校验。

SVG 保留 beautiful-mermaid 的布局，将 CSS 颜色变量解析为固定颜色并移除远程字体 import，以支持离线渲染与文档图片加载。没有用其他布局器替换 beautiful-mermaid。PNG 已栅格化中文，接收方不需要安装字体即可查看。

`check` 重新渲染并检查 SVG/PNG/HTML/清单与源码一致；节点状态、源码符号或图文件过期会报错。清单含渲染器版本、字体哈希、源码/图片哈希、状态数量和全部节点链接。
