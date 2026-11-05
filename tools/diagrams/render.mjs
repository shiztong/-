import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { parseMermaid, renderMermaidSVG } from 'beautiful-mermaid';
import { Resvg } from '@resvg/resvg-js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const output = path.join(root, 'docs/recall/diagrams');
const check = process.argv.includes('--check');
const github = 'https://github.com/shiztong/-/blob/master/';
const fontFiles = process.env.RECALL_DIAGRAM_FONTS?.split(path.delimiter) ?? [
  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
  '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
];
for (const filename of fontFiles) await access(filename);
const hash = value => createHash('sha256').update(value).digest('hex');
const xml = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const theme = { bg: '#f8fafc', fg: '#0f172a', line: '#64748b', accent: '#475569', muted: '#475569', surface: '#ffffff', border: '#cbd5e1' };
const colors = {
  '--_text': theme.fg, '--_text-sec': theme.muted, '--_text-muted': theme.muted,
  '--_text-faint': '#bac1ca', '--_line': theme.line, '--_arrow': theme.accent,
  '--_node-fill': theme.surface, '--_node-stroke': theme.border, '--_group-fill': theme.bg,
  '--_group-hdr': '#e9edf2', '--_inner-stroke': '#dde2e9', '--_key-badge': '#e1e6ed',
  '--bg': theme.bg, '--fg': theme.fg,
};
const models = symbol => ({ file: 'recommender/retrieval/models.py', symbol });
const experiment = symbol => ({ file: 'recommender/retrieval/experiment.py', symbol });
const roadmap = github + 'docs/recall/README.md';
const definitions = [
  {
    id: '01-evolution', title: '召回演进路线 · 能力分支与落地状态',
    subtitle: '从协同与双塔到会话、多兴趣、语义、图协同，再到跨层对齐与前沿研究。',
    note: 'L 是本地参考实现，不是原厂精确复现；E 仅表示外部源码已核验。升级需验证收益和成本。',
    links: { R0: { file: 'recommender/engine.py', symbol: 'Recommender' }, R1: models('MatrixFactorization'),
      R2: models('PooledTwoTower'), R3: models('SessionGateRetriever'), R4: models('MINDReference'),
      R5A: models('ComiRecSAReference'), R5B: models('ComiRecDRReference'),
      R6: roadmap + '#r6-内容语义升级合同', R8: roadmap + '#r8-跨层协同升级合同',
      R7: 'https://github.com/wujcan/SGL/blob/58fe93149ca6fb733639f556343c41f12206ba55/model/general_recommender/SGL.py',
      EG: roadmap, R9: github + 'docs/recommendation/13-industry-evidence.md',
      R10: roadmap + '#r10-生成式增量通道合同', FUSE: roadmap, GATE: roadmap },
  },
  {
    id: '02-current-dataflow', title: '当前实现 · 从样本到召回结果的数据流',
    subtitle: '时间留出、六模型训练、选模、内容向量导出、版本校验、精确检索与已看过滤。',
    note: '当前 index.pt 是小目录精确向量表；ANN、采样 loss、实时特征与增量更新仍未实现。',
    links: { DATA: experiment('generate'), CHECK: experiment('load_data'), SPLIT: experiment('require_global_time_split'),
      TRAIN: experiment('make_examples'), VALID: experiment('evaluate'), TEST: experiment('evaluate'), MODEL: experiment('train'),
      CKPT: experiment('train'), VECTORS: models('encode_items'), INDEX: experiment('train'), LOAD: experiment('load_export'),
      REQUEST: experiment('query'), USER: models('build_model'), SCORE: models('scores'), FILTER: experiment('query'),
      RESULT: experiment('evaluate'), ANN: roadmap },
  },
  {
    id: '03-production-target', title: '生产目标 · 召回训练、服务与反馈闭环',
    subtitle: '目标架构，全图 P：日志回流、在线状态、语义编码、ANN、版本包与分发实验尚未贯通。',
    note: '该图是后续建设合同，不能视为已部署系统；实线表示目标数据流，不表示节点已实现。',
    links: {},
  },
  {
    id: '04-multi-interest-flow', title: '单向量与多兴趣 · 训练和服务的关键区别',
    subtitle: '训练可用正例进行 hard readout；检索不能使用未知真实目标，需评估所有有效兴趣。',
    note: 'K 增加 query 向量数，不等于业务 QPS 提升 K 倍；需比较同候选数和同服务预算。',
    links: { INPUT: experiment('tensor_batch'), SINGLE: models('PooledTwoTower'), ROUTE: models('RoutingReference'),
      ONE: models('finish'), MANY: models('finish_multi'), POS: models('loss'), READOUT: models('loss'), LOSS: models('loss'),
      CATALOG: models('encode_items'), DOT: models('scores'), TOP: experiment('query'), ANN: roadmap },
  },
];

async function resolveLink(value) {
  if (!value) return null;
  if (typeof value === 'string') return value;
  const text = await readFile(path.join(root, value.file), 'utf8');
  const line = text.split('\n').findIndex(s => new RegExp(`(?:class|def) ${value.symbol}\\b`).test(s));
  assert(line >= 0, `Missing source symbol: ${value.file}:${value.symbol}`);
  return github + value.file + '#L' + (line + 1);
}

async function render(definition) {
  const source = await readFile(path.join(output, definition.id + '.mmd'), 'utf8');
  const graph = parseMermaid(source);
  const statusCounts = { L: 0, E: 0, P: 0, X: 0 };
  const links = {};
  for (const [id, node] of graph.nodes) {
    const status = / · ([LEPX])(?:<br\s*\/?>|\n|$)/.exec(node.label)?.[1];
    assert(status, `${definition.id}:${id} needs an explicit implementation status`);
    statusCounts[status]++;
    links[id] = await resolveLink(definition.links[id] ?? (definition.id === '03-production-target' ? github + 'docs/recommendation/14-production-execution.md' : null));
    assert(links[id], `Missing source or planning reference for ${id}`);
  }
  let raw = renderMermaidSVG(source, { ...theme, font: 'Noto Sans CJK SC', padding: 40, nodeSpacing: 32, layerSpacing: 52, thoroughness: 7 });
  const dimensions = /viewBox="0 0 ([\d.]+) ([\d.]+)"/.exec(raw);
  assert(dimensions, 'Expected zero-origin SVG viewBox');
  const graphWidth = Number(dimensions[1]), graphHeight = Number(dimensions[2]);
  const width = Math.ceil(Math.max(graphWidth, 1100)), height = Math.ceil(graphHeight + 174);
  // Keep beautiful-mermaid layout; resolve its CSS variables for offline SVG/PNG consumers.
  raw = raw.replace(/<style>[\s\S]*?<\/style>/g, '')
    .replace(/^<svg[^>]*>/, '').replace(/<\/svg>\s*$/, '');
  for (const [name, value] of Object.entries(colors)) raw = raw.replaceAll(`var(${name})`, value);
  assert(!raw.includes('var(') && !raw.includes('@import'), 'Unexpected unresolved CSS/external font');
  raw = raw.replace(/<g class="node" data-id="([^"]+)"([\s\S]*?)<\/g>/g, (all, id) =>
    `<a href="${xml(links[id])}" target="_blank" rel="noopener noreferrer">${all}</a>`);
  // Namespace marker IDs so all four SVGs can coexist in the offline gallery.
  raw = raw.replace(/id="arrowhead/g, `id="${definition.id}-arrowhead`)
    .replace(/url\(#arrowhead/g, `url(#${definition.id}-arrowhead`);
  const legend = [['L 本地已运行', '#16845b'], ['E 外部已核验', '#3b6dd8'], ['P 规划待实现', '#c88718'], ['X 证据不足', '#82708e']]
    .map(([label, color], i) => `<circle cx="${44 + i * 192}" cy="91" r="5" fill="${color}"/><text x="${58 + i * 192}" y="96" font-size="13" fill="#475569">${label}</text>`).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${definition.id}-title ${definition.id}-desc">
<title id="${definition.id}-title">${xml(definition.title)}</title><desc id="${definition.id}-desc">${xml(definition.subtitle + ' ' + definition.note)}</desc>
<style>text{font-family:'Noto Sans CJK SC',sans-serif}a:hover .node>rect{stroke-width:2}</style>
<rect width="100%" height="100%" fill="${theme.bg}"/>
<text x="38" y="39" font-size="25" font-weight="700" fill="#0f172a">${xml(definition.title)}</text>
<text x="40" y="65" font-size="13" fill="#475569">${xml(definition.subtitle)}</text>
${legend}<g transform="translate(${(width - graphWidth) / 2},112)">${raw}</g>
<text x="40" y="${height - 32}" font-size="12" fill="#475569">${xml(definition.note)}</text>
<text x="40" y="${height - 12}" font-size="11" fill="#64748b">beautiful-mermaid 1.1.3 · 节点可点击查看代码或任务 · 连线表示能力依赖或数据流</text>
</svg>\n`;
  assert(!/@import|<script|<foreignObject|var\(/.test(svg));
  const png = new Resvg(svg, { fitTo: { mode: 'width', value: Math.min(2400, Math.ceil(width * 1.5)) },
    font: { fontFiles, loadSystemFonts: false, defaultFontFamily: 'Noto Sans CJK SC' } }).render().asPng();
  return { ...definition, source, svg, png, width, height, statusCounts, links };
}

function gallery(diagrams) {
  const nav = diagrams.map((d, i) => `<button class="tab" data-index="${i}" aria-selected="${i === 0}">${i + 1}. ${xml(d.title.split(' · ')[0])}</button>`).join('');
  const sections = diagrams.map((d, i) => `<section class="panel" ${i ? 'hidden' : ''} data-width="${d.width}" data-height="${d.height}"><h2>${xml(d.title)}</h2><p>${xml(d.subtitle)}</p><div class="downloads"><a href="${d.id}.svg" download>下载 SVG</a><a href="${d.id}.png" download>下载 PNG</a><a href="${d.id}.mmd">Mermaid 源码</a></div><div class="viewport"><div class="canvas">${d.svg}</div></div><p class="note">${xml(d.note)}</p></section>`).join('');
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>召回演进与数据流图册</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172033;font:15px/1.65 'Noto Sans CJK SC',system-ui,sans-serif}main{max-width:1540px;margin:auto;padding:28px}h1{font-size:30px;margin:0}header p{color:#526074;margin:4px 0 18px}nav,.toolbar,.downloads{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}button,a{font:inherit}button{background:white;border:1px solid #ccd5df;border-radius:8px;padding:8px 14px;cursor:pointer}button[aria-selected=true]{background:#0f513d;color:white;border-color:#0f513d}a{color:#2362b5}section{background:white;padding:20px;border:1px solid #dce2ea;border-radius:14px}section[hidden]{display:none}h2{margin:0;font-size:20px}p{margin:5px 0 12px}section>p,.note{color:#59677b}.viewport{height:68vh;overflow:auto;background:#f8fafc;border:1px solid #e3e8ef;border-radius:10px;overscroll-behavior:contain}.canvas{padding:16px;min-width:100%}.canvas>svg{display:block;max-width:none;height:auto}.downloads a{padding-right:12px}.toolbar{align-items:center}.toolbar input{width:180px}footer{color:#66758a;margin-top:18px}.note{margin-top:14px}button:focus-visible,a:focus-visible{outline:3px solid #579aff;outline-offset:3px}@media(max-width:700px){main{padding:12px}h1{font-size:24px}.viewport{height:62vh}section{padding:12px}}
</style></head><body><main><header><h1>召回演进与数据流图册</h1><p>四张图对应能力路线、当前实现、生产目标，以及训练与检索差异。L / E / P / X 明确区分落地状态。</p><a href="../README.md">召回总路标与收益证据</a> · <a href="../IMPLEMENTATION.md">训练、导出与查询手册</a></header><nav aria-label="选择图表">${nav}</nav><div class="toolbar"><button id="fit">适应宽度</button><button id="actual">原始尺寸</button><label>缩放 <input id="zoom" type="range" min="25" max="200" value="100"></label><output id="scale">100%</output><span>可滚动查看，点击节点跳转代码</span></div>${sections}<footer>离线文件，无 CDN、远程字体或预览服务器依赖。代码链接需要网络；PNG 包含中文字体渲染结果。源码与生成器保存在仓库。</footer></main><script>
const panels=[...document.querySelectorAll('.panel')],tabs=[...document.querySelectorAll('.tab')];let active=0;
const zoom=document.querySelector('#zoom'),label=document.querySelector('#scale');
function apply(value){const panel=panels[active];zoom.value=value;label.value=Math.round(value)+'%';panel.querySelector('.canvas>svg').style.width=(Number(panel.dataset.width)*value/100)+'px';}
function fit(){const p=panels[active],v=p.querySelector('.viewport');apply(Math.max(25,Math.min(100,100*(v.clientWidth-34)/Number(p.dataset.width))));}
tabs.forEach((tab,i)=>tab.addEventListener('click',()=>{active=i;tabs.forEach((t,j)=>t.setAttribute('aria-selected',String(j===i)));panels.forEach((p,j)=>p.hidden=j!==i);fit();}));
zoom.addEventListener('input',()=>apply(Number(zoom.value)));document.querySelector('#fit').addEventListener('click',fit);document.querySelector('#actual').addEventListener('click',()=>apply(100));window.addEventListener('resize',fit);fit();
</script></body></html>\n`;
}

await mkdir(output, { recursive: true });
const diagrams = [];
for (const definition of definitions) diagrams.push(await render(definition));
const manifest = {
  renderer: 'beautiful-mermaid', version: '1.1.3', rasterizer: '@resvg/resvg-js@2.6.2',
  fonts: await Promise.all(fontFiles.map(async file => ({ file: path.basename(file), sha256: hash(await readFile(file)) }))),
  diagrams: diagrams.map(d => ({ id: d.id, title: d.title, width: d.width, height: d.height,
    statusCounts: d.statusCounts, sourceSha256: hash(d.source), svgSha256: hash(d.svg), pngSha256: hash(d.png), links: d.links })),
};
const generated = new Map([['index.html', gallery(diagrams)], ['manifest.json', JSON.stringify(manifest, null, 2) + '\n']]);
for (const d of diagrams) { generated.set(d.id + '.svg', d.svg); generated.set(d.id + '.png', d.png); }
for (const [filename, value] of generated) {
  if (check) assert(Buffer.from(await readFile(path.join(output, filename))).equals(Buffer.from(value)), `Regenerate stale diagram: ${filename}`);
  else await writeFile(path.join(output, filename), value);
}
for (const d of diagrams) console.log(`${check ? 'Verified' : 'Rendered'} ${d.id}: ${d.width}x${d.height}, ${Object.entries(d.statusCounts).map(([k, v]) => k + '=' + v).join(' ')}`);
