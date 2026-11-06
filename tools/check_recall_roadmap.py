"""Keep the consolidated delivery ledger, documentation and diagram statuses aligned."""

import argparse
import json
import os
import re
from pathlib import Path


START = '<!-- RECALL-LEDGER:START -->'
END = '<!-- RECALL-LEDGER:END -->'
STATES = {'L': '模型/服务 smoke', 'C': '组件已测', 'E': '外部源码', 'P': '待建设', 'X': '证据不接受'}


def load_and_validate(root):
    data = json.loads((root / 'configs/recall/roadmap.json').read_text())
    items = data['items']
    by_id = {item['id']: item for item in items}
    if len(by_id) != len(items):
        raise ValueError('duplicate roadmap IDs')
    valid_scopes = {'L': {'synthetic_training', 'teaching_service'}, 'C': {'component_tests'},
                    'E': {'external_source'}, 'P': {'design'}, 'X': {'unaccepted_source'}}
    for item in items:
        if item['status'] not in STATES or item['priority'] not in ('P0', 'P1', 'P2'):
            raise ValueError(f"invalid state/priority: {item['id']}")
        if item['evidence_scope'] not in valid_scopes[item['status']]:
            raise ValueError(f"evidence scope does not justify state: {item['id']}")
        if not (root / item['doc']).is_file() or not item['exit_gate'] or not item['scope']:
            raise ValueError(f"missing documentation or gate: {item['id']}")
        if not set(item['adoption_requires']) <= by_id.keys():
            raise ValueError(f"unknown adoption dependency: {item['id']}")
        if item['status'] in ('L', 'C') and not item['core_code']:
            raise ValueError(f"local claim without core code: {item['id']}")
        for code in item['core_code']:
            if 'path' in code:
                text = (root / code['path']).read_text()
                if not re.search(r'\b(?:class|def) ' + re.escape(code['symbol']) + r'\b', text):
                    raise ValueError(f'missing core symbol: {code}')
            elif not code.get('url', '').startswith('https://github.com/'):
                raise ValueError(f'invalid external code reference: {code}')
    visited, active = set(), set()

    def visit(key):
        if key in active:
            raise ValueError(f'adoption dependency cycle at {key}')
        if key in visited:
            return
        active.add(key)
        for dependency in by_id[key]['adoption_requires']:
            visit(dependency)
        active.remove(key)
        visited.add(key)
    for key in by_id:
        visit(key)
    return data


def render_ledger(root, data):
    base = root / 'docs/recall'

    def relative(filename):
        return Path(os.path.relpath(root / filename, base)).as_posix()

    lines = [START, '', '本表由 [roadmap.json](../../configs/recall/roadmap.json) 自动生成，是当前交付状态台账。',
             '**L/C/E 均不表示生产就绪**；依赖列是进入真实业务的验收依赖，不是运行 toy 代码的依赖。', '',
             '| ID / 优先级 | 能力 / 当前状态 | 已有证据与适用范围 | 核心代码 | 上线依赖 → 退出门槛 |',
             '|---|---|---|---|---|']
    for item in data['items']:
        code_links = []
        for code in item['core_code']:
            if 'path' in code:
                text = (root / code['path']).read_text().splitlines()
                line = next(i for i, s in enumerate(text, 1) if re.search(r'\b(?:class|def) ' + re.escape(code['symbol']) + r'\b', s))
                code_links.append(f"[`{code['symbol']}`]({relative(code['path'])}#L{line})")
            else:
                code_links.append(f"[固定外部源码]({code['url']})")
        dependencies = ', '.join(item['adoption_requires']) or '—'
        lines.append(f"| {item['id']} / {item['priority']} | [{item['title']}]({relative(item['doc'])}) / **{item['status']} {STATES[item['status']]}** | {item['scope']} | {' / '.join(code_links) or '待建设；不标记为已实现'} | {dependencies} → {item['exit_gate']} |")
    lines.extend(['', 'SDM/序列的 S1/S2、位置消偏的 B1/B2 已进入同一台账；当前 C 组件尚未接入六模型训练器。', '', END])
    return '\n'.join(lines)


def render_readiness_graph(data):
    selected = {'D0', 'T0', 'S1', 'S2', 'B2', 'B1', 'V0', 'I0', 'F0', 'SF0', 'K0', 'E0', 'C0', 'AB0'}
    items = [i for i in data['items'] if i['id'] in selected]
    lines = ['graph TD']
    for item in items:
        lines.append(f'  {item["id"]}["{item["id"]} {item["title"]} · {item["status"]}<br/>{item["priority"]} / {STATES[item["status"]]}"]')
    for item in items:
        for dependency in item['adoption_requires']:
            if dependency in selected:
                lines.append(f'  {dependency} -.-> {item["id"]}')
    styles = {'L': 'fill:#e7f5ee,stroke:#16845b,color:#145239',
              'C': 'fill:#e1f4f5,stroke:#087f8c,color:#075c65',
              'E': 'fill:#e9f0ff,stroke:#3b6dd8,color:#234681',
              'P': 'fill:#fff3dc,stroke:#c88718,color:#785014',
              'X': 'fill:#f1edf4,stroke:#82708e,color:#574561'}
    for status, style in styles.items():
        ids = [i['id'] for i in items if i['status'] == status]
        if ids:
            lines += [f'  classDef state{status} {style}', f'  class {",".join(ids)} state{status}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='refresh the generated ledger before rendering diagrams')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = load_and_validate(root)
    target = root / 'docs/recall/README.md'
    source = target.read_text()
    rendered = render_ledger(root, data)
    if START not in source or END not in source:
        raise ValueError('README ledger markers are missing')
    expected = source[:source.index(START)] + rendered + source[source.index(END) + len(END):]
    if args.write:
        target.write_text(expected)
    elif source != expected:
        raise ValueError('stale delivery ledger; run python3 -m tools.check_recall_roadmap --write')
    graph_path = root / 'docs/recall/diagrams/07-delivery-gates.mmd'
    graph_source = render_readiness_graph(data)
    if args.write:
        graph_path.write_text(graph_source)
    elif graph_path.read_text() != graph_source:
        raise ValueError('stale readiness graph source; run --write then render diagrams')
    if not args.write:
        diagram_manifest = json.loads((root / 'docs/recall/diagrams/manifest.json').read_text())
        graphs = {d['id']: d for d in diagram_manifest['diagrams']}
        for item in data['items']:
            for diagram, nodes in item['graph_nodes'].items():
                for node in nodes:
                    if graphs[diagram]['nodeStatuses'].get(node) != item['status']:
                        raise ValueError(f"diagram status mismatch: {item['id']} {diagram}:{node}")
    counts = {s: sum(i['status'] == s for i in data['items']) for s in STATES}
    print(json.dumps(dict(capabilities=len(data['items']), states=counts, production_readiness='not established'), ensure_ascii=False))


if __name__ == '__main__':
    main()
