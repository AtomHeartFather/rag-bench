#!/usr/bin/env python3
"""Aggregate results/{variant}_{scale}/metrics.json -> summary.csv + report.md."""
import csv, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
VARIANTS = ['file', 'fts', 'rag']
SCALES = ['1k', '10k', '50k', '100k', 'real-mh', 'real-rag-10k', 'real-rag-66k']

ROWS = []  # (variant, scale, metric, value)


def add(variant, scale, metric, value):
    if value is None:
        return
    ROWS.append((variant, scale, metric, value))


def load_metrics(v, s):
    p = BASE / 'results' / f'{v}_{s}' / 'metrics.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    for v in VARIANTS:
        for s in SCALES:
            m = load_metrics(v, s)
            if not m:
                continue
            # index
            add(v, s, 'index_wall_s', m.get('index_wall_s'))
            add(v, s, 'index_ram_peak_kb', m.get('index_ram_peak_kb'))
            add(v, s, 'index_cpu_s', m.get('index_cpu_s'))
            add(v, s, 'index_disk_bytes', m.get('index_disk_bytes'))
            add(v, s, 'embed_requests', m.get('embed_requests'))
            add(v, s, 'embed_tokens', m.get('embed_tokens'))
            add(v, s, 'embed_tok_per_s', m.get('embed_tok_per_s'))
            add(v, s, 'host_qdrant_ram_peak_kb', (m.get('host_ram_peak') or {}).get('qdrant_ram_peak_kb'))
            add(v, s, 'host_embed_ram_peak_kb', (m.get('host_ram_peak') or {}).get('embed_ram_peak_kb'))
            # search
            add(v, s, 'search_latency_p50_ms', m.get('latency_p50_ms'))
            add(v, s, 'search_latency_p95_ms', m.get('latency_p95_ms'))
            add(v, s, 'search_latency_p99_ms', m.get('latency_p99_ms'))
            add(v, s, 'search_ram_peak_kb', m.get('search_ram_peak_kb'))
            # relevance overall
            ov = m.get('overall') or {}
            for k in ('hit@1', 'hit@3', 'hit@5', 'hit@10', 'mrr@10', 'overlap@5'):
                add(v, s, f'rel_{k.replace("@", "_")}', ov.get(k))
            # relevance by level
            for lv, d in (m.get('by_level') or {}).items():
                for k in ('hit@10', 'mrr@10'):
                    add(v, s, f'rel_{lv}_{k.replace("@", "_")}', d.get(k))
            # judge
            j = m.get('judge') or {}
            if j:
                add(v, s, 'judge_correct_rate', j.get('e2e_correct_rate'))
                add(v, s, 'judge_e2e_p50_ms', j.get('e2e_p50_ms'))
                add(v, s, 'judge_e2e_p95_ms', j.get('e2e_p95_ms'))
                add(v, s, 'judge_llm_prompt_tokens', j.get('llm_prompt_tokens'))
                add(v, s, 'judge_llm_completion_tokens', j.get('llm_completion_tokens'))

    out = BASE / 'results'
    out.mkdir(exist_ok=True)
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'scale', 'metric', 'value'])
        w.writerows(ROWS)

    # report.md
    def fmt(x, pct=False):
        if x is None:
            return '—'
        if pct:
            return f'{x * 100:.1f}%'
        if isinstance(x, float):
            return f'{x:.2f}'
        return str(x)

    L = []
    L.append('# RAG-бенчмарк: file (BM25) vs fts (SQLite FTS5) vs rag (Qdrant+bge-m3)\n')
    L.append('Матрица: 3 варианта × масштабы × 300 запросов (5 уровней × 60).\n')

    def table(title, metric, variants, scales, pct=False, key=lambda m: m):
        L.append(f'\n## {title}\n')
        hdr = 'масштаб | ' + ' | '.join(variants)
        L.append(hdr)
        L.append('---|' + '---|' * len(variants))
        for s in scales:
            row = [s]
            for v in variants:
                m = load_metrics(v, s) or {}
                val = key(m)
                row.append(fmt(val, pct))
            L.append(' | '.join(row))

    active_scales = [s for s in SCALES if any((BASE / 'results' / f'{v}_{s}' / 'metrics.json').exists() for v in VARIANTS)]
    table('Релевантность (реtrieval, 300 запросов)', 'Hit@10', VARIANTS, active_scales, pct=True,
          key=lambda m: (m.get('overall') or {}).get('hit@10'))
    table('МRR@10', 'MRR@10', VARIANTS, active_scales, key=lambda m: (m.get('overall') or {}).get('mrr@10'))
    table('Hit@5', 'Hit@5', VARIANTS, active_scales, pct=True,
          key=lambda m: (m.get('overall') or {}).get('hit@5'))
    table('Overlap@5', 'Overlap@5', VARIANTS, active_scales,
          key=lambda m: (m.get('overall') or {}).get('overlap@5'))
    table('Latency p50 (мс)', 'p50', VARIANTS, active_scales, key=lambda m: m.get('latency_p50_ms'))
    table('Latency p95 (мс)', 'p95', VARIANTS, active_scales, key=lambda m: m.get('latency_p95_ms'))
    table('Latency p99 (мс)', 'p99', VARIANTS, active_scales, key=lambda m: m.get('latency_p99_ms'))
    table('Индексация: wall (с)', 'index wall', VARIANTS, active_scales, key=lambda m: m.get('index_wall_s'))
    table('Индексация: пик RAM (МБ)', 'ram', VARIANTS, active_scales,
          key=lambda m: round(m.get('index_ram_peak_kb', 0) / 1024, 1) if m.get('index_ram_peak_kb') else None)
    table('Индексация: CPU-с', 'cpu', VARIANTS, active_scales, key=lambda m: m.get('index_cpu_s'))
    table('Индексация: диск (МБ)', 'disk', VARIANTS, active_scales,
          key=lambda m: round(m.get('index_disk_bytes', 0) / 1024 / 1024, 1) if m.get('index_disk_bytes') else None)

    # per-level Hit@10
    L.append('\n## Hit@10 по уровням сложности\n')
    L.append('масштаб | variant | L1 | L2 | L3 | L4 | L5')
    L.append('---|---|---|---|---|---|---')
    for s in active_scales:
        for v in VARIANTS:
            m = load_metrics(v, s) or {}
            bl = m.get('by_level') or {}
            row = [s, v]
            for lv in ('L1', 'L2', 'L3', 'L4', 'L5'):
                d = bl.get(lv) or {}
                h = d.get('hit@10')
                row.append(f'{h * 100:.0f}%' if h is not None else '—')
            L.append(' | '.join(row))

    # cost
    L.append('\n## Стоимость (токены)\n')
    L.append('масштаб | variant | embed-токены (индексация) | embed tok/с | judge LLM prompt | judge LLM compl')
    L.append('---|---|---|---|---|---')
    for s in active_scales:
        for v in VARIANTS:
            m = load_metrics(v, s) or {}
            j = m.get('judge') or {}
            L.append(f"{s} | {v} | {m.get('embed_tokens') or '—'} | {m.get('embed_tok_per_s') or '—'} | "
                     f"{j.get('llm_prompt_tokens') or '—'} | {j.get('llm_completion_tokens') or '—'}")

    # corpus gen cost
    L.append('\n## Генерация корпуса (LLM)\n')
    L.append('масштаб | сущности | чанки | символы | LLM-вызовы | prompt ток | completion ток | wall с')
    L.append('---|---|---|---|---|---|---|---')
    for s in active_scales:
        p = BASE / 'results' / f'gen_corpus_{s}.json'
        if p.exists():
            d = json.loads(p.read_text())
            L.append(f"{s} | {d.get('entities')} | {d.get('chunks')} | {d.get('chars')} | "
                     f"{d.get('llm_calls')} | {d.get('llm_prompt_tokens')} | {d.get('llm_completion_tokens')} | {d.get('llm_wall_s')}")

    (out / 'report.md').write_text('\n'.join(L))
    print(f'wrote summary.csv ({len(ROWS)} rows) and report.md')


if __name__ == '__main__':
    main()
