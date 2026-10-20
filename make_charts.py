#!/usr/bin/env python3
"""Generate charts for README from results/summary.csv (real datasets)."""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(BASE, "results", "summary.csv"))))

data = defaultdict(dict)
for r in rows:
    data[(r["variant"], r["scale"])][r["metric"]] = float(r["value"])

SCALES = ["real-mh", "real-rag-10k", "real-rag-66k"]
LABELS = ["MultiHop-RAG\n(5k chunks, 300q)", "RGB\n(11k passages, 500q)",
          "HotpotQA fullwiki\n(66k paragraphs, 1000q)"]
VARIANTS = ["file", "fts", "rag"]
COLORS = {"file": "#9aa0a6", "fts": "#f9ab00", "rag": "#1a73e8"}
VNAMES = {"file": "file (BM25)", "fts": "fts (SQLite FTS5)", "rag": "rag (Qdrant+bge-m3)"}

# ---- Chart 1: Hit@10 by dataset ----
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.26
for i, v in enumerate(VARIANTS):
    vals = [data[(v, s)].get("rel_hit_10", 0) * 100 for s in SCALES]
    ax.bar([x + i * w for x in range(len(SCALES))], vals, w,
           label=VNAMES[v], color=COLORS[v])
    for x, val in zip(range(len(SCALES)), vals):
        ax.text(x + i * w, val + 0.8, f"{val:.0f}", ha="center", fontsize=9)
ax.set_xticks([x + w for x in range(len(SCALES))])
ax.set_xticklabels(LABELS)
ax.set_ylim(0, 110)
ax.set_ylabel("Hit@10: % of queries where the right document is in top-10")
ax.set_title("Search accuracy on real datasets\nsame corpus and queries for all 3 variants")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "accuracy_by_scale.png"), dpi=140)
plt.close()

# ---- Chart 2: p95 latency (log) ----
fig, ax = plt.subplots(figsize=(9, 5))
for v in VARIANTS:
    vals = [data[(v, s)].get("search_latency_p95_ms", 0) for s in SCALES]
    ax.plot(LABELS, vals, marker="o", label=VNAMES[v], color=COLORS[v], linewidth=2)
    for x, val in zip(range(len(SCALES)), vals):
        ax.annotate(f"{val:.0f}ms" if val >= 10 else f"{val:.1f}ms",
                    (x, val), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
ax.set_yscale("log")
ax.set_ylabel("search latency p95, ms (log scale)")
ax.set_title("Search speed: 95th percentile\nthe file variant re-scores the whole corpus per query and degrades with size")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "latency_p95_by_scale.png"), dpi=140)
plt.close()

# ---- Chart 3: index cost: RAM peak + wall time ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
for i, v in enumerate(VARIANTS):
    ram = [data[(v, s)].get("index_ram_peak_kb", 0) / 1024 / 1024 for s in SCALES]
    wall = [data[(v, s)].get("index_wall_s", 0) for s in SCALES]
    a1.bar([x + i * w for x in range(len(SCALES))], ram, w, label=VNAMES[v], color=COLORS[v])
    a2.bar([x + i * w for x in range(len(SCALES))], wall, w, label=VNAMES[v], color=COLORS[v])
for a in (a1, a2):
    a.set_xticks([x + w for x in range(len(SCALES))])
    a.set_xticklabels(LABELS, fontsize=8)
    a.legend()
    a.grid(axis="y", alpha=0.3)
a1.set_ylabel("peak RAM during index build, GB")
a2.set_ylabel("index build wall time, s")
a1.set_title("RAM"); a2.set_title("Time")
fig.suptitle("One-time cost of building the index (each corpus is ingested once)")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "index_cost.png"), dpi=140)
plt.close()

# ---- Chart 4: Hit@10 by query difficulty (MultiHop-RAG, multi-hop questions) ----
fig, ax = plt.subplots(figsize=(9, 5))
levels = ["rel_L3_hit_10", "rel_L4_hit_10", "rel_L5_hit_10"]
llabels = ["L3 bridge\n(question spans 2+ docs)", "L4 comparison\n(compare several entities)", "L5 inference\n(answer not stated verbatim)"]
for v in VARIANTS:
    vals = [data[(v, "real-mh")].get(l, 0) * 100 for l in levels]
    ax.plot(range(3), vals, marker="o", label=VNAMES[v], color=COLORS[v], linewidth=2)
    for x, val in zip(range(3), vals):
        ax.annotate(f"{val:.0f}%", (x, val), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(llabels)
ax.set_ylabel("Hit@10, %")
ax.set_ylim(40, 108)
ax.set_title("Query difficulty matters: on multi-hop questions (MultiHop-RAG)\nkeyword search drops hard, semantic search holds")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "difficulty_by_level.png"), dpi=140)
plt.close()
print("charts ok")
