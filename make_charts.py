#!/usr/bin/env python3
"""Generate charts for README from results/summary.csv."""
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

SCALES = ["1k", "10k", "real-mh", "real-rag-10k", "real-rag-66k"]
LABELS = ["1k (500 chunks)", "10k (5k chunks)", "real-mh (5k, multi-hop)",
          "real-rag-10k (11k)", "real-rag-66k (66k)"]
VARIANTS = ["file", "fts", "rag"]
COLORS = {"file": "#9aa0a6", "fts": "#f9ab00", "rag": "#1a73e8"}

# ---- Chart 1: Hit@10 by scale ----
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.26
for i, v in enumerate(VARIANTS):
    vals = [data[(v, s)].get("rel_hit_10", 0) * 100 for s in SCALES]
    ax.bar([x + i * w for x in range(len(SCALES))], vals, w,
           label=v, color=COLORS[v])
    for x, val in zip(range(len(SCALES)), vals):
        ax.text(x + i * w, val + 0.6, f"{val:.0f}", ha="center", fontsize=8)
ax.set_xticks([x + w for x in range(len(SCALES))])
ax.set_xticklabels(LABELS, rotation=12)
ax.set_ylim(0, 108)
ax.set_ylabel("Hit@10, % of queries where the right document is in top-10")
ax.set_title("Search accuracy by corpus size\n3 memory types, same corpus and queries")
ax.legend(title="variant")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "accuracy_by_scale.png"), dpi=140)
plt.close()

# ---- Chart 2: p95 latency (log) by scale ----
fig, ax = plt.subplots(figsize=(9, 5))
for i, v in enumerate(VARIANTS):
    vals = [data[(v, s)].get("search_latency_p95_ms", 0) for s in SCALES]
    ax.plot(LABELS, vals, marker="o", label=v, color=COLORS[v], linewidth=2)
    for x, val in zip(range(len(SCALES)), vals):
        ax.annotate(f"{val:.0f}ms" if val >= 10 else f"{val:.1f}ms",
                    (x, val), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("search latency p95, ms (log scale)")
ax.set_title("Search speed: 95th percentile\n(file variant degrades with corpus size — it re-scores everything)")
ax.legend(title="variant")
ax.grid(alpha=0.3)
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "latency_p95_by_scale.png"), dpi=140)
plt.close()

# ---- Chart 3: index cost: RAM peak + wall time ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
for i, v in enumerate(VARIANTS):
    ram = [data[(v, s)].get("index_ram_peak_kb", 0) / 1024 / 1024 for s in SCALES]
    wall = [data[(v, s)].get("index_wall_s", 0) for s in SCALES]
    a1.bar([x + i * w for x in range(len(SCALES))], ram, w, label=v, color=COLORS[v])
    a2.bar([x + i * w for x in range(len(SCALES))], wall, w, label=v, color=COLORS[v])
a1.set_xticks([x + w for x in range(len(SCALES))]); a1.set_xticklabels(LABELS, rotation=12, fontsize=8)
a2.set_xticks([x + w for x in range(len(SCALES))]); a2.set_xticklabels(LABELS, rotation=12, fontsize=8)
a1.set_ylabel("peak RAM during index build, GB"); a2.set_ylabel("index build wall time, s")
a1.set_title("RAM"); a2.set_title("Time")
fig.suptitle("One-time cost of building the index (all data is ingested once)")
for a in (a1, a2):
    a.legend(title="variant"); a.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "index_cost.png"), dpi=140)
plt.close()

# ---- Chart 4: Hit@10 by difficulty (1k and 10k) ----
fig, ax = plt.subplots(figsize=(9, 4.8))
levels = ["rel_L1_hit_10", "rel_L2_hit_10", "rel_L3_hit_10", "rel_L4_hit_10", "rel_L5_hit_10"]
llabels = ["L1 exact fact", "L2 paraphrase", "L3 multi-doc", "L4 distractors", "L5 aggregation"]
for i, v in enumerate(VARIANTS):
    for s, off in (("1k", -0.18), ("10k", 0.18)):
        vals = [data[(v, s)].get(l, 0) * 100 for l in levels]
        ax.plot(range(5), vals, marker="o", linestyle="-" if s == "1k" else "--",
                label=f"{v} 1k" if s == "1k" else f"{v} 10k", color=COLORS[v],
                linewidth=1.8 if s == "1k" else 1.2, alpha=1.0 if s == "1k" else 0.55)
ax.set_xticks(range(5)); ax.set_xticklabels(llabels)
ax.set_ylabel("Hit@10, %")
ax.set_ylim(40, 106)
ax.set_title("Accuracy by query difficulty (solid = 1k corpus, dashed = 10k)\nfile breaks on L5 aggregation; all methods are ~100% on simple questions")
ax.legend(ncol=3, fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "images", "difficulty_by_level.png"), dpi=140)
plt.close()
print("charts ok")
