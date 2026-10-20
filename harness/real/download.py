#!/usr/bin/env python3
"""Stage 1: download real datasets (RGB + HotpotQA) into corpus/raw_real/, analyze."""
import json, io, os, sys, time
from pathlib import Path
import requests

PY_DIR = Path(__file__).resolve().parent.parent.parent
RAW = PY_DIR / "corpus" / "raw_real"
RAW.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (rag-bench/1.0)"

def get(url, dest, force=False):
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  cached {dest.name} {dest.stat().st_size/1e6:.1f} MB")
        return dest
    t0 = time.time()
    print(f"  GET {url}")
    r = S.get(url, timeout=600, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for ch in r.iter_content(1 << 20):
            f.write(ch)
    print(f"    {dest.name} {dest.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s")
    return dest

print("== RGB (github raw):")
for name in ["en.json", "en_refine.json", "en_int.json", "en_fact.json"]:
    get(f"https://raw.githubusercontent.com/chen700564/RGB/master/data/{name}", RAW / name)

print("== HotpotQA fullwiki (HF parquet):")
for name in ["fullwiki_validation.parquet", "distractor_validation.parquet"]:
    cfg = "fullwiki" if "fullwiki" in name else "distractor"
    get(f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/{cfg}/validation-00000-of-00001.parquet",
        RAW / name)

# ---- analysis ----
print("\n== ANALYSIS ==")
import pyarrow.parquet as pq

def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

en = load_jsonl(RAW / "en.json")
en_refine = load_jsonl(RAW / "en_refine.json")
en_int = load_jsonl(RAW / "en_int.json")
en_fact = load_jsonl(RAW / "en_fact.json")
print(f"RGB: en={len(en)} refine={len(en_refine)} int={len(en_int)} fact={len(en_fact)}")

# passage lengths
import statistics
def plen(rows, key):
    ls = []
    for x in rows:
        for p in x.get(key, []):
            if isinstance(p, list):
                ls += [len(s) for s in p if isinstance(s, str)]
            else:
                ls.append(len(p))
    return ls
for rows, name in [(en, "en"), (en_refine, "en_refine"), (en_int, "en_int")]:
    pl = plen(rows, "positive") + plen(rows, "negative")
    if pl:
        print(f"{name}: passages={len(pl)} len p50={statistics.median(pl):.0f} p95={sorted(pl)[int(len(pl)*0.95)]} max={max(pl)}")

# ids alignment
ids_en = {x["id"] for x in en}
ids_refine = {x["id"] for x in en_refine}
print(f"en ids {min(ids_en)}-{max(ids_en)}; refine ids align with en: {ids_refine <= ids_en or ids_en <= ids_refine}")

# gold coverage in en (does gold appear in positives?)
def flatten(x, key):
    out = []
    for p in x.get(key, []):
        if isinstance(p, list):
            out += [s for s in p if isinstance(s, str)]
        else:
            out.append(p)
    return out
ok = 0
for x in en:
    ans = x.get("answer") or []
    flat = []
    for a in ans:
        if isinstance(a, list):
            flat += a
        else:
            flat.append(a)
    pos = " ".join(flatten(x, "positive"))
    if flat and any(a.lower() in pos.lower() for a in flat):
        ok += 1
print(f"en: {ok}/{len(en)} have answer text inside positive passages")

# hotpot: build gold-passage union
tbl = pq.read_table(io.BytesIO((RAW / "fullwiki_validation.parquet").read_bytes())).to_pylist()
print(f"\nHotpotQA fullwiki val: {len(tbl)} rows")
titles = {}
gold_titles = set()
for r in tbl:
    for t, s in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
        gold_titles.add(t)
print(f"distinct gold paragraph titles: {len(gold_titles)}")
# distractor val: how many paragraphs total (10/row), union
td = pq.read_table(io.BytesIO((RAW / "distractor_validation.parquet").read_bytes())).to_pylist()
all_paras = {}
for r in td:
    for t, sents in zip(r["context"]["title"], r["context"]["sentences"]):
        if t not in all_paras:
            all_paras[t] = " ".join(sents)
print(f"distinct paragraphs in distractor-val contexts: {len(all_paras)}")
gl = [len(all_paras[t]) for t in gold_titles if t in all_paras]
print(f"gold paragraphs found in that pool: {len(gl)}/{len(gold_titles)}; len p50={statistics.median(gl) if gl else 0:.0f} mean={statistics.fmean(gl) if gl else 0:.0f}")
import collections
lv = collections.Counter(r["level"] for r in tbl)
ty = collections.Counter(r["type"] for r in tbl)
print("hotpot level dist:", dict(lv), " type dist:", dict(ty))
print("\nDONE")
