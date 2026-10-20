#!/usr/bin/env python3
"""Convert downloaded real datasets (corpus/raw_real/) to harness format:
  corpus/chunks_{scale}.jsonl  +  queries/queries_{scale}.jsonl
Scales:
  real-mh       : MultiHop-RAG  (609 news articles corpus, 300 queries)
  real-rag-10k  : RGB           (~10k passages corpus, 500 queries)
  real-rag-66k  : HotpotQA      (66,581 paragraphs corpus, 1000 queries)
Chunk schema: {chunk_id, entity_id, doc_type, text, ts, topic_tags}
Query schema: {query_id, level, text, entity_ids, gold_chunk_ids, answer_keys, difficulty, field}
Deterministic: seed=42, no LLM calls.
"""
import json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
import bench  # noqa: E402  (make_chunks, tokenize)

BASE = bench.BASE
RAW = BASE / "corpus" / "raw_real"
rng = random.Random(42)
TS = "2026-10-20"

def slug(s, maxlen=60):
    s = re.sub(r"[^a-zа-яё0-9]+", "_", s.lower()).strip("_")
    return (s or "doc")[:maxlen]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.relative_to(BASE)}: {len(rows)} rows")

def make_chunk(cid, entity_id, doc_type, text, tags):
    return {"chunk_id": cid, "entity_id": entity_id, "doc_type": doc_type,
            "text": text, "ts": TS, "topic_tags": tags}

# ================= MultiHop-RAG -> real-mh =================
def convert_mh():
    scale = "real-mh"
    print("== MultiHop-RAG ->", scale)
    corpus = json.loads((RAW / "corpus.json").read_text())
    qa = json.loads((RAW / "MultiHopRAG.json").read_text())
    chunks, title2chunks = [], {}
    for i, art in enumerate(corpus):
        body = art.get("body") or ""
        for ctext in bench.make_chunks(body, bench.CHUNK_TARGET):
            cid = f"mh_{len(chunks):05d}"
            chunks.append(make_chunk(cid, slug(art["url"]), "article", ctext,
                                     ["news", art.get("category") or "general"]))
            title2chunks[cid] = len(chunks) - 1
    # gold: evidence fact text -> containing chunks
    def find_chunks(fact):
        fact_l = fact.lower()
        hits = []
        for c in chunks:
            if fact_l in c["text"].lower():
                hits.append(c["chunk_id"])
        if hits:
            return hits
        # fallback: token overlap
        ft = set(bench.tokenize(fact))
        if not ft:
            return []
        best = []
        for c in chunks:
            ct = set(bench.tokenize(c["text"]))
            if len(ct):
                ov = len(ft & ct) / len(ft)
                if ov >= 0.6:
                    best.append((ov, c["chunk_id"]))
        best.sort(reverse=True)
        return [cid for _ov, cid in best[:3]]
    lvl_map = {"bridge_query": "L3", "composition_query": "L3",
               "comparison_query": "L4", "inference_query": "L5"}
    rng.shuffle(qa)
    queries, dropped = [], 0
    for q in qa:
        gold = set()
        for ev in q["evidence_list"]:
            gold |= set(find_chunks(ev.get("fact") or ""))
        if not gold:
            dropped += 1
            continue
        queries.append({"query_id": f"MH-{len(queries)+1:03d}",
                        "level": lvl_map.get(q["question_type"], "L3"),
                        "text": q["query"], "entity_ids": [],
                        "gold_chunk_ids": sorted(gold),
                        "answer_keys": [q["answer"]],
                        "difficulty": 3, "field": q["question_type"]})
        if len(queries) >= 300:
            break
    write_jsonl(BASE / "corpus" / f"chunks_{scale}.jsonl", chunks)
    write_jsonl(BASE / "queries" / f"queries_{scale}.jsonl", queries)
    print(f"  dropped (no gold found): {dropped}; corpus chars: {sum(len(c['text']) for c in chunks)}")

# ================= RGB -> real-rag-10k =================
def _rgb_passages(rows, key):
    out = []
    for x in rows:
        for p in x.get(key, []):
            if isinstance(p, list):
                out += [s for s in p if isinstance(s, str)]
            else:
                out.append(p)
    return out

def convert_rgb():
    scale = "real-rag-10k"
    print("== RGB ->", scale)
    def loadl(name):
        return [json.loads(l) for l in (RAW / name).read_text().splitlines() if l.strip()]
    en = loadl("en.json")
    en_refine = {x["id"]: x for x in loadl("en_refine.json")}
    en_int = loadl("en_int.json")
    en_fact = loadl("en_fact.json")
    # corpus: refine (300) + int (100) + fact (100) passages, dedup by normalized text
    seen, chunks, text2cid = set(), [], {}
    def add_passage(p, owner, kind):
        key = " ".join(p.lower().split())
        if key in seen or len(p.strip()) < 40:
            return text2cid.get(key)
        seen.add(key)
        cid = f"rgb_{len(chunks):05d}"
        chunks.append(make_chunk(cid, owner, kind, p.strip(), ["rgb", kind]))
        text2cid[key] = cid
        return cid
    q2cid = {}  # (qid, pos_idx, kind) -> cid
    def qpassages(qid, row):
        """gold positive passage chunk-ids for a row (refine preferred)."""
        ids = []
        idx = 0
        for p in _rgb_passages([row], "positive"):
            owner = f"rgb_q{qid:03d}"
            cid = text2cid.get(" ".join(p.lower().split()))
            if cid is None:
                cid = add_passage(p, owner, "gold")
            ids.append(cid)
            idx += 1
        return ids
    # order: refine positives+negatives, then int, then fact (stable dedup)
    for qid in sorted(en_refine):
        row = en_refine[qid]
        for p in _rgb_passages([row], "positive"):
            add_passage(p, f"rgb_q{qid:03d}", "gold")
        for p in _rgb_passages([row], "negative"):
            add_passage(p, f"rgb_q{qid:03d}", "noise")
    for row in en_int:
        for p in _rgb_passages([row], "positive"):
            add_passage(p, f"rgb_int{row['id']:03d}", "gold")
        for p in _rgb_passages([row], "negative"):
            add_passage(p, f"rgb_int{row['id']:03d}", "noise")
    for row in en_fact:
        for p in _rgb_passages([row], "positive"):
            add_passage(p, f"rgb_fact{row['id']:03d}", "gold")
        for p in _rgb_passages([row], "negative"):
            add_passage(p, f"rgb_fact{row['id']:03d}", "noise")

    def flat_answers(row):
        out = []
        for a in row.get("answer", []):
            if isinstance(a, list):
                out += [str(x) for x in a if isinstance(x, str)]
            else:
                out.append(str(a))
        return out

    queries = []
    # L1: 300 factoid (refine corrected)
    for qid in sorted(en_refine):
        row = en_refine[qid]
        gold = []
        for p in _rgb_passages([row], "positive"):
            key = " ".join(p.lower().split())
            cid = text2cid.get(key)
            if cid:
                gold.append(cid)
        if gold:
            queries.append({"query_id": f"RAG-L1-{len([q for q in queries if q['level']=='L1'])+1:03d}",
                            "level": "L1", "text": row["query"], "entity_ids": [],
                            "gold_chunk_ids": sorted(set(gold)),
                            "answer_keys": flat_answers(row),
                            "difficulty": 1, "field": "factoid"})
    # L3: 100 information-integration (multi-passage, 2-part answer)
    for row in en_int:
        gold = []
        for p in _rgb_passages([row], "positive"):
            key = " ".join(p.lower().split())
            cid = text2cid.get(key)
            if cid:
                gold.append(cid)
        if gold:
            queries.append({"query_id": f"RAG-L3-{len([q for q in queries if q['level']=='L3'])+1:03d}",
                            "level": "L3", "text": row["query"], "entity_ids": [],
                            "gold_chunk_ids": sorted(set(gold)),
                            "answer_keys": flat_answers(row),
                            "difficulty": 3, "field": "integration"})
    # L4: 100 counterfactual robustness (false premise)
    for row in en_fact:
        gold = []
        for p in _rgb_passages([row], "positive"):
            key = " ".join(p.lower().split())
            cid = text2cid.get(key)
            if cid:
                gold.append(cid)
        if gold:
            queries.append({"query_id": f"RAG-L4-{len([q for q in queries if q['level']=='L4'])+1:03d}",
                            "level": "L4", "text": row["query"], "entity_ids": [],
                            "gold_chunk_ids": sorted(set(gold)),
                            "answer_keys": flat_answers(row),
                            "difficulty": 4, "field": "counterfactual"})
    # renumber query_ids cleanly per level
    n = {}
    for q in queries:
        n[q["level"]] = n.get(q["level"], 0) + 1
        q["query_id"] = f"RAG-{q['level']}-{n[q['level']]:03d}"
    write_jsonl(BASE / "corpus" / f"chunks_{scale}.jsonl", chunks)
    write_jsonl(BASE / "queries" / f"queries_{scale}.jsonl", queries)
    print(f"  corpus chars: {sum(len(c['text']) for c in chunks)}")

# ================= HotpotQA -> real-rag-66k =================
def convert_hotpot():
    scale = "real-rag-66k"
    print("== HotpotQA ->", scale)
    import io
    import pyarrow.parquet as pq
    td = pq.read_table(io.BytesIO((RAW / "distractor_validation.parquet").read_bytes())).to_pylist()
    fw = pq.read_table(io.BytesIO((RAW / "fullwiki_validation.parquet").read_bytes())).to_pylist()
    # corpus: union of distractor-val context paragraphs
    paras = {}
    for r in td:
        for t, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            if t not in paras:
                paras[t] = " ".join(s for s in sents if isinstance(s, str))
    title2cid = {}
    chunks = []
    for i, t in enumerate(sorted(paras)):
        cid = f"hp_{i:05d}"
        chunks.append(make_chunk(cid, slug(t), "paragraph", paras[t], ["wikipedia", "hotpotqa"]))
        title2cid[t] = cid
    queries = []
    by_type = {"bridge": [], "comparison": []}
    for r in fw:
        by_type.setdefault(r["type"], []).append(r)
    for ty in ("bridge", "comparison"):
        rows = by_type.get(ty, [])
        rng.shuffle(rows)
        for r in rows[:500]:
            gold = set()
            for t in r["supporting_facts"]["title"]:
                if t in title2cid:
                    gold.add(title2cid[t])
            if not gold:
                continue
            level = "L3" if ty == "bridge" else "L4"
            queries.append({"query_id": f"HP-{level}-{len([q for q in queries if q['level']==level])+1:03d}",
                            "level": level, "text": r["question"], "entity_ids": [],
                            "gold_chunk_ids": sorted(gold),
                            "answer_keys": [str(r["answer"])],
                            "difficulty": 3 if ty == "bridge" else 2,
                            "field": ty, "hotpot_level": r["level"]})
    n = {}
    for q in queries:
        n[q["level"]] = n.get(q["level"], 0) + 1
        q["query_id"] = f"HP-{q['level']}-{n[q['level']]:03d}"
    write_jsonl(BASE / "corpus" / f"chunks_{scale}.jsonl", chunks)
    write_jsonl(BASE / "queries" / f"queries_{scale}.jsonl", queries)
    print(f"  corpus chars: {sum(len(c['text']) for c in chunks)}")

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else ["mh", "rgb", "hotpot"]
    if "mh" in only:
        convert_mh()
    if "rgb" in only:
        convert_rgb()
    if "hotpot" in only:
        convert_hotpot()
    print("DONE")
