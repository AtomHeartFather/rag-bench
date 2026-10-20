#!/usr/bin/env bash
# Overnight pipeline for scale 10k: gen-corpus -> gen-queries -> index x3 -> search x3 -> report.
# Gates: stop if gen-corpus fails (don't index partial data). Progress -> logs/pipeline_status.txt.
cd /home/egor/workspace/tasks/rag-benchmark || exit 99
PY=/home/egor/.hermes/installs/b9b919e706bd0f05/environments/a626f35f6f6549eca5868f86a6a2bb32/venv/bin/python3
S=10k
STATUS=logs/pipeline_status.txt
: > "$STATUS"
echo "=== [$(date '+%F %T')] PIPELINE_START $S (cache=$(ls corpus/llm_cache | wc -l)) ===" >> "$STATUS"

run() {
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] START $name ===" >> "$STATUS"
  if "$@" >> "logs/pipe_${name}.log" 2>&1; then
    echo "=== [$(date '+%F %T')] OK $name ===" >> "$STATUS"
  else
    local rc=$?
    echo "=== [$(date '+%F %T')] FAIL $name (exit $rc) ===" >> "$STATUS"
    return 1
  fi
}

run gen_corpus     "$PY" harness/bench.py gen-corpus  --scale "$S"   || { echo "GEN_CORPUS_FAILED"  >> "$STATUS"; exit 1; }
run gen_queries    "$PY" harness/bench.py gen-queries --scale "$S"   || { echo "GEN_QUERIES_FAILED" >> "$STATUS"; exit 1; }
for V in file fts rag; do
  run "index_${V}"   "$PY" harness/bench.py index  --variant "$V" --scale "$S" || { echo "INDEX_${V}_FAILED"   >> "$STATUS"; exit 1; }
done
for V in file fts rag; do
  run "search_${V}"  "$PY" harness/bench.py search --variant "$V" --scale "$S" --k 10 || { echo "SEARCH_${V}_FAILED" >> "$STATUS"; exit 1; }
done
run report        "$PY" harness/report.py                                             || { echo "REPORT_FAILED" >> "$STATUS"; exit 1; }
echo "=== [$(date '+%F %T')] PIPELINE_DONE $S (cache=$(ls corpus/llm_cache | wc -l)) ===" >> "$STATUS"
