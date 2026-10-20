#!/usr/bin/env bash
# Full run over real-data scales: index x3 + search x3 per scale, then report.
cd /home/egor/workspace/tasks/rag-benchmark || exit 99
PY=/home/egor/.hermes/installs/b9b919e706bd0f05/environments/a626f35f6f6549eca5868f86a6a2bb32/venv/bin/python3
STATUS=logs/real_status.txt
: > "$STATUS"
echo "=== [$(date '+%F %T')] REAL_PIPELINE_START ===" >> "$STATUS"
run() {
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] START $name ===" >> "$STATUS"
  if "$@" >> "logs/real_${name}.log" 2>&1; then
    echo "=== [$(date '+%F %T')] OK $name ===" >> "$STATUS"
  else
    echo "=== [$(date '+%F %T')] FAIL $name (exit $?) ===" >> "$STATUS"
    return 1
  fi
}
for S in real-mh real-rag-10k real-rag-66k; do
  for V in file fts rag; do
    run "index_${V}_${S}"  "$PY" harness/bench.py index  --variant "$V" --scale "$S" || { echo "FAILED index $V $S" >> "$STATUS"; exit 1; }
  done
  for V in file fts rag; do
    run "search_${V}_${S}" "$PY" harness/bench.py search --variant "$V" --scale "$S" --k 10 || { echo "FAILED search $V $S" >> "$STATUS"; exit 1; }
  done
done
run report "$PY" harness/report.py || { echo "FAILED report" >> "$STATUS"; exit 1; }
echo "=== [$(date '+%F %T')] REAL_PIPELINE_DONE ===" >> "$STATUS"
