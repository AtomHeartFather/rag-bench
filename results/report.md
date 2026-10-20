# RAG-бенчмарк: file (BM25) vs fts (SQLite FTS5) vs rag (Qdrant+bge-m3)

Матрица: 3 варианта × 3 реальных датасета (5k–66k чанков), 1800 запросов.


## Релевантность (retrieval, Hit@10)

масштаб | file | fts | rag
---|---|---|---|
real-mh | 85.7% | 65.3% | 83.7%
real-rag-10k | 88.2% | 90.4% | 96.6%
real-rag-66k | 94.1% | 96.8% | 98.9%

## МRR@10

масштаб | file | fts | rag
---|---|---|---|
real-mh | 0.55 | 0.38 | 0.50
real-rag-10k | 0.60 | 0.64 | 0.78
real-rag-66k | 0.73 | 0.80 | 0.93

## Hit@5

масштаб | file | fts | rag
---|---|---|---|
real-mh | 74.7% | 56.3% | 70.0%
real-rag-10k | 81.0% | 82.2% | 92.2%
real-rag-66k | 86.3% | 91.6% | 98.1%

## Overlap@5

масштаб | file | fts | rag
---|---|---|---|
real-mh | 0.33 | 0.23 | 0.34
real-rag-10k | 0.51 | 0.54 | 0.63
real-rag-66k | 0.31 | 0.31 | 0.33

## Latency p50 (мс)

масштаб | file | fts | rag
---|---|---|---|
real-mh | 30.98 | 1.15 | 21.90
real-rag-10k | 18.21 | 1.41 | 11.87
real-rag-66k | 317.35 | 5.90 | 13.48

## Latency p95 (мс)

масштаб | file | fts | rag
---|---|---|---|
real-mh | 45.92 | 1.70 | 25.34
real-rag-10k | 31.10 | 2.73 | 13.65
real-rag-66k | 642.58 | 13.60 | 16.98

## Latency p99 (мс)

масштаб | file | fts | rag
---|---|---|---|
real-mh | 50.32 | 2.28 | 26.71
real-rag-10k | 36.05 | 3.25 | 14.71
real-rag-66k | 972.18 | 19.37 | 18.62

## Индексация: wall (с)

масштаб | file | fts | rag
---|---|---|---|
real-mh | 0.41 | 0.78 | 169.15
real-rag-10k | 0.60 | 1.01 | 366.31
real-rag-66k | 3.65 | 4.06 | 1451.13

## Индексация: пик RAM (МБ)

масштаб | file | fts | rag
---|---|---|---|
real-mh | 1830.40 | 222.00 | 222.00
real-rag-10k | 1900.60 | 231.30 | 230.40
real-rag-66k | 2483.70 | 326.00 | 325.10

## Индексация: CPU-с

масштаб | file | fts | rag
---|---|---|---|
real-mh | 3.06 | 0.77 | 4.80
real-rag-10k | 3.34 | 0.99 | 10.37
real-rag-66k | 6.39 | 4.02 | 55.80

## Индексация: диск (МБ)

масштаб | file | fts | rag
---|---|---|---|
real-mh | — | — | —
real-rag-10k | — | — | —
real-rag-66k | — | — | —

## Hit@10 по уровням сложности

масштаб | variant | L1 | L2 | L3 | L4 | L5
---|---|---|---|---|---|---
real-mh | file | — | — | 74% | 92% | 87%
real-mh | fts | — | — | 56% | 73% | 65%
real-mh | rag | — | — | 79% | 86% | 85%
real-rag-10k | file | 94% | — | 81% | 77% | —
real-rag-10k | fts | 95% | — | 85% | 81% | —
real-rag-10k | rag | 99% | — | 96% | 91% | —
real-rag-66k | file | — | — | 95% | 93% | —
real-rag-66k | fts | — | — | 96% | 97% | —
real-rag-66k | rag | — | — | 98% | 100% | —

## Стоимость (токены)

масштаб | variant | embed-токены (индексация) | embed tok/с | judge LLM prompt | judge LLM compl
---|---|---|---|---|---
real-mh | file | — | — | — | —
real-mh | fts | — | — | — | —
real-mh | rag | 1571891 | 9588.4 | — | —
real-rag-10k | file | — | — | — | —
real-rag-10k | fts | — | — | — | —
real-rag-10k | rag | 2294719 | 6459.9 | — | —
real-rag-66k | file | — | — | — | —
real-rag-66k | fts | — | — | — | —
real-rag-66k | rag | 8899371 | 6393.8 | — | —