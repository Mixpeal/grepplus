# Progressive JIT Indexing for Agent Code Search

**Working title.** Draft outline for Phase 3 paper.

## Abstract

Agent code search tools face a tension: lexical search (grep) is instant but misses paraphrases; semantic search requires upfront embedding of the entire corpus. We present **grepplus**, a hybrid CLI that never requires a full corpus embed before the first query. A progressive embed ladder—sketch shell → budgeted JIT reheat → per-file HOT cache—matches grep's zero-setup model while reaching full-index recall after session warmup (~70ms/query on Sendtrill).

## Contributions

1. **JIT progressive embed ladder** with per-file temperature (HOT/COLD/COOL) and content-hash invalidation.
2. **Hybrid candidate funnel**: Laser ∪ SketchBeam (MinHash + BM25) → PQ4 semantic rank, with RRF fusion.
3. **Learned PQ4 projection** (Track 2): margin-trained linear projection preserving kNN order under 4-bit quantization vs PCA/baseline ablations.
4. **AgentCode eval harness**: L/P/H/X query taxonomy with cold@1 / warm@2+ latency metrics.
5. **Learned router** (Track 3): feature + trace-trained route selection (grep vs semantic vs hybrid).

## Evaluation plan

| Experiment | Corpus | Metrics |
|------------|--------|---------|
| Mode comparison | Sendtrill (30q), AgentCode (100q) | recall@10, MRR, cold@1, warm@2+ |
| Projection ablation | AgentCode | baseline vs PCA vs learned PQ4 |
| JIT economics | Sendtrill | recall vs cumulative embed budget |
| Router ablation | AgentCode traces | route accuracy, end-to-end success |

## Commands (reproducibility)

```bash
grepplus eval report ./eval/agentcode/repos/mini \
  --suite ./eval/agentcode/queries-100.jsonl \
  --modes laser,vector,hybrid,jit --ensure-index --isolate-modes

grepplus research pq4 ablate ./eval/agentcode/repos/mini \
  --suite ./eval/agentcode/queries-100.jsonl --yes-download

grepplus router train --traces ~/.grepplus/traces/routes.jsonl
```

## Related work

- *Is Grep All You Need?* — grep vs vector in agent harnesses; we add hybrid + JIT indexing.
- Cursor / Cody index-at-open — full upfront embed; grepplus defers embed cost.

## Status

Phase 2–3 implementation complete. Full AgentCode release and multi-corpus tables: Phase 4.
