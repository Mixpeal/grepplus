# AgentCode — Track 1 eval corpus

Seed query set for grep+ hybrid search evaluation.

## Layout

- `repos/` — small checked-in repos (or fetch script)
- `queries.jsonl` — 20 seed queries with oracle annotations
- `queries-100.jsonl` — 100 expanded queries (Phase 2 scale set)

## Query categories

| Code | Meaning |
|------|---------|
| L | Literal — symbol/string search |
| P | Paraphrase — concept ≠ surface text |
| H | Hybrid — symbol + concept |
| X | Cross-file — evidence split across modules |

## Usage

```bash
grepplus eval run ./eval/agentcode/repos/mini --suite ./eval/agentcode/queries-100.jsonl
grepplus eval compare ./eval/agentcode/repos/mini --modes laser,vector,hybrid,jit --ensure-index
```
