# grep+

**grep+** (`grepplus`, `gp`) is a hybrid code search CLI for developers and coding agents. It stays **fast by default** (grep works with zero setup), adds meaning-aware search when you need it, and uses **progressive JIT indexing** so you are not forced to embed an entire repo before the first query.

- **Lexical:** parallel in-process regex engine (ripgrep optional for exact channel)
- **Semantic:** local ONNX embeddings via [ONNX Runtime](https://onnxruntime.ai/) — no cloud API
- **Hybrid:** Laser (exact) ∪ SketchBeam (MinHash + BM25) → semantic rank, fused with RRF
- **JIT:** sketch shell first, budgeted per-query reheat of cold files, per-file HOT/COLD/COOL cache

---

## Install

### Homebrew

```bash
brew install Mixpeal/grepplus/grepplus
```

To install the latest `main` branch:

```bash
brew install --HEAD Mixpeal/grepplus/grepplus
```

### From source

Requires **Rust 1.75+** (2021 edition).

```bash
git clone https://github.com/Mixpeal/grepplus.git
cd grepplus
cargo install --path crates/gp-cli --force
```

Binaries: `grepplus` and `gp` (identical).

Data lives under `~/.grepplus/`:

| Path                      | Purpose                                                  |
| ------------------------- | -------------------------------------------------------- |
| `~/.grepplus/models/`     | Downloaded ONNX weights + `manifest.json` per install    |
| `~/.grepplus/cache/`      | Per-repo search indexes (never inside your project tree) |
| `~/.grepplus/config.toml` | User config (created on first edit)                      |
| `.grepplus.toml`          | Optional repo-local config override                      |

---

## Quick start

```bash
# Status + tips (no pattern)
grepplus

# Literal search — no model required
grepplus paymentWebhook ./src

# Hybrid semantic + lexical — indexes automatically on first query
grepplus --hybrid "retry logic" ./src

# First-time model setup (interactive picker)
grepplus models install

# Optional: preheat the whole repo upfront (faster repeat queries)
grepplus index ./src --ensure-model
```

---

## Search

```bash
grepplus [OPTIONS] <PATTERN> [PATHS]...
```

Default path is `.`. With no pattern and no subcommand, grep+ prints a welcome banner (active model, quick start).

### Route selection

grep+ picks a **route** automatically from the query and repo state (index warm?, model installed?). Override with flags:

| Flag                                      | Route                              | Needs model |
| ----------------------------------------- | ---------------------------------- | ----------- |
| *(default)*                               | auto (heuristic or learned router) | depends     |
| `--semantic`                              | semantic vector search             | yes         |
| `--hybrid`                                | lexical + semantic fusion          | yes         |
| `--prefocus`                              | SketchBeam pre-focus → semantic    | yes         |
| `--route <grep|semantic|hybrid|prefocus>` | explicit route                     | if not grep |

If semantic/hybrid is requested but no model is installed, grep+ errors with setup instructions. For auto route without `--semantic`/`--hybrid`, it **falls back to grep** when no model is available.

```bash
grepplus --route-debug "auth middleware" ./src   # show route + rationale
```

### Grep-compatible flags

| Flag                    | Description                              |
| ----------------------- | ---------------------------------------- |
| `-i`, `--ignore-case`   | Case-insensitive                         |
| `-F`, `--fixed-strings` | Literal substring (no regex)             |
| `-n`, `--line-numbers`  | Print `file:line` prefixes (default: on) |

### Index & model flags

| Flag             | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `--ensure-model` | Download/install active model if missing (non-interactive)                 |
| `--yes-download` | Same, for scripts (alias behavior with ensure-model)                       |
| `--ensure-index` | Force sketch/warm build before search (search already auto-creates sketch) |
| `--warm-index`   | With `--ensure-index`, preheat full warm index when missing                |
| `--local-traces` | Append route/latency trace for router training                             |

### Examples

```bash
grepplus -i "TODO" ./src
grepplus --semantic "where is the webhook handler" .
grepplus --hybrid --ensure-index --warm-index "session refresh" ./api
grepplus --route grep 'fn main' .
```

---

## Models

Embedding models are **ONNX-only** (downloaded from Hugging Face). Each install gets a `manifest.json` with `quant`, `base_id`, pooling, and dimensions. Multiple quants of the same model are separate install IDs (e.g. `e5-small-v2-model_o4`).

### Recommended catalog

| ID                      | ~Size   | Notes                                            |
| ----------------------- | ------- | ------------------------------------------------ |
| `qwen3-embedding-0.6b`  | 573 MB  | **Default** — Qwen3 0.6B ONNX INT8, MRL 32–1024d |
| `bge-small-en-v1.5`     | 126 MB  | Fast English                                     |
| `nomic-embed-text-v1.5` | 105 MB+ | Multilingual; quants ~105–261 MB                 |
| `e5-small-v2`           | 32 MB+  | Multilingual; quants ~32–63 MB                   |
| `harrier-oss-v1-0.6b`   | 337 MB  | Code-oriented, 8k context                        |

### `models list`

Show installed variants grouped by family, with quant label, install id, size, and `[active]` marker.

```bash
grepplus models list
```

### `models install` / `models use`

Both open the **same interactive picker** when no id is given:

1. Catalog models (install or pick another quant)
2. Non-catalog installs from prior `models pull` (activate)
3. **Find more on Hugging Face…** (opens browser + paste `org/repo`)
4. **Skip for now**

```bash
grepplus models install    # first-time / guided setup
grepplus models use        # switch or install (same picker)
grepplus models use e5-small-v2-model_o4   # activate by id
```

Quant picker groups **Full precision** and **Quantized** variants; installed quants show `(installed)`.

### `models pull`

Install from catalog id or any Hugging Face repo with ONNX exports.

```bash
grepplus models pull qwen3-embedding-0.6b
grepplus models pull intfloat/e5-small-v2 --quant model_qint8_avx512_vnni
grepplus models pull nixiesearch/all-MiniLM-L6-v2-onnx --set-active
```

| Flag               | Description                                         |
| ------------------ | --------------------------------------------------- |
| `--revision <rev>` | HF revision (default: `main`)                       |
| `--quant <label>`  | ONNX variant (e.g. `model_q4f16`, `model`)          |
| `--as-id <id>`     | Override local install directory name               |
| `--yes-download`   | Skip quant picker; use recommended variant          |
| `--include-full`   | Show multi-GB full-precision `model.onnx` in picker |
| `--force`          | Re-download even if already installed               |
| `--pin`            | Save entry to user catalog override                 |
| `--set-active`     | Activate after install                              |

Two-step flow for bare `org/repo` pulls: ONNX export mirror (if needed) → quantization picker.

### `models remove`

```bash
grepplus models remove e5-small-v2-model_o4
```

### Environment

| Variable                          | Purpose                                           |
| --------------------------------- | ------------------------------------------------- |
| `HF_TOKEN` or `GREPPLUS_HF_TOKEN` | Hugging Face token for gated models / rate limits |

---

## Index

**You do not need to run `grepplus index` before semantic search.** The first `--semantic`, `--hybrid`, or `--prefocus` query on a repo creates a **sketch shell** automatically (chunk list + MinHash). grep+ then **JIT-reheats** files as you search — embedding cold files within a per-query budget and promoting frequently hit files to HOT in the cache.

`grepplus index` is for **preheating**: building or warming the cache *ahead of time* so later queries skip cold-start work. Literal grep never touches the index.

### What the index is

For hybrid/semantic routes, grep+ needs a local picture of the repo — source split into **chunks**, a **MinHash sketch** for fast lexical shortlist (SketchBeam), and **quantized embedding vectors** per chunk (filled upfront on warm build, or via JIT). The cache tracks per-file **temperature** (HOT / COLD / COOL) and content hashes so JIT reheat stays correct when files change.

| Artifact                      | Purpose                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| **Chunk text**                | Source passages with file + line range (what results point to)               |
| **MinHash sketch**            | Which files/chunks look textually related to the query                       |
| **Vector codes**              | Compact quantized embeddings per chunk (upfront or JIT)                      |
| **Per-file metadata**         | Hashes, chunk lists, temperature for progressive embed                       |
| **ANN graph** *(large repos)* | Approximate nearest-neighbor links when chunk count exceeds `ann_min_chunks` |

The index is tied to a **model id** and **embedding dimension**. Change model or dim → rebuild or let JIT repopulate under the new settings.

### Automatic vs manual

|                   | **Automatic (on search)**             | **Manual (`grepplus index`)**                     |
| ----------------- | ------------------------------------- | ------------------------------------------------- |
| **When**          | First semantic/hybrid query on a path | When you want to preheat before searching         |
| **Default build** | Sketch shell only (all files COLD)    | Your choice: `--sketch-only` or full warm         |
| **Embeddings**    | JIT per query + git to HOT over time  | `--ensure-model` warm build embeds everything now |
| **Typical use**   | Just search; zero prep                | CI, agents, or `--watch` for a always-warm repo   |

`--ensure-index` / `--warm-index` on search or `serve` are optional flags to force sketch or warm build *before* the query runs (useful in scripts). Normal interactive search already ensures a sketch shell internally.

### Sketch-only vs warm

| Mode                                                             | What’s stored upfront                          |
| ---------------------------------------------------------------- | ---------------------------------------------- |
| **Sketch-only** (auto on first search, or `index --sketch-only`) | Chunks + MinHash — no embeddings (files COLD)  |
| **Warm** (`index` without `--sketch-only`)                       | Full vectors for every chunk (all files HOT)   |

Warm indexes trade upfront time and disk for lower latency and higher first-query recall. Sketch + JIT is the default path: no full-repo embed required before the first question.

### Where it lives

Indexes live under `~/.grepplus/cache/`, keyed by repo path — never inside your project (`GREPPLUS_CACHE_DIR` to override).

```bash
grepplus index ./src --ensure-model          # preheat: full warm index
grepplus index ./src --sketch-only           # preheat: sketch shell only
grepplus index ./src --status                # chunk count, model, temperature stats
grepplus index ./src --purge                 # delete index for path
grepplus index ./src --watch                 # keep index updated on file changes
```

| Flag            | Description                                      |
| --------------- | ------------------------------------------------ |
| `--status`      | Print index manifest + HOT/COLD/COOL file counts |
| `--sketch-only` | MinHash sketch without embedding all chunks      |
| `--ensure-model`| Ensure embedding model before build              |
| `--yes-download`| Non-interactive model download                   |
| `--purge`       | Remove cached index                              |
| `--watch`       | Watch filesystem and incrementally update          |

**Temperature (JIT):** files start COLD; hits promote toward HOT (embedded in cache). Reheat budget per query: `jit_embed_budget`, `jit_reheat_file_cap` in config.

---

## Serve (HTTP daemon)

For agent integrations and editor plugins.

```bash
grepplus serve
grepplus serve --bind 127.0.0.1:9470 --ensure-index --token "$GREPPLUS_SERVE_TOKEN"
```

| Endpoint  | Method | Description                                                        |
| --------- | ------ | ------------------------------------------------------------------ |
| `/health` | GET    | Version, model loaded, auth required                               |
| `/search` | POST   | JSON body: `{ "query", "path", "route"? }` → `{ "route", "hits" }` |

| Flag                 | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `--bind <addr>`      | Listen address (default: `127.0.0.1:9470`)           |
| `--ensure-index`     | Ensure sketch/warm index per search                  |
| `--warm-index`       | Warm index when missing                              |
| `--yes-download`     | Auto-download model if needed                        |
| `--token <secret>`   | Require `Authorization: Bearer <token>` on `/search` |
| `--no-cors`          | Disable CORS                                         |
| `--no-reload-config` | Disable hot-reload of `~/.grepplus/config.toml`      |

`GREPPLUS_SERVE_TOKEN` is used when `--token` is omitted.

Example:

```bash
curl -s http://127.0.0.1:9470/health
curl -s -X POST http://127.0.0.1:9470/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"retry logic","path":"./src","route":"hybrid"}'
```

---

## Configuration

Load order: defaults → `~/.grepplus/config.toml` → `./.grepplus.toml`.

```toml
[embedder]
active = "qwen3-embedding-0.6b"
dim = 256
query_instruct = "Given a code search query, retrieve relevant source passages"

[index]
sketch = "beam"
chunk_mode = "line"
cache_ttl_days = 7
exclude = ["node_modules", "target", ".git"]

[search]
fusion = "rrf"
jit_enabled = true
jit_embed_budget = 64
laser_candidate_cap = 500
sketch_beam_width = 50

[router]
mode = "heuristic"   # or "learned"
contrib_traces = false
model_path = "router/model.json"

[grep]
backend = "parallel"   # parallel | ripgrep | auto
```

| Variable               | Description                          |
| ---------------------- | ------------------------------------ |
| `GREPPLUS_CACHE_DIR`   | Override index cache root            |
| `RUST_LOG` / `tracing` | Log level (default: `warn,ort=warn`) |

---

## How hybrid search works

```text
Query
  │
  ├─ route=grep ──────────────────────────► parallel regex scan
  │
  └─ route=semantic|hybrid|prefocus
        │
        ├─ Laser channel (exact / regex candidates)
        ├─ SketchBeam (MinHash + BM25 shortlist)
        ├─ JIT reheat (embed cold/cool files in budget)
        ├─ semantic score on candidates
        └─ RRF fusion → ranked chunks (file:line-range + preview)
```

**Progressive embed ladder:** sketch shell → budgeted JIT reheat → per-file HOT cache. Matches grep’s zero-setup model while improving recall after warmup.

---

## License

Apache-2.0. See workspace `Cargo.toml` for crate metadata.
