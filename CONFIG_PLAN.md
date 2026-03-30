# MemoryCo Config Plan

Comprehensive list of config options — what exists, what's being added, and what's proposed.

## Current State (post config-to-toml migration)

All brain config lives in `config.toml` under `[brain]`. LLM config lives under `[llm]` or as flat `llm_*` keys. Loaded once on startup, immutable for process lifetime. `config_set` writes to TOML (takes effect on restart).

---

## `[brain]` — Memory Substrate

### Existing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `decay_rate_per_day` | float | 0.05 | Energy loss per day of non-use (0.0-1.0) |
| `decay_interval_hours` | float | 1.0 | Minimum hours between decay checks |
| `propagation_damping` | float | 0.5 | Signal reduction to neighbors (0.0-1.0) |
| `hebbian_learning_rate` | float | 0.1 | Association strength boost on co-access |
| `recall_strength` | float | 0.2 | Energy boost when recalling a memory |
| `association_decay_rate` | float | 1.0 | Relative to memory decay. 1.0 = same rate |
| `min_association_weight` | float | 0.05 | Prune associations below this on startup |

### Existing — Search

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `search_follow_associations` | bool | true | Follow associations during search |
| `search_association_depth` | int | 1 | Hops to follow (1 = direct only) |
| `embedding_model` | string | "AllMiniLML6V2" | Embedding model. Change triggers re-embedding on restart |
| `rerank_mode` | string | "cross-encoder" | Reranking strategy (see below) |
| `rerank_candidates` | int | 30 | Candidates for cross-encoder reranking pass |
| `hybrid_search_enabled` | bool | true | BM25 + vector fusion via RRF |
| `query_expansion_enabled` | bool | true | LLM/mechanical query variant expansion |

### Rerank Modes

`rerank_mode` controls how search results are reordered after initial retrieval:

| Value | Pipeline | Description |
|-------|----------|-------------|
| `"off"` | cosine order | No reranking. Results ordered by blended cosine similarity × energy score. Fastest but weakest relevance. |
| `"cross-encoder"` | cosine → cross-encoder | BGERerankerBase scores all (query, memory) pairs. Fast (~100-200ms), no context limit, good at surface-level relevance. Handles 60-100+ candidates easily. Current default. |
| `"llm"` | cosine → LLM | Top N candidates (controlled by `llm_rerank_candidates`) sent to local LLM. LLM can infer semantic relationships cosine misses ("destress" → "pottery class"). Context-length limited. |
| `"hybrid"` | cosine → cross-encoder → LLM | **Proposed.** Two-stage pipeline. Cross-encoder narrows 60-100 candidates to a tight top N (fast, no context limit). Then LLM reorders that refined set (smart, inference-capable). Cross-encoder is the bouncer, LLM is the sommelier. Best of both: the cross-encoder catches candidates the LLM would never see due to context limits, then the LLM applies reasoning the cross-encoder can't. |

**Why hybrid?** With `"llm"` mode alone, the LLM only sees the top 20 candidates by blended cosine score. If a relevant memory (like "Melanie's pottery class") landed at position 35 in the cosine ranking, the LLM never gets a chance to promote it. With `"hybrid"`, the cross-encoder first promotes that memory from position 35 to maybe position 12 (it's decent at relevance), putting it in the LLM's window where the LLM recognizes the inferential connection and pushes it to top 5.

**Implementation:** The hybrid flow in search.rs would be:
1. Run cross-encoder on full `scored` list (same as `"cross-encoder"` mode)
2. Take cross-encoder's top N (controlled by `llm_rerank_candidates`, default 20)
3. Feed those N candidates to LLM for final reordering
4. LLM returns ordered indices, assigned synthetic rank-based scores

### Proposed — Search

| Key | Type | Default | Why |
|-----|------|---------|-----|
| `llm_rerank_candidates` | int | 20 | Candidate cap for LLM rerank stage (used in both `"llm"` and `"hybrid"` modes). Separate from `rerank_candidates` because LLM is context-length constrained. At `llm_context_length` 2048 → cap ~20. At 4096 → can push to 30-40. |
| `search_min_score` | float | 0.3 | Server-side default floor for similarity. Currently only set by callers (bench uses 0.3, MCP default is 0.4). A config default means consistent behavior regardless of caller. |
| `composite_limit_min` | int | 15 | Minimum effective_limit for composite/list-style queries. Currently hardcoded. Directly affects MH recall — higher = more memories returned for "what activities does X do" questions. |
| `composite_limit_max` | int | 30 | Maximum effective_limit for composite queries. Cap to prevent runaway result sets. |
| `association_cap_min` | int | 5 | Minimum number of association discoveries to merge into results. Currently hardcoded clamp(5, 12). |
| `association_cap_max` | int | 12 | Maximum association discoveries. |

### Maybe Later — Search (hardcoded, probably fine)

These are currently hardcoded. They're tunable but unlikely to need user-facing config. Documenting for awareness.

| Constant | Value | Location | Notes |
|----------|-------|----------|-------|
| `SEMANTIC_DEDUP_THRESHOLD` | 0.85 | search.rs | Cosine sim for dedup. 0.85 is well-tested. |
| `SPARSE_THRESHOLD` | 3 | search.rs | Results below this triggers Phase 2 fallback expansion |
| Inferential relaxed factor | 0.6 | search.rs | `min_score * 0.6` for inferential queries |
| Inferential floor | 0.15 | search.rs | Absolute floor for relaxed min_score |
| Diversity Jaccard penalty | 0.20 | search.rs | Overlap penalty coefficient |
| Diversity cue bonus | 0.05 | search.rs | Per-cue coverage bonus |
| Diversity source penalty | 0.08 | search.rs | Per-source-bucket overrepresentation penalty |
| `fetch_count` multiplier | 3× | search.rs | `effective_limit * 3` for initial candidate pool |
| Fallback max terms | 3 | search.rs | Max fallback term queries for composite questions |

---

## `[llm]` — Local LLM

### Existing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_enable` / `enabled` | bool | false | Enable local LLM |
| `llm_model` / `model_path` | string | none | Path to GGUF model file |
| `llm_context_length` / `context_length` | int | **2048** | Context window size |
| `llm_gpu_layers` / `n_gpu_layers` | int | 0 | GPU layers (-1 = all) |
| `llm_threads` / `threads` | int | 4 | CPU threads for inference |
| `llm_lazy_load` / `lazy_load` | bool | true | Load model on first use vs startup |

### Proposed

| Key | Type | Default | Why |
|-----|------|---------|-----|
| (none right now) | | | The immediate win is bumping `llm_context_length` to 4096. Qwen3-4B supports it. More context = better enrichment quality + more LLM rerank candidates. Pure config change, no code needed. |

---

## Immediate Actions

1. **Bump `llm_context_length` to 4096** in Brandon's real `~/.memoryco/config.toml`. Free win.
2. **Add `llm_rerank_candidates`** to `[brain]` config (default 20). Wire it into search.rs LLM rerank path replacing the hardcoded `20.min(scored.len())`.
3. **Add `search_min_score`** to `[brain]` config (default 0.3). Use as fallback when caller doesn't specify.
4. **Add `"hybrid"` rerank mode.** Implement the cross-encoder → LLM pipeline in search.rs.

## Next Batch

5. **`composite_limit_min`/`max`** — expose the 15/30 magic numbers. These directly affect MH benchmark scores.
6. **`association_cap_min`/`max`** — expose the 5/12 clamp range.

## Bench-Specific Notes

The bench harness (`bench/locomo/config.py`) has its own search settings:
- `SEARCH_LIMIT = 10` (passed to memory_search as `limit`)
- `SEARCH_MIN_SCORE = 0.3` (passed as `min_score`)

These are caller-side. The server-side config provides defaults when callers don't specify. The bench always specifies, so server defaults only matter for real MCP usage (Claude Desktop, etc).

The bench config.toml (copied into each temp MEMORY_HOME) should set:
- `rerank_mode = "hybrid"` to test the full pipeline
- `llm_context_length = 4096` to give the LLM rerank stage more room
- `llm_rerank_candidates = 30` once context length supports it
