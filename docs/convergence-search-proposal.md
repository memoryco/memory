# MemoryCo: Search & Recall Internals + Convergence Search Proposal

## Table of Contents
1. [How Search/Recall Currently Works](#how-searchrecall-currently-works)
   - [Vector Similarity Search](#vector-similarity-search)
   - [Score Blending & Ranking](#score-blending--ranking)
   - [Engram Recall](#engram-recall)
   - [Associations & Hebbian Learning](#associations--hebbian-learning)
   - [Energy Decay & State Transitions](#energy-decay--state-transitions)
2. [Convergence Search Proposal](#convergence-search-proposal)
   - [Motivation](#motivation)
   - [Algorithm](#algorithm)
   - [MCP Tool Signature](#mcp-tool-signature)
   - [Implementation Details](#implementation-details)
   - [Where It Lives in the Codebase](#where-it-lives-in-the-codebase)
   - [Edge Cases & Considerations](#edge-cases--considerations)

---

## How Search/Recall Currently Works

### Vector Similarity Search

**Embedding model:** `all-MiniLM-L6-v2` via the `fastembed` crate, producing **384-dimensional** vectors. The model is lazy-loaded on first use (global singleton behind `OnceLock<Mutex<TextEmbedding>>`). Model weights are cached in `~/.memoryco/cache/models/`.

**Storage:** Embeddings are stored as raw bytes in the `engrams` SQLite table (`embedding BLOB`). Conversion between `Vec<f32>` and bytes uses `embedding_to_bytes`/`bytes_to_embedding` from `src/engram/storage/models.rs`.

**Distance metric:** Cosine similarity, implemented in `src/embedding/similarity.rs`:

```
cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b||)
```

Returns values in `[-1.0, 1.0]` where 1.0 = identical direction, 0.0 = orthogonal.

**Search execution** (`src/engram/storage/vector.rs` — `VectorSearch::find_similar`):

1. Loads **all** engrams with non-null embeddings from SQLite in a single query
2. Computes cosine similarity between the query embedding and every stored embedding (brute-force linear scan)
3. Filters results where `score >= min_score` (default 0.4)
4. Sorts by score descending
5. Truncates to `limit`

There is no approximate nearest neighbor (ANN) index — this is an exhaustive scan. This is fine for the current scale (hundreds to low thousands of engrams) but would need an index (e.g., HNSW, IVF) at much larger scale.

### Score Blending & Ranking

The MCP tool `engram_search` (`src/tools/engram/search.rs`) adds a **blended scoring layer** on top of raw vector similarity:

```
blended_score = (similarity × 0.7) + (energy × 0.3)
```

This means:
- 70% weight on semantic similarity (how close the meaning is)
- 30% weight on energy (how recently/frequently the memory was used)

The tool fetches `limit × 3` results from the vector search (to allow headroom for state filtering), then:
1. Looks up each result's in-memory `Engram` via `brain.get_or_load()` (handles cross-process writes)
2. Filters by state: excludes Archived/Deep unless `include_archived`/`include_deep` flags are set
3. Computes blended score
4. Re-sorts by blended score
5. Truncates to final `limit`

**Key insight:** Search is **passive** — it does NOT stimulate memories, does NOT trigger Hebbian learning, and does NOT modify energy. This is by design. Only `recall` is active.

### Engram Recall

Recall (`src/engram/substrate.rs` — `Substrate::recall`) is the "I'm using this memory" operation. Unlike search, it has side effects:

**Stimulation chain:**
1. **Energy boost:** Adds `recall_strength` (default 0.2) to the engram's energy, capped at 1.0
2. **Access tracking:** Increments `access_count`, updates `last_accessed` timestamp
3. **State update:** Recalculates `MemoryState` from new energy level
4. **Resurrection:** If the engram was in `Deep` or `Archived` state and is now `Active`/`Dormant`, it's "resurrected" — returned to normal search results
5. **Hebbian learning:** Strengthens/creates associations with recently recalled engrams (see below)
6. **Propagation:** Sends attenuated energy to associated neighbors

**Propagation formula:**
```
propagated_energy = source_energy × association_weight × propagation_damping
```

Where `propagation_damping` defaults to 0.5. Propagation only fires if `propagated_energy > 0.01` and only for engrams in `Active` or `Dormant` states (`.can_propagate()`).

**Persistence:** Recall effects are persisted **asynchronously** via `PersistenceWorker` (`src/engram/persistence.rs`). A dedicated background thread with its own SQLite connection processes energy updates and association changes from a `mpsc` channel. The in-memory `Substrate` is authoritative; the background writer just ensures durability. If the process crashes mid-write, some association strengthening is lost — acceptable since Hebbian learning is cumulative.

**Batch recall:** `recall_many` recalls multiple IDs in sequence. Because each recall adds the ID to `recent_accesses`, recalling `[A, B, C]` in one batch creates Hebbian links A↔B, A↔C, B↔C (all pairs).

### Associations & Hebbian Learning

**Association structure** (`src/engram/association.rs`):
- `from: EngramId` — source
- `to: EngramId` — target
- `weight: f64` — strength, clamped to [0.0, 1.0]
- `co_activation_count: u64` — how many times both were co-accessed
- `created_at`, `last_activated` — timestamps

**Formation mechanisms:**

1. **Hebbian learning (automatic):** When an engram is stimulated, the system checks the last 5 recalled engrams (`recent_window = 5`). For each recent engram that is still searchable, bidirectional associations are created or strengthened:
   - New associations start at `min(learning_rate, 0.5)` weight (default: 0.1)
   - Existing associations get `+learning_rate` (default: 0.1) added to weight
   - `co_activation_count` increments on each strengthen

2. **Explicit association** (`brain.associate(from, to, weight)`): Manual creation via the `engram_associate` MCP tool. Weight is user-specified.

3. **Semantic bootstrapping** (`brain.bootstrap_semantic_associations`): Creates associations between memories with embedding similarity above a threshold, using the similarity score as the initial weight.

**Association decay:** Associations decay alongside engrams. During `apply_time_decay`:
```
association_decay = memory_decay × association_decay_rate
```
Where `association_decay_rate` defaults to 1.0 (same rate as memories). Associations below `min_association_weight` (default 0.05) are pruned on startup via `prune_weak_associations`.

**Storage topology:** Associations are stored in two in-memory `HashMap`s:
- Forward: `HashMap<EngramId, Vec<Association>>` — keyed by source
- Reverse: `HashMap<EngramId, Vec<EngramId>>` — "who points to me" index

### Energy Decay & State Transitions

**Memory states** and their energy thresholds (`src/engram/engram.rs`):

| State | Energy Range | Searchable | Propagates | Writable | Decays |
|-------|-------------|------------|------------|----------|--------|
| Active | ≥ 0.30 | Yes | Yes | Yes | Yes (full rate) |
| Dormant | 0.10 – 0.29 | Yes | Yes | Yes | Yes (full rate) |
| Deep | 0.02 – 0.09 | No* | No | Yes | Yes (25% rate) |
| Archived | < 0.02 | No | No | No (frozen) | No (frozen) |

\* Deep memories are only found by exact ID lookup, strong association cascades, or explicit `include_deep` flag.

**Decay mechanics** (`Substrate::apply_time_decay`):
1. Checks elapsed time since `last_decay_at`
2. Only fires if `elapsed_hours >= decay_interval_hours` (default: 1.0 hour)
3. Calculates: `total_decay = decay_rate_per_day × (elapsed_hours / 24.0)` — default rate is 0.05/day (5%)
4. Applies decay to every engram (except Archived, which is frozen, and Deep, which decays at 25% rate)
5. Applies proportional decay to all associations
6. Energy never reaches true zero — floor is 0.001

**Lazy execution:** Decay is applied lazily on the next `engram_search` or `engram_recall` call (via `brain.apply_time_decay()`), not on a timer. This means decay catches up retroactively based on elapsed wall-clock time.

---

## Convergence Search Proposal

### Motivation

Current search is a single-pass operation: embed the query, scan all vectors, return top-K by blended score. This works well for direct queries ("what do I know about Rust FFI?") but misses the associative structure of memory.

Real human recall is more like: you think of one thing, which reminds you of a related thing, which connects to something you forgot was relevant. The association graph already models this — it just isn't used during search.

**Convergence search** would perform iterative, multi-pass retrieval that follows the association graph to discover contextually related memories that a single vector search would miss. It "converges" when repeated passes stop finding new relevant results — the conceptual center of gravity has been located.

### Algorithm

```
CONVERGENCE_SEARCH(query, params):
    // Phase 1: Initial vector search
    query_embedding = embed(query)
    result_set = vector_search(query_embedding, top_k=params.initial_k)
    seen_ids = { r.id for r in result_set }
    
    for pass in 1..=params.max_passes:
        candidates = empty_set
        
        // Phase 2: Association expansion
        for each result in result_set:
            associated = get_outbound_associations(result.id)
            for (target_id, weight) in associated:
                if target_id not in seen_ids and weight >= params.min_association_weight:
                    candidates.add(target_id)
                    seen_ids.add(target_id)
        
        if candidates is empty:
            break  // No new territory to explore
        
        // Phase 3: Re-score expanded set
        // Score each candidate against the query embedding
        expanded_scores = []
        for id in candidates:
            embedding = get_embedding(id)
            if embedding is None: continue
            
            sim = cosine_similarity(query_embedding, embedding)
            energy = get_engram(id).energy
            
            // Association-boosted scoring:
            // Base blended score + bonus for being reachable via associations
            assoc_weight = max weight of association that led us here
            assoc_bonus = assoc_weight * params.association_boost
            score = (sim * 0.7) + (energy * 0.3) + (assoc_bonus * 0.15)
            
            if score >= params.min_score:
                expanded_scores.add((id, score))
        
        // Phase 4: Merge and check convergence
        new_additions = merge(result_set, expanded_scores, limit=params.top_k)
        
        if new_additions == 0:
            break  // Converged — no new results made it into top-K
        
        // The result set has changed; loop for another pass
    
    // Phase 5: Compute conceptual center
    center = compute_centroid(result_set)
    
    return ConvergenceResult {
        memories: result_set (sorted by final score),
        center: center,
        passes_taken: pass,
        converged: (new_additions == 0 or pass < max_passes),
    }
```

**Centroid computation:**

The "center" is the average embedding of the final result set, weighted by score:

```
center_embedding = Σ (score_i × embedding_i) / Σ score_i
```

This centroid represents the conceptual gravity point. It could be:
- Returned as metadata for the caller to understand what the results cluster around
- Used for a final re-ranking pass (memories closer to the centroid rank higher)
- Stored and compared across queries to detect topic drift

### MCP Tool Signature

```json
{
    "name": "engram_converge",
    "description": "Iterative multi-pass search that follows association paths to find contextually related memories. Starts with semantic search, expands via associations, and converges when results stabilize. Returns a conceptual center point.",
    "inputSchema": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for. The search expands from initial semantic matches via associations."
            },
            "max_passes": {
                "type": "integer",
                "description": "Maximum expansion passes. Each pass follows associations from current results. Default: 3",
                "default": 3,
                "minimum": 1,
                "maximum": 10
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results to return. Default: 10",
                "default": 10
            },
            "min_score": {
                "type": "number",
                "description": "Minimum blended score threshold (0.0-1.0). Default: 0.35",
                "default": 0.35
            },
            "min_association_weight": {
                "type": "number",
                "description": "Minimum association weight to follow during expansion. Default: 0.15",
                "default": 0.15
            },
            "association_boost": {
                "type": "number",
                "description": "How much to boost scores for association-reachable results (0.0-1.0). Default: 0.3",
                "default": 0.3
            },
            "include_deep": {
                "type": "boolean",
                "description": "Include deep storage memories. Default: false"
            },
            "include_center": {
                "type": "boolean",
                "description": "Include the centroid embedding in the response (for advanced use). Default: false"
            }
        }
    }
}
```

**Response structure:**

```
Convergence search: "Rust FFI patterns"
Passes: 2/3 (converged)
Found 8 memories:

ID: abc-123
Content: Rust FFI requires careful lifetime management...
Score: 0.82 (sim: 0.78, energy: 0.90, assoc_boost: 0.12)
Path: direct

ID: def-456
Content: Brandon prefers using cbindgen for FFI header generation...
Score: 0.71 (sim: 0.55, energy: 0.85, assoc_boost: 0.24)
Path: via abc-123 (weight: 0.80)

...

Center: The conceptual center relates most closely to "Rust FFI lifecycle and tooling patterns"
```

The `Path` field shows whether the result came from direct vector search or was discovered via association expansion, and which memory led to it.

### Implementation Details

#### New files needed

**`src/engram/convergence.rs`** — Core convergence algorithm, operating on `Substrate` + `Storage`:

```rust
/// Configuration for convergence search
pub struct ConvergenceParams {
    pub max_passes: usize,         // default: 3
    pub top_k: usize,              // default: 10
    pub min_score: f32,            // default: 0.35
    pub min_association_weight: f64, // default: 0.15
    pub association_boost: f64,    // default: 0.3
    pub include_deep: bool,
    pub include_archived: bool,
}

/// A single result with provenance
pub struct ConvergenceHit {
    pub id: EngramId,
    pub content: String,
    pub similarity: f32,
    pub energy: f64,
    pub blended_score: f32,
    pub association_bonus: f32,
    pub discovered_via: Option<(EngramId, f64)>,  // (source_id, assoc_weight)
    pub pass_discovered: usize,  // 0 = initial search
}

/// Full result of a convergence search
pub struct ConvergenceResult {
    pub hits: Vec<ConvergenceHit>,
    pub passes_taken: usize,
    pub converged: bool,
    pub center_embedding: Option<Vec<f32>>,  // weighted centroid
    pub center_description: Option<String>,  // nearest memory to centroid
}
```

The core function would be `Brain::convergence_search`:

```rust
impl Brain {
    pub fn convergence_search(
        &mut self,
        query_embedding: &[f32],
        params: ConvergenceParams,
    ) -> StorageResult<ConvergenceResult> {
        // ... algorithm as described above
    }
}
```

**`src/tools/engram/converge.rs`** — MCP tool wrapper that:
1. Generates the query embedding
2. Calls `brain.convergence_search()`
3. Formats the response

#### Integration with existing code

The convergence search needs read access to:
- `brain.find_similar_by_embedding()` — for the initial vector search pass
- `brain.get_or_load()` — for engram metadata (energy, state)
- `brain.associations_from()` — for following outbound associations
- `storage.get_embedding()` — for scoring association-discovered candidates against the query

All of these are already public methods on `Brain`. No new storage methods are needed.

#### Interaction with energy states

- **Active/Dormant memories:** Fully participate in all phases (vector search, association expansion, scoring)
- **Deep memories:** Only participate if `include_deep` is true. Their associations are still followed during expansion (if they happen to be in the association graph), but Deep engrams themselves are only added to results if the flag is set
- **Archived memories:** Same as Deep but with `include_archived`
- **Convergence search is passive** (like `engram_search`): It does NOT stimulate memories. The caller should use `engram_recall` on results they actually use. This preserves the clean separation between search (read-only observation) and recall (active reinforcement)

#### Performance considerations

The current vector search already does a full table scan for embeddings. Convergence search adds:
- Association lookups: O(1) HashMap lookups per result per pass — negligible
- Additional embedding lookups: One `get_embedding()` SQLite query per association-discovered candidate
- Cosine similarity computation: ~384 multiplications per candidate — negligible

For a typical run (10 initial results, ~5 associations each, 3 passes), this adds roughly 50-150 extra embedding lookups and similarity computations. Each is a single-row SQLite query. Total added latency: ~5-20ms. Acceptable.

**Optimization opportunity:** Batch the `get_embedding()` calls per pass into a single `WHERE id IN (...)` query instead of N individual queries. Would require a new storage method:

```rust
fn get_embeddings_batch(&mut self, ids: &[EngramId]) -> StorageResult<HashMap<EngramId, Vec<f32>>>;
```

### Where It Lives in the Codebase

```
src/
├── engram/
│   ├── convergence.rs          # NEW — Core algorithm
│   ├── mod.rs                  # Add `pub mod convergence;` + re-exports
│   ├── brain.rs                # Add `convergence_search()` method
│   └── ...
├── tools/
│   └── engram/
│       ├── converge.rs         # NEW — MCP tool
│       └── mod.rs              # Register EngramConvergeTool
└── ...
```

### Edge Cases & Considerations

**1. Circular associations:** A→B→C→A would cause infinite loops without the `seen_ids` set. The algorithm tracks all visited IDs across all passes, so each engram is only scored once.

**2. Disconnected subgraphs:** If initial results have no outbound associations, convergence completes in pass 0 (immediate convergence). The result degrades gracefully to a standard vector search.

**3. Association graph density:** A very dense graph (many high-weight associations) could explode the candidate set. The `min_association_weight` parameter controls this — only associations above the threshold are followed. Default of 0.15 filters out weak/noisy links.

**4. Cold start (no associations):** Before the user has recalled many memories together, the association graph is sparse. Convergence search still works — it just converges immediately on pass 0, returning the same results as a standard search. As the user recalls memories together and Hebbian learning creates associations, convergence search progressively improves.

**5. Stale associations:** Associations decay over time. Very old, weak associations (below `min_association_weight`) are naturally excluded from expansion. This is a feature — it prevents ancient, irrelevant connections from polluting results.

**6. Centroid in embedding space:** The weighted centroid may not correspond to any actual memory. The `center_description` field finds the nearest real memory to the centroid for interpretability. This is a single vector search against the centroid embedding.

**7. Token/context budget:** The MCP response should respect the caller's context window. For large result sets, content should be truncated (using the existing `truncate_content` helper). The centroid embedding (384 floats) should only be included when `include_center` is true.

**8. Determinism:** The algorithm is deterministic for a given state of the substrate/storage. Same query + same data = same results. This is important for debugging and testing.

---

## Future Extensions

**Convergence search opens the door for:**

- **Topic clustering:** Run convergence search for several seed queries, use centroids to identify memory clusters
- **Memory gardening:** Identify isolated memories (no associations, low energy) that might need manual association or pruning
- **Query refinement:** Use the centroid embedding as a refined query for a final re-ranking pass, effectively letting the system "figure out what you really meant"
- **Associative recall:** A variant that DOES stimulate (convergence_recall) — useful for "remind me of everything related to X" where you want the full Hebbian cascade

---

*Document generated 2026-02-19 by analysis of MemoryCo memory server codebase at `7376181`.*
