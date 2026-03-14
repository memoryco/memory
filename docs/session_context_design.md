# Session Context: Design Document

**Status:** Draft
**Author:** Brandon Sneed + Porter
**Date:** March 14, 2026
**Inspired by:** DeepSeek Engram paper (arXiv:2601.07372) — context-aware gating mechanism

---

## Problem

When MemoryCo's memory MCP server receives an `engram_recall` query, it has no knowledge
of the broader conversation context. A query for "Mochi" returns results ranked purely by
semantic/keyword relevance — the server can't distinguish between the user's cat and Japanese
rice cakes because it doesn't know what the conversation is about.

DeepSeek's Engram paper solved an analogous problem inside the transformer by gating
retrieved memory against the current hidden state. We need the same capability at the MCP
layer, but we can't see the model's hidden state — we only see tool call parameters.

## Insight

We already receive signal for free. Every tool call is a breadcrumb:
- `engram_recall("twilio destination implementation")` → conversation is about Rust/Twilio work
- `engram_recall("mock server encryption")` → topic narrows
- `engram_create(...)` → content tells us what's being discussed
- `identity_get(session_id=...)` → session starts, we know who this is

We don't need the LLM to summarize context for us. We don't need the caller to pass extra
parameters. We just accumulate the signal we already get and use it to bias results.

## Key Constraint: LLM Context Budget

The local LLM (Qwen3 etc.) runs at `context_length = 2048` by default. Existing prompts
(rerank, query expansion) are carefully designed to fit within this budget. **We cannot feed
session history into LLM prompts** — it would blow the context window.

**Solution:** Session context biasing is pure vector math. No LLM tokens required.

## Design

### Session as a Persistent Record

Sessions are NOT ephemeral in-memory state. They are rows in a database table that persist
across server restarts and conversation boundaries.

- A `session_id` arrives → check DB → load if exists, create if not
- Tool calls accumulate signal into that row
- When the conversation ends → nothing happens, row persists
- Same `session_id` returns later → pick up where we left off
- Sessions older than `session_expire_days` are deleted on server startup

This eliminates the session lifecycle/cleanup problem entirely. There is no "session end" —
just a hard expiry for sessions that haven't been touched in a long time.

### Who Generates session_id?

The **server** generates the `session_id` during `identity_get`, which is the mandatory
first call of every conversation. The caller never creates session IDs — it receives one
from identity_get and echoes it back on subsequent calls.

**Format:** 16-character hex string = 8 chars unix timestamp (seconds) + 8 chars random
(from UUID v4). Example: `67d4a1b2c3f8e901`. This gives natural sort order by creation
time and effectively zero collision risk — you'd need ~65K sessions in the same second.

### session_id is Required, Per-Call, and Echoed

`session_id` is a **required parameter** on `engram_recall`, `engram_search`, and
`engram_create`. Every response from these tools echoes the session_id as the first
line of output:

```
session_id: 67d4a1b2c3f8e901

[actual results]
```

**Why echo?** In long conversations, LLM context compaction discards earlier messages.
The `identity_get` response from turn 1 is the first thing to disappear. By echoing
the session_id on every tool response, the most recent result always has it visible,
so the AI can pass it on the next call regardless of compaction.

**Why required, not optional?** Optional params get dropped when the AI loses context
(exactly the scenario we're solving). Required params cause a hard failure if missing,
which the AI self-corrects on. Silent fallback to unbiased retrieval is worse than a
retryable error.

**Why per-call, not per-connection?** `session_id` is NOT stashed on the server's
`Context` struct.

Why: In stdio mode (today), one process = one client, so stashing on Context would work.
But once we support server+port, multiple clients share the same server process. Stashing
`session_id` on shared state would cause session bleed between concurrent conversations.
The stateless per-call design works correctly in both transport modes.

### Which Tools Accept session_id?

`session_id` is a **required parameter** on these tools:

| Tool | Signal contributed | Rationale |
|------|-------------------|-----------|
| `engram_recall` | Recalled memory embeddings | Direct topic signal — what the user is asking about |
| `engram_search` | Query string | Same as recall — semantic search query = topic signal |
| `engram_create` | Memory content | If you're creating memories about a topic, strong signal |

`identity_get` **generates** the session_id and passes it to any piggy-backed searches.

All other tools are excluded:

| Tool | Why excluded |
|------|-------------|
| `engram_associate` | Structural operation on IDs. The IDs were already found via recall/search, so the topic signal already entered the session. Would require extra DB lookups for marginal gain. |
| `engram_get` | Fetch by ID — same reasoning as associate. Already found via recall/search. |
| `engram_delete` | Destructive housekeeping, not topic signal. |
| `identity_*` | Persona/config, not conversation topic. |
| `reference_search` | Different domain (clinical references). Don't want DSM-5 queries biasing personal memory retrieval. |
| `plan_*`, `lenses_*`, `config_*` | Administrative, not topic signal. |

### Storage Schema

New table in `brain.db`:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    queries       TEXT NOT NULL DEFAULT '[]',  -- JSON array of query strings
    centroid      BLOB,                        -- running embedding centroid vector
    query_count   INTEGER NOT NULL DEFAULT 0,  -- number of accumulated queries
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
);
```

Estimated row size: ~3-4 KB (mostly the centroid BLOB at float32 * embedding_dim).
10,000 sessions ≈ 30-40 MB. Negligible.

### Accumulation (Write Path)

On every tool call that includes a `session_id`:

1. Load session from DB (or create)
2. Update `last_seen_at`
3. Extract topic signal from the call:
   - `engram_recall` / `engram_search`: the query string
   - `engram_create`: the memory content
4. If signal was extracted:
   a. Append text to `queries` JSON array (capped at `session_max_queries`)
   b. Embed the text using the existing embedding model
   c. Update running centroid via exponential moving average:
      `centroid = α * new_embedding + (1 - α) * centroid`
      where α = `session_centroid_smoothing` (default 0.1)
5. Write session back to DB

### Retrieval Biasing (Read Path)

When `engram_recall` or `engram_search` is called with a `session_id` that has an
accumulated centroid:

1. Run existing retrieval pipeline (FTS5 + vector search)
2. For each candidate result that has an embedding:
   `session_affinity = cosine(candidate_embedding, session_centroid)`
3. Apply as multiplicative gate on relevance score:
   `final_score = retrieval_score * (1.0 + β * session_affinity)`
4. Re-sort by `final_score`
5. Pass top candidates to LLM reranker if enabled (existing prompt, unchanged)

**β (session_weight):** Configurable in `config.toml` under `[brain]`. Default 0.3.
Start conservative — session context should nudge results, not dominate them.

### Config

```toml
[brain]
# ... existing keys ...
session_context_weight = 0.3    # β: how much session context biases retrieval (0.0 = off)
session_max_queries = 50        # max queries to retain per session
session_centroid_smoothing = 0.1 # α for EMA (0 = equal weight all queries, higher = more recency bias)
session_expire_days = 90        # delete sessions not touched in this many days (0 = never expire)
```

### Pipeline Diagram

```
Query arrives with session_id
        │
        ├──► Update session in DB (append query, update centroid)
        │         [vector math only, no LLM]
        │
        ├──► Existing retrieval (FTS5 + vector search)
        │         │
        │         ▼
        │    Candidate results with scores
        │         │
        │         ▼
        ├──► Session affinity gate ◄── session centroid from DB
        │    final_score = score * (1 + β * cosine_sim)
        │         [vector math only, no LLM]
        │         │
        │         ▼
        │    Re-sorted candidates
        │         │
        │         ▼
        └──► LLM reranker (if enabled, existing prompt, unchanged)
                  │
                  ▼
             Final results returned to caller
```

**LLM context budget impact: ZERO.** All session biasing is cosine similarity on embeddings.

### Session Expiry

Sessions use a simple age-based expiry rather than the Hebbian energy/decay model used
by engrams. This is intentional — sessions are lightweight operational state, not memories.
They don't need to "fade" gradually; they're either still useful or they're not.

- On server startup (in `apply_maintenance`), delete all sessions where
  `last_seen_at` is older than `session_expire_days` from now
- Default: 90 days. Configurable via `config.toml`
- Setting `session_expire_days = 0` disables expiry (sessions persist forever)
- This keeps the sessions table bounded without any runtime overhead — cleanup
  only happens at startup, same pattern as `apply_time_decay` and `prune_weak_associations`

Why not Hebbian decay? Because session centroids are mathematical artifacts (running
averages of embedding vectors), not semantic content. A centroid doesn't "mean less"
over time the way a memory does — it either represents a conversation that might resume,
or it's dead weight. Hard expiry is the right model.

### Centroid Update Strategy: Exponential Moving Average

The centroid uses EMA (exponential moving average) rather than a simple average. This
means recent queries are weighted more heavily than older ones.

Rationale: Conversations drift. If you start talking about Twilio and pivot to skateboarding,
you want the centroid to track the pivot, not stay anchored at the average of both topics.
A smoothing factor of 0.1 means recent queries dominate within about 5-10 calls, which
roughly matches how fast a typical conversation shifts topic.

The `session_centroid_smoothing` config controls α:
- `0.1` (default): Smooth transition, good for most conversations
- Higher values (0.2-0.3): More responsive to topic shifts
- Lower values (0.05): More stable, better for long single-topic sessions

## What This Enables

### Conversational Continuity
Same `session_id` returns days later → session centroid is still there → memory recall
is already biased toward the right topic cluster. "Hey, where were we on that thing?"
just works.

### Multi-Conversation Isolation
Multiple agents/clients provide different `session_id` values. Each gets independent
session state. A coding agent talking about Rust doesn't pollute a personal conversation
about weekend plans.

### Predictive Prefetch (Future)
Once we have session centroids, we can precompute "memories this session is likely to
need" and have them ready before the next tool call arrives. The centroid tells us the
topic neighborhood — we can prefetch the top N engrams in that neighborhood.

## Action Items

- [x] Add `sessions` table to brain.db schema
- [x] Add `SessionContext` struct with accumulation + centroid update logic
- [x] ~~Add `session_id` as optional param~~ → Made `session_id` **required** on `engram_recall`, `engram_search`, `engram_create`
- [x] Add `generate_session_id()` — timestamp-prefixed hex (16 chars), generated in `identity_get`
- [x] Echo `session_id: <id>` at top of every tool response (survives context compaction)
- [x] Forward session_id through internal call paths (identity_get→search, recall→create)
- [x] Implement centroid update on query/content accumulation
- [x] Implement session affinity gate in recall pipeline (after retrieval, before LLM rerank)
- [x] Add session config keys to config.toml (`session_context_weight`, `session_max_queries`, `session_centroid_smoothing`, `session_expire_days`)
- [x] Add session expiry to `apply_maintenance` startup routine
- [ ] Benchmark with LOCOMO (compare recall quality with/without session context)
- [ ] Add session stats to dashboard
