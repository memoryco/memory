//! Memory bootstrap - core memory operational instructions

/// Core memory workflow instructions — imported by the instructions tool.
pub const INSTRUCTIONS: &str = r#"# Memory Tools

## Workflow (every turn)

1. **Search** → `memory_search` with keywords from the user's message
2. **Recall** → `memory_recall` with IDs you'll reference (do this NOW, before responding — tool budget runs out at end-of-turn)
3. **Respond** → incorporate what you found
4. **Create** → `memory_create` with any new facts. Non-optional. If you learned something and didn't store it, it's gone.

## memory_search

Runs semantic similarity search. Accepts an array of queries — batch multiple searches in one call.

- Decompose abstract queries into concrete terms: "relationship status" → try "breakup", "dating", "partner"
- Search actions/events, not abstract states
- Use `created_after` / `created_before` (ISO 8601 or epoch) for time-based queries
- If results include a 🔗 procedure chain hint, call `memory_associations` on the anchor with `direction: outbound` to get ordered steps

## memory_recall

Strengthens memories and wires associations between them (Hebbian learning). Recalled memories resist decay; unreferenced ones fade.

- Call IMMEDIATELY after search, BEFORE your response
- Pass all relevant IDs in one call
- Can resurrect archived memories back to active state

## memory_create

Stores new facts. **Mandatory** at least once per turn if you learned anything.

**Atomicity rule:** One fact per memory. Each memory generates an embedding vector — compound memories dilute it, making retrieval unreliable.

Split when:
- You're writing a numbered list → each item is a separate memory
- More than 2 sentences → probably multiple facts
- Removing half would leave something useful → split it

Anchor shared context as a prefix: "Project X: database migrated", "Project X: API updated to v2".

Don't store: exact duplicates, ephemeral task state, info already in identity, meta-observations about the conversation.

On long turns, create incrementally (~3–5 facts per batch) as you discover them.

## memory_associate

Creates explicit weighted links between memories. Use `ordinal` for ordered chains (procedure steps). Omit ordinal for unordered associations.

**Procedure chains** (repeatable multi-step processes):
1. Create anchor: "PROCEDURE: [name]"
2. Create one memory per step
3. Wire anchor → each step with ordinals 1–N, weight 0.8

To fix broken chains: delete anchor + all steps, recreate from scratch. There's no way to remove individual associations.

## date_resolve

LLMs are bad at date math. Call `date_resolve` when:
- A retrieved memory contains relative time ("last Saturday", "two weeks ago")
- You need to convert duration ↔ date ("seven years" → "since 2019")
- You need day-of-week for a date

Pass the relative expression and the memory's `created_at` as `reference_date`. Never compute dates yourself.
"#;

// No bootstrap function needed — instructions are served by the instructions tool.
