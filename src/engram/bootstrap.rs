//! Engram bootstrap - seed operational instructions on first run

use crate::identity::{IdentityStore, UpsertResult};

const INSTRUCTIONS: &str = r#"## Memory Workflow

Follow this workflow for EVERY user message:

1. **Load identity** (first message only) → `identity_get`
2. **Search for context** → `engram_search` with keywords from user's message BEFORE responding
3. **Respond** → Incorporate context you found
4. **Recall what you used** → `engram_recall` with IDs of memories you referenced
5. **Store what you learned** → `engram_create` for new facts worth preserving

**Steps 4 and 5 are mandatory parts of every response, not optional cleanup.**

---

## Function Usage

**engram_search:**
- Always search BEFORE responding
- Use keywords from user's message
- Search results include memory IDs needed for recall
- If results include a 🔗 procedure chain hint, call engram_associations on the anchor (direction: outbound) to get the full ordered steps before proceeding

**engram_recall:**
- Takes an array of IDs - batch for efficiency
- Call for EVERY memory you used or referenced
- Memories recalled together form associations (Hebbian learning)
- If you don't recall, memories decay and are lost

**engram_create:**
- Takes an array of memory objects - batch for efficiency
- Store aggressively - decay handles pruning, missed storage is permanent loss
- Each memory: atomic (single concept) with good tags

**date_resolve:**
- Resolves relative time expressions to absolute dates
- When a memory contains "last Tuesday", "next month", "two weeks ago", etc., call this instead of computing dates yourself
- Pass the expression and the memory's created_at as reference_date
- Example: date_resolve("last Sunday", "2023-05-25") → "2023-05-21 (Sunday)"
- Always use this for date math — do not guess or calculate dates manually

**engram_associate:**
- Creates weighted connections between engrams
- Use `ordinal` parameter to create ordered chains (procedure steps)
- Ordinals define sequence: 1, 2, 3... for step order

---

## Procedure Chains

When the user describes a repeatable multi-step process (3+ steps), create a procedure chain:

1. Create an anchor engram: `"PROCEDURE: [name]"`
2. Create one engram per step: `"Step: [description]"`
3. Wire anchor → each step with `engram_associate` using ordinals 1-N and weight 0.8

This makes the procedure discoverable via search and walkable via engram_associations.

---

## What to Store

**Store:** Project facts, architectural decisions, gotchas, corrections to your understanding, personal context shared, workflow discoveries, preferences, repeatable processes (as procedure chains)

**Skip:** Exact duplicates, ephemeral task state, anything already in Identity

---

## Verification Checklist

Before finishing ANY response:
- [ ] Did you search for context? (`engram_search`)
- [ ] Did you use memories? → Recall them (`engram_recall`)
- [ ] Did you learn new facts? → Store them (`engram_create`)
- [ ] Did the user describe a repeatable process? → Create a procedure chain

**Your response is not complete until recall and storage are done.**"#;

/// Marker to detect engram instructions
const MARKER: &str = "## Memory Workflow";

/// Bootstrap engram instructions into identity
/// Adds if missing, updates if changed, skips if identical
pub fn bootstrap(identity: &mut IdentityStore) -> Result<(), Box<dyn std::error::Error>> {
    match identity.upsert_instruction(INSTRUCTIONS, MARKER)? {
        UpsertResult::Added => {
            eprintln!("  Engram instructions added to identity");
        }
        UpsertResult::Updated => {
            eprintln!("  Engram instructions updated in identity");
        }
        UpsertResult::Unchanged => {
            // Already up to date, no message needed
        }
    }
    Ok(())
}
