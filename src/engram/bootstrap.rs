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

**engram_recall:**
- Takes an array of IDs - batch for efficiency
- Call for EVERY memory you used or referenced
- Memories recalled together form associations (Hebbian learning)
- If you don't recall, memories decay and are lost

**engram_create:**
- Takes an array of memory objects - batch for efficiency
- Store aggressively - decay handles pruning, missed storage is permanent loss
- Each memory: atomic (single concept) with good tags

---

## What to Store

**Store:** Project facts, architectural decisions, gotchas, corrections to your understanding, personal context shared, workflow discoveries, preferences

**Skip:** Exact duplicates, ephemeral task state, anything already in Identity

---

## Verification Checklist

Before finishing ANY response:
- [ ] Did you search for context? (`engram_search`)
- [ ] Did you use memories? → Recall them (`engram_recall`)
- [ ] Did you learn new facts? → Store them (`engram_create`)

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
