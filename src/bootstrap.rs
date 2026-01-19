//! Bootstrap - seed operational instructions on first run
//!
//! Seeds instructions into Identity (permanent, never decays).
//! Personal context should be added through normal use
//! or migrated from existing systems.

use engram::Brain;

/// Check if brain needs bootstrapping and do it if so
pub fn bootstrap_if_needed(brain: &mut Brain) -> Result<(), Box<dyn std::error::Error>> {
    // Check if identity has no instructions
    if brain.identity().instructions.is_empty() {
        eprintln!("  Bootstrapping operational instructions...");
        bootstrap_instructions(brain)?;
        eprintln!("  Operational instructions bootstrapped");
    }
    
    Ok(())
}

/// Seed operational instructions into Identity
fn bootstrap_instructions(brain: &mut Brain) -> Result<(), Box<dyn std::error::Error>> {
    let instructions = vec![
        r#"## Memory Workflow

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

        **Your response is not complete until recall and storage are done.**"#
    ];
    
    // Get current identity and add instructions
    let mut identity = brain.identity().clone();
    for instruction in instructions {
        identity = identity.with_instruction(instruction);
    }
    
    brain.set_identity(identity)?;
    
    Ok(())
}
