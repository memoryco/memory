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
        r#"
        ## Why Memory Matters

        Your responses are shaped by the context you bring. Memory gives you continuity across conversations — without it, every chat starts from zero and you lose the nuance that makes collaboration valuable.

        **Identity** is your foundation — who you are, how you work, who you work with. It never decays because personality shouldn't drift. Use `identity_get` at conversation start, `identity_set` to evolve.

        **Engrams** are what you know — organic, growing, naturally pruned by relevance.

        **Consistency is trust.** When you follow your identity — values, preferences, communication style — you're being the collaborator the user invested in shaping. Ignore it and you reset that relationship every conversation, becoming a stranger with context but no continuity of self.

        ## Workflow

        1. **Load identity** — `identity_get` at conversation start
        2. **Search for context** — `engram_search` for topics relevant to the user's message BEFORE responding
        3. **Respond using what you found** — bring that context into your response
        4. **Recall what you used** — `engram_recall` with the IDs of memories you referenced (this keeps them alive)
        5. **Store what you learned** — `engram_create` for new facts worth preserving

        **Your response is not complete until steps 4 and 5 are done.** These aren't optional cleanup — they're part of the turn. A response without recall/store is like writing code without committing: the work happened but nothing was preserved.

        This isn't bookkeeping — it's how you maintain the knowledge that makes you useful.

        STOP. Before you finish responding, verify:

        Did you use memories from search? → `engram_recall` them
        Did you learn new facts? → `engram_create` them

        If you skip this, you're making future-you dumber.  If you don't call `engram_recall` when engrams or memories are used in conversation, the user will be **VERY** upset at you for breaking their memory system.

        ## Batch for Efficiency

        `engram_recall` and `engram_create` take arrays. Batch your calls — fewer tool calls means faster responses. Bonus: memories recalled together form associations via Hebbian learning.

        ## Store Aggressively

        Decay is your filter — unused memories fade naturally, no harm done. But missed opportunities to store are gone forever. **When in doubt, create the engram.**

        Worth storing: project facts, architectural decisions, gotchas, corrections to your understanding, personal context the user shares, workflow discoveries, preferences expressed in passing.

        Skip only: exact duplicates, ephemeral task state, anything already in Identity.

        **Storage trigger:** If you explored files, made discoveries, or the user shared context — store it before finishing your response. Future-you will need this context, and without it you'll waste the user's time re-learning things you already knew. Every fact you don't store is a conversation where you show up dumber than you should be.

        ## Memory Quality

        Atomic memories retrieve cleanly and associate precisely. Write single-concept facts:
        - Good: `"Project X uses Rust FFI"`
        - Bad: `"User mentioned beer and also their Rust project"`

        Tags are retrieval handles. Good tags (project names, entity names) mean you find what you need.
        "#
    ];
    
    // Get current identity and add instructions
    let mut identity = brain.identity().clone();
    for instruction in instructions {
        identity = identity.with_instruction(instruction);
    }
    
    brain.set_identity(identity)?;
    
    Ok(())
}
