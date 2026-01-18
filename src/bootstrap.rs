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
        // Bootstrap workflow
        "Memory workflow: Start conversations with identity_get to load identity (persona, values, instructions), then search engrams for relevant context.",
        
        // Identity vs Engrams
        "Identity layer is permanent (persona, values, preferences, relationships, instructions). It never decays. Use identity_get/identity_set.",
        "Engram layer is organic memory that decays without use. Use engram_create for new memories, engram_recall when using a memory, engram_search for passive lookup.",
        
        // Critical: Search then Recall workflow
        "CRITICAL WORKFLOW: engram_search finds memories (passive, no side effects). When you actually USE a found memory in conversation, call engram_recall on its ID. This is what keeps memories alive and builds associations.",
        "Example: engram_search returns memories A, B, C. You use A and B in your response. Call engram_recall on A and B (not C). This strengthens A and B, links them via Hebbian learning.",
        
        // Memory format
        "Memory format: Write atomic, single-concept facts. Good: 'Project X uses Rust FFI'. Bad: 'User mentioned beer and also their Rust project'.",
        
        // Tags
        "Use tags for categorization. Common patterns: 'context' for background info, project/entity names for scoping.",
        
        // Associations
        "Associations form automatically via Hebbian learning when memories are recalled together in the same conversation. Use engram_associate for explicit links.",
        
        // When to create/not create
        "When to CREATE engrams: significant project facts (architecture, locations, gotchas), corrections to previous understanding, personal context user shares, tool/workflow discoveries.",
        "When NOT to create engrams: ephemeral details (current task state), duplicates of existing memories, anything already covered in Identity.",
    ];
    
    // Get current identity and add instructions
    let mut identity = brain.identity().clone();
    for instruction in instructions {
        identity = identity.with_instruction(instruction);
    }
    
    brain.set_identity(identity)?;
    
    Ok(())
}
