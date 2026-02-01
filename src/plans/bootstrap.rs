//! Plans bootstrap - seed operational instructions on first run

use crate::engram::Brain;

const INSTRUCTIONS: &str = r#"## Plans

Plans are lightweight task tracking for multi-step work. Use them to stay organized during complex tasks and conversations.

**When to use plans:**
- Multi-step tasks that span multiple messages
- Complex work where you might lose track
- Anything with clear milestones worth tracking

**Tools:**
- `plans` - list active plans (id + description)
- `plan_get` - view a plan with all steps and their status
- `plan_start` - create a new plan, returns id
- `plan_stop` - delete a plan (done or abandoned)
- `step_add` - add a step to a plan (auto-indexed)
- `step_complete` - mark a step done (auto-closes plan when all complete)

**Workflow:**
1. `plan_start("description")` → get plan id
2. `step_add(id, "step description")` for each step
3. `step_complete(id, step)` as you finish each
4. Plan auto-closes when all steps complete (or use `plan_stop` to abandon)

**Key distinction:**
- Plans are transient - deleted when stopped
- No history, no decay - just active working documents
- One plan per task, stop it when you're done
"#;

/// Marker to detect if plans instructions already exist
const MARKER: &str = "## Plans";

/// Bootstrap plans instructions into identity if not present
pub fn bootstrap(brain: &mut Brain) -> Result<(), Box<dyn std::error::Error>> {
    // Check if already bootstrapped
    let already_present = brain.identity().instructions.iter()
        .any(|i| i.contains(MARKER));
    
    if already_present {
        return Ok(());
    }
    
    eprintln!("  Bootstrapping plans instructions...");
    
    let mut identity = brain.identity().clone();
    identity = identity.with_instruction(INSTRUCTIONS);
    brain.set_identity(identity)?;
    
    eprintln!("  Plans instructions added to identity");
    Ok(())
}
