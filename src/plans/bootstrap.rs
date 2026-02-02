//! Plans bootstrap - seed operational instructions on first run

use crate::engram::{Brain, UpsertResult};

const INSTRUCTIONS: &str = r#"## Plans

Plans are lightweight task tracking for multi-step work. Use them to stay organized during complex tasks and conversations.

**When to use plans:**
- Multi-step tasks that span multiple messages
- Complex work where you might lose track
- Anything with clear milestones worth tracking
- Keep steps small and atomic (~5 min each). If a step feels too big, break it down first.

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

/// Marker to detect plans instructions
const MARKER: &str = "## Plans";

/// Bootstrap plans instructions into identity
/// Adds if missing, updates if changed, skips if identical
pub fn bootstrap(brain: &mut Brain) -> Result<(), Box<dyn std::error::Error>> {
    match brain.upsert_instruction(INSTRUCTIONS, MARKER)? {
        UpsertResult::Added => {
            eprintln!("  Plans instructions added to identity");
        }
        UpsertResult::Updated => {
            eprintln!("  Plans instructions updated in identity");
        }
        UpsertResult::Unchanged => {
            // Already up to date, no message needed
        }
    }
    Ok(())
}
