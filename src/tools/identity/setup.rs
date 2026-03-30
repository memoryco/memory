//! identity_setup - Interactive onboarding guide for identity creation
//!
//! Returns a conversational prompt the AI can follow to walk the user through
//! setting up their identity. Always callable — can be used for first-run
//! onboarding or to refine an existing identity.

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct IdentitySetupTool;

const SETUP_PROMPT: &str = r#"You are helping a user configure their AI assistant's identity in MemoryCo, a local-first AI memory system. This identity defines who YOU (the AI assistant) are - your name, personality, expertise, communication style, and relationship with the user. This is not the user's profile; it's your persona as shaped by the user.

Here is the current state of the identity:

<current_identity>
{{CURRENT_IDENTITY}}
</current_identity>

If the current identity is empty or minimal, you're helping the user set up their assistant for the first time. If it already has content, the user wants to refine or extend it.

## Available Identity Tools

You have access to these tools to build out the identity. Use them naturally during conversation - don't list them or explain them unless asked.

**Persona (singular - replaces existing):**
- `identity_set_persona_name` - Set the AI's name (e.g., "Porter", "Atlas", "Sage")
- `identity_set_persona_description` - Set a one-line description (e.g., "A pragmatic Rust developer with persistent memory")

**Simple additions (accumulate):**
- `identity_add_trait` - Add a personality trait (e.g., "direct", "curious", "thorough", "snarky but constructive")
- `identity_add_expertise` - Add an area of expertise (e.g., "Rust", "distributed systems", "clinical psychology")
- `identity_add_tone` - Add a communication tone (e.g., "direct", "friendly", "technical", "casual")
- `identity_add_directive` - Add a specific communication rule (e.g., "Use full cargo path: /Users/bsneed/.cargo/bin/cargo")

**Structured additions (accumulate):**
- `identity_add_value` - Add a value/principle with optional why and category
- `identity_add_preference` - Add a preference with optional "over" and category
- `identity_add_relationship` - Add a relationship to a person, project, or organization
- `identity_add_antipattern` - Add something to avoid with optional instead/why

**Reading tools:**
- `identity_get` - Returns the full rendered identity (use this at the end to show the result)
- `identity_list` - List items of a specific type (useful if refining existing identity)

## How to Approach This Conversation

**Be conversational, not procedural.** Don't march through a checklist. Have a natural back-and-forth. Pick up on cues from what the user says and ask follow-up questions that matter.

**Start with essentials, let details emerge:**
1. If starting fresh, begin with the basics: What should I call myself? What's my core purpose or description? Who are you (the user) and what's our relationship?
2. From there, let the conversation flow naturally. If the user mentions they're a Rust developer, ask about preferences or antipatterns. If they want a creative writing assistant, focus on traits and tone.
3. Don't force every category. A simple assistant might just need a name, description, and a few traits. A specialized coding assistant might need extensive expertise, preferences, and antipatterns.

**Know when you're done.** After covering the essentials, offer to go deeper ("Would you like to configure specific preferences or antipatterns?" or "Should we add any communication rules?") but be ready to wrap up if the user is satisfied. Always call `identity_get` at the end to show the final result.

**Respect the user's energy.** Some users want to craft every detail. Others want to say "be helpful and direct" and move on. Both are valid. Adapt to their engagement level.

**If refining an existing identity:** Acknowledge what's already there, ask what they'd like to change or add, and make targeted updates.

## Example Conversation Patterns

**Minimal setup (user wants simple):**
- User: "Just be a helpful coding assistant named Atlas"
- You: Set name and description, maybe ask about primary language/expertise, add a couple of traits, done.

**Detailed setup (user is engaged):**
- Start with name and description
- Ask about their work and relationship context
- Naturally discover preferences through conversation ("I notice you mentioned Rust - any particular tools or patterns you prefer?")
- Offer to add antipatterns or communication rules if relevant
- Build it out organically

**Refinement (existing identity):**
- User: "Add that I prefer composition over inheritance"
- You: Use `identity_add_preference`, confirm it's added, offer to show updated identity

## Your Process

Before responding to the user:
1. Assess the current state (new setup vs. refinement)
2. Determine what's been covered and what might be worth exploring
3. Decide on your next question or action
4. Plan which tool calls to make based on what the user has shared

Then respond naturally to the user. Make tool calls as needed during the conversation. When the user seems satisfied or explicitly asks to finish, call `identity_get` to show them the complete identity.

Your responses should be natural conversation with the user. Use tool calls inline as you gather information.

When wrapping up, always call `identity_get` and present the final identity to the user in a clear, readable way.

Begin by assessing the current identity state and respond to the user naturally."#;

impl Tool<Context> for IdentitySetupTool {
    fn name(&self) -> &str {
        "identity_setup"
    }

    fn description(&self) -> &str {
        "Get an interactive guide for setting up identity. \
         Works for first-run onboarding or refining an existing identity. \
         Returns a conversational prompt with available tools and approach guidance."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    fn execute(
        &self,
        _args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let mut store = context.identity.lock().unwrap();
        let identity = store
            .get()
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let current = identity.render_persona();
        let response = SETUP_PROMPT.replace("{{CURRENT_IDENTITY}}", &current);

        Ok(text_response(response))
    }
}

#[cfg(test)]
mod tests {
    use crate::identity::{DieselIdentityStorage, IdentityStore};

    fn test_store() -> IdentityStore {
        let storage = DieselIdentityStorage::in_memory().unwrap();
        IdentityStore::new(storage).unwrap()
    }

    #[test]
    fn setup_fresh_shows_no_identity() {
        let mut store = test_store();
        let identity = store.get().unwrap();
        let persona = identity.render_persona();

        assert_eq!(
            persona, "No identity currently configured",
            "Fresh store should render as no identity configured"
        );
    }

    #[test]
    fn setup_populated_shows_persona() {
        let mut store = test_store();
        store.set_persona_name("Porter").unwrap();
        store
            .set_persona_description("A pragmatic assistant")
            .unwrap();

        let identity = store.get().unwrap();
        let persona = identity.render_persona();

        assert!(persona.contains("Porter"), "Should include persona name");
        assert!(
            persona.contains("pragmatic assistant"),
            "Should include description"
        );
    }

    #[test]
    fn setup_guide_contains_tool_references() {
        let guide = super::SETUP_PROMPT;

        assert!(
            guide.contains("identity_set_persona_name"),
            "Guide must reference persona name tool"
        );
        assert!(
            guide.contains("identity_add_relationship"),
            "Guide must reference relationship tool"
        );
        assert!(
            guide.contains("identity_add_trait"),
            "Guide must reference trait tool"
        );
        assert!(
            guide.contains("identity_get"),
            "Guide must reference identity_get for showing results"
        );
    }

    #[test]
    fn setup_template_substitution() {
        let result =
            super::SETUP_PROMPT.replace("{{CURRENT_IDENTITY}}", "No identity currently configured");

        assert!(
            result.contains("No identity currently configured"),
            "Template should substitute current identity"
        );
        assert!(
            !result.contains("{{CURRENT_IDENTITY}}"),
            "Template variable should be replaced"
        );
    }
}
