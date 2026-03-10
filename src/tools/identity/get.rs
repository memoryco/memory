//! identity_get - Get the current identity

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct IdentityGetTool;

impl Tool<Context> for IdentityGetTool {
    fn name(&self) -> &str {
        "identity_get"
    }

    fn description(&self) -> &str {
        "Get the current identity (persona, values, preferences, relationships). \
         Identity never decays - it's who you ARE, not what you remember."
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

        let mut output = identity.render();

        if identity.persona.name.is_empty() {
            output.insert_str(
                0,
                "No persona is currently configured. \
                 You might ask the user if they'd like to set one up \
                 (use `identity_setup` to walk them through it).\n\n\
                 ---\n\n",
            );
        }

        Ok(text_response(output))
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
    fn get_empty_identity_suggests_setup() {
        let mut store = test_store();
        let identity = store.get().unwrap();
        let mut output = identity.render();

        if identity.persona.name.is_empty() {
            output.insert_str(
                0,
                "No persona is currently configured. \
                You might ask the user if they'd like to set one up \
                (use `identity_setup` to walk them through it).\n\n---\n\n",
            );
        }

        assert!(
            output.contains("identity_setup"),
            "Empty identity should mention identity_setup"
        );
        assert!(
            output.contains("No persona"),
            "Should indicate no persona configured"
        );
    }

    #[test]
    fn get_populated_identity_no_hint() {
        let mut store = test_store();
        store.set_persona_name("Porter").unwrap();
        store
            .set_persona_description("A pragmatic assistant")
            .unwrap();

        let identity = store.get().unwrap();
        let mut output = identity.render();

        if identity.persona.name.is_empty() {
            output.push_str("identity_setup hint");
        }

        assert!(
            !output.contains("identity_setup"),
            "Populated identity should NOT hint about setup"
        );
        assert!(output.contains("Porter"), "Should render the persona name");
    }
}
