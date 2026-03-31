//! identity_get - Get the current identity, optionally with memory search results

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::memory::MemorySearchTool;
use crate::tools::{extract_text, text_response};

pub struct IdentityGetTool;

#[derive(Deserialize, Default)]
struct Args {
    /// Optional search queries to run alongside identity loading
    #[serde(default)]
    queries: Option<Vec<String>>,
}

impl Tool<Context> for IdentityGetTool {
    fn name(&self) -> &str {
        "identity_get"
    }

    fn description(&self) -> &str {
        "Load persistent identity on FIRST message of every conversation. Pass search \
         queries from the user's message to retrieve relevant memories in the same call. \
         Call `instructions` after this."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional array of search queries to run alongside \
                        identity loading. Results are returned after identity. Derive \
                        queries from keywords and concepts in the user's first message."
                }
            }
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args = serde_json::from_value(args).unwrap_or_default();

        // Generate a session ID for this conversation. Every subsequent tool
        // call that accepts session_id will echo it back so it survives compaction.
        let session_id = crate::memory_core::generate_session_id();

        // Phase 1: Load identity (the framing lens)
        let mut output = format!("session_id: {}\n\n", session_id);
        output += &{
            let mut store = context.identity.lock().unwrap();
            let identity = store
                .get()
                .map_err(|e| McpError::ToolError(e.to_string()))?;

            let mut text = identity.render();

            if identity.persona.name.is_empty() {
                text.insert_str(
                    0,
                    "No persona is currently configured. \
                     You might ask the user if they'd like to set one up \
                     (use `identity_setup` to walk them through it).\n\n\
                     ---\n\n",
                );
            }

            text
        }; // identity lock released here

        // Phase 2: Run search queries if provided (piggyback on the same call)
        if let Some(queries) = &args.queries {
            if !queries.is_empty() {
                let search_tool = MemorySearchTool;
                output.push_str("\n\n---\n\n# Memory Search Results\n\n");

                let search_args = json!({ "queries": queries, "session_id": session_id });
                match search_tool.execute(search_args, context, env) {
                    Ok(result) => {
                        output.push_str(&extract_text(&result));
                    }
                    Err(e) => {
                        output.push_str(&format!("Search failed: {}\n", e));
                    }
                }
            }
        }

        Ok(text_response(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{DieselIdentityStorage, IdentityStore};
    use sml_mcps::Content;

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

    // --- Args deserialization ---

    #[test]
    fn args_empty_object_defaults_to_no_queries() {
        let args: Args = serde_json::from_value(json!({})).unwrap();
        assert!(args.queries.is_none());
    }

    #[test]
    fn args_with_queries_deserializes() {
        let args: Args =
            serde_json::from_value(json!({ "queries": ["topic A", "topic B"] })).unwrap();
        let queries = args.queries.unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0], "topic A");
        assert_eq!(queries[1], "topic B");
    }

    #[test]
    fn args_with_empty_queries_array() {
        let args: Args = serde_json::from_value(json!({ "queries": [] })).unwrap();
        let queries = args.queries.unwrap();
        assert!(queries.is_empty());
    }

    #[test]
    fn args_null_value_defaults() {
        // Handles the case where the LLM sends null or invalid args
        let args: Args = serde_json::from_value(json!(null)).unwrap_or_default();
        assert!(args.queries.is_none());
    }

    // --- extract_text helper ---

    #[test]
    fn extract_text_from_text_content() {
        let result = CallToolResult::text("hello world");
        assert_eq!(extract_text(&result), "hello world");
    }

    #[test]
    fn extract_text_skips_non_text_content() {
        let result = CallToolResult {
            content: vec![
                Content::Text {
                    text: "text part".to_string(),
                },
                Content::Image {
                    data: "base64data".to_string(),
                    mime_type: "image/png".to_string(),
                },
            ],
            is_error: false,
        };
        assert_eq!(extract_text(&result), "text part");
    }

    #[test]
    fn extract_text_joins_multiple_text_blocks() {
        let result = CallToolResult {
            content: vec![
                Content::Text {
                    text: "first".to_string(),
                },
                Content::Text {
                    text: "second".to_string(),
                },
            ],
            is_error: false,
        };
        assert_eq!(extract_text(&result), "first\nsecond");
    }

    #[test]
    fn extract_text_empty_content() {
        let result = CallToolResult {
            content: vec![],
            is_error: false,
        };
        assert_eq!(extract_text(&result), "");
    }
}
