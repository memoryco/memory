//! reference_list - List loaded reference sources

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::tools::text_response;

pub struct ReferenceListTool;

impl Tool<Context> for ReferenceListTool {
    fn name(&self) -> &str {
        "reference_list"
    }

    fn description(&self) -> &str {
        "List all loaded reference sources. Shows source names that can be used with \
         other reference tools."
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
        let references = context.references.lock().unwrap();
        let sources = references.sources();
        
        if sources.is_empty() {
            return Ok(text_response(format!(
                "No reference sources loaded.\n\nTo add sources, place PDF files in: {}/references/",
                context.memory_home.display()
            )));
        }

        let mut output = format!("Loaded reference sources ({}):\n\n", sources.len());
        for name in &sources {
            // Try to get citation for richer display
            if let Some(citation) = references.get_citation(name) {
                output.push_str(&format!("• {} - {} ({})\n", name, citation.title, citation.year));
            } else {
                output.push_str(&format!("• {}\n", name));
            }
        }

        Ok(text_response(output))
    }
}
