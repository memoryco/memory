//! reference_sections - List sections in a reference source

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::tools::text_response;

pub struct ReferenceSectionsTool;

impl Tool<Context> for ReferenceSectionsTool {
    fn name(&self) -> &str {
        "reference_sections"
    }

    fn description(&self) -> &str {
        "List all top-level sections in a reference source. Useful for browsing \
         available content before searching or getting specific sections."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source name (from reference_list)"
                }
            },
            "required": ["source"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let source_name = args.get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("source is required".into()))?;

        let mut references = context.references.lock().unwrap();

        let sections = match references.list_sections(source_name) {
            Ok(s) => s,
            Err(e) => return Ok(text_response(format!("Error: {}", e))),
        };

        if sections.is_empty() {
            return Ok(text_response(format!(
                "No sections found in {}.\n\nThis source may be page-indexed (no section structure). \
                 Use reference_search to find content.",
                source_name
            )));
        }

        let mut output = format!("Sections in {} ({}):\n\n", source_name, sections.len());
        for section in &sections {
            output.push_str(&format!("• {}\n", section));
        }

        output.push_str("\nUse reference_get with exact title to retrieve section content.");

        Ok(text_response(output))
    }
}
