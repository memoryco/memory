//! reference_citation - Get citation info for a source

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::tools::text_response;

pub struct ReferenceCitationTool;

impl Tool<Context> for ReferenceCitationTool {
    fn name(&self) -> &str {
        "reference_citation"
    }

    fn description(&self) -> &str {
        "Get APA 7 citation for a reference source. Returns both in-text and full reference formats."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source name (from reference_list)"
                },
                "page_start": {
                    "type": "integer",
                    "description": "Optional: starting page for in-text citation"
                },
                "page_end": {
                    "type": "integer",
                    "description": "Optional: ending page for in-text citation"
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
        
        let page_start = args.get("page_start").and_then(|v| v.as_u64()).map(|p| p as usize);
        let page_end = args.get("page_end").and_then(|v| v.as_u64()).map(|p| p as usize);

        let references = context.references.lock().unwrap();

        let citation = match references.get_citation(source_name) {
            Some(c) => c,
            None => return Ok(text_response(format!(
                "No citation metadata found for: {}\n\n\
                 To add citation info, create a {}.meta.json file next to the PDF.",
                source_name, source_name
            ))),
        };

        let mut output = String::new();

        // In-text citation
        output.push_str("**In-text citation:**\n");
        if let (Some(start), Some(end)) = (page_start, page_end) {
            output.push_str(&citation.format_inline(start, end));
        } else if let Some(start) = page_start {
            output.push_str(&citation.format_inline(start, start));
        } else {
            output.push_str(&citation.format_inline_short());
        }
        
        output.push_str("\n\n**Full reference:**\n");
        output.push_str(&citation.format_reference());

        Ok(text_response(output))
    }
}
