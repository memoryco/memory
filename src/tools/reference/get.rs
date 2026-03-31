//! reference_get - Get a specific section by title

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct ReferenceGetTool;

impl Tool<Context> for ReferenceGetTool {
    fn name(&self) -> &str {
        "reference_get"
    }

    fn description(&self) -> &str {
        "Get full section content by exact title. Use reference_sections to find titles."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source name (from reference_list)"
                },
                "title": {
                    "type": "string",
                    "description": "Exact section title (from reference_sections or search results)"
                }
            },
            "required": ["source", "title"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let source_name = args
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("source is required".into()))?;

        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("title is required".into()))?;

        let mut references = context.references.lock().unwrap();

        let result = match references.get_section(source_name, title) {
            Ok(Some(r)) => r,
            Ok(None) => {
                return Ok(text_response(format!(
                    "Section not found: \"{}\" in {}\n\nUse reference_sections to list available sections.",
                    title, source_name
                )));
            }
            Err(e) => return Ok(text_response(format!("Error: {}", e))),
        };

        let mut output = format!("# {}\n\n", result.title);

        // Show parent section if present
        if let Some(ref parent) = result.parent {
            output.push_str(&format!("**Parent:** {}\n", parent));
        }

        // Show ICD codes if present
        if !result.codes.is_empty() {
            output.push_str(&format!("**Codes:** {}\n", result.codes.join(", ")));
        }

        // Page range
        if result.page_start == result.page_end {
            output.push_str(&format!("**Page:** {}\n", result.page_start));
        } else {
            output.push_str(&format!(
                "**Pages:** {}-{}\n",
                result.page_start, result.page_end
            ));
        }

        output.push_str("\n---\n\n");

        // Full content (stored in snippet field for get_section)
        output.push_str(&result.snippet);

        // Citation
        output.push_str("\n\n---\n\n");
        if let Some(citation) = references.get_citation(source_name) {
            output.push_str(&format!(
                "**Citation:** {}\n\n**Reference:** {}",
                citation.format_inline(result.page_start, result.page_end),
                citation.format_reference()
            ));
        }

        Ok(text_response(output))
    }
}
