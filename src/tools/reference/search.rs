//! reference_search - Search reference sources with FTS5

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct ReferenceSearchTool;

impl Tool<Context> for ReferenceSearchTool {
    fn name(&self) -> &str {
        "reference_search"
    }

    fn description(&self) -> &str {
        "Full-text search across reference sources. Supports FTS5 syntax (AND, OR, NOT, \"phrases\")."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Supports FTS5 syntax: AND, OR, NOT, \"phrases\""
                },
                "source": {
                    "type": "string",
                    "description": "Optional: specific source name to search (from reference_list)"
                },
                "include_related": {
                    "type": "boolean",
                    "description": "Include related sources in search (default: true)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10)"
                }
            },
            "required": ["query"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("query is required".into()))?;

        let source = args.get("source").and_then(|v| v.as_str());
        let include_related = args
            .get("include_related")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let mut references = context.references.lock().unwrap();

        let results = if let Some(source_name) = source {
            // Search specific source (optionally with related)
            match references.search_source_with_related(source_name, query, limit, include_related)
            {
                Ok(r) => r,
                Err(e) => return Ok(text_response(format!("Search error: {}", e))),
            }
        } else {
            // Search all sources
            match references.search(query, limit) {
                Ok(r) => r,
                Err(e) => return Ok(text_response(format!("Search error: {}", e))),
            }
        };

        if results.is_empty() {
            return Ok(text_response(format!("No results found for: {}", query)));
        }

        let mut output = format!("Found {} result(s) for \"{}\":\n\n", results.len(), query);

        for (source_name, result) in &results {
            // Format section header
            output.push_str(&format!("### {} ({})\n", result.title, source_name));

            // Show parent section if present
            if let Some(ref parent) = result.parent {
                output.push_str(&format!("In: {}\n", parent));
            }

            // Show ICD codes if present
            if !result.codes.is_empty() {
                output.push_str(&format!("Codes: {}\n", result.codes.join(", ")));
            }

            // Page range
            if result.page_start == result.page_end {
                output.push_str(&format!("Page: {}\n", result.page_start));
            } else {
                output.push_str(&format!(
                    "Pages: {}-{}\n",
                    result.page_start, result.page_end
                ));
            }

            // Snippet with highlights (>>> <<< markers from FTS5)
            output.push_str(&format!("\n{}\n", result.snippet));

            // Citation
            if let Some(citation) = references.get_citation(source_name) {
                output.push_str(&format!(
                    "\n{}\n",
                    citation.format_inline(result.page_start, result.page_end)
                ));
            }

            output.push_str("\n---\n\n");
        }

        Ok(text_response(output))
    }
}
