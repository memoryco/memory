//! engram_search - Passively search memories (no side effects)

use engram::{Engram, SearchOptions, TagMatchMode};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::{text_response, format_engram};

pub struct EngramSearchTool;

#[derive(Deserialize)]
struct Args {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    tag_mode: Option<String>,
    #[serde(default)]
    include_deep: Option<bool>,
    #[serde(default)]
    include_archived: Option<bool>,
    #[serde(default)]
    limit: Option<usize>,
}

impl Tool<Context> for EngramSearchTool {
    fn name(&self) -> &str {
        "engram_search"
    }

    fn description(&self) -> &str {
        "Search memories by content or tags. This is a passive operation - \
         it does NOT stimulate memories or trigger learning. Use engram_recall \
         when you actually want to use a memory."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in memory content."
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Filter by tags."
                },
                "tag_mode": {
                    "type": "string",
                    "enum": ["all", "any"],
                    "description": "Match all tags or any tag. Default: any"
                },
                "include_deep": {
                    "type": "boolean",
                    "description": "Include deep storage memories. Default: false"
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived memories. Default: false"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return."
                }
            }
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let brain = context.brain.lock().unwrap();

        let mut options = SearchOptions::default();
        if args.include_archived.unwrap_or(false) {
            options = options.include_all();
        } else if args.include_deep.unwrap_or(false) {
            options = options.include_deep();
        }
        if let Some(limit) = args.limit {
            options = options.with_limit(limit);
        }

        let results: Vec<&Engram> = if !args.tags.is_empty() {
            let tag_refs: Vec<&str> = args.tags.iter().map(|s| s.as_str()).collect();
            let mode = match args.tag_mode.as_deref() {
                Some("all") => TagMatchMode::All,
                _ => TagMatchMode::Any,
            };
            brain.search_by_tags(&tag_refs, mode)
        } else if let Some(query) = &args.query {
            // Use FTS search for tokenized matching
            match brain.search_fts(query) {
                Ok(mut fts_results) => {
                    // Apply state filtering based on options
                    let include_archived = args.include_archived.unwrap_or(false);
                    let include_deep = args.include_deep.unwrap_or(false);
                    
                    fts_results.retain(|e| {
                        if include_archived {
                            true // Include all states
                        } else if include_deep {
                            !e.is_archived() // Include Active, Dormant, Deep
                        } else {
                            e.is_searchable() // Only Active and Dormant
                        }
                    });
                    
                    // Apply limit
                    if let Some(limit) = args.limit {
                        fts_results.truncate(limit);
                    }
                    fts_results
                }
                Err(_) => {
                    // Fallback to substring search if FTS fails
                    brain.search_with_options(query, options)
                }
            }
        } else {
            brain.searchable_engrams().take(args.limit.unwrap_or(10)).collect()
        };

        if results.is_empty() {
            return Ok(text_response("No memories found.".to_string()));
        }

        let mut output = String::from(
            "⚡ **REQUIRED:** Call engram_recall on IDs you use. \n\
             💾 **REQUIRED:** Call engram_create if you learn ANY new facts this turn.\n\
             ---\n\n"
        );
        output.push_str(&format!("Found {} memories:\n\n", results.len()));
        for engram in results {
            output.push_str(&format_engram(engram));
            output.push_str("\n\n");
        }

        Ok(text_response(output))
    }
}
