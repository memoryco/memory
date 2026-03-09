//! engram_recall - Actively recall memories (batch)

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::{text_response, format_engram};

pub struct EngramRecallTool;

#[derive(Deserialize)]
struct Args {
    ids: Vec<String>,
    #[serde(default)]
    strength: Option<f64>,
}

impl Tool<Context> for EngramRecallTool {
    fn name(&self) -> &str {
        "engram_recall"
    }

    fn description(&self) -> &str {
        "Actively recall memories. Accepts an array of IDs to recall in one call. \
         Stimulates memories (increases energy), triggers Hebbian learning between \
         all recalled memories, and can resurrect archived memories. Memories \
         recalled together form associations automatically."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of UUIDs to recall. All memories will be linked via Hebbian learning."
                },
                "strength": {
                    "type": "number",
                    "description": "Stimulation strength (0.0-1.0). Default uses config value."
                }
            },
            "required": ["ids"]
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

        let mut brain = context.brain.lock().unwrap();
        
        // Lazy maintenance: decay + cross-process sync
        let _ = brain.apply_time_decay();
        let _ = brain.sync_from_storage();
        
        let mut output = String::new();
        let mut recalled_count = 0;
        let mut not_found_count = 0;
        let mut total_affected = 0;

        for id_str in &args.ids {
            let id: EngramId = id_str.parse()
                .map_err(|e| McpError::InvalidParams(format!("Invalid UUID '{}': {}", id_str, e)))?;

            let result = if let Some(s) = args.strength {
                brain.recall_with_strength(id, s)
            } else {
                brain.recall(id)
            }.map_err(|e| McpError::ToolError(e.to_string()))?;

            if !result.found() {
                not_found_count += 1;
                output.push_str(&format!("Memory {} not found.\n\n", id));
                continue;
            }

            recalled_count += 1;
            total_affected += result.affected_count();

            let engram = result.engram.as_ref().unwrap();
            output.push_str(&format_engram(engram));

            if result.resurrected {
                output.push_str(&format!(
                    "\n🔄 RESURRECTED from {:?}!",
                    result.previous_state.unwrap()
                ));
            }
            output.push_str("\n\n");
        }

        // Log the search→recall cycle for membed training data extraction
        {
            let query_text = context.last_search_query.lock().ok()
                .and_then(|mut q| q.take());
            let result_ids = context.last_search_result_ids.lock().ok()
                .map(|mut r| std::mem::take(&mut *r))
                .unwrap_or_default();

            if let Some(query) = query_text {
                let recalled: Vec<EngramId> = args.ids.iter()
                    .filter_map(|id_str| id_str.parse().ok())
                    .collect();
                if let Err(e) = brain.log_access(&query, &result_ids, &recalled) {
                    eprintln!("[recall] access log write failed: {}", e);
                }
            }
        }

        let reminder = "💾 **REQUIRED:** If you explored files or learned ANY new facts, call engram_create BEFORE responding.\n---\n\n";
        
        let header = format!(
            "Recalled {} memories ({} not found). Affected {} total via Hebbian learning.\n\n",
            recalled_count, not_found_count, total_affected
        );

        Ok(text_response(format!("{}{}{}", reminder, header, output.trim())))
    }
}
