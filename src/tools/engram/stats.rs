//! engram_stats - Get substrate statistics

use serde_json::{json, Value as JsonValue};
use sovran_mcp::server::server::{McpTool, McpToolEnvironment};
use sovran_mcp::types::{CallToolResponse, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct EngramStatsTool;

impl McpTool<Context> for EngramStatsTool {
    fn name(&self) -> &str {
        "engram_stats"
    }

    fn description(&self) -> &str {
        "Get statistics about the memory substrate."
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let brain = context.brain.lock().unwrap();
        let stats = brain.stats();

        let output = format!(
            "Memory Substrate Statistics:\n\n\
             Total memories: {}\n\
             ✨ Active: {}\n\
             💤 Dormant: {}\n\
             🌊 Deep: {}\n\
             🧊 Archived: {}\n\n\
             Total associations: {}\n\
             Average energy: {:.2}",
            stats.total_engrams,
            stats.active_engrams,
            stats.dormant_engrams,
            stats.deep_engrams,
            stats.archived_engrams,
            stats.total_associations,
            stats.average_energy
        );

        Ok(text_response(output))
    }
}
