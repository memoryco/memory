//! engram_stats - Get substrate statistics

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct EngramStatsTool;

impl Tool<Context> for EngramStatsTool {
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
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let mut brain = context.brain.write().unwrap();

        // Lazy maintenance: decay + cross-process sync
        let _ = brain.apply_time_decay();
        let _ = brain.sync_from_storage();

        let stats = brain.stats();
        let enrichment_count = brain.count_enrichments().unwrap_or(0);

        let output = format!(
            "Memory Substrate Statistics:\n\n\
             Total memories: {}\n\
             ✨ Active: {}\n\
             💤 Dormant: {}\n\
             🌊 Deep: {}\n\
             🧊 Archived: {}\n\n\
             Total associations: {}\n\
             Enrichment vectors: {}\n\
             Average energy: {:.2}",
            stats.total_engrams,
            stats.active_engrams,
            stats.dormant_engrams,
            stats.deep_engrams,
            stats.archived_engrams,
            stats.total_associations,
            enrichment_count,
            stats.average_energy
        );

        Ok(text_response(output))
    }
}
