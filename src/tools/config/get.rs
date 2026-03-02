//! config_get - Get current configuration

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::tools::text_response;

pub struct ConfigGetTool;

impl Tool<Context> for ConfigGetTool {
    fn name(&self) -> &str {
        "config_get"
    }

    fn description(&self) -> &str {
        "Get the current memory system configuration."
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
        let brain = context.brain.lock().unwrap();
        let config = brain.config();

        let output = format!(
            "Memory Configuration:\n\n\
             decay_rate_per_day: {:.2} ({}% energy loss per day of non-use)\n\
             decay_interval_hours: {:.1} (minimum hours between decay checks)\n\
             propagation_damping: {:.2} (signal reduction to neighbors)\n\
             hebbian_learning_rate: {:.2} (association strength on co-access)\n\
             recall_strength: {:.2} (energy boost when recalling)\n\
             search_follow_associations: {} (follow associations during search)\n\
             search_association_depth: {} (hops to follow)\n\
             rerank_enabled: {} (cross-encoder re-ranking on search)\n\
             rerank_candidates: {} (candidates for re-ranking pass)\n\
             hybrid_search_enabled: {} (BM25+vector fusion via RRF)\n\
             query_expansion_enabled: {} (expand queries with variants before retrieval)",
            config.decay_rate_per_day,
            config.decay_rate_per_day * 100.0,
            config.decay_interval_hours,
            config.propagation_damping,
            config.hebbian_learning_rate,
            config.recall_strength,
            config.search_follow_associations,
            config.search_association_depth,
            config.rerank_enabled,
            config.rerank_candidates,
            config.hybrid_search_enabled,
            config.query_expansion_enabled
        );

        Ok(text_response(output))
    }
}
