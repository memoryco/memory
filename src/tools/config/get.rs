//! config_get - Get current configuration

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

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
             embedding_model: {} (desired embedding model)\n\
             embedding_model_active: {} (model used for current embeddings)\n\
             rerank_mode: {} (reranking mode: off, cross-encoder, or llm)\n\
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
            config.embedding_model,
            config.embedding_model_active.as_deref().unwrap_or("(none)"),
            config.rerank_mode,
            config.rerank_candidates,
            config.hybrid_search_enabled,
            config.query_expansion_enabled
        );

        Ok(text_response(output))
    }
}
