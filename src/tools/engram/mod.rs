//! Engram (memory) tools - CRUD and graph operations

mod associate;
mod associations;
mod create;
mod delete;
mod get;
mod graph;
mod query_expansion;
mod recall;
mod search;
mod stats;

pub use associate::EngramAssociateTool;
pub use associations::EngramAssociationsTool;
pub use create::EngramCreateTool;
pub use delete::EngramDeleteTool;
pub use get::EngramGetTool;
pub use graph::EngramGraphTool;
pub use recall::EngramRecallTool;
pub use search::EngramSearchTool;
pub use stats::EngramStatsTool;

/// Accumulate a text signal into a session's running centroid.
///
/// Loads or creates the session, appends the text to its query history,
/// updates the centroid via EMA, and saves. Failures are logged but never
/// propagated — session context is best-effort.
///
/// Optional ID tracking: pass `search_result_ids` or `created_ids` to record
/// which engrams were touched in this session, avoiding a separate load/save cycle.
///
/// Returns the embedding generated for the text, if successful. Callers can
/// reuse this embedding (e.g. to store on the engram) to avoid regenerating it.
pub(crate) fn accumulate_session_signal(
    context: &crate::Context,
    session_id: &str,
    text: &str,
    search_result_ids: Option<&[uuid::Uuid]>,
    created_ids: Option<&[uuid::Uuid]>,
) -> Option<Vec<f32>> {
    let brain = context.brain.read().unwrap();
    let config = brain.config();
    let max_queries = config.session_max_queries;
    let smoothing = config.session_centroid_smoothing;

    let mut session = brain
        .load_session(session_id)
        .unwrap_or(None)
        .unwrap_or_else(|| crate::engram::SessionContext::new(session_id));

    session.touch();
    session.add_query(text, max_queries);

    let generator = crate::embedding::EmbeddingGenerator::new();
    let generated = generator.generate(text).ok();

    if let Some(ref embedding) = generated {
        session.update_centroid(embedding, smoothing);
    }

    if let Some(ids) = search_result_ids {
        session.add_search_results(ids);
    }
    if let Some(ids) = created_ids {
        session.add_created(ids);
    }

    if let Err(e) = brain.save_session(&session) {
        eprintln!("[session] Failed to save session {}: {}", session_id, e);
    }

    generated
}
