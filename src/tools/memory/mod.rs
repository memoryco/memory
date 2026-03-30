//! Memory (memory) tools - CRUD and graph operations

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

pub use associate::MemoryAssociateTool;
pub use associations::MemoryAssociationsTool;
pub use create::MemoryCreateTool;
pub use delete::MemoryDeleteTool;
pub use get::MemoryGetTool;
pub use graph::MemoryGraphTool;
pub use recall::MemoryRecallTool;
pub use search::MemorySearchTool;
pub use stats::MemoryStatsTool;

/// Accumulate a text signal into a session's running centroid.
///
/// Loads or creates the session, appends the text to its query history,
/// updates the centroid via EMA, and saves. Failures are logged but never
/// propagated — session context is best-effort.
///
/// Optional ID tracking: pass `search_result_ids` or `created_ids` to record
/// which memories were touched in this session, avoiding a separate load/save cycle.
///
/// Returns the embedding generated for the text, if successful. Callers can
/// reuse this embedding (e.g. to store on the memory) to avoid regenerating it.
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
        .unwrap_or_else(|| crate::memory_core::SessionContext::new(session_id));

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

/// Wire bidirectional associations between all created and recalled memories
/// in a session. Uses create-if-absent semantics at weight 0.5 — existing
/// associations are left untouched (no within-session strengthening).
///
/// Requires a write lock on brain. Callers should invoke this as a separate
/// lock phase after session accumulation is complete.
pub(crate) fn wire_session_associations(context: &crate::Context, session_id: &str) {
    let mut brain = context.brain.write().unwrap();

    // Load session to get the current ID sets
    let session = match brain.load_session(session_id) {
        Ok(Some(s)) => s,
        _ => return,
    };

    // Combine created + recalled IDs — these are the high-confidence tiers.
    // Dedup since a memory could appear in both created_ids and recalled_ids.
    let mut seen = std::collections::HashSet::new();
    let all_ids: Vec<uuid::Uuid> = session
        .created_ids
        .iter()
        .chain(session.recalled_ids.iter())
        .filter_map(|s| s.parse::<uuid::Uuid>().ok())
        .filter(|id| seen.insert(*id))
        .collect();

    if all_ids.len() < 2 {
        return;
    }

    let mut wired = 0usize;
    for i in 0..all_ids.len() {
        for j in (i + 1)..all_ids.len() {
            let a = all_ids[i];
            let b = all_ids[j];
            // Bidirectional: A→B and B→A
            if let Ok(true) = brain.associate_if_absent(a, b, 0.5) {
                wired += 1;
            }
            if let Ok(true) = brain.associate_if_absent(b, a, 0.5) {
                wired += 1;
            }
        }
    }

    if wired > 0 {
        eprintln!(
            "[session] Wired {} new associations from {} session IDs (session {})",
            wired,
            all_ids.len(),
            session_id
        );
    }
}
