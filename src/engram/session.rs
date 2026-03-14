//! Session context for conversation-aware memory retrieval.
//!
//! A session accumulates query signal across tool calls, building a running
//! centroid embedding that biases memory recall toward the conversation topic.
//!
//! Sessions are persisted to `brain.db` and survive server restarts. They
//! don't need an explicit "end" — stale sessions are expired at startup via
//! `session_expire_days` config.

/// Generate a compact, timestamp-prefixed session ID (16 hex chars).
///
/// Format: 8 chars unix timestamp (seconds) + 8 chars random from UUID v4.
/// This gives natural sort order by creation time and effectively zero
/// collision risk — you'd need ~65K sessions in the same second to worry.
pub fn generate_session_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32;
    let random_bytes = uuid::Uuid::new_v4();
    let rand = &random_bytes.as_bytes()[..4];
    format!(
        "{:08x}{:02x}{:02x}{:02x}{:02x}",
        ts, rand[0], rand[1], rand[2], rand[3]
    )
}

/// Running context for a single conversation session.
///
/// Accumulates query strings and a rolling centroid embedding representing
/// the semantic topic of the conversation so far. The centroid is updated
/// via exponential moving average (EMA) so recent queries weigh more.
#[derive(Debug, Clone)]
pub struct SessionContext {
    /// Opaque identifier supplied by the caller (e.g. a thread/conversation ID).
    pub session_id: String,
    /// Recent query strings, capped at `session_max_queries`.
    pub queries: Vec<String>,
    /// Running centroid of accumulated query embeddings (EMA), L2-normalized.
    pub centroid: Option<Vec<f32>>,
    /// Total number of queries accumulated (may exceed `queries.len()` after capping).
    pub query_count: usize,
    /// Unix epoch seconds when this session was created.
    pub created_at: i64,
    /// Unix epoch seconds when this session was last touched.
    pub last_seen_at: i64,
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

impl SessionContext {
    /// Create a new, empty session with timestamps set to now.
    pub fn new(session_id: &str) -> Self {
        let now = now_secs();
        Self {
            session_id: session_id.to_string(),
            queries: Vec::new(),
            centroid: None,
            query_count: 0,
            created_at: now,
            last_seen_at: now,
        }
    }

    /// Append a query string to the session history.
    ///
    /// Trims the oldest entry from the front if `queries.len()` would exceed
    /// `max_queries`. Always increments `query_count`. A `max_queries` of 0
    /// means unlimited.
    pub fn add_query(&mut self, query: &str, max_queries: usize) {
        self.queries.push(query.to_string());
        if max_queries > 0 && self.queries.len() > max_queries {
            self.queries.remove(0);
        }
        self.query_count += 1;
    }

    /// Update the running centroid via exponential moving average.
    ///
    /// On the first call (no existing centroid), sets the centroid directly.
    /// On subsequent calls: `centroid = smoothing * new + (1 - smoothing) * old`.
    /// The result is always L2-normalized.
    ///
    /// `smoothing` is α — higher values weight recent queries more heavily.
    pub fn update_centroid(&mut self, new_embedding: &[f32], smoothing: f32) {
        let mut updated = match &self.centroid {
            None => new_embedding.to_vec(),
            Some(old) => old
                .iter()
                .zip(new_embedding.iter())
                .map(|(o, n)| smoothing * n + (1.0 - smoothing) * o)
                .collect(),
        };
        l2_normalize(&mut updated);
        self.centroid = Some(updated);
    }

    /// Update `last_seen_at` to the current time.
    pub fn touch(&mut self) {
        self.last_seen_at = now_secs();
    }
}

/// Normalize a vector in-place to unit length (L2 norm). No-op for zero vectors.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session_has_correct_defaults() {
        let before = now_secs();
        let session = SessionContext::new("test-session-id");
        let after = now_secs();

        assert_eq!(session.session_id, "test-session-id");
        assert!(session.queries.is_empty());
        assert!(session.centroid.is_none());
        assert_eq!(session.query_count, 0);
        assert!(session.created_at >= before && session.created_at <= after);
        assert!(session.last_seen_at >= before && session.last_seen_at <= after);
    }

    #[test]
    fn test_add_query_appends_and_caps() {
        let mut session = SessionContext::new("test");
        session.add_query("alpha", 3);
        session.add_query("beta", 3);
        session.add_query("gamma", 3);
        assert_eq!(session.queries.len(), 3);
        assert_eq!(session.query_count, 3);

        // Adding a 4th with cap=3 should evict "alpha"
        session.add_query("delta", 3);
        assert_eq!(session.queries.len(), 3);
        assert_eq!(session.query_count, 4);
        assert_eq!(session.queries[0], "beta");
        assert_eq!(session.queries[2], "delta");
    }

    #[test]
    fn test_update_centroid_cold_start() {
        let mut session = SessionContext::new("test");
        // [3, 4] normalized = [0.6, 0.8]
        session.update_centroid(&[3.0_f32, 4.0_f32], 0.1);
        let c = session.centroid.as_ref().unwrap();
        assert_eq!(c.len(), 2);
        assert!((c[0] - 0.6).abs() < 1e-5, "c[0]={}", c[0]);
        assert!((c[1] - 0.8).abs() < 1e-5, "c[1]={}", c[1]);
    }

    #[test]
    fn test_update_centroid_ema() {
        let mut session = SessionContext::new("test");
        // First: [1, 0] normalized = [1, 0]
        session.update_centroid(&[1.0_f32, 0.0_f32], 0.1);
        // Blend in [0, 1] with smoothing=0.1:
        // blended = 0.1*[0,1] + 0.9*[1,0] = [0.9, 0.1]
        session.update_centroid(&[0.0_f32, 1.0_f32], 0.1);

        let c = session.centroid.as_ref().unwrap();
        let expected_norm = (0.81_f32 + 0.01_f32).sqrt();
        let expected_x = 0.9 / expected_norm;
        let expected_y = 0.1 / expected_norm;
        assert!((c[0] - expected_x).abs() < 1e-5, "c[0]={} expected={}", c[0], expected_x);
        assert!((c[1] - expected_y).abs() < 1e-5, "c[1]={} expected={}", c[1], expected_y);
    }

    #[test]
    fn test_touch_updates_timestamp() {
        let mut session = SessionContext::new("test");
        session.last_seen_at = 0;
        session.touch();
        assert!(session.last_seen_at > 0);
    }

    #[test]
    fn test_generate_session_id_format() {
        let id = generate_session_id();
        assert_eq!(id.len(), 16, "session ID should be 16 hex chars, got: {}", id);
        // Must be valid hex
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "session ID should be all hex chars, got: {}",
            id
        );
    }

    #[test]
    fn test_generate_session_id_uniqueness() {
        let ids: Vec<String> = (0..100).map(|_| generate_session_id()).collect();
        let mut deduped = ids.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(ids.len(), deduped.len(), "100 generated IDs should all be unique");
    }

    #[test]
    fn test_generate_session_id_timestamp_prefix() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        let id = generate_session_id();
        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        let ts_hex = &id[..8];
        let ts = u32::from_str_radix(ts_hex, 16).expect("first 8 chars should be valid hex u32");
        assert!(
            ts >= before && ts <= after,
            "timestamp prefix {} should be between {} and {}",
            ts, before, after
        );
    }

    // Storage round-trip tests use the real SQLite backend.
    #[cfg(feature = "sqlite")]
    mod storage_tests {
        use super::*;
        use crate::engram::storage::{EngramStorage, Storage};

        fn make_storage() -> EngramStorage {
            let mut s = EngramStorage::in_memory().unwrap();
            s.initialize().unwrap();
            s
        }

        #[test]
        fn test_save_and_load_session() {
            let mut storage = make_storage();
            let mut session = SessionContext::new("round-trip-test");
            session.add_query("hello world", 50);
            session.update_centroid(&[1.0_f32, 0.0_f32, 0.0_f32], 0.1);

            storage.save_session(&session).unwrap();

            let loaded = storage.load_session("round-trip-test").unwrap().unwrap();
            assert_eq!(loaded.session_id, session.session_id);
            assert_eq!(loaded.queries, session.queries);
            assert_eq!(loaded.query_count, session.query_count);
            assert_eq!(loaded.created_at, session.created_at);
            assert_eq!(loaded.last_seen_at, session.last_seen_at);

            let c = loaded.centroid.unwrap();
            let orig = session.centroid.unwrap();
            assert_eq!(c.len(), orig.len());
            for (a, b) in c.iter().zip(orig.iter()) {
                assert!((a - b).abs() < 1e-6, "centroid mismatch: {} vs {}", a, b);
            }
        }

        #[test]
        fn test_load_nonexistent_session_returns_none() {
            let mut storage = make_storage();
            let result = storage.load_session("does-not-exist").unwrap();
            assert!(result.is_none());
        }

        #[test]
        fn test_delete_expired_sessions() {
            let mut storage = make_storage();
            let mut old_session = SessionContext::new("old-session");
            old_session.last_seen_at = 1000; // very old timestamp
            old_session.created_at = 1000;
            storage.save_session(&old_session).unwrap();

            let cutoff = now_secs() - 86400; // 1 day ago
            let deleted = storage.delete_expired_sessions(cutoff).unwrap();
            assert_eq!(deleted, 1);

            let loaded = storage.load_session("old-session").unwrap();
            assert!(loaded.is_none());
        }

        #[test]
        fn test_delete_expired_sessions_preserves_recent() {
            let mut storage = make_storage();
            let recent = SessionContext::new("recent-session");
            storage.save_session(&recent).unwrap();

            let cutoff = now_secs() - 86400; // 1 day ago
            let deleted = storage.delete_expired_sessions(cutoff).unwrap();
            assert_eq!(deleted, 0);

            let loaded = storage.load_session("recent-session").unwrap();
            assert!(loaded.is_some());
        }
    }
}
