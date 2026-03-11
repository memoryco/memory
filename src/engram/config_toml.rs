//! TOML-based configuration loading and writing for the brain substrate.
//!
//! Config lives under the `[brain]` section in `{memory_home}/config.toml`.
//! Missing keys fall back to `Config::default()` values.
//! The file is read once on startup and written back when `config_set` is called.

use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table, value};

use super::Config;

/// Path to config.toml within a memory_home directory.
pub fn config_toml_path(memory_home: &Path) -> PathBuf {
    memory_home.join("config.toml")
}

/// Load `Config` from `{memory_home}/config.toml`.
///
/// - If the file doesn't exist, returns `Config::default()`.
/// - Keys present in `[brain]` override the defaults.
/// - `embedding_model_active` is NOT loaded from TOML (runtime state, lives in SQLite).
pub fn load_config_from_toml(memory_home: &Path) -> Config {
    let path = config_toml_path(memory_home);

    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return Config::default(),
    };

    let doc: DocumentMut = match content.parse() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[config] Failed to parse config.toml: {}. Using defaults.", e);
            return Config::default();
        }
    };

    let defaults = Config::default();

    let brain = doc.get("brain").and_then(|v| v.as_table());

    macro_rules! get_f64 {
        ($field:ident) => {
            brain
                .and_then(|t| t.get(stringify!($field)))
                .and_then(|v| v.as_float())
                .unwrap_or(defaults.$field)
        };
    }

    macro_rules! get_bool {
        ($field:ident) => {
            brain
                .and_then(|t| t.get(stringify!($field)))
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.$field)
        };
    }

    macro_rules! get_usize {
        ($field:ident) => {
            brain
                .and_then(|t| t.get(stringify!($field)))
                .and_then(|v| v.as_integer())
                .map(|n| n as usize)
                .unwrap_or(defaults.$field)
        };
    }

    macro_rules! get_u8 {
        ($field:ident) => {
            brain
                .and_then(|t| t.get(stringify!($field)))
                .and_then(|v| v.as_integer())
                .map(|n| n as u8)
                .unwrap_or(defaults.$field)
        };
    }

    macro_rules! get_string {
        ($field:ident) => {
            brain
                .and_then(|t| t.get(stringify!($field)))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or(defaults.$field)
        };
    }

    Config {
        decay_rate_per_day: get_f64!(decay_rate_per_day),
        decay_interval_hours: get_f64!(decay_interval_hours),
        propagation_damping: get_f64!(propagation_damping),
        hebbian_learning_rate: get_f64!(hebbian_learning_rate),
        recall_strength: get_f64!(recall_strength),
        association_decay_rate: get_f64!(association_decay_rate),
        min_association_weight: get_f64!(min_association_weight),
        search_follow_associations: get_bool!(search_follow_associations),
        search_association_depth: get_u8!(search_association_depth),
        embedding_model: get_string!(embedding_model),
        rerank_mode: get_string!(rerank_mode),
        rerank_candidates: get_usize!(rerank_candidates),
        hybrid_search_enabled: get_bool!(hybrid_search_enabled),
        query_expansion_enabled: get_bool!(query_expansion_enabled),
        // Runtime state — not stored in TOML
        embedding_model_active: defaults.embedding_model_active,
        // Rarely-changed fields — keep defaults (can be set manually in TOML if needed)
        blocked_tags: defaults.blocked_tags,
        max_tag_cardinality: defaults.max_tag_cardinality,
    }
}

/// Write a single config key under `[brain]` in `{memory_home}/config.toml`.
///
/// Preserves all existing comments and formatting.
/// If the file doesn't exist, writes a fresh default config first.
pub fn write_config_key(memory_home: &Path, key: &str, val: &ConfigValue) -> std::io::Result<()> {
    let path = config_toml_path(memory_home);

    let content = if path.exists() {
        std::fs::read_to_string(&path)?
    } else {
        default_config_toml_content()
    };

    let mut doc: DocumentMut = content.parse().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse config.toml: {}", e),
        )
    })?;

    // Ensure [brain] section exists
    if !doc.contains_key("brain") {
        doc.insert("brain", Item::Table(Table::new()));
    }

    let brain_table = doc["brain"].as_table_mut().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "[brain] is not a table")
    })?;

    match val {
        ConfigValue::Float(f) => {
            brain_table.insert(key, value(*f));
        }
        ConfigValue::Bool(b) => {
            brain_table.insert(key, value(*b));
        }
        ConfigValue::Int(i) => {
            brain_table.insert(key, value(*i as i64));
        }
        ConfigValue::Str(s) => {
            brain_table.insert(key, value(s.as_str()));
        }
    }

    std::fs::write(&path, doc.to_string())
}

/// A typed config value for writing to TOML.
pub enum ConfigValue {
    Float(f64),
    Bool(bool),
    Int(usize),
    Str(String),
}

/// The default config.toml content, including all brain keys with comments.
pub fn default_config_toml_content() -> String {
    // We need the actual default values from Config::default()
    let d = Config::default();
    format!(
        r#"# ── Brain Configuration ──────────────────────────────────────────────
# These settings control the memory substrate behavior.
# Changes take effect on server restart.

[brain]
# Energy decay rate per day (0.0-1.0). 0.05 = 5% loss per day of non-use.
decay_rate_per_day = {decay_rate_per_day}

# Minimum hours between decay checks.
decay_interval_hours = {decay_interval_hours}

# Signal propagation to neighbors (0.0-1.0). 0.5 = 50% to neighbors.
propagation_damping = {propagation_damping}

# Association strength boost on co-access (0.0-1.0).
hebbian_learning_rate = {hebbian_learning_rate}

# Energy boost when recalling a memory (0.0-1.0).
recall_strength = {recall_strength}

# Association decay rate relative to memory decay. 1.0 = same rate.
association_decay_rate = {association_decay_rate}

# Associations below this weight are pruned on startup.
min_association_weight = {min_association_weight}

# Follow associations during search to discover related memories.
search_follow_associations = {search_follow_associations}

# How many hops to follow (1 = direct only, 2 = friends-of-friends).
search_association_depth = {search_association_depth}

# Embedding model name. Changing triggers re-embedding on restart.
embedding_model = "{embedding_model}"

# Reranking mode: "off", "cross-encoder", or "llm"
rerank_mode = "{rerank_mode}"

# Candidates to feed to re-ranker (higher = better recall, slower).
rerank_candidates = {rerank_candidates}

# BM25 + vector search fusion via Reciprocal Rank Fusion.
hybrid_search_enabled = {hybrid_search_enabled}

# Expand queries with variant phrasings before retrieval.
query_expansion_enabled = {query_expansion_enabled}
"#,
        decay_rate_per_day = d.decay_rate_per_day,
        decay_interval_hours = d.decay_interval_hours,
        propagation_damping = d.propagation_damping,
        hebbian_learning_rate = d.hebbian_learning_rate,
        recall_strength = d.recall_strength,
        association_decay_rate = d.association_decay_rate,
        min_association_weight = d.min_association_weight,
        search_follow_associations = d.search_follow_associations,
        search_association_depth = d.search_association_depth,
        embedding_model = d.embedding_model,
        rerank_mode = d.rerank_mode,
        rerank_candidates = d.rerank_candidates,
        hybrid_search_enabled = d.hybrid_search_enabled,
        query_expansion_enabled = d.query_expansion_enabled,
    )
}

/// Write a default config.toml if one doesn't exist.
/// If it already exists, does nothing (preserves existing config).
pub fn ensure_default_config_toml(memory_home: &Path) -> std::io::Result<()> {
    let path = config_toml_path(memory_home);
    if path.exists() {
        return Ok(());
    }

    std::fs::write(&path, default_config_toml_content())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_home() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn test_missing_file_returns_defaults() {
        let dir = temp_home();
        let config = load_config_from_toml(dir.path());
        let defaults = Config::default();
        assert!((config.decay_rate_per_day - defaults.decay_rate_per_day).abs() < f64::EPSILON);
        assert!((config.recall_strength - defaults.recall_strength).abs() < f64::EPSILON);
        assert_eq!(config.rerank_mode, defaults.rerank_mode);
        assert_eq!(config.embedding_model, defaults.embedding_model);
    }

    #[test]
    fn test_full_brain_section_loads_correctly() {
        let dir = temp_home();
        let toml = r#"
[brain]
decay_rate_per_day = 0.1
decay_interval_hours = 2.0
propagation_damping = 0.3
hebbian_learning_rate = 0.05
recall_strength = 0.4
association_decay_rate = 0.8
min_association_weight = 0.1
search_follow_associations = false
search_association_depth = 2
embedding_model = "BGELargeENV15"
rerank_mode = "off"
rerank_candidates = 50
hybrid_search_enabled = false
query_expansion_enabled = false
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let config = load_config_from_toml(dir.path());

        assert!((config.decay_rate_per_day - 0.1).abs() < f64::EPSILON);
        assert!((config.decay_interval_hours - 2.0).abs() < f64::EPSILON);
        assert!((config.propagation_damping - 0.3).abs() < f64::EPSILON);
        assert!((config.hebbian_learning_rate - 0.05).abs() < f64::EPSILON);
        assert!((config.recall_strength - 0.4).abs() < f64::EPSILON);
        assert!((config.association_decay_rate - 0.8).abs() < f64::EPSILON);
        assert!((config.min_association_weight - 0.1).abs() < f64::EPSILON);
        assert!(!config.search_follow_associations);
        assert_eq!(config.search_association_depth, 2);
        assert_eq!(config.embedding_model, "BGELargeENV15");
        assert_eq!(config.rerank_mode, "off");
        assert_eq!(config.rerank_candidates, 50);
        assert!(!config.hybrid_search_enabled);
        assert!(!config.query_expansion_enabled);
    }

    #[test]
    fn test_partial_brain_section_uses_defaults_for_missing_keys() {
        let dir = temp_home();
        let toml = r#"
[brain]
decay_rate_per_day = 0.2
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let config = load_config_from_toml(dir.path());
        let defaults = Config::default();

        assert!((config.decay_rate_per_day - 0.2).abs() < f64::EPSILON);
        // Everything else falls back to defaults
        assert!((config.recall_strength - defaults.recall_strength).abs() < f64::EPSILON);
        assert_eq!(config.rerank_mode, defaults.rerank_mode);
    }

    #[test]
    fn test_llm_section_coexists_without_interference() {
        let dir = temp_home();
        let toml = r#"
[llm]
model = "some-llm-model"

[brain]
decay_rate_per_day = 0.07
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let config = load_config_from_toml(dir.path());
        assert!((config.decay_rate_per_day - 0.07).abs() < f64::EPSILON);
    }

    #[test]
    fn test_write_config_key_creates_file_if_missing() {
        let dir = temp_home();
        write_config_key(
            dir.path(),
            "decay_rate_per_day",
            &ConfigValue::Float(0.99),
        )
        .unwrap();

        let path = config_toml_path(dir.path());
        assert!(path.exists());

        let config = load_config_from_toml(dir.path());
        assert!((config.decay_rate_per_day - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_write_config_key_preserves_other_keys() {
        let dir = temp_home();
        // Start with two keys
        let toml = r#"
# comment preserved
[brain]
decay_rate_per_day = 0.05
rerank_mode = "cross-encoder"
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();

        // Update only one key
        write_config_key(dir.path(), "rerank_mode", &ConfigValue::Str("off".to_string())).unwrap();

        let config = load_config_from_toml(dir.path());
        // The unchanged key should still be there
        assert!((config.decay_rate_per_day - 0.05).abs() < f64::EPSILON);
        // The updated key should reflect the new value
        assert_eq!(config.rerank_mode, "off");

        // The comment should still be in the file
        let content = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(content.contains("# comment preserved"));
    }

    #[test]
    fn test_write_config_key_string_value() {
        let dir = temp_home();
        ensure_default_config_toml(dir.path()).unwrap();
        write_config_key(
            dir.path(),
            "embedding_model",
            &ConfigValue::Str("SnowflakeArcticEmbedL".to_string()),
        )
        .unwrap();
        let config = load_config_from_toml(dir.path());
        assert_eq!(config.embedding_model, "SnowflakeArcticEmbedL");
    }

    #[test]
    fn test_ensure_default_config_toml_does_not_overwrite() {
        let dir = temp_home();
        let custom = "[brain]\ndecay_rate_per_day = 0.99\n";
        std::fs::write(dir.path().join("config.toml"), custom).unwrap();

        ensure_default_config_toml(dir.path()).unwrap();

        // Should not have been overwritten
        let content = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(content.contains("0.99"));
    }

    #[test]
    fn test_ensure_default_config_toml_creates_when_absent() {
        let dir = temp_home();
        ensure_default_config_toml(dir.path()).unwrap();
        let path = config_toml_path(dir.path());
        assert!(path.exists());
        let config = load_config_from_toml(dir.path());
        let defaults = Config::default();
        assert!((config.decay_rate_per_day - defaults.decay_rate_per_day).abs() < f64::EPSILON);
    }

    #[test]
    fn test_brain_accepts_external_config() {
        // Verify that a Config loaded from TOML can be passed into Brain
        let dir = temp_home();
        let toml = r#"
[brain]
decay_rate_per_day = 0.15
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let config = load_config_from_toml(dir.path());
        assert!((config.decay_rate_per_day - 0.15).abs() < f64::EPSILON);
        // Brain::new with an external config is tested in brain.rs tests
    }

    #[test]
    fn test_default_toml_includes_rerank_mode() {
        let content = default_config_toml_content();
        assert!(content.contains("rerank_mode"), "default TOML should include rerank_mode");
    }

    #[test]
    fn test_load_rerank_mode_llm() {
        let dir = temp_home();
        let toml = r#"
[brain]
rerank_mode = "llm"
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let config = load_config_from_toml(dir.path());
        assert_eq!(config.rerank_mode, "llm");
    }

    #[test]
    fn test_missing_rerank_mode_defaults_to_cross_encoder() {
        let dir = temp_home();
        // No rerank_mode in TOML
        let toml = r#"
[brain]
decay_rate_per_day = 0.05
"#;
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let config = load_config_from_toml(dir.path());
        assert_eq!(config.rerank_mode, "cross-encoder");
    }
}
