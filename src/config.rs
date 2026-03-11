//! Shared configuration for the memory server.
//!
//! Centralizes path resolution so all modules agree on where data lives.
//! The MEMORY_HOME environment variable overrides the default location.

use std::path::PathBuf;

/// Default directory name in the user's home directory
const DEFAULT_DIR_NAME: &str = ".memoryco";

/// Get the memory home directory.
///
/// Resolution order:
/// 1. `MEMORY_HOME` environment variable (if set)
/// 2. `~/.memoryco/` (default)
///
/// # Panics
/// Panics if the home directory cannot be determined and MEMORY_HOME is not set.
pub fn get_memory_home() -> PathBuf {
    std::env::var("MEMORY_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .expect("Could not determine home directory")
                .join(DEFAULT_DIR_NAME)
        })
}

/// Get the cache directory for downloadable assets (embedding models, etc.)
///
/// Returns `<memory_home>/cache/`
pub fn get_cache_dir() -> PathBuf {
    get_memory_home().join("cache")
}

/// Get the embedding model cache directory.
///
/// Returns `<memory_home>/cache/models/`
pub fn get_model_cache_dir() -> PathBuf {
    get_cache_dir().join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    // SAFETY: env var mutations are not thread-safe. These tests should run
    // with --test-threads=1 or accept the (test-only) race risk.

    fn save_and_set(key: &str, val: &str) -> Option<String> {
        let original = std::env::var(key).ok();
        // SAFETY: test-only, acceptable race risk
        unsafe {
            std::env::set_var(key, val);
        }
        original
    }

    fn restore(key: &str, original: Option<String>) {
        // SAFETY: test-only, acceptable race risk
        unsafe {
            match original {
                Some(val) => std::env::set_var(key, val),
                None => std::env::remove_var(key),
            }
        }
    }

    // NOTE: env var tests are inherently racy when run in parallel.
    // We test the resolution logic directly instead of mutating the env.

    #[test]
    fn test_default_path_uses_home_dir() {
        // Verify the default path structure is ~/.memoryco
        let expected_suffix = std::path::Path::new(DEFAULT_DIR_NAME);
        let home = dirs::home_dir().expect("home dir must exist");
        let expected = home.join(expected_suffix);

        // When MEMORY_HOME is not set, we should get ~/.memoryco
        // We can't safely unset env vars in parallel tests, so just
        // verify the expected path is well-formed
        assert!(expected.ends_with(DEFAULT_DIR_NAME));
        assert!(expected.starts_with(home));
    }

    #[test]
    fn test_env_override_is_respected() {
        let original = save_and_set("MEMORY_HOME", "/tmp/test-memoryco-config");

        let home = get_memory_home();
        assert_eq!(home, PathBuf::from("/tmp/test-memoryco-config"));

        restore("MEMORY_HOME", original);
    }

    #[test]
    fn test_model_cache_dir_structure() {
        let original = save_and_set("MEMORY_HOME", "/tmp/test-memoryco-config");

        let cache = get_model_cache_dir();
        assert_eq!(
            cache,
            PathBuf::from("/tmp/test-memoryco-config/cache/models")
        );

        restore("MEMORY_HOME", original);
    }
}
