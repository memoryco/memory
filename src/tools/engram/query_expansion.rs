//! Query expansion for search — generates variant queries from the original
//! to improve recall by searching for stop-word-stripped and individual-term variants.

/// English stop words for conversational queries
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "what", "when", "where",
    "which", "who", "whom", "whose", "why", "how", "that", "this",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "not", "no", "and", "but", "or", "so", "yet",
    "for", "from", "if", "in", "into", "of", "on", "to", "up", "with",
    "at", "by", "about", "after", "before", "between", "during", "than",
    "too", "very", "just", "also", "am", "out", "over", "own", "same",
    "still", "then", "under", "until", "again", "once", "only",
];

/// Maximum number of query variants to generate (including original).
const MAX_VARIANTS: usize = 5;

/// Clean punctuation/possessive noise from query tokens while preserving case.
fn clean_token(raw: &str) -> Option<String> {
    let mut t = raw
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’' && c != '`')
        .replace('’', "'")
        .replace('`', "");

    if t.ends_with("'s") || t.ends_with("'S") {
        let new_len = t.len().saturating_sub(2);
        t.truncate(new_len);
    }

    let trimmed = t.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Generate query variants for expansion (Level 1 only).
/// Returns the original query plus a stop-word-stripped variant.
/// The original query is ALWAYS first in the returned vec.
pub fn expand_query(query: &str) -> Vec<String> {
    let mut variants = vec![query.to_string()];

    let words: Vec<&str> = query.split_whitespace().collect();
    if words.len() <= 1 {
        return variants; // Single word, nothing to expand
    }

    // Level 1: Stop words removed
    let significant: Vec<String> = words
        .iter()
        .filter_map(|w| clean_token(w))
        .filter(|cleaned| {
            let lower = cleaned.to_lowercase();
            !STOP_WORDS.contains(&lower.as_str())
        })
        .collect();

    if !significant.is_empty() && significant.len() < words.len() {
        let stripped = significant.join(" ");
        if stripped != query && !stripped.is_empty() {
            variants.push(stripped);
        }
    }

    variants
}

/// Generate fallback single-term queries (Level 2).
/// Only call this when primary search returned too few results.
/// Returns individual significant terms (names and 4+ char words).
pub fn fallback_terms(query: &str) -> Vec<String> {
    let words: Vec<&str> = query.split_whitespace().collect();

    let significant: Vec<String> = words
        .iter()
        .filter_map(|w| clean_token(w))
        .filter(|cleaned| {
            let lower = cleaned.to_lowercase();
            !STOP_WORDS.contains(&lower.as_str())
        })
        .collect();

    if significant.len() < 2 {
        return Vec::new(); // Not enough terms to split
    }

    let mut terms = Vec::new();
    for term in &significant {
        if terms.len() >= MAX_VARIANTS {
            break;
        }
        let first_char = term.chars().next().unwrap_or('a');
        if first_char.is_uppercase() || term.len() >= 4 {
            let t_str = term.to_string();
            if !terms.contains(&t_str) {
                terms.push(t_str);
            }
        }
    }

    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_expansion_for_single_word() {
        let result = expand_query("Caroline");
        assert_eq!(result, vec!["Caroline"]);
    }

    #[test]
    fn stop_words_stripped() {
        let result = expand_query("What did Caroline research");
        assert!(result[0] == "What did Caroline research", "Original must be first");
        assert!(result.contains(&"Caroline research".to_string()),
            "Stop-word-stripped variant should be present: {:?}", result);
        // Level 2 terms should NOT be in expand_query anymore
        assert!(!result.contains(&"Caroline".to_string()),
            "Individual terms should not be in primary expansion: {:?}", result);
    }

    #[test]
    fn fallback_terms_extracted() {
        let result = fallback_terms("What did Caroline research");
        assert!(result.contains(&"Caroline".to_string()),
            "Capitalized name should be extracted: {:?}", result);
        assert!(result.contains(&"research".to_string()),
            "4+ char term should be extracted: {:?}", result);
    }

    #[test]
    fn fallback_terms_short_words_skipped() {
        // "go" and "run" are 2-3 chars and lowercase — should not be extracted
        let result = fallback_terms("go run far");
        assert!(result.is_empty(),
            "Short generic terms should not produce fallback terms: {:?}", result);
    }

    #[test]
    fn fallback_terms_single_word_returns_empty() {
        let result = fallback_terms("Caroline");
        assert!(result.is_empty(), "Single word should not produce fallback terms");
    }

    #[test]
    fn original_always_first() {
        let queries = vec![
            "What did Caroline research",
            "hello world",
            "test",
            "Where is Melanie going camping",
        ];
        for q in queries {
            let result = expand_query(q);
            assert_eq!(result[0], q, "Original query must always be first for '{}'", q);
        }
    }

    #[test]
    fn empty_query_handled() {
        let result = expand_query("");
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn already_clean_query_no_level2() {
        // A query with no stop words should only return the original (no Level 1 variant)
        let result = expand_query("Caroline research");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Caroline research");
    }

    #[test]
    fn max_fallback_terms_capped() {
        let result = fallback_terms("Caroline Melanie Brandon Portland Oregon camping hiking");
        assert!(result.len() <= MAX_VARIANTS,
            "Should not exceed {} fallback terms, got {}: {:?}", MAX_VARIANTS, result.len(), result);
    }

    #[test]
    fn complex_conversational_query() {
        let result = expand_query("When is Melanie planning on going camping");
        assert_eq!(result[0], "When is Melanie planning on going camping");
        assert!(result.contains(&"Melanie planning going camping".to_string()),
            "Stop-word-stripped variant should be present: {:?}", result);
        // Level 2 terms should NOT be here
        assert_eq!(result.len(), 2, "Should only have original + stripped: {:?}", result);
    }

    #[test]
    fn possessive_tokens_are_cleaned() {
        let result = expand_query("What is Caroline's relationship status?");
        assert!(result.contains(&"Caroline relationship status".to_string()),
            "Possessive should be cleaned in stripped variant: {:?}", result);

        let fallback = fallback_terms("What is Caroline's relationship status?");
        assert!(fallback.contains(&"Caroline".to_string()));
        assert!(fallback.contains(&"relationship".to_string()));
    }
}
