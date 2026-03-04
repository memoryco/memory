//! Language-specific processing for MemoryCo.
//!
//! This module provides trait-based language dispatch for operations that
//! require natural language understanding — temporal expression resolution,
//! number word parsing, etc.
//!
//! # Architecture
//!
//! Each supported language gets its own submodule (e.g. `english.rs`) that
//! implements the [`LanguageSupport`] trait. The [`detect`] function identifies
//! the language of input text, and [`resolver_for`] returns the appropriate
//! implementation.
//!
//! # Known Limitations
//!
//! Only English is currently implemented. Unrecognized languages fall through
//! to the English resolver rather than failing, because the LLM interface
//! layer almost always normalizes input to English before calling tools.
//! The English resolver's own error path handles unrecognized patterns.
//!
//! To add a new language:
//! 1. Create `src/lang/<language>.rs`
//! 2. Implement `LanguageSupport` for your language
//! 3. Register it in `resolver_for()` below

pub mod english;

use chrono::NaiveDate;

/// The result of resolving a temporal expression.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalResult {
    /// A single resolved date with an optional label (e.g. day name).
    Date(NaiveDate, Option<String>),
    /// A date range (start, end) with an optional label.
    Range(NaiveDate, NaiveDate, Option<String>),
}

/// Language detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    // Future: Spanish, French, Japanese, Portuguese, German, etc.
    Unknown,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::English => write!(f, "en"),
            Language::Unknown => write!(f, "unknown"),
        }
    }
}

/// Trait defining language-specific operations.
///
/// Each supported language implements this trait. The trait is intentionally
/// narrow — only operations that genuinely vary by language belong here.
/// Language-agnostic operations (ISO date parsing, arithmetic) stay in
/// shared code.
pub trait LanguageSupport: Send + Sync {
    /// Which language this implementation covers.
    fn language(&self) -> Language;

    /// Resolve a relative temporal expression against a reference date.
    ///
    /// Examples (English):
    /// - "last Sunday" + 2023-05-25 → 2023-05-21
    /// - "two weekends before 17 July 2023" → 2023-07-08 to 2023-07-09
    /// - "three days ago" + 2023-05-25 → 2023-05-22
    fn resolve_temporal(
        &self,
        expression: &str,
        reference: NaiveDate,
    ) -> Result<TemporalResult, String>;

    /// Parse a number from a word in this language.
    ///
    /// Examples (English): "one" → 1, "twelve" → 12
    /// Returns None if the word isn't a recognized number.
    fn parse_number_word(&self, word: &str) -> Option<u32>;

    /// Parse month name or abbreviation in this language.
    ///
    /// Examples (English): "january" → 1, "sep" → 9
    fn parse_month_name(&self, word: &str) -> Option<u32>;

    /// Parse a weekday name or abbreviation in this language.
    ///
    /// Examples (English): "monday" → Weekday::Mon, "thu" → Weekday::Thu
    fn parse_weekday_name(&self, word: &str) -> Option<chrono::Weekday>;

    /// Parse an inline date from natural language text.
    ///
    /// Examples (English): "17 July 2023", "July 17, 2023", "1st September 2023"
    /// ISO dates (2023-07-17) are handled by shared code and don't need this.
    fn parse_natural_date(&self, text: &str) -> Option<NaiveDate>;
}

/// Detect the language of a text string.
///
/// Uses the `whatlang` crate for statistical n-gram detection.
/// Falls back to `Language::Unknown` for short or ambiguous input.
pub fn detect(text: &str) -> Language {
    // Short-circuit: very short text can't be reliably detected.
    // Temporal expressions are typically 2-5 words ("3 days ago",
    // "the weekend before 20 October 2023") — way too short for
    // n-gram language detection. Default to English since it's our
    // only implementation anyway.
    if text.split_whitespace().count() < 8 {
        return Language::English;
    }

    match whatlang::detect_lang(text) {
        Some(whatlang::Lang::Eng) => Language::English,
        Some(_) => Language::Unknown,
        None => Language::English, // ambiguous → default to English
    }
}

/// Get the language resolver for detected (or specified) language.
///
/// Returns the English resolver for all languages. When we don't have a
/// native resolver for the detected language, we fall through to English
/// because the LLM interface layer almost always normalizes input to
/// English before calling tools. The English resolver has its own error
/// path for unrecognized patterns — that's the right place to reject
/// garbage, not the language gate.
pub fn resolver_for(lang: Language) -> &'static dyn LanguageSupport {
    match lang {
        Language::English => &english::English,
        // Fall through to English rather than failing closed.
        // See module docs for rationale.
        Language::Unknown => &english::English,
    }
}

/// Convenience: detect language and return the appropriate resolver.
pub fn resolver_for_text(text: &str) -> &'static dyn LanguageSupport {
    resolver_for(detect(text))
}

// NOTE: FallbackResolver was removed intentionally. Unknown languages fall
// through to the English resolver because:
// 1. The LLM interface layer normalizes input to English before calling tools
// 2. Short phrases ("last Sunday") don't have enough signal for detection
// 3. The English resolver already returns Err for unrecognized patterns
// 4. Failing open through English is better than failing closed on detection
//
// When we add a second language, the dispatch in resolver_for() gains a new
// arm and Unknown continues to fall through to English as the default.
