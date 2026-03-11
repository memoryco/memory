//! English language support for MemoryCo.
//!
//! # Module structure
//!
//! - `temporal` — Relative date/time expression resolution
//! - `parsing` — Number words, month/weekday names, natural date formats
//!
//! To add a new capability, create a new file in this directory and
//! wire it into the `LanguageSupport` impl below.

#[cfg(test)]
mod bench_cases;
pub(crate) mod parsing;
pub(crate) mod temporal;

use super::{Language, LanguageSupport, TemporalResult};
use chrono::{NaiveDate, Weekday};

/// English language resolver. Stateless unit struct.
pub struct English;

impl LanguageSupport for English {
    fn language(&self) -> Language {
        Language::English
    }

    fn resolve_temporal(
        &self,
        expression: &str,
        reference: NaiveDate,
    ) -> Result<TemporalResult, String> {
        temporal::resolve(expression, reference)
    }

    fn parse_number_word(&self, word: &str) -> Option<u32> {
        parsing::number_word(word)
    }

    fn parse_month_name(&self, word: &str) -> Option<u32> {
        parsing::month_name(word)
    }

    fn parse_weekday_name(&self, word: &str) -> Option<Weekday> {
        parsing::weekday_name(word)
    }

    fn parse_natural_date(&self, text: &str) -> Option<NaiveDate> {
        parsing::natural_date(text)
    }
}
