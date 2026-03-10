//! date_resolve - Resolve relative time expressions to absolute dates
//!
//! This is a thin tool wrapper that delegates to the language-specific
//! temporal resolvers in `crate::lang`.

use chrono::NaiveDate;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::lang::{self, TemporalResult};
use crate::tools::text_response;

pub struct DateResolveTool;

#[derive(Deserialize)]
struct Args {
    expression: String,
    reference_date: String,
}

impl Tool<Context> for DateResolveTool {
    fn name(&self) -> &str {
        "date_resolve"
    }

    fn description(&self) -> &str {
        "Resolve a relative time expression to an absolute date. ALWAYS call this tool for \
         ANY date computation — never calculate dates yourself. LLMs are notoriously bad at \
         weekday arithmetic and off-by-one errors. Supports: 'last Sunday', \
         'the sunday before 2023-05-25', 'two weeks ago', 'next month', etc. Pass the \
         expression and the memory's created_at as reference_date. For 'the X before/after Y' \
         patterns, pass the full expression."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The relative time expression to resolve. Examples: \
                                    'last Sunday', 'next month', 'two weeks ago', 'yesterday', \
                                    'the week before'"
                },
                "reference_date": {
                    "type": "string",
                    "description": "The date to resolve relative to, in ISO 8601 format \
                                    (YYYY-MM-DD or full datetime). Typically the memory's \
                                    created_at timestamp."
                }
            },
            "required": ["expression", "reference_date"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        _context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let reference =
            parse_reference_date(&args.reference_date).map_err(|e| McpError::InvalidParams(e))?;

        // Detect language and dispatch to the appropriate resolver
        let resolver = lang::resolver_for_text(&args.expression);
        match resolver.resolve_temporal(&args.expression, reference) {
            Ok(result) => Ok(text_response(format_result(&result))),
            Err(e) => Ok(text_response(e)),
        }
    }
}

// =============================================================================
// Date parsing (language-agnostic)
// =============================================================================

/// Parse a reference date from ISO 8601 format.
/// Accepts "YYYY-MM-DD" or full datetime like "2023-05-25T14:30:00Z".
fn parse_reference_date(input: &str) -> Result<NaiveDate, String> {
    let trimmed = input.trim();

    // Try YYYY-MM-DD first
    if let Ok(date) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return Ok(date);
    }

    // Try full ISO 8601 datetime — extract just the date portion
    if trimmed.len() >= 10 {
        let date_part = &trimmed[..10];
        if let Ok(date) = NaiveDate::parse_from_str(date_part, "%Y-%m-%d") {
            return Ok(date);
        }
    }

    Err(format!(
        "Could not parse reference_date '{}'. Expected YYYY-MM-DD or ISO 8601 datetime.",
        input
    ))
}

// =============================================================================
// Formatting
// =============================================================================

/// Format a TemporalResult as a human-readable string.
fn format_result(result: &TemporalResult) -> String {
    match result {
        TemporalResult::Date(date, label) => {
            let formatted = date.format("%Y-%m-%d").to_string();
            match label {
                Some(l) => format!("{} ({})", formatted, l),
                None => formatted,
            }
        }
        TemporalResult::Range(start, end, label) => {
            let s = start.format("%Y-%m-%d").to_string();
            let e = end.format("%Y-%m-%d").to_string();
            match label {
                Some(l) => format!("{} to {} ({})", s, e, l),
                None => format!("{} to {}", s, e),
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    /// Helper: resolve an expression via the lang module (same path as the tool).
    fn resolve(expression: &str, reference: NaiveDate) -> Result<TemporalResult, String> {
        let resolver = lang::resolver_for_text(expression);
        resolver.resolve_temporal(expression, reference)
    }

    // --- Date parsing ---

    #[test]
    fn parse_iso_date() {
        assert_eq!(
            parse_reference_date("2023-05-25").unwrap(),
            date(2023, 5, 25)
        );
    }

    #[test]
    fn parse_iso_datetime() {
        assert_eq!(
            parse_reference_date("2023-05-25T14:30:00Z").unwrap(),
            date(2023, 5, 25)
        );
    }

    #[test]
    fn parse_iso_datetime_with_offset() {
        assert_eq!(
            parse_reference_date("2023-05-25T14:30:00+05:00").unwrap(),
            date(2023, 5, 25)
        );
    }

    #[test]
    fn parse_invalid_date() {
        assert!(parse_reference_date("not-a-date").is_err());
        assert!(parse_reference_date("").is_err());
    }

    // --- Assertion helpers ---

    fn assert_single_date(result: &TemporalResult, expected: NaiveDate) {
        match result {
            TemporalResult::Date(d, _) => {
                assert_eq!(*d, expected, "Expected {}, got {}", expected, d)
            }
            TemporalResult::Range(s, e, _) => {
                panic!(
                    "Expected single date {}, got range {} to {}",
                    expected, s, e
                )
            }
        }
    }

    fn assert_date_range(
        result: &TemporalResult,
        expected_start: NaiveDate,
        expected_end: NaiveDate,
    ) {
        match result {
            TemporalResult::Range(s, e, _) => {
                assert_eq!(
                    *s, expected_start,
                    "Range start: expected {}, got {}",
                    expected_start, s
                );
                assert_eq!(
                    *e, expected_end,
                    "Range end: expected {}, got {}",
                    expected_end, e
                );
            }
            TemporalResult::Date(d, _) => {
                panic!(
                    "Expected range {} to {}, got single date {}",
                    expected_start, expected_end, d
                )
            }
        }
    }

    // --- Relative days (integration: tool layer → lang module) ---

    #[test]
    fn yesterday() {
        let result = resolve("yesterday", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn today() {
        let result = resolve("today", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 25));
    }

    #[test]
    fn tomorrow() {
        let result = resolve("tomorrow", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn three_days_ago() {
        let result = resolve("3 days ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn three_days_ago_word() {
        let result = resolve("three days ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn in_five_days() {
        let result = resolve("in 5 days", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 30));
    }

    #[test]
    fn a_day_ago() {
        let result = resolve("a day ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    // --- Relative weekdays ---

    #[test]
    fn last_sunday_from_thursday() {
        let result = resolve("last sunday", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn last_monday_from_thursday() {
        let result = resolve("last monday", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn last_sunday_from_sunday() {
        let result = resolve("last sunday", date(2023, 5, 21)).unwrap();
        assert_single_date(&result, date(2023, 5, 14));
    }

    #[test]
    fn next_friday_from_monday() {
        let result = resolve("next friday", date(2023, 5, 22)).unwrap();
        assert_single_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn this_wednesday_from_monday() {
        let result = resolve("this wednesday", date(2023, 5, 22)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn this_monday_from_wednesday() {
        let result = resolve("this monday", date(2023, 5, 24)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    // --- Relative weeks ---

    #[test]
    fn last_week() {
        let result = resolve("last week", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 15), date(2023, 5, 21));
    }

    #[test]
    fn next_week() {
        let result = resolve("next week", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 29), date(2023, 6, 4));
    }

    #[test]
    fn this_week() {
        let result = resolve("this week", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 22), date(2023, 5, 28));
    }

    #[test]
    fn two_weeks_ago() {
        let result = resolve("2 weeks ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 11));
    }

    // --- Relative months ---

    #[test]
    fn last_month() {
        let result = resolve("last month", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 4, 1), date(2023, 4, 30));
    }

    #[test]
    fn next_month_from_may() {
        let result = resolve("next month", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 6, 1), date(2023, 6, 30));
    }

    #[test]
    fn this_month() {
        let result = resolve("this month", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 1), date(2023, 5, 31));
    }

    #[test]
    fn two_months_ago() {
        let result = resolve("2 months ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 3, 25));
    }

    // --- Relative years ---

    #[test]
    fn last_year() {
        let result = resolve("last year", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2022, 1, 1), date(2022, 12, 31));
    }

    #[test]
    fn next_year() {
        let result = resolve("next year", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2024, 1, 1), date(2024, 12, 31));
    }

    #[test]
    fn two_years_ago() {
        let result = resolve("2 years ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2021, 5, 25));
    }

    // --- Combined / the X before/after ---

    #[test]
    fn the_sunday_before_may_25() {
        let result = resolve("the sunday before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn the_friday_after_may_25() {
        let result = resolve("the friday after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn the_week_before_june_9() {
        let result = resolve("the week before 2023-06-09", date(2023, 6, 9)).unwrap();
        assert_date_range(&result, date(2023, 6, 2), date(2023, 6, 8));
    }

    #[test]
    fn the_week_after_may_25() {
        let result = resolve("the week after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 26), date(2023, 6, 1));
    }

    #[test]
    fn the_month_before_may() {
        let result = resolve("the month before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 4, 1), date(2023, 4, 30));
    }

    #[test]
    fn the_month_after_may() {
        let result = resolve("the month after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 6, 1), date(2023, 6, 30));
    }

    #[test]
    fn the_sunday_before_natural_language_date() {
        let result = resolve("the sunday before 3 July 2023", date(2023, 7, 10)).unwrap();
        assert_single_date(&result, date(2023, 7, 2));
    }

    #[test]
    fn weekend_before_natural_language_date() {
        let result = resolve("the weekend before 20 October 2023", date(2023, 10, 25)).unwrap();
        assert_date_range(&result, date(2023, 10, 14), date(2023, 10, 15));
    }

    #[test]
    fn two_weekends_before_anchor() {
        let result = resolve("two weekends before 17 July 2023", date(2023, 7, 20)).unwrap();
        assert_date_range(&result, date(2023, 7, 8), date(2023, 7, 9));
    }

    #[test]
    fn first_weekend_of_month() {
        let result = resolve("first weekend of August 2023", date(2023, 8, 10)).unwrap();
        assert_date_range(&result, date(2023, 8, 5), date(2023, 8, 6));
    }

    #[test]
    fn season_of_year_summer() {
        let result = resolve("summer of 2022", date(2023, 8, 10)).unwrap();
        assert_date_range(&result, date(2022, 6, 1), date(2022, 8, 31));
    }

    #[test]
    fn the_week_before_no_date() {
        let result = resolve("the week before", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 18), date(2023, 5, 24));
    }

    // --- Edge cases ---

    #[test]
    fn case_insensitive() {
        let result = resolve("LAST SUNDAY", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn abbreviations() {
        let result = resolve("last sun", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn abbreviation_mon() {
        let result = resolve("next mon", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 29));
    }

    #[test]
    fn whitespace_trimmed() {
        let result = resolve("  yesterday  ", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn unrecognized_expression() {
        let result = resolve("when pigs fly", date(2023, 5, 25));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Could not resolve"));
        assert!(err.contains("Supported patterns"));
    }

    // --- Formatting ---

    #[test]
    fn format_single_date_with_label() {
        let result = TemporalResult::Date(date(2023, 5, 21), Some("Sunday".to_string()));
        assert_eq!(format_result(&result), "2023-05-21 (Sunday)");
    }

    #[test]
    fn format_range_with_label() {
        let result = TemporalResult::Range(
            date(2023, 6, 1),
            date(2023, 6, 30),
            Some("June 2023".to_string()),
        );
        assert_eq!(
            format_result(&result),
            "2023-06-01 to 2023-06-30 (June 2023)"
        );
    }

    #[test]
    fn format_range_without_label() {
        let result = TemporalResult::Range(date(2023, 6, 2), date(2023, 6, 8), None);
        assert_eq!(format_result(&result), "2023-06-02 to 2023-06-08");
    }

    // --- Month boundary edge cases ---

    #[test]
    fn last_month_from_january() {
        let result = resolve("last month", date(2023, 1, 15)).unwrap();
        assert_date_range(&result, date(2022, 12, 1), date(2022, 12, 31));
    }

    #[test]
    fn next_month_from_december() {
        let result = resolve("next month", date(2023, 12, 15)).unwrap();
        assert_date_range(&result, date(2024, 1, 1), date(2024, 1, 31));
    }

    #[test]
    fn month_ago_clamps_day() {
        let result = resolve("a month ago", date(2023, 3, 31)).unwrap();
        assert_single_date(&result, date(2023, 2, 28));
    }

    #[test]
    fn leap_year_february() {
        let result = resolve("a month ago", date(2024, 3, 31)).unwrap();
        assert_single_date(&result, date(2024, 2, 29));
    }

    // --- The original motivating example ---

    #[test]
    fn charity_race_last_sunday() {
        let reference = parse_reference_date("2023-05-25T14:30:00Z").unwrap();
        let result = resolve("last Sunday", reference).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
        assert_eq!(format_result(&result), "2023-05-21 (Sunday)");
    }
}
