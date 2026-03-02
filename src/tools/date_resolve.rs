//! date_resolve - Resolve relative time expressions to absolute dates

use chrono::{Datelike, NaiveDate, Weekday, Duration};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct DateResolveTool;

#[derive(Deserialize)]
struct Args {
    expression: String,
    reference_date: String,
}

/// Result of resolving a date expression.
#[derive(Debug)]
enum DateResult {
    /// A single resolved date with an optional label (e.g. day name).
    SingleDate(NaiveDate, Option<String>),
    /// A date range with an optional label.
    DateRange(NaiveDate, NaiveDate, Option<String>),
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
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let reference = parse_reference_date(&args.reference_date)
            .map_err(|e| McpError::InvalidParams(e))?;

        match resolve_expression(&args.expression, reference) {
            Ok(result) => Ok(text_response(format_result(&result))),
            Err(e) => Ok(text_response(e)),
        }
    }
}

// =============================================================================
// Date parsing
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

/// Parse an inline date from an expression (e.g. "2023-05-25" within "the sunday before 2023-05-25").
fn parse_inline_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
}

// =============================================================================
// Expression resolver
// =============================================================================

/// Resolve a relative time expression against a reference date.
fn resolve_expression(expression: &str, reference: NaiveDate) -> Result<DateResult, String> {
    let expr = expression.trim().to_lowercase();

    // Simple literals
    match expr.as_str() {
        "today" => return Ok(DateResult::SingleDate(reference, Some(day_name(reference)))),
        "yesterday" => {
            let d = reference - Duration::days(1);
            return Ok(DateResult::SingleDate(d, Some(day_name(d))));
        }
        "tomorrow" => {
            let d = reference + Duration::days(1);
            return Ok(DateResult::SingleDate(d, Some(day_name(d))));
        }
        _ => {}
    }

    // "X days/weeks/months/years ago"
    if let Some(result) = try_n_units_ago(&expr, reference) {
        return result;
    }

    // "in X days/weeks/months/years"
    if let Some(result) = try_in_n_units(&expr, reference) {
        return result;
    }

    // "last/next/this [weekday]"
    if let Some(result) = try_relative_weekday(&expr, reference) {
        return result;
    }

    // "last/next/this week/month/year"
    if let Some(result) = try_relative_period(&expr, reference) {
        return result;
    }

    // "the [weekday] before/after [date]"
    if let Some(result) = try_weekday_before_after(&expr, reference) {
        return result;
    }

    // "the week/month before/after [date]"
    if let Some(result) = try_period_before_after(&expr, reference) {
        return result;
    }

    Err(format!(
        "Could not resolve expression '{}'. Supported patterns:\n\
         - yesterday, today, tomorrow\n\
         - X days/weeks/months/years ago\n\
         - in X days/weeks/months/years\n\
         - last/next/this [weekday]\n\
         - last/next/this week/month/year\n\
         - the [weekday] before/after [date]\n\
         - the week/month before/after [date]",
        expression.trim()
    ))
}

// =============================================================================
// Pattern matchers
// =============================================================================

/// Match "X days/weeks/months/years ago" or "a day/week/month/year ago"
fn try_n_units_ago(expr: &str, reference: NaiveDate) -> Option<Result<DateResult, String>> {
    let stripped = expr.strip_suffix(" ago")?;
    let (n, unit) = parse_n_unit(stripped)?;

    Some(offset_date(reference, -(n as i64), unit))
}

/// Match "in X days/weeks/months/years"
fn try_in_n_units(expr: &str, reference: NaiveDate) -> Option<Result<DateResult, String>> {
    let stripped = expr.strip_prefix("in ")?;
    let (n, unit) = parse_n_unit(stripped)?;

    Some(offset_date(reference, n as i64, unit))
}

/// Parse "X unit(s)" or "a/an/one unit" from a string.
/// Returns (count, unit_name) where unit_name is one of "day", "week", "month", "year".
fn parse_n_unit(s: &str) -> Option<(u32, &'static str)> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let n = parse_number(parts[0])?;
    let unit = normalize_unit(parts[1])?;
    Some((n, unit))
}

/// Parse a number from a word — supports digits and common English words.
fn parse_number(s: &str) -> Option<u32> {
    if let Ok(n) = s.parse::<u32>() {
        return Some(n);
    }
    match s {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        _ => None,
    }
}

/// Normalize unit names to singular form.
fn normalize_unit(s: &str) -> Option<&'static str> {
    match s {
        "day" | "days" => Some("day"),
        "week" | "weeks" => Some("week"),
        "month" | "months" => Some("month"),
        "year" | "years" => Some("year"),
        _ => None,
    }
}

/// Apply an offset to a date by count and unit.
fn offset_date(reference: NaiveDate, count: i64, unit: &str) -> Result<DateResult, String> {
    match unit {
        "day" => {
            let d = reference + Duration::days(count);
            Ok(DateResult::SingleDate(d, Some(day_name(d))))
        }
        "week" => {
            let d = reference + Duration::weeks(count);
            Ok(DateResult::SingleDate(d, Some(day_name(d))))
        }
        "month" => {
            let d = add_months(reference, count as i32)
                .ok_or_else(|| "Date out of range after month offset".to_string())?;
            Ok(DateResult::SingleDate(d, Some(day_name(d))))
        }
        "year" => {
            let d = add_months(reference, count as i32 * 12)
                .ok_or_else(|| "Date out of range after year offset".to_string())?;
            Ok(DateResult::SingleDate(d, Some(day_name(d))))
        }
        _ => Err(format!("Unknown unit: {}", unit)),
    }
}

/// Match "last/next/this [weekday]"
fn try_relative_weekday(expr: &str, reference: NaiveDate) -> Option<Result<DateResult, String>> {
    let (direction, rest) = split_direction(expr)?;
    let weekday = parse_weekday(rest)?;

    let result = match direction {
        "last" => {
            let d = find_previous_weekday(reference, weekday);
            DateResult::SingleDate(d, Some(day_name(d)))
        }
        "next" => {
            let d = find_next_weekday(reference, weekday);
            DateResult::SingleDate(d, Some(day_name(d)))
        }
        "this" => {
            let d = find_this_weekday(reference, weekday);
            DateResult::SingleDate(d, Some(day_name(d)))
        }
        _ => return None,
    };

    Some(Ok(result))
}

/// Match "last/next/this week/month/year"
fn try_relative_period(expr: &str, reference: NaiveDate) -> Option<Result<DateResult, String>> {
    let (direction, rest) = split_direction(expr)?;

    match rest {
        "week" => {
            let (start, end) = match direction {
                "last" => week_range(reference - Duration::weeks(1)),
                "next" => week_range(reference + Duration::weeks(1)),
                "this" => week_range(reference),
                _ => return None,
            };
            let label = format!("{} week", direction);
            Some(Ok(DateResult::DateRange(start, end, Some(label))))
        }
        "month" => {
            let (start, end) = match direction {
                "last" => {
                    let d = add_months(reference, -1)?;
                    month_range(d.year(), d.month())
                }
                "next" => {
                    let d = add_months(reference, 1)?;
                    month_range(d.year(), d.month())
                }
                "this" => month_range(reference.year(), reference.month()),
                _ => return None,
            };
            let label = format!("{} {}", month_name(start.month()), start.year());
            Some(Ok(DateResult::DateRange(start, end, Some(label))))
        }
        "year" => {
            let year = match direction {
                "last" => reference.year() - 1,
                "next" => reference.year() + 1,
                "this" => reference.year(),
                _ => return None,
            };
            let start = NaiveDate::from_ymd_opt(year, 1, 1)?;
            let end = NaiveDate::from_ymd_opt(year, 12, 31)?;
            let label = format!("{}", year);
            Some(Ok(DateResult::DateRange(start, end, Some(label))))
        }
        _ => None,
    }
}

/// Match "the [weekday] before/after [date]" or "the [weekday] before/after"
fn try_weekday_before_after(expr: &str, reference: NaiveDate) -> Option<Result<DateResult, String>> {
    let rest = expr.strip_prefix("the ")?;
    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return None;
    }

    let weekday = parse_weekday(parts[0])?;
    let direction = parts[1];

    if direction != "before" && direction != "after" {
        return None;
    }

    let anchor = if parts.len() == 3 {
        match parse_inline_date(parts[2]) {
            Some(d) => d,
            None => return None,
        }
    } else {
        reference
    };

    let result = match direction {
        "before" => {
            let d = find_previous_weekday(anchor, weekday);
            DateResult::SingleDate(d, Some(day_name(d)))
        }
        "after" => {
            let d = find_next_weekday(anchor, weekday);
            DateResult::SingleDate(d, Some(day_name(d)))
        }
        _ => return None,
    };

    Some(Ok(result))
}

/// Match "the week/month before/after [date]" or "the week/month before/after"
fn try_period_before_after(expr: &str, reference: NaiveDate) -> Option<Result<DateResult, String>> {
    let rest = expr.strip_prefix("the ")?;
    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return None;
    }

    let period = parts[0];
    if period != "week" && period != "month" {
        return None;
    }

    let direction = parts[1];
    if direction != "before" && direction != "after" {
        return None;
    }

    let anchor = if parts.len() == 3 {
        match parse_inline_date(parts[2]) {
            Some(d) => d,
            None => return None,
        }
    } else {
        reference
    };

    match (period, direction) {
        ("week", "before") => {
            let end = anchor - Duration::days(1);
            let start = end - Duration::days(6);
            Some(Ok(DateResult::DateRange(start, end, None)))
        }
        ("week", "after") => {
            let start = anchor + Duration::days(1);
            let end = start + Duration::days(6);
            Some(Ok(DateResult::DateRange(start, end, None)))
        }
        ("month", "before") => {
            let d = add_months(anchor, -1)?;
            let (start, end) = month_range(d.year(), d.month());
            let label = format!("{} {}", month_name(start.month()), start.year());
            Some(Ok(DateResult::DateRange(start, end, Some(label))))
        }
        ("month", "after") => {
            let d = add_months(anchor, 1)?;
            let (start, end) = month_range(d.year(), d.month());
            let label = format!("{} {}", month_name(start.month()), start.year());
            Some(Ok(DateResult::DateRange(start, end, Some(label))))
        }
        _ => None,
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Split "last/next/this foo" into (direction, rest).
fn split_direction(s: &str) -> Option<(&str, &str)> {
    for prefix in ["last ", "next ", "this "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return Some((prefix.trim(), rest));
        }
    }
    None
}

/// Parse a weekday name (full or abbreviated, case-insensitive — input already lowered).
fn parse_weekday(s: &str) -> Option<Weekday> {
    match s {
        "monday" | "mon" => Some(Weekday::Mon),
        "tuesday" | "tue" | "tues" => Some(Weekday::Tue),
        "wednesday" | "wed" => Some(Weekday::Wed),
        "thursday" | "thu" | "thur" | "thurs" => Some(Weekday::Thu),
        "friday" | "fri" => Some(Weekday::Fri),
        "saturday" | "sat" => Some(Weekday::Sat),
        "sunday" | "sun" => Some(Weekday::Sun),
        _ => None,
    }
}

/// Find the most recent occurrence of a weekday strictly before the given date.
fn find_previous_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let mut d = from - Duration::days(1);
    while d.weekday() != target {
        d = d - Duration::days(1);
    }
    d
}

/// Find the next occurrence of a weekday strictly after the given date.
fn find_next_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let mut d = from + Duration::days(1);
    while d.weekday() != target {
        d = d + Duration::days(1);
    }
    d
}

/// Find the given weekday in the current ISO week (Mon-Sun).
fn find_this_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let current_iso = from.weekday().num_days_from_monday() as i64;
    let target_iso = target.num_days_from_monday() as i64;
    let diff = target_iso - current_iso;
    from + Duration::days(diff)
}

/// Get the Monday-Sunday range for the week containing the given date.
fn week_range(date: NaiveDate) -> (NaiveDate, NaiveDate) {
    let days_since_monday = date.weekday().num_days_from_monday() as i64;
    let monday = date - Duration::days(days_since_monday);
    let sunday = monday + Duration::days(6);
    (monday, sunday)
}

/// Get the first and last day of a given month.
fn month_range(year: i32, month: u32) -> (NaiveDate, NaiveDate) {
    let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let end = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap() - Duration::days(1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap() - Duration::days(1)
    };
    (start, end)
}

/// Add months to a date, clamping the day to the last day of the target month.
fn add_months(date: NaiveDate, months: i32) -> Option<NaiveDate> {
    let total_months = date.year() * 12 + date.month() as i32 - 1 + months;
    let year = total_months.div_euclid(12);
    let month = (total_months.rem_euclid(12) + 1) as u32;
    let day = date.day().min(days_in_month(year, month));
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Get the number of days in a given month.
fn days_in_month(year: i32, month: u32) -> u32 {
    if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .map(|d| (d - NaiveDate::from_ymd_opt(year, month, 1).unwrap()).num_days() as u32)
    .unwrap_or(30)
}

/// Get the English name for a weekday.
fn day_name(date: NaiveDate) -> String {
    match date.weekday() {
        Weekday::Mon => "Monday",
        Weekday::Tue => "Tuesday",
        Weekday::Wed => "Wednesday",
        Weekday::Thu => "Thursday",
        Weekday::Fri => "Friday",
        Weekday::Sat => "Saturday",
        Weekday::Sun => "Sunday",
    }
    .to_string()
}

/// Get the English name for a month number (1-12).
fn month_name(month: u32) -> &'static str {
    match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "Unknown",
    }
}

// =============================================================================
// Formatting
// =============================================================================

/// Format a DateResult as a human-readable string.
fn format_result(result: &DateResult) -> String {
    match result {
        DateResult::SingleDate(date, label) => {
            let formatted = date.format("%Y-%m-%d").to_string();
            match label {
                Some(l) => format!("{} ({})", formatted, l),
                None => formatted,
            }
        }
        DateResult::DateRange(start, end, label) => {
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

    // --- Date parsing ---

    #[test]
    fn parse_iso_date() {
        assert_eq!(parse_reference_date("2023-05-25").unwrap(), date(2023, 5, 25));
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

    // --- Relative days ---

    #[test]
    fn yesterday() {
        let result = resolve_expression("yesterday", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn today() {
        let result = resolve_expression("today", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 25));
    }

    #[test]
    fn tomorrow() {
        let result = resolve_expression("tomorrow", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn three_days_ago() {
        let result = resolve_expression("3 days ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn three_days_ago_word() {
        let result = resolve_expression("three days ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn in_five_days() {
        let result = resolve_expression("in 5 days", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 30));
    }

    #[test]
    fn a_day_ago() {
        let result = resolve_expression("a day ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    // --- Relative weekdays ---

    #[test]
    fn last_sunday_from_thursday() {
        // 2023-05-25 is Thursday → last Sunday = 2023-05-21
        let result = resolve_expression("last sunday", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn last_monday_from_thursday() {
        // 2023-05-25 is Thursday → last Monday = 2023-05-22
        let result = resolve_expression("last monday", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn last_sunday_from_sunday() {
        // 2023-05-21 is Sunday → last Sunday should be 2023-05-14, NOT same day
        let result = resolve_expression("last sunday", date(2023, 5, 21)).unwrap();
        assert_single_date(&result, date(2023, 5, 14));
    }

    #[test]
    fn next_friday_from_monday() {
        // 2023-05-22 is Monday → next Friday = 2023-05-26
        let result = resolve_expression("next friday", date(2023, 5, 22)).unwrap();
        assert_single_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn this_wednesday_from_monday() {
        // 2023-05-22 is Monday → this Wednesday = 2023-05-24
        let result = resolve_expression("this wednesday", date(2023, 5, 22)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn this_monday_from_wednesday() {
        // 2023-05-24 is Wednesday → this Monday = 2023-05-22
        let result = resolve_expression("this monday", date(2023, 5, 24)).unwrap();
        assert_single_date(&result, date(2023, 5, 22));
    }

    // --- Relative weeks ---

    #[test]
    fn last_week() {
        // 2023-05-25 is Thursday → current week Mon=2023-05-22
        // last week = Mon 2023-05-15 to Sun 2023-05-21
        let result = resolve_expression("last week", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 15), date(2023, 5, 21));
    }

    #[test]
    fn next_week() {
        let result = resolve_expression("next week", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 29), date(2023, 6, 4));
    }

    #[test]
    fn this_week() {
        let result = resolve_expression("this week", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 22), date(2023, 5, 28));
    }

    #[test]
    fn two_weeks_ago() {
        let result = resolve_expression("2 weeks ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 11));
    }

    // --- Relative months ---

    #[test]
    fn last_month() {
        // Reference May 2023 → last month = April 2023
        let result = resolve_expression("last month", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 4, 1), date(2023, 4, 30));
    }

    #[test]
    fn next_month_from_may() {
        // Reference May 2023 → next month = June 2023
        let result = resolve_expression("next month", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 6, 1), date(2023, 6, 30));
    }

    #[test]
    fn this_month() {
        let result = resolve_expression("this month", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 1), date(2023, 5, 31));
    }

    #[test]
    fn two_months_ago() {
        // Reference May 2023 → 2 months ago = March 25, 2023
        let result = resolve_expression("2 months ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 3, 25));
    }

    // --- Relative years ---

    #[test]
    fn last_year() {
        let result = resolve_expression("last year", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2022, 1, 1), date(2022, 12, 31));
    }

    #[test]
    fn next_year() {
        let result = resolve_expression("next year", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2024, 1, 1), date(2024, 12, 31));
    }

    #[test]
    fn two_years_ago() {
        let result = resolve_expression("2 years ago", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2021, 5, 25));
    }

    // --- Combined / the X before/after ---

    #[test]
    fn the_sunday_before_may_25() {
        // 2023-05-25 is Thursday → Sunday before = 2023-05-21
        let result =
            resolve_expression("the sunday before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn the_friday_after_may_25() {
        // 2023-05-25 is Thursday → Friday after = 2023-05-26
        let result =
            resolve_expression("the friday after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn the_week_before_june_9() {
        // "the week before 2023-06-09" → 2023-06-02 to 2023-06-08
        let result =
            resolve_expression("the week before 2023-06-09", date(2023, 6, 9)).unwrap();
        assert_date_range(&result, date(2023, 6, 2), date(2023, 6, 8));
    }

    #[test]
    fn the_week_after_may_25() {
        // "the week after 2023-05-25" → 2023-05-26 to 2023-06-01
        let result =
            resolve_expression("the week after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 5, 26), date(2023, 6, 1));
    }

    #[test]
    fn the_month_before_may() {
        let result =
            resolve_expression("the month before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 4, 1), date(2023, 4, 30));
    }

    #[test]
    fn the_month_after_may() {
        let result =
            resolve_expression("the month after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date_range(&result, date(2023, 6, 1), date(2023, 6, 30));
    }

    #[test]
    fn the_week_before_no_date() {
        // "the week before" with no inline date → same as "last week"
        let result = resolve_expression("the week before", date(2023, 5, 25)).unwrap();
        // anchor = reference → week before = day before (May 24) back 7 days
        // end = 2023-05-24, start = 2023-05-18
        assert_date_range(&result, date(2023, 5, 18), date(2023, 5, 24));
    }

    // --- Edge cases ---

    #[test]
    fn case_insensitive() {
        let result = resolve_expression("LAST SUNDAY", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn abbreviations() {
        let result = resolve_expression("last sun", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn abbreviation_mon() {
        let result = resolve_expression("next mon", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 29));
    }

    #[test]
    fn whitespace_trimmed() {
        let result = resolve_expression("  yesterday  ", date(2023, 5, 25)).unwrap();
        assert_single_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn unrecognized_expression() {
        let result = resolve_expression("when pigs fly", date(2023, 5, 25));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Could not resolve"));
        assert!(err.contains("Supported patterns"));
    }

    // --- Formatting ---

    #[test]
    fn format_single_date_with_label() {
        let result = DateResult::SingleDate(date(2023, 5, 21), Some("Sunday".to_string()));
        assert_eq!(format_result(&result), "2023-05-21 (Sunday)");
    }

    #[test]
    fn format_range_with_label() {
        let result = DateResult::DateRange(
            date(2023, 6, 1),
            date(2023, 6, 30),
            Some("June 2023".to_string()),
        );
        assert_eq!(format_result(&result), "2023-06-01 to 2023-06-30 (June 2023)");
    }

    #[test]
    fn format_range_without_label() {
        let result = DateResult::DateRange(date(2023, 6, 2), date(2023, 6, 8), None);
        assert_eq!(format_result(&result), "2023-06-02 to 2023-06-08");
    }

    // --- Month boundary edge cases ---

    #[test]
    fn last_month_from_january() {
        // January → last month = December of previous year
        let result = resolve_expression("last month", date(2023, 1, 15)).unwrap();
        assert_date_range(&result, date(2022, 12, 1), date(2022, 12, 31));
    }

    #[test]
    fn next_month_from_december() {
        let result = resolve_expression("next month", date(2023, 12, 15)).unwrap();
        assert_date_range(&result, date(2024, 1, 1), date(2024, 1, 31));
    }

    #[test]
    fn month_ago_clamps_day() {
        // March 31 → 1 month ago → Feb doesn't have 31 days, clamp to 28
        let result = resolve_expression("a month ago", date(2023, 3, 31)).unwrap();
        assert_single_date(&result, date(2023, 2, 28));
    }

    #[test]
    fn leap_year_february() {
        // March 31 2024 → 1 month ago → Feb 2024 is leap year, clamp to 29
        let result = resolve_expression("a month ago", date(2024, 3, 31)).unwrap();
        assert_single_date(&result, date(2024, 2, 29));
    }

    // --- The original motivating example ---

    #[test]
    fn charity_race_last_sunday() {
        // "Melanie mentioned she ran a charity race last Sunday"
        // Memory created 2023-05-25T14:30:00Z (Thursday)
        let reference = parse_reference_date("2023-05-25T14:30:00Z").unwrap();
        let result = resolve_expression("last Sunday", reference).unwrap();
        assert_single_date(&result, date(2023, 5, 21));
        assert_eq!(format_result(&result), "2023-05-21 (Sunday)");
    }

    // --- Assertion helpers ---

    fn assert_single_date(result: &DateResult, expected: NaiveDate) {
        match result {
            DateResult::SingleDate(d, _) => assert_eq!(*d, expected, "Expected {}, got {}", expected, d),
            DateResult::DateRange(s, e, _) => {
                panic!("Expected single date {}, got range {} to {}", expected, s, e)
            }
        }
    }

    fn assert_date_range(result: &DateResult, expected_start: NaiveDate, expected_end: NaiveDate) {
        match result {
            DateResult::DateRange(s, e, _) => {
                assert_eq!(*s, expected_start, "Range start: expected {}, got {}", expected_start, s);
                assert_eq!(*e, expected_end, "Range end: expected {}, got {}", expected_end, e);
            }
            DateResult::SingleDate(d, _) => {
                panic!(
                    "Expected range {} to {}, got single date {}",
                    expected_start, expected_end, d
                )
            }
        }
    }
}
