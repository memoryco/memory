//! English temporal expression resolution.
//!
//! Resolves relative time expressions like "last Sunday", "two weeks ago",
//! "the weekend before 17 July 2023" against a reference date.
//!
//! # Pattern priority
//!
//! Patterns are matched in the order listed in `resolve()`. More specific
//! patterns (e.g. "the [weekday] before [date]") should come after simpler
//! ones (e.g. "last [weekday]") to avoid false matches.
//!
//! # Migration note
//!
//! The pattern matchers here are being migrated from `tools/date_resolve.rs`.
//! During migration, both files exist. Once migration is complete,
//! `date_resolve.rs` becomes a thin tool wrapper that delegates here.

use super::parsing;
use crate::lang::TemporalResult;
use chrono::{Datelike, Duration, NaiveDate, Weekday};

/// Resolve an English temporal expression against a reference date.
pub fn resolve(expression: &str, reference: NaiveDate) -> Result<TemporalResult, String> {
    let expr = expression.trim().to_lowercase();

    // Simple literals
    match expr.as_str() {
        "today" => {
            return Ok(TemporalResult::Date(
                reference,
                Some(parsing::day_name(reference)),
            ));
        }
        "yesterday" => {
            let d = reference - Duration::days(1);
            return Ok(TemporalResult::Date(d, Some(parsing::day_name(d))));
        }
        "tomorrow" => {
            let d = reference + Duration::days(1);
            return Ok(TemporalResult::Date(d, Some(parsing::day_name(d))));
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

    // "the weekend before/after [date]", "two weekends before [date]",
    // "first weekend of August 2023"
    if let Some(result) = try_weekend_patterns(&expr, reference) {
        return result;
    }

    // "the week/month before/after [date]"
    if let Some(result) = try_period_before_after(&expr, reference) {
        return result;
    }

    // "summer of 2022", "winter 2023", etc.
    if let Some(result) = try_season_of_year(&expr) {
        return result;
    }

    Err(format!(
        "Could not resolve English temporal expression '{}'. Supported patterns:\n\
         - yesterday, today, tomorrow\n\
         - X days/weeks/months/years ago\n\
         - in X days/weeks/months/years\n\
         - last/next/this [weekday]\n\
         - last/next/this week/month/year\n\
         - the [weekday] before/after [date]\n\
         - the weekend before/after [date]\n\
         - X weekends before/after [date]\n\
         - first/second/third/fourth/last weekend of [month] [year]\n\
         - the week/month before/after [date]\n\
         - season of year (e.g. 'summer of 2022')",
        expression.trim()
    ))
}

// =============================================================================
// Pattern matchers
// =============================================================================

/// Match "X days/weeks/months/years ago" or "a day/week/month/year ago"
fn try_n_units_ago(expr: &str, reference: NaiveDate) -> Option<Result<TemporalResult, String>> {
    let stripped = expr.strip_suffix(" ago")?;
    let (n, unit) = parse_n_unit(stripped)?;
    Some(offset_date(reference, -(n as i64), unit))
}

/// Match "in X days/weeks/months/years"
fn try_in_n_units(expr: &str, reference: NaiveDate) -> Option<Result<TemporalResult, String>> {
    let stripped = expr.strip_prefix("in ")?;
    let (n, unit) = parse_n_unit(stripped)?;
    Some(offset_date(reference, n as i64, unit))
}

/// Match "last/next/this [weekday]"
fn try_relative_weekday(
    expr: &str,
    reference: NaiveDate,
) -> Option<Result<TemporalResult, String>> {
    let (direction, rest) = split_direction(expr)?;
    let weekday = parsing::weekday_name(rest)?;

    let result = match direction {
        "last" => {
            let d = find_previous_weekday(reference, weekday);
            TemporalResult::Date(d, Some(parsing::day_name(d)))
        }
        "next" => {
            let d = find_next_weekday(reference, weekday);
            TemporalResult::Date(d, Some(parsing::day_name(d)))
        }
        "this" => {
            let d = find_this_weekday(reference, weekday);
            TemporalResult::Date(d, Some(parsing::day_name(d)))
        }
        _ => return None,
    };

    Some(Ok(result))
}

/// Match "last/next/this week/month/year"
fn try_relative_period(expr: &str, reference: NaiveDate) -> Option<Result<TemporalResult, String>> {
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
            Some(Ok(TemporalResult::Range(start, end, Some(label))))
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
            let label = format!(
                "{} {}",
                parsing::month_name_from_number(start.month()),
                start.year()
            );
            Some(Ok(TemporalResult::Range(start, end, Some(label))))
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
            Some(Ok(TemporalResult::Range(start, end, Some(label))))
        }
        _ => None,
    }
}

/// Match "the [weekday] before/after [date]" or "the [weekday] before/after"
fn try_weekday_before_after(
    expr: &str,
    reference: NaiveDate,
) -> Option<Result<TemporalResult, String>> {
    let rest = expr.strip_prefix("the ")?;
    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return None;
    }

    let weekday = parsing::weekday_name(parts[0])?;
    let direction = parts[1];

    if direction != "before" && direction != "after" {
        return None;
    }

    let anchor = if parts.len() == 3 {
        parsing::natural_date(parts[2])?
    } else {
        reference
    };

    let result = match direction {
        "before" => {
            let d = find_previous_weekday(anchor, weekday);
            TemporalResult::Date(d, Some(parsing::day_name(d)))
        }
        "after" => {
            let d = find_next_weekday(anchor, weekday);
            TemporalResult::Date(d, Some(parsing::day_name(d)))
        }
        _ => return None,
    };

    Some(Ok(result))
}

/// Match "the week/month before/after [date]" or "the week/month before/after"
fn try_period_before_after(
    expr: &str,
    reference: NaiveDate,
) -> Option<Result<TemporalResult, String>> {
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
        parsing::natural_date(parts[2])?
    } else {
        reference
    };

    match (period, direction) {
        ("week", "before") => {
            let end = anchor - Duration::days(1);
            let start = end - Duration::days(6);
            Some(Ok(TemporalResult::Range(start, end, None)))
        }
        ("week", "after") => {
            let start = anchor + Duration::days(1);
            let end = start + Duration::days(6);
            Some(Ok(TemporalResult::Range(start, end, None)))
        }
        ("month", "before") => {
            let d = add_months(anchor, -1)?;
            let (start, end) = month_range(d.year(), d.month());
            let label = format!(
                "{} {}",
                parsing::month_name_from_number(start.month()),
                start.year()
            );
            Some(Ok(TemporalResult::Range(start, end, Some(label))))
        }
        ("month", "after") => {
            let d = add_months(anchor, 1)?;
            let (start, end) = month_range(d.year(), d.month());
            let label = format!(
                "{} {}",
                parsing::month_name_from_number(start.month()),
                start.year()
            );
            Some(Ok(TemporalResult::Range(start, end, Some(label))))
        }
        _ => None,
    }
}

/// Match weekend patterns.
fn try_weekend_patterns(
    expr: &str,
    reference: NaiveDate,
) -> Option<Result<TemporalResult, String>> {
    if let Some(result) = try_n_weekends_before_after(expr, reference) {
        return Some(result);
    }
    if let Some(result) = try_nth_weekend_of_month(expr) {
        return Some(result);
    }
    None
}

/// Match "weekend before/after [date]" and "X weekends before/after [date]".
fn try_n_weekends_before_after(
    expr: &str,
    reference: NaiveDate,
) -> Option<Result<TemporalResult, String>> {
    let rest = expr.strip_prefix("the ").unwrap_or(expr);
    let parts: Vec<&str> = rest.split_whitespace().collect();
    let dir_idx = parts.iter().position(|p| *p == "before" || *p == "after")?;
    if dir_idx == 0 {
        return None;
    }

    let direction = parts[dir_idx];
    let left = &parts[..dir_idx];
    let right = parts[dir_idx + 1..].join(" ");

    let n = match left {
        ["weekend"] | ["weekends"] => 1,
        [count, unit] if *unit == "weekend" || *unit == "weekends" => parsing::number_word(count)?,
        _ => return None,
    };

    let anchor = if right.is_empty() {
        reference
    } else {
        parsing::natural_date(&right)?
    };

    let (start, end) = match direction {
        "before" => nth_weekend_before(anchor, n),
        "after" => nth_weekend_after(anchor, n),
        _ => return None,
    };

    Some(Ok(TemporalResult::Range(
        start,
        end,
        Some(format!("{} weekend{}", n, if n == 1 { "" } else { "s" })),
    )))
}

/// Match "first/second/third/fourth/last weekend of [month] [year]".
fn try_nth_weekend_of_month(expr: &str) -> Option<Result<TemporalResult, String>> {
    let rest = expr.strip_prefix("the ").unwrap_or(expr);
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    if parts[1] != "weekend" || parts[2] != "of" {
        return None;
    }

    let ordinal = parts[0];
    let month = parsing::month_name(parts[3])?;
    let year = parsing::year(parts[4])?;

    let (start, end) = match ordinal {
        "first" => nth_weekend_in_month(year, month, 1)?,
        "second" => nth_weekend_in_month(year, month, 2)?,
        "third" => nth_weekend_in_month(year, month, 3)?,
        "fourth" => nth_weekend_in_month(year, month, 4)?,
        "last" => last_weekend_in_month(year, month)?,
        _ => return None,
    };

    Some(Ok(TemporalResult::Range(
        start,
        end,
        Some(format!(
            "{} weekend of {} {}",
            ordinal,
            parsing::month_name_from_number(month),
            year
        )),
    )))
}

/// Match season expressions like "summer of 2022" or "winter 2023".
fn try_season_of_year(expr: &str) -> Option<Result<TemporalResult, String>> {
    let rest = expr.strip_prefix("the ").unwrap_or(expr);
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }

    let (season, year_token) = if tokens.len() >= 3 && tokens[1] == "of" {
        (tokens[0], tokens[2])
    } else {
        (tokens[0], tokens[1])
    };

    let year = parsing::year(year_token)?;

    let result = match season {
        "spring" => TemporalResult::Range(
            NaiveDate::from_ymd_opt(year, 3, 1)?,
            NaiveDate::from_ymd_opt(year, 5, 31)?,
            Some(format!("Spring {}", year)),
        ),
        "summer" => TemporalResult::Range(
            NaiveDate::from_ymd_opt(year, 6, 1)?,
            NaiveDate::from_ymd_opt(year, 8, 31)?,
            Some(format!("Summer {}", year)),
        ),
        "fall" | "autumn" => TemporalResult::Range(
            NaiveDate::from_ymd_opt(year, 9, 1)?,
            NaiveDate::from_ymd_opt(year, 11, 30)?,
            Some(format!("Fall {}", year)),
        ),
        "winter" => {
            let next_year = year + 1;
            let end_day = days_in_month(next_year, 2);
            TemporalResult::Range(
                NaiveDate::from_ymd_opt(year, 12, 1)?,
                NaiveDate::from_ymd_opt(next_year, 2, end_day)?,
                Some(format!("Winter {}-{}", year, next_year)),
            )
        }
        _ => return None,
    };

    Some(Ok(result))
}

// =============================================================================
// Internal helpers (language-agnostic calendar math)
// =============================================================================

/// Parse "X unit(s)" or "a/an/one unit".
fn parse_n_unit(s: &str) -> Option<(u32, &'static str)> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }
    let n = parsing::number_word(parts[0])?;
    let unit = normalize_unit(parts[1])?;
    Some((n, unit))
}

fn normalize_unit(s: &str) -> Option<&'static str> {
    match s {
        "day" | "days" => Some("day"),
        "week" | "weeks" => Some("week"),
        "month" | "months" => Some("month"),
        "year" | "years" => Some("year"),
        _ => None,
    }
}

fn offset_date(reference: NaiveDate, count: i64, unit: &str) -> Result<TemporalResult, String> {
    match unit {
        "day" => {
            let d = reference + Duration::days(count);
            Ok(TemporalResult::Date(d, Some(parsing::day_name(d))))
        }
        "week" => {
            let d = reference + Duration::weeks(count);
            Ok(TemporalResult::Date(d, Some(parsing::day_name(d))))
        }
        "month" => {
            let d = add_months(reference, count as i32)
                .ok_or_else(|| "Date out of range after month offset".to_string())?;
            Ok(TemporalResult::Date(d, Some(parsing::day_name(d))))
        }
        "year" => {
            let d = add_months(reference, count as i32 * 12)
                .ok_or_else(|| "Date out of range after year offset".to_string())?;
            Ok(TemporalResult::Date(d, Some(parsing::day_name(d))))
        }
        _ => Err(format!("Unknown unit: {}", unit)),
    }
}

fn split_direction(s: &str) -> Option<(&str, &str)> {
    for prefix in ["last ", "next ", "this "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return Some((prefix.trim(), rest));
        }
    }
    None
}

fn find_previous_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let mut d = from - Duration::days(1);
    while d.weekday() != target {
        d = d - Duration::days(1);
    }
    d
}

fn find_next_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let mut d = from + Duration::days(1);
    while d.weekday() != target {
        d = d + Duration::days(1);
    }
    d
}

fn find_this_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let current_iso = from.weekday().num_days_from_monday() as i64;
    let target_iso = target.num_days_from_monday() as i64;
    from + Duration::days(target_iso - current_iso)
}

fn week_range(date: NaiveDate) -> (NaiveDate, NaiveDate) {
    let days_since_monday = date.weekday().num_days_from_monday() as i64;
    let monday = date - Duration::days(days_since_monday);
    let sunday = monday + Duration::days(6);
    (monday, sunday)
}

fn nth_weekend_before(anchor: NaiveDate, n: u32) -> (NaiveDate, NaiveDate) {
    let mut sunday = anchor - Duration::days(1);
    while sunday.weekday() != Weekday::Sun {
        sunday = sunday - Duration::days(1);
    }
    if n > 1 {
        sunday = sunday - Duration::weeks((n - 1) as i64);
    }
    let saturday = sunday - Duration::days(1);
    (saturday, sunday)
}

fn nth_weekend_after(anchor: NaiveDate, n: u32) -> (NaiveDate, NaiveDate) {
    let mut saturday = anchor + Duration::days(1);
    while saturday.weekday() != Weekday::Sat {
        saturday = saturday + Duration::days(1);
    }
    if n > 1 {
        saturday = saturday + Duration::weeks((n - 1) as i64);
    }
    let sunday = saturday + Duration::days(1);
    (saturday, sunday)
}

fn nth_weekend_in_month(year: i32, month: u32, n: u32) -> Option<(NaiveDate, NaiveDate)> {
    if n == 0 {
        return None;
    }
    let first_day = NaiveDate::from_ymd_opt(year, month, 1)?;
    let mut first_saturday = first_day;
    while first_saturday.weekday() != Weekday::Sat {
        first_saturday = first_saturday + Duration::days(1);
    }
    let saturday = first_saturday + Duration::weeks((n - 1) as i64);
    let sunday = saturday + Duration::days(1);
    if saturday.month() != month || sunday.month() != month {
        return None;
    }
    Some((saturday, sunday))
}

fn last_weekend_in_month(year: i32, month: u32) -> Option<(NaiveDate, NaiveDate)> {
    let (_, month_end) = month_range(year, month);
    let mut saturday = month_end;
    while saturday.weekday() != Weekday::Sat {
        saturday = saturday - Duration::days(1);
    }
    let sunday = saturday + Duration::days(1);
    if sunday.month() != month {
        let saturday_prev = saturday - Duration::weeks(1);
        let sunday_prev = saturday_prev + Duration::days(1);
        if saturday_prev.month() == month && sunday_prev.month() == month {
            return Some((saturday_prev, sunday_prev));
        }
        return None;
    }
    Some((saturday, sunday))
}

fn month_range(year: i32, month: u32) -> (NaiveDate, NaiveDate) {
    let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let end = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap() - Duration::days(1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap() - Duration::days(1)
    };
    (start, end)
}

fn add_months(date: NaiveDate, months: i32) -> Option<NaiveDate> {
    let total_months = date.year() * 12 + date.month() as i32 - 1 + months;
    let year = total_months.div_euclid(12);
    let month = (total_months.rem_euclid(12) + 1) as u32;
    let day = date.day().min(days_in_month(year, month));
    NaiveDate::from_ymd_opt(year, month, day)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .map(|d| (d - NaiveDate::from_ymd_opt(year, month, 1).unwrap()).num_days() as u32)
    .unwrap_or(30)
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

    fn assert_date(result: &TemporalResult, expected: NaiveDate) {
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

    fn assert_range(result: &TemporalResult, expected_start: NaiveDate, expected_end: NaiveDate) {
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

    // --- Simple literals ---

    #[test]
    fn yesterday() {
        let result = resolve("yesterday", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn today() {
        let result = resolve("today", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 25));
    }

    #[test]
    fn tomorrow() {
        let result = resolve("tomorrow", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 26));
    }

    // --- N units ago ---

    #[test]
    fn three_days_ago() {
        let result = resolve("3 days ago", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn three_days_ago_word() {
        let result = resolve("three days ago", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 22));
    }

    #[test]
    fn a_day_ago() {
        let result = resolve("a day ago", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn two_weeks_ago() {
        let result = resolve("2 weeks ago", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 11));
    }

    #[test]
    fn two_months_ago() {
        let result = resolve("2 months ago", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 3, 25));
    }

    // --- In N units ---

    #[test]
    fn in_five_days() {
        let result = resolve("in 5 days", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 30));
    }

    // --- Relative weekdays ---

    #[test]
    fn last_sunday_from_thursday() {
        let result = resolve("last sunday", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn last_sunday_from_sunday() {
        let result = resolve("last sunday", date(2023, 5, 21)).unwrap();
        assert_date(&result, date(2023, 5, 14));
    }

    #[test]
    fn next_friday_from_monday() {
        let result = resolve("next friday", date(2023, 5, 22)).unwrap();
        assert_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn this_wednesday_from_monday() {
        let result = resolve("this wednesday", date(2023, 5, 22)).unwrap();
        assert_date(&result, date(2023, 5, 24));
    }

    // --- Relative periods ---

    #[test]
    fn last_week() {
        let result = resolve("last week", date(2023, 5, 25)).unwrap();
        assert_range(&result, date(2023, 5, 15), date(2023, 5, 21));
    }

    #[test]
    fn last_month() {
        let result = resolve("last month", date(2023, 5, 25)).unwrap();
        assert_range(&result, date(2023, 4, 1), date(2023, 4, 30));
    }

    #[test]
    fn last_year() {
        let result = resolve("last year", date(2023, 5, 25)).unwrap();
        assert_range(&result, date(2022, 1, 1), date(2022, 12, 31));
    }

    // --- Weekday before/after date ---

    #[test]
    fn the_sunday_before_may_25() {
        let result = resolve("the sunday before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn the_friday_after_may_25() {
        let result = resolve("the friday after 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 26));
    }

    #[test]
    fn the_sunday_before_natural_date() {
        let result = resolve("the sunday before 3 July 2023", date(2023, 7, 10)).unwrap();
        assert_date(&result, date(2023, 7, 2));
    }

    // --- Weekends ---

    #[test]
    fn weekend_before_date() {
        let result = resolve("the weekend before 20 October 2023", date(2023, 10, 25)).unwrap();
        assert_range(&result, date(2023, 10, 14), date(2023, 10, 15));
    }

    #[test]
    fn two_weekends_before_date() {
        let result = resolve("two weekends before 17 July 2023", date(2023, 7, 20)).unwrap();
        assert_range(&result, date(2023, 7, 8), date(2023, 7, 9));
    }

    #[test]
    fn first_weekend_of_august() {
        let result = resolve("first weekend of August 2023", date(2023, 8, 10)).unwrap();
        assert_range(&result, date(2023, 8, 5), date(2023, 8, 6));
    }

    // --- Week/month before/after ---

    #[test]
    fn the_week_before_date() {
        let result = resolve("the week before 2023-06-09", date(2023, 6, 9)).unwrap();
        assert_range(&result, date(2023, 6, 2), date(2023, 6, 8));
    }

    #[test]
    fn the_month_before_date() {
        let result = resolve("the month before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_range(&result, date(2023, 4, 1), date(2023, 4, 30));
    }

    // --- Seasons ---

    #[test]
    fn summer_of_2022() {
        let result = resolve("summer of 2022", date(2023, 8, 10)).unwrap();
        assert_range(&result, date(2022, 6, 1), date(2022, 8, 31));
    }

    // --- Edge cases ---

    #[test]
    fn case_insensitive() {
        let result = resolve("LAST SUNDAY", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 21));
    }

    #[test]
    fn whitespace_trimmed() {
        let result = resolve("  yesterday  ", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 24));
    }

    #[test]
    fn unrecognized_expression() {
        let result = resolve("when pigs fly", date(2023, 5, 25));
        assert!(result.is_err());
    }

    // --- Month boundary edge cases ---

    #[test]
    fn month_ago_clamps_day() {
        let result = resolve("a month ago", date(2023, 3, 31)).unwrap();
        assert_date(&result, date(2023, 2, 28));
    }

    #[test]
    fn leap_year_february() {
        let result = resolve("a month ago", date(2024, 3, 31)).unwrap();
        assert_date(&result, date(2024, 2, 29));
    }

    // --- The LOCOMO motivating example ---

    #[test]
    fn charity_race_last_sunday() {
        let result = resolve("last Sunday", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 21));
    }
}
