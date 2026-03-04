//! English natural language parsing — numbers, months, weekdays, dates.

use chrono::{Datelike, NaiveDate, Weekday};

/// Parse a number from an English word.
///
/// Supports digits and common words ("one" through "twelve", plus "a"/"an").
pub fn number_word(word: &str) -> Option<u32> {
    if let Ok(n) = word.parse::<u32>() {
        return Some(n);
    }
    match word.to_lowercase().as_str() {
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

/// Parse an English month name or abbreviation → 1-12.
pub fn month_name(word: &str) -> Option<u32> {
    let cleaned = word
        .trim_matches(|c: char| !c.is_alphabetic())
        .to_lowercase();
    match cleaned.as_str() {
        "january" | "jan" => Some(1),
        "february" | "feb" => Some(2),
        "march" | "mar" => Some(3),
        "april" | "apr" => Some(4),
        "may" => Some(5),
        "june" | "jun" => Some(6),
        "july" | "jul" => Some(7),
        "august" | "aug" => Some(8),
        "september" | "sep" | "sept" => Some(9),
        "october" | "oct" => Some(10),
        "november" | "nov" => Some(11),
        "december" | "dec" => Some(12),
        _ => None,
    }
}

/// Parse an English weekday name or abbreviation.
pub fn weekday_name(word: &str) -> Option<Weekday> {
    match word.to_lowercase().as_str() {
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

/// Parse a day number token, supporting ordinal suffixes (1st, 2nd, 3rd, 4th).
pub fn day_number(token: &str) -> Option<u32> {
    let cleaned = token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    let stripped = cleaned
        .trim_end_matches("st")
        .trim_end_matches("nd")
        .trim_end_matches("rd")
        .trim_end_matches("th");
    let day = stripped.parse::<u32>().ok()?;
    if (1..=31).contains(&day) { Some(day) } else { None }
}

/// Parse a year token.
pub fn year(token: &str) -> Option<i32> {
    let cleaned = token.trim_matches(|c: char| !c.is_ascii_digit());
    let y = cleaned.parse::<i32>().ok()?;
    if (1000..=9999).contains(&y) { Some(y) } else { None }
}

/// Normalize free-form date text for parsing.
///
/// Strips punctuation (commas, backticks, periods) that often appears in
/// natural-language dates while keeping alphanumeric chars, whitespace,
/// and dashes.
pub fn normalize_date_text(input: &str) -> String {
    let cleaned: String = input
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '-' {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parse an inline date from English natural language text.
///
/// Handles:
/// - ISO: "2023-07-17"
/// - Day-Month-Year: "17 July 2023", "15 June, 2023", "1st September 2023"
/// - Month-Day-Year: "July 17 2023"
pub fn natural_date(text: &str) -> Option<NaiveDate> {
    let normalized = normalize_date_text(text);

    // ISO first (fast path)
    if let Ok(date) = NaiveDate::parse_from_str(&normalized, "%Y-%m-%d") {
        return Some(date);
    }

    let tokens: Vec<&str> = normalized.split_whitespace().collect();
    if tokens.len() < 3 {
        return None;
    }

    // Day-Month-Year: "17 July 2023"
    if let (Some(d), Some(m), Some(y)) = (
        day_number(tokens[0]),
        month_name(tokens[1]),
        year(tokens[2]),
    ) {
        if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
            return Some(date);
        }
    }

    // Month-Day-Year: "July 17 2023"
    if let (Some(m), Some(d), Some(y)) = (
        month_name(tokens[0]),
        day_number(tokens[1]),
        year(tokens[2]),
    ) {
        if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
            return Some(date);
        }
    }

    None
}

/// Get the English name for a weekday.
pub fn day_name(date: NaiveDate) -> String {
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
pub fn month_name_from_number(month: u32) -> &'static str {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn number_words() {
        assert_eq!(number_word("one"), Some(1));
        assert_eq!(number_word("twelve"), Some(12));
        assert_eq!(number_word("a"), Some(1));
        assert_eq!(number_word("5"), Some(5));
        assert_eq!(number_word("treize"), None);
    }

    #[test]
    fn month_names() {
        assert_eq!(month_name("january"), Some(1));
        assert_eq!(month_name("Sep"), Some(9));
        assert_eq!(month_name("sept"), Some(9));
        assert_eq!(month_name("enero"), None);
    }

    #[test]
    fn weekday_names() {
        assert_eq!(weekday_name("monday"), Some(Weekday::Mon));
        assert_eq!(weekday_name("Thu"), Some(Weekday::Thu));
        assert_eq!(weekday_name("thurs"), Some(Weekday::Thu));
        assert_eq!(weekday_name("lundi"), None);
    }

    #[test]
    fn day_numbers_with_ordinals() {
        assert_eq!(day_number("1st"), Some(1));
        assert_eq!(day_number("2nd"), Some(2));
        assert_eq!(day_number("3rd"), Some(3));
        assert_eq!(day_number("4th"), Some(4));
        assert_eq!(day_number("17"), Some(17));
        assert_eq!(day_number("0"), None);
        assert_eq!(day_number("32"), None);
    }

    #[test]
    fn natural_date_dmy() {
        assert_eq!(natural_date("17 July 2023"), Some(date(2023, 7, 17)));
        assert_eq!(natural_date("1st September 2023"), Some(date(2023, 9, 1)));
        assert_eq!(natural_date("15 June, 2023"), Some(date(2023, 6, 15)));
    }

    #[test]
    fn natural_date_mdy() {
        assert_eq!(natural_date("July 17 2023"), Some(date(2023, 7, 17)));
    }

    #[test]
    fn natural_date_iso() {
        assert_eq!(natural_date("2023-07-17"), Some(date(2023, 7, 17)));
    }

    #[test]
    fn natural_date_with_noise() {
        assert_eq!(natural_date("3` July 2023"), Some(date(2023, 7, 3)));
    }

    #[test]
    fn normalize_strips_punctuation() {
        assert_eq!(normalize_date_text("15 June, 2023"), "15 June 2023");
        assert_eq!(normalize_date_text("3` July 2023"), "3 July 2023");
    }
}
