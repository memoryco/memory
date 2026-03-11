//! LOCOMO benchmark regression tests.
//!
//! These tests are derived directly from bench_fails.md analysis of
//! run_20260303_114058. Each test maps to a specific benchmark failure case
//! and documents whether the failure is in our temporal resolver or upstream.
//!
//! Only cases testable at the lang/temporal level live here. Retrieval and
//! entity disambiguation failures are tracked in bench_fails.md but can't
//! be caught by unit tests.
//!
//! ## Case mapping
//!
//! | Bench Case | Category | Testable here? | Status       |
//! |-----------|----------|----------------|--------------|
//! | 1         | R1       | Yes            | DATASET_BUG  |
//! | 2         | R3       | No (retrieval) | —            |
//! | 3         | R4       | Yes            | VERIFY_MATH  |
//! | 4         | R2       | No (entity)    | —            |
//! | 5         | R1       | Partially      | NEEDS_PATTERN |
//! | 6         | R4       | Yes (duration) | NEEDS_MODULE  |
//! | 7         | R1       | Yes            | VERIFY        |
//! | 8         | R3       | No (retrieval) | —            |
//! | 9         | R1       | Yes            | VERIFY        |
//! | 10        | R3       | No (retrieval) | —            |
//! | 11        | R4       | Yes (parsing)  | VERIFY        |
//! | 12        | R1       | Yes            | VERIFY        |
//! | 13        | R2       | No (entity)    | —            |
//! | 14        | R2       | No (entity)    | —            |

#[cfg(test)]
mod tests {
    use crate::lang::TemporalResult;
    use crate::lang::english::parsing;
    use crate::lang::english::temporal;
    use chrono::NaiveDate;

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

    // =========================================================================
    // Case 1 (R1): Melanie's charity race — "last Saturday" from May 25
    //
    // Memory text: "ran a charity race last Saturday" in a May 25, 2023 message.
    // Gold answer: "The sunday before 25 May 2023" (2023-05-21)
    // Our answer:  2023-05-20 (Saturday)
    //
    // DATASET BUG: "last Saturday" from Thursday May 25 is May 20 (Saturday).
    // Gold says Sunday. We're correct — the source says Saturday, not Sunday.
    // This test asserts OUR correct answer. The benchmark mismatch is a gold
    // quality issue, not a resolver bug.
    // =========================================================================

    #[test]
    fn bench_case_01_charity_race_last_saturday() {
        let reference = date(2023, 5, 25); // Thursday
        let result = temporal::resolve("last saturday", reference).unwrap();
        assert_date(&result, date(2023, 5, 20)); // Saturday, May 20
    }

    // Also test the gold's phrasing to document what IT resolves to
    #[test]
    fn bench_case_01_gold_phrasing_sunday_before() {
        let result = temporal::resolve("the sunday before 2023-05-25", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 21)); // Sunday, May 21
    }

    // =========================================================================
    // Case 3 (R4): Melanie camping — "two weekends before 17 July 2023"
    //
    // Gold answer: "two weekends before 17 July 2023" (expects Jul 1–2)
    // Our answer:  Jul 8–9
    //
    // 17 July 2023 is a Monday.
    // Weekend immediately before: Jul 15–16
    // Two weekends before: Jul 8–9
    //
    // Gold expects Jul 1–2, which is THREE weekends back. "Two weekends
    // before" most naturally means: find the weekend before the anchor, then
    // go back one more. Jul 15-16 → Jul 8-9. We believe our math is correct
    // and this is an ambiguity in how you count "two weekends before."
    //
    // We test our interpretation. If we later decide to match gold, change
    // the expected values and document why.
    // =========================================================================

    #[test]
    fn bench_case_03_camping_two_weekends_before() {
        let result =
            temporal::resolve("two weekends before 17 July 2023", date(2023, 7, 20)).unwrap();
        // Our interpretation: Jul 8–9 (two weekends back from Monday Jul 17)
        assert_range(&result, date(2023, 7, 8), date(2023, 7, 9));
    }

    // =========================================================================
    // Case 5 (R1): Caroline's hike — "recently" relative to Aug 17
    //
    // Memory text: "Recently" in a message timestamped 2023-08-17
    // Gold answer: "The week before 25 August 2023"
    // Our answer:  Unknown (resolver doesn't handle "recently")
    //
    // "recently" is fuzzy. Best we can do is map it to a bounded window
    // relative to the message timestamp. This test documents the gap and
    // will pass once we add the pattern.
    // =========================================================================

    #[test]
    fn bench_case_05_hike_recently_is_not_yet_supported() {
        let reference = date(2023, 8, 17);
        let result = temporal::resolve("recently", reference);
        // Currently: Err (unrecognized pattern)
        // TODO: Once "recently" pattern is added, change this to assert
        // a range like (reference - 7days, reference) or similar.
        assert!(
            result.is_err(),
            "Expected 'recently' to fail until pattern is implemented. \
             If this test breaks, update it to assert the resolved range."
        );
    }

    // =========================================================================
    // Case 6 (R4): Melanie practicing art — "seven years" vs "since 2016"
    //
    // Gold: "Since 2016"
    // Predicted: "Seven years"
    //
    // These are semantically equivalent given a 2023 anchor. This isn't a
    // temporal resolution problem — it's a normalization / equivalence
    // problem. Belongs in a future duration.rs module.
    //
    // For now, test that we can at least resolve "seven years ago" if given
    // the right phrasing, and that we can do the arithmetic.
    // =========================================================================

    #[test]
    fn bench_case_06_art_practice_duration_to_year() {
        // If the phrasing were "seven years ago" we'd get 2016.
        let reference = date(2023, 5, 25);
        let result = temporal::resolve("seven years ago", reference).unwrap();
        assert_date(&result, date(2016, 5, 25));
    }

    // TODO: Add duration equivalence test once duration.rs exists:
    // assert_equivalent("seven years", "since 2016", anchor=2023)

    // =========================================================================
    // Case 7 (R1): Melanie's roadtrip — "weekend before 20 October 2023"
    //
    // Gold: "The weekend before 20 October 2023"
    // Predicted: 2023-10-19 (a Thursday — wrong)
    //
    // 20 October 2023 is a Friday.
    // Weekend before Friday = Sat Oct 14 – Sun Oct 15.
    // The benchmark agent returned a weekday instead of the weekend range.
    // Our resolver should get this right — the agent just didn't call it.
    // =========================================================================

    #[test]
    fn bench_case_07_roadtrip_weekend_before() {
        let result =
            temporal::resolve("the weekend before 20 October 2023", date(2023, 10, 25)).unwrap();
        assert_range(&result, date(2023, 10, 14), date(2023, 10, 15));
    }

    // =========================================================================
    // Case 9 (R1): John's veterans party — "the Friday before 20 May 2023"
    //
    // Gold: "The Friday before 20 May 2023"
    // Predicted: May 20, 2023 (the anchor date itself — wrong)
    //
    // 20 May 2023 is a Saturday.
    // Friday before Saturday May 20 = Friday May 19.
    // Agent returned the anchor date instead of computing the prior Friday.
    // =========================================================================

    #[test]
    fn bench_case_09_veterans_party_friday_before() {
        let result = temporal::resolve("the friday before 20 May 2023", date(2023, 5, 25)).unwrap();
        assert_date(&result, date(2023, 5, 19));
    }

    // =========================================================================
    // Case 11 (R4): John's firefighter call — malformed gold "3` July 2023"
    //
    // Gold: "The sunday before 3` July 2023" (note the backtick)
    // Predicted: 2023-07-30
    //
    // The backtick is a typo in the gold data. Our parser should handle it
    // gracefully. 3 July 2023 is a Monday.
    // Sunday before Monday Jul 3 = Sunday Jul 2.
    // =========================================================================

    #[test]
    fn bench_case_11_firefighter_malformed_gold_date() {
        // Parsing resilience: backtick in date string
        let parsed = parsing::natural_date("3` July 2023");
        assert_eq!(parsed, Some(date(2023, 7, 3)));
    }

    #[test]
    fn bench_case_11_firefighter_sunday_before() {
        let result = temporal::resolve("the sunday before 3 July 2023", date(2023, 7, 10)).unwrap();
        assert_date(&result, date(2023, 7, 2));
    }

    // =========================================================================
    // Case 12 (R1): John's 5K charity run — "first weekend of August 2023"
    //
    // Gold: "first weekend of August 2023"
    // Predicted: August 9, 2023 (a Wednesday — wrong)
    //
    // First weekend of August 2023 = Sat Aug 5 – Sun Aug 6.
    // Agent returned a weekday in the second week. Our resolver should nail
    // this since we have the nth_weekend_of_month pattern.
    // =========================================================================

    #[test]
    fn bench_case_12_charity_run_first_weekend() {
        let result = temporal::resolve("first weekend of August 2023", date(2023, 8, 10)).unwrap();
        assert_range(&result, date(2023, 8, 5), date(2023, 8, 6));
    }
}
