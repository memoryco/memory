//! Filename sanitization and PDF validation for the reference system.
//!
//! Handles filenames from user-dropped files that may contain URL prefixes,
//! query parameters, percent-encoded characters, and other path-unsafe content
//! that would break JSON parsing or filesystem operations.

use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SanitizeError {
    #[error("not a PDF file (missing .pdf extension)")]
    NotAPdf,

    #[error("invalid PDF: missing %PDF- magic bytes")]
    InvalidMagicBytes,

    #[error("file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("empty filename after sanitization")]
    EmptyFilename,
}

/// Decode percent-encoded sequences in a string (e.g., `%E2%80%90` → actual bytes).
/// Also decodes `+` as space (URL form encoding).
fn percent_decode(input: &str) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut chars = input.bytes();

    while let Some(b) = chars.next() {
        match b {
            b'%' => {
                let hi = chars.next();
                let lo = chars.next();
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    if let (Some(h), Some(l)) = (hex_val(hi), hex_val(lo)) {
                        bytes.push(h << 4 | l);
                        continue;
                    }
                    // Not valid hex — emit the raw bytes
                    bytes.push(b'%');
                    bytes.push(hi);
                    bytes.push(lo);
                } else {
                    bytes.push(b'%');
                    if let Some(hi) = hi {
                        bytes.push(hi);
                    }
                }
            }
            b'+' => bytes.push(b' '),
            _ => bytes.push(b),
        }
    }

    String::from_utf8_lossy(&bytes).into_owned()
}

/// Convert an ASCII hex digit to its numeric value.
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Characters that are dangerous in filenames or break JSON parsing.
const DANGEROUS_CHARS: &[char] = &[':', '?', '&', '#', '\\', '/', '"', '\''];

/// Sanitize a filename for safe filesystem and JSON usage.
///
/// Strips URL prefixes, replaces dangerous characters, decodes percent-encoded
/// sequences, collapses redundant separators, and preserves the `.pdf` extension.
pub fn sanitize_filename(name: &str) -> String {
    // Step 1: Percent-decode first so encoded chars get sanitized too
    let decoded = percent_decode(name);

    // Step 2: Strip URL scheme prefixes (blob:https://..., https://..., etc.)
    // Handle nested schemes like blob:https:...
    let stripped = strip_url_prefix(&decoded);

    // Step 3: Separate the .pdf extension (case-insensitive) before sanitizing
    let (stem, ext) = split_pdf_extension(stripped);

    // Step 4: Replace dangerous characters with underscore
    let mut sanitized = String::with_capacity(stem.len());
    for ch in stem.chars() {
        if DANGEROUS_CHARS.contains(&ch) || ch.is_control() {
            sanitized.push('_');
        } else {
            sanitized.push(ch);
        }
    }

    // Step 5: Collapse consecutive underscores and spaces into single underscore
    let collapsed = collapse_separators(&sanitized);

    // Step 6: Trim leading/trailing underscores and whitespace
    let trimmed = collapsed.trim_matches(|c: char| c == '_' || c.is_whitespace());

    // Step 7: Reassemble with extension, or use fallback
    if trimmed.is_empty() {
        "unnamed_reference.pdf".to_string()
    } else {
        format!("{}{}", trimmed, ext)
    }
}

/// Strip URL scheme prefixes from a string.
/// Handles `blob:https:...`, `https://...`, `http:...`, `file:///...`, etc.
fn strip_url_prefix(s: &str) -> &str {
    let mut s = s;

    // Strip blob: prefix first
    if let Some(rest) = s.strip_prefix("blob:") {
        s = rest;
    }

    // Strip scheme prefixes (https:, http:, file:)
    // Handle both `https://` and `https:` (colons used as separators)
    for scheme in &["https://", "http://", "file:///", "file://", "https:", "http:", "file:"] {
        if let Some(rest) = s.strip_prefix(scheme) {
            s = rest;
            break;
        }
    }

    s
}

/// Split a filename into (stem, extension) where extension is `.pdf` (case-insensitive).
/// If the name doesn't end with .pdf, returns the whole name and `.pdf` as fallback.
fn split_pdf_extension(name: &str) -> (&str, &str) {
    if name.len() >= 4 {
        let (stem, ext) = name.split_at(name.len() - 4);
        if ext.eq_ignore_ascii_case(".pdf") {
            return (stem, ".pdf");
        }
    }
    // No .pdf extension found — treat entire string as stem
    (name, ".pdf")
}

/// Collapse consecutive underscores and/or whitespace into a single underscore.
fn collapse_separators(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_was_sep = false;

    for ch in s.chars() {
        if ch == '_' || ch.is_whitespace() {
            if !prev_was_sep {
                result.push('_');
            }
            prev_was_sep = true;
        } else {
            prev_was_sep = false;
            result.push(ch);
        }
    }

    result
}

/// Validate that a file is a PDF.
///
/// Checks existence, `.pdf` extension (case-insensitive), and `%PDF-` magic bytes.
pub fn validate_pdf(path: &Path) -> Result<(), SanitizeError> {
    if !path.exists() {
        return Err(SanitizeError::FileNotFound(path.to_path_buf()));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if !ext.eq_ignore_ascii_case("pdf") {
        return Err(SanitizeError::NotAPdf);
    }

    // Check magic bytes
    let mut buf = [0u8; 5];
    let mut file = std::fs::File::open(path)?;
    std::io::Read::read_exact(&mut file, &mut buf)?;
    if &buf != b"%PDF-" {
        return Err(SanitizeError::InvalidMagicBytes);
    }

    Ok(())
}

/// Validate, sanitize, and copy a PDF to a destination directory.
///
/// Returns the path of the new file. Handles duplicate filenames by appending
/// `_2`, `_3`, etc. before the extension.
pub fn sanitize_and_copy(source: &Path, dest_dir: &Path) -> Result<PathBuf, SanitizeError> {
    validate_pdf(source)?;

    let original_name = source
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or(SanitizeError::EmptyFilename)?;

    let sanitized = sanitize_filename(original_name);

    // Find a non-colliding name
    let dest_path = deduplicate_path(dest_dir, &sanitized);

    std::fs::copy(source, &dest_path)?;

    Ok(dest_path)
}

/// Generate a unique path in `dir` for `filename`, appending `_2`, `_3`, etc. if needed.
fn deduplicate_path(dir: &Path, filename: &str) -> PathBuf {
    let candidate = dir.join(filename);
    if !candidate.exists() {
        return candidate;
    }

    let (stem, ext) = split_pdf_extension(filename);

    let mut counter = 2u32;
    loop {
        let new_name = format!("{}_{}{}", stem, counter, ext);
        let path = dir.join(&new_name);
        if !path.exists() {
            return path;
        }
        counter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    // --- sanitize_filename tests ---

    #[test]
    fn normal_filename_passes_through() {
        assert_eq!(sanitize_filename("CBT_workbook.pdf"), "CBT_workbook.pdf");
    }

    #[test]
    fn url_blob_prefix_stripped() {
        let input = "blob:https:patient.labcorp.com:8f4ce6f2-25a3-4135-8fa9-869cc6372d52.pdf";
        let result = sanitize_filename(input);
        assert!(!result.contains("blob"));
        assert!(!result.contains("https"));
        assert!(result.ends_with(".pdf"));
        assert!(!result.contains(':'));
    }

    #[test]
    fn url_with_query_params() {
        let input = "https:reserves.usc.edu:ares:ares.dll?SessionID=H123818983B&Action=10&Type=10&Value=410416.pdf";
        let result = sanitize_filename(input);
        assert!(result.ends_with(".pdf"));
        assert!(!result.contains('?'));
        assert!(!result.contains('&'));
        assert!(!result.contains(':'));
    }

    #[test]
    fn hash_prefix() {
        let result = sanitize_filename("#10   PST Training.pdf");
        assert!(result.ends_with(".pdf"));
        assert!(!result.contains('#'));
        // Multiple spaces should collapse
        assert!(!result.contains("  "));
        // Should look something like "10_PST_Training.pdf"
        assert!(result.contains("10"));
        assert!(result.contains("PST"));
        assert!(result.contains("Training"));
    }

    #[test]
    fn url_encoded_characters() {
        let result = sanitize_filename("%E2%80%90analysis.pdf");
        assert!(result.ends_with(".pdf"));
        // The percent-encoded chars should be decoded, not raw
        assert!(!result.contains('%'));
        assert!(result.contains("analysis"));
    }

    #[test]
    fn plus_signs_decoded() {
        let result = sanitize_filename("UNIT+3_NEEDS+ASSESSMENT.pdf");
        assert!(result.ends_with(".pdf"));
        assert!(!result.contains('+'));
        assert!(result.contains("UNIT"));
        assert!(result.contains("NEEDS"));
        assert!(result.contains("ASSESSMENT"));
    }

    #[test]
    fn multiple_consecutive_special_chars_collapse() {
        assert_eq!(sanitize_filename("foo:::bar???baz.pdf"), "foo_bar_baz.pdf");
    }

    #[test]
    fn empty_input_gives_fallback() {
        assert_eq!(sanitize_filename(""), "unnamed_reference.pdf");
    }

    #[test]
    fn whitespace_only_gives_fallback() {
        assert_eq!(sanitize_filename("   "), "unnamed_reference.pdf");
    }

    #[test]
    fn just_pdf_extension_gives_fallback() {
        assert_eq!(sanitize_filename(".pdf"), "unnamed_reference.pdf");
    }

    #[test]
    fn leading_trailing_whitespace_trimmed() {
        let result = sanitize_filename("  Basic Risk Assessment Flow Chart.pdf");
        assert_eq!(result, "Basic_Risk_Assessment_Flow_Chart.pdf");
    }

    #[test]
    fn only_dangerous_chars_gives_fallback() {
        assert_eq!(sanitize_filename(":::???.pdf"), "unnamed_reference.pdf");
    }

    #[test]
    fn https_double_slash_prefix() {
        let result = sanitize_filename("https://example.com/docs/file.pdf");
        assert!(result.ends_with(".pdf"));
        assert!(!result.contains("https"));
        assert!(!result.contains("//"));
    }

    #[test]
    fn file_scheme_prefix() {
        let result = sanitize_filename("file:///home/user/docs/report.pdf");
        assert!(result.ends_with(".pdf"));
        assert!(!result.contains("file"));
        assert!(result.contains("report"));
    }

    // --- validate_pdf tests ---

    fn create_fake_pdf(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"%PDF-1.4 fake pdf content").unwrap();
        path
    }

    fn create_text_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"This is not a PDF file").unwrap();
        path
    }

    #[test]
    fn validate_valid_pdf() {
        let dir = TempDir::new().unwrap();
        let pdf = create_fake_pdf(dir.path(), "test.pdf");
        assert!(validate_pdf(&pdf).is_ok());
    }

    #[test]
    fn validate_invalid_magic_bytes() {
        let dir = TempDir::new().unwrap();
        let path = create_text_file(dir.path(), "fake.pdf");
        let err = validate_pdf(&path).unwrap_err();
        assert!(matches!(err, SanitizeError::InvalidMagicBytes));
    }

    #[test]
    fn validate_file_not_found() {
        let path = Path::new("/nonexistent/path/to/missing.pdf");
        let err = validate_pdf(path).unwrap_err();
        assert!(matches!(err, SanitizeError::FileNotFound(_)));
    }

    #[test]
    fn validate_non_pdf_extension() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("document.txt");
        std::fs::File::create(&path).unwrap();
        let err = validate_pdf(&path).unwrap_err();
        assert!(matches!(err, SanitizeError::NotAPdf));
    }

    #[test]
    fn validate_pdf_case_insensitive_extension() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.PDF");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"%PDF-1.7").unwrap();
        assert!(validate_pdf(&path).is_ok());
    }

    // --- sanitize_and_copy tests ---

    #[test]
    fn sanitize_and_copy_basic() {
        let src_dir = TempDir::new().unwrap();
        let dst_dir = TempDir::new().unwrap();

        let src = create_fake_pdf(src_dir.path(), "CBT_workbook.pdf");
        let result = sanitize_and_copy(&src, dst_dir.path()).unwrap();

        assert!(result.exists());
        assert_eq!(result.file_name().unwrap(), "CBT_workbook.pdf");
    }

    #[test]
    fn sanitize_and_copy_sanitizes_name() {
        let src_dir = TempDir::new().unwrap();
        let dst_dir = TempDir::new().unwrap();

        let src = create_fake_pdf(src_dir.path(), "foo:::bar.pdf");
        let result = sanitize_and_copy(&src, dst_dir.path()).unwrap();

        assert!(result.exists());
        assert_eq!(result.file_name().unwrap(), "foo_bar.pdf");
    }

    #[test]
    fn sanitize_and_copy_deduplicates() {
        let src_dir = TempDir::new().unwrap();
        let dst_dir = TempDir::new().unwrap();

        let src = create_fake_pdf(src_dir.path(), "report.pdf");

        let first = sanitize_and_copy(&src, dst_dir.path()).unwrap();
        assert_eq!(first.file_name().unwrap(), "report.pdf");

        let second = sanitize_and_copy(&src, dst_dir.path()).unwrap();
        assert_eq!(second.file_name().unwrap(), "report_2.pdf");

        let third = sanitize_and_copy(&src, dst_dir.path()).unwrap();
        assert_eq!(third.file_name().unwrap(), "report_3.pdf");
    }

    #[test]
    fn sanitize_and_copy_rejects_non_pdf() {
        let src_dir = TempDir::new().unwrap();
        let dst_dir = TempDir::new().unwrap();

        let src = create_text_file(src_dir.path(), "notes.txt");
        let err = sanitize_and_copy(&src, dst_dir.path()).unwrap_err();
        assert!(matches!(err, SanitizeError::NotAPdf));
    }

    #[test]
    fn sanitize_and_copy_rejects_missing_file() {
        let dst_dir = TempDir::new().unwrap();
        let src = Path::new("/nonexistent/missing.pdf");
        let err = sanitize_and_copy(src, dst_dir.path()).unwrap_err();
        assert!(matches!(err, SanitizeError::FileNotFound(_)));
    }

    // --- percent_decode tests ---

    #[test]
    fn percent_decode_basic() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
    }

    #[test]
    fn percent_decode_plus() {
        assert_eq!(percent_decode("hello+world"), "hello world");
    }

    #[test]
    fn percent_decode_invalid_hex_passthrough() {
        // %ZZ is not valid hex — should be left as-is
        assert_eq!(percent_decode("%ZZfoo"), "%ZZfoo");
    }

    #[test]
    fn percent_decode_utf8_sequence() {
        // %E2%80%90 is the UTF-8 encoding of U+2010 HYPHEN (‐)
        let result = percent_decode("%E2%80%90");
        assert_eq!(result, "\u{2010}");
    }
}
