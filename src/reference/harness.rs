//! Reference pipeline test harness.
//!
//! Runs every file in a test corpus through each stage of the reference
//! ingestion pipeline, catching panics and errors per-file so one bad
//! apple doesn't kill the whole run.
//!
//! Run with:
//!   cargo test -p memoryco reference_harness -- --nocapture
//!
//! Set HARNESS_DIR to override the default corpus path:
//!   HARNESS_DIR=/some/path cargo test -p memoryco reference_harness -- --nocapture

#[cfg(test)]
mod tests {
    use crate::reference::extractor::{default_extractor, PdfExtractor};
    use crate::reference::indexer::Indexer;
    use crate::reference::profiles::ProfileRegistry;
    use crate::reference::sanitize::{sanitize_filename, validate_pdf};
    use crate::reference::searcher::Searcher;
    use crate::reference::source::ReferenceSource;
    use std::fmt;
    use std::fs;
    use std::panic;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    const DEFAULT_CORPUS: &str = "/Users/bsneed/work/memoryco/ingrid_references";

    // ── Stage results ──────────────────────────────────────────────

    #[derive(Debug, Clone)]
    enum StageResult {
        Pass(String),
        Fail(String),
        Skip(String),
        Panic(String),
    }

    impl StageResult {
        fn is_pass(&self) -> bool {
            matches!(self, StageResult::Pass(_))
        }

        fn is_fail(&self) -> bool {
            matches!(self, StageResult::Fail(_) | StageResult::Panic(_))
        }

        fn icon(&self) -> &'static str {
            match self {
                StageResult::Pass(_) => "✅",
                StageResult::Fail(_) => "❌",
                StageResult::Skip(_) => "⏭️ ",
                StageResult::Panic(_) => "💥",
            }
        }
    }

    impl fmt::Display for StageResult {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let (label, msg) = match self {
                StageResult::Pass(m) => ("PASS", m.as_str()),
                StageResult::Fail(m) => ("FAIL", m.as_str()),
                StageResult::Skip(m) => ("SKIP", m.as_str()),
                StageResult::Panic(m) => ("PANIC", m.as_str()),
            };
            write!(f, "[{}] {}", label, msg)
        }
    }

    // ── Per-file report ────────────────────────────────────────────

    struct FileReport {
        original_name: String,
        stages: Vec<(&'static str, StageResult)>,
    }

    impl FileReport {
        fn new(name: &str) -> Self {
            Self {
                original_name: name.to_string(),
                stages: Vec::new(),
            }
        }

        fn add(&mut self, stage: &'static str, result: StageResult) {
            self.stages.push((stage, result));
        }

        fn has_failures(&self) -> bool {
            self.stages.iter().any(|(_, r)| r.is_fail())
        }

        fn print(&self) {
            let status = if self.has_failures() { "💀" } else { "👍" };
            println!("\n{} {}", status, self.original_name);
            println!("{}", "─".repeat(60));
            for (stage, result) in &self.stages {
                println!("  {} {}: {}", result.icon(), stage, result);
            }
        }
    }

    // ── Stage runners ──────────────────────────────────────────────

    /// Stage 1: Can we get a valid UTF-8 filename from the dir entry?
    fn stage_filename(path: &Path) -> StageResult {
        match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => StageResult::Pass(format!("\"{}\"", name)),
            None => StageResult::Fail("file_name() returned None or invalid UTF-8".into()),
        }
    }

    /// Stage 2: Does sanitize_filename produce something reasonable?
    fn stage_sanitize(filename: &str) -> StageResult {
        let result = panic::catch_unwind(|| sanitize_filename(filename));
        match result {
            Ok(sanitized) => {
                if sanitized == "unnamed_reference.pdf" && !filename.is_empty() {
                    StageResult::Fail(format!(
                        "sanitized to fallback 'unnamed_reference.pdf' from \"{}\"",
                        filename
                    ))
                } else {
                    StageResult::Pass(format!("\"{}\" → \"{}\"", filename, sanitized))
                }
            }
            Err(e) => {
                let msg = panic_message(&e);
                StageResult::Panic(format!("sanitize_filename panicked: {}", msg))
            }
        }
    }

    /// Stage 3: Is this a valid PDF? (extension + magic bytes)
    fn stage_validate_pdf(path: &Path) -> StageResult {
        match validate_pdf(path) {
            Ok(()) => StageResult::Pass("valid PDF (extension + magic bytes)".into()),
            Err(e) => StageResult::Fail(format!("{}", e)),
        }
    }

    /// Stage 4: Can pdf-extract read the content?
    fn stage_extract(path: &Path, extractor: &dyn PdfExtractor) -> StageResult {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| extractor.extract(path)));
        match result {
            Ok(Ok(pages)) => {
                let total_chars: usize = pages.iter().map(|p| p.text.len()).sum();
                StageResult::Pass(format!(
                    "{} pages, {} chars total",
                    pages.len(),
                    total_chars
                ))
            }
            Ok(Err(e)) => StageResult::Fail(format!("extraction error: {}", e)),
            Err(e) => {
                let msg = panic_message(&e);
                StageResult::Panic(format!("pdf-extract panicked: {}", msg))
            }
        }
    }

    /// Stage 5: Can we build an FTS5 index?
    /// Copies the PDF to a temp dir with sanitized name first.
    fn stage_index(
        path: &Path,
        sanitized_name: &str,
        tmp: &Path,
        extractor: &dyn PdfExtractor,
        profiles: &ProfileRegistry,
    ) -> (StageResult, Option<PathBuf>) {
        // Copy to temp dir with sanitized name
        let dest = tmp.join(sanitized_name);
        if let Err(e) = fs::copy(path, &dest) {
            return (
                StageResult::Fail(format!("failed to copy to temp dir: {}", e)),
                None,
            );
        }

        let indexer = Indexer::new(extractor, profiles);
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| indexer.build(&dest)));
        match result {
            Ok(Ok(())) => {
                let idx_path = dest.with_extension("idx");
                let idx_size = fs::metadata(&idx_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                (
                    StageResult::Pass(format!(
                        "index built ({} bytes)",
                        idx_size
                    )),
                    Some(idx_path),
                )
            }
            Ok(Err(e)) => (StageResult::Fail(format!("index error: {}", e)), None),
            Err(e) => {
                let msg = panic_message(&e);
                (
                    StageResult::Panic(format!("indexer panicked: {}", msg)),
                    None,
                )
            }
        }
    }

    /// Stage 6: Can we open the index and run a smoke-test search?
    fn stage_search(idx_path: &Path) -> StageResult {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let searcher = Searcher::open(idx_path)?;
            let results = searcher.search("the", 5)?;
            Ok::<usize, crate::reference::ReferenceError>(results.len())
        }));
        match result {
            Ok(Ok(count)) => StageResult::Pass(format!("{} results for smoke query", count)),
            Ok(Err(e)) => StageResult::Fail(format!("search error: {}", e)),
            Err(e) => {
                let msg = panic_message(&e);
                StageResult::Panic(format!("searcher panicked: {}", msg))
            }
        }
    }

    /// Stage 7: Full ReferenceSource round-trip (new → ensure_index → search)
    fn stage_full_roundtrip(
        sanitized_pdf: &Path,
        extractor: &dyn PdfExtractor,
        profiles: &ProfileRegistry,
    ) -> StageResult {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let mut source = ReferenceSource::new(sanitized_pdf);
            source.ensure_index(extractor, profiles)?;
            let sections = source.list_sections()?;
            let search = source.search("the", 3)?;
            Ok::<(usize, usize), crate::reference::ReferenceError>((
                sections.len(),
                search.len(),
            ))
        }));
        match result {
            Ok(Ok((sections, hits))) => StageResult::Pass(format!(
                "{} sections, {} search hits",
                sections, hits
            )),
            Ok(Err(e)) => StageResult::Fail(format!("roundtrip error: {}", e)),
            Err(e) => {
                let msg = panic_message(&e);
                StageResult::Panic(format!("roundtrip panicked: {}", msg))
            }
        }
    }

    // ── Helpers ────────────────────────────────────────────────────

    fn panic_message(e: &Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".into()
        }
    }

    // ── Main harness ───────────────────────────────────────────────

    #[test]
    #[ignore] // slow: processes entire PDF corpus; run explicitly with --ignored
    fn reference_harness() {
        let corpus_dir = std::env::var("HARNESS_DIR")
            .unwrap_or_else(|_| DEFAULT_CORPUS.to_string());
        let corpus = Path::new(&corpus_dir);

        println!("\n{}", "=".repeat(60));
        println!("  REFERENCE PIPELINE HARNESS");
        println!("  corpus: {}", corpus.display());
        println!("{}", "=".repeat(60));

        assert!(corpus.exists(), "Corpus directory does not exist: {}", corpus.display());

        // Collect all entries (not just .pdf — we want to see what gets filtered)
        let mut entries: Vec<PathBuf> = fs::read_dir(corpus)
            .expect("failed to read corpus directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();

        println!("  {} files found\n", entries.len());

        let extractor = default_extractor();
        let profiles = ProfileRegistry::new();
        let tmp = TempDir::new().expect("failed to create temp dir");
        let mut reports: Vec<FileReport> = Vec::new();

        for path in &entries {
            let filename = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => {
                    let mut report = FileReport::new(&format!("{}", path.display()));
                    report.add("1. filename", StageResult::Fail("invalid UTF-8".into()));
                    reports.push(report);
                    continue;
                }
            };

            let mut report = FileReport::new(&filename);

            // Stage 1: Filename
            report.add("1. filename", stage_filename(path));

            // Stage 2: Sanitize
            let sanitized = sanitize_filename(&filename);
            report.add("2. sanitize", stage_sanitize(&filename));

            // Skip non-PDF files after sanitize (we still want to see sanitize results for .idx etc)
            let is_pdf_ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("pdf"))
                .unwrap_or(false);

            if !is_pdf_ext {
                report.add("3. validate", StageResult::Skip("not a .pdf file".into()));
                report.add("4. extract", StageResult::Skip("not a .pdf file".into()));
                report.add("5. index", StageResult::Skip("not a .pdf file".into()));
                report.add("6. search", StageResult::Skip("not a .pdf file".into()));
                report.add("7. roundtrip", StageResult::Skip("not a .pdf file".into()));
                reports.push(report);
                continue;
            }

            // Stage 3: Validate PDF
            let valid = stage_validate_pdf(path);
            let pdf_valid = valid.is_pass();
            report.add("3. validate", valid);

            if !pdf_valid {
                report.add("4. extract", StageResult::Skip("invalid PDF".into()));
                report.add("5. index", StageResult::Skip("invalid PDF".into()));
                report.add("6. search", StageResult::Skip("invalid PDF".into()));
                report.add("7. roundtrip", StageResult::Skip("invalid PDF".into()));
                reports.push(report);
                continue;
            }

            // Stage 4: Extract text
            let extract = stage_extract(path, extractor.as_ref());
            let extract_ok = extract.is_pass();
            report.add("4. extract", extract);

            if !extract_ok {
                report.add("5. index", StageResult::Skip("extraction failed".into()));
                report.add("6. search", StageResult::Skip("extraction failed".into()));
                report.add("7. roundtrip", StageResult::Skip("extraction failed".into()));
                reports.push(report);
                continue;
            }

            // Stage 5: Build index (in temp dir with sanitized name)
            let (index_result, idx_path) =
                stage_index(path, &sanitized, tmp.path(), extractor.as_ref(), &profiles);
            let index_ok = index_result.is_pass();
            report.add("5. index", index_result);

            if !index_ok || idx_path.is_none() {
                report.add("6. search", StageResult::Skip("index build failed".into()));
                report.add("7. roundtrip", StageResult::Skip("index build failed".into()));
                reports.push(report);
                continue;
            }

            // Stage 6: Search smoke test
            let idx = idx_path.unwrap();
            report.add("6. search", stage_search(&idx));

            // Stage 7: Full ReferenceSource round-trip
            // Use the sanitized copy in temp dir
            let sanitized_pdf = tmp.path().join(&sanitized);
            report.add(
                "7. roundtrip",
                stage_full_roundtrip(&sanitized_pdf, extractor.as_ref(), &profiles),
            );

            reports.push(report);
        }

        // ── Print reports ──────────────────────────────────────────

        // Failures first
        let failures: Vec<&FileReport> = reports.iter().filter(|r| r.has_failures()).collect();
        let passes: Vec<&FileReport> = reports.iter().filter(|r| !r.has_failures()).collect();

        if !failures.is_empty() {
            println!("\n\n{}", "=".repeat(60));
            println!("  FAILURES ({} files)", failures.len());
            println!("{}", "=".repeat(60));
            for report in &failures {
                report.print();
            }
        }

        if !passes.is_empty() {
            println!("\n\n{}", "=".repeat(60));
            println!("  PASSES ({} files)", passes.len());
            println!("{}", "=".repeat(60));
            for report in &passes {
                report.print();
            }
        }

        // ── Summary ────────────────────────────────────────────────

        let total = reports.len();
        let failed_count = failures.len();
        let passed_count = passes.len();

        // Count failures by stage
        let stage_names = [
            "1. filename",
            "2. sanitize",
            "3. validate",
            "4. extract",
            "5. index",
            "6. search",
            "7. roundtrip",
        ];
        println!("\n\n{}", "=".repeat(60));
        println!("  SUMMARY");
        println!("{}", "=".repeat(60));
        println!("  Total files: {}", total);
        println!("  Clean:       {} ✅", passed_count);
        println!("  Broken:      {} ❌", failed_count);
        println!();

        for stage in &stage_names {
            let fails: usize = reports
                .iter()
                .filter(|r| {
                    r.stages
                        .iter()
                        .any(|(name, result)| name == stage && result.is_fail())
                })
                .count();
            let panics: usize = reports
                .iter()
                .filter(|r| {
                    r.stages
                        .iter()
                        .any(|(name, result)| {
                            name == stage && matches!(result, StageResult::Panic(_))
                        })
                })
                .count();
            if fails > 0 || panics > 0 {
                println!(
                    "  {}: {} failures, {} panics",
                    stage, fails, panics
                );
            }
        }

        println!("\n{}\n", "=".repeat(60));

        // Don't assert — we EXPECT failures, that's the point.
        // But print a clear status line.
        if failed_count > 0 {
            println!(
                "⚠️  {} of {} files have issues to investigate.\n",
                failed_count, total
            );
        } else {
            println!("🎉 All {} files passed cleanly!\n", total);
        }
    }
}
