//! Embedding and enrichment generation.
//!
//! Shared logic for `memoryco generate`. Extracted from server.rs so it can
//! be called synchronously from the CLI (with progress output) rather than
//! exclusively in a background thread at server startup.

use crate::embedding::EmbeddingGenerator;
use crate::memory_core::Brain;
use std::io::Write;
use std::sync::{Arc, RwLock};

/// Stats returned by [`generate_embeddings`].
pub struct EmbeddingStats {
    pub generated: usize,
    pub errors: usize,
}

/// Stats returned by [`generate_enrichments`].
pub struct EnrichmentStats {
    pub generated: usize,
    pub total: usize,
    /// Total enrichment vectors stored after generation.
    pub vectors: usize,
}

/// Generate embeddings for every memory that is missing one.
///
/// Uses read locks throughout — safe to call while the server is running.
/// When `show_progress` is true, prints a `\r`-overwritten progress line to
/// stderr and a final newline on completion.
pub fn generate_embeddings(brain: Arc<RwLock<Brain>>, show_progress: bool) -> EmbeddingStats {
    let count = brain
        .read()
        .ok()
        .and_then(|b| b.count_without_embeddings().ok())
        .unwrap_or(0);

    if count == 0 {
        return EmbeddingStats {
            generated: 0,
            errors: 0,
        };
    }

    let generator = EmbeddingGenerator::new();
    let mut processed = 0usize;
    let mut errors = 0usize;

    loop {
        let ids = match brain
            .read()
            .ok()
            .and_then(|b| b.get_ids_without_embeddings(50).ok())
        {
            Some(ids) if ids.is_empty() => break,
            Some(ids) => ids,
            None => break,
        };

        let items: Vec<_> = match brain.read() {
            Ok(b) => ids
                .iter()
                .filter_map(|id| b.get(id).map(|e| (*id, e.content.clone())))
                .collect(),
            Err(_) => break,
        };

        let texts: Vec<&str> = items.iter().map(|(_, c)| c.as_str()).collect();
        match generator.generate_batch(&texts) {
            Ok(embeddings) => {
                if let Ok(b) = brain.read() {
                    for ((id, _), embedding) in items.iter().zip(embeddings.iter()) {
                        if b.set_embedding(id, embedding).is_ok() {
                            processed += 1;
                        } else {
                            errors += 1;
                        }
                    }
                }
            }
            Err(e) => {
                if show_progress {
                    eprintln!("\n  Batch error: {}", e);
                }
                errors += items.len();
                if errors > 100 {
                    if show_progress {
                        eprintln!("  Too many errors — aborting embedding generation");
                    }
                    break;
                }
            }
        }

        if show_progress {
            let pct = if count > 0 {
                (processed * 100) / count
            } else {
                100
            };
            eprint!(
                "\rGenerating embeddings... {}/{} ({}%)   ",
                processed, count, pct
            );
            std::io::stderr().flush().ok();
        }
    }

    if show_progress {
        eprintln!(); // newline after final progress line
    }

    EmbeddingStats {
        generated: processed,
        errors,
    }
}

/// Generate enrichment embeddings for memories that don't have them yet.
///
/// Enrichments are multi-vector representations built from LLM-generated
/// training queries. They require an available LLM service.
///
/// Skips memories that already have enrichments — safe to interrupt and re-run.
/// Uses read locks throughout — safe to call while the server is running.
/// When `show_progress` is true, prints a `\r`-overwritten progress line to
/// stderr and a final newline on completion.
pub fn generate_enrichments(
    brain: Arc<RwLock<Brain>>,
    llm: crate::llm::SharedLlmService,
    show_progress: bool,
) -> EnrichmentStats {
    // Only fetch IDs that have no enrichments yet — skip already-enriched memories.
    let ids: Vec<crate::memory_core::MemoryId> = match brain.read() {
        Ok(b) => b.get_ids_without_enrichments().unwrap_or_default(),
        Err(_) => {
            return EnrichmentStats {
                generated: 0,
                total: 0,
                vectors: 0,
            }
        }
    };

    if ids.is_empty() {
        return EnrichmentStats {
            generated: 0,
            total: 0,
            vectors: 0,
        };
    }

    let total = ids.len();
    let generator = EmbeddingGenerator::new();
    let mut enriched = 0usize;

    for (i, id) in ids.iter().enumerate() {
        let content = match brain.read() {
            Ok(b) => match b.get(id) {
                Some(e) => e.content.clone(),
                None => continue,
            },
            Err(_) => break,
        };

        if let Ok(queries) = llm.generate_training_queries(&content, 3) {
            let embeddings: Vec<Vec<f32>> = queries
                .iter()
                .filter_map(|q| generator.generate(q).ok())
                .collect();
            if !embeddings.is_empty() {
                if let Ok(b) = brain.read() {
                    let _ = b.set_enrichment_embeddings(id, &embeddings, "llm");
                    enriched += 1;
                }
            }
        }

        if show_progress {
            let pct = ((i + 1) * 100) / total.max(1);
            eprint!(
                "\rGenerating enrichments... {}/{} ({}%)   ",
                i + 1,
                total,
                pct
            );
            std::io::stderr().flush().ok();
        }
    }

    if show_progress {
        eprintln!();
    }

    let vectors = brain
        .read()
        .ok()
        .and_then(|b| b.count_enrichments().ok())
        .unwrap_or(0);

    EnrichmentStats {
        generated: enriched,
        total,
        vectors,
    }
}
