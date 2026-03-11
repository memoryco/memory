//! Search interface for reference indexes.

use super::error::{ReferenceError, Result};
use rusqlite::Connection;
use std::path::Path;

/// A search result from the index.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Section or page title
    pub title: String,
    /// Parent section (if applicable)
    pub parent: Option<String>,
    /// Starting page
    pub page_start: usize,
    /// Ending page
    pub page_end: usize,
    /// ICD codes (if applicable)
    pub codes: Vec<String>,
    /// Section type
    pub section_type: String,
    /// Content snippet with highlights
    pub snippet: String,
    /// FTS5 rank score
    pub rank: f64,
}

/// Query interface for a reference index.
pub struct Searcher {
    conn: Connection,
    index_type: String,
}

impl Searcher {
    /// Open an existing index for searching.
    pub fn open(index_path: &Path) -> Result<Self> {
        if !index_path.exists() {
            return Err(ReferenceError::SourceNotFound(
                index_path.display().to_string(),
            ));
        }

        let conn = Connection::open(index_path)?;

        let index_type: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'index_type'",
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "pages".to_string());

        Ok(Self { conn, index_type })
    }

    /// Get the index type (sections or pages).
    pub fn index_type(&self) -> &str {
        &self.index_type
    }

    /// Get the profile ID if this was profile-indexed.
    pub fn profile(&self) -> Option<String> {
        self.conn
            .query_row("SELECT value FROM meta WHERE key = 'profile'", [], |row| {
                row.get(0)
            })
            .ok()
    }

    /// Search the index with FTS5.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // Use FTS5 query syntax - user can use AND, OR, NOT, phrases, etc.
        // snippet(fts, column_index, before, after, ellipsis, max_tokens)
        let mut stmt = self.conn.prepare(
            r#"
            SELECT 
                rowid,
                snippet(fts, 1, '>>>', '<<<', '...', 40) as snippet,
                rank
            FROM fts 
            WHERE fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
            "#,
        )?;

        let rows = stmt.query_map(rusqlite::params![query, limit], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (rowid, snippet, rank) = row?;

            if let Some(result) = self.fetch_result(rowid, snippet, rank)? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Search within a specific section type (only for section-indexed docs).
    pub fn search_type(
        &self,
        query: &str,
        section_type: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if self.index_type != "sections" {
            return self.search(query, limit);
        }

        let mut stmt = self.conn.prepare(
            r#"
            SELECT 
                fts.rowid,
                snippet(fts, 1, '>>>', '<<<', '...', 32) as snippet,
                fts.rank
            FROM fts
            JOIN sections ON sections.id = fts.rowid
            WHERE fts MATCH ?1 AND sections.section_type = ?2
            ORDER BY fts.rank
            LIMIT ?3
            "#,
        )?;

        let rows = stmt.query_map(rusqlite::params![query, section_type, limit], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (rowid, snippet, rank) = row?;
            if let Some(result) = self.fetch_result(rowid, snippet, rank)? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Get a section by exact title.
    pub fn get_section(&self, title: &str) -> Result<Option<SearchResult>> {
        if self.index_type != "sections" {
            return Ok(None);
        }

        let result = self.conn.query_row(
            r#"
            SELECT id, title, parent, page_start, page_end, codes, section_type, content
            FROM sections
            WHERE title = ?1
            "#,
            [title],
            |row| {
                Ok(SearchResult {
                    title: row.get(1)?,
                    parent: row.get(2)?,
                    page_start: row.get::<_, i64>(3)? as usize,
                    page_end: row.get::<_, i64>(4)? as usize,
                    codes: parse_codes(&row.get::<_, String>(5)?),
                    section_type: row.get(6)?,
                    snippet: row.get(7)?, // Full content
                    rank: 0.0,
                })
            },
        );

        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all top-level sections (disorders for DSM).
    pub fn list_sections(&self) -> Result<Vec<String>> {
        if self.index_type != "sections" {
            return Ok(Vec::new());
        }

        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT title FROM sections WHERE parent IS NULL ORDER BY title")?;

        let titles = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(titles)
    }

    /// Fetch full result data from rowid.
    fn fetch_result(&self, rowid: i64, snippet: String, rank: f64) -> Result<Option<SearchResult>> {
        if rowid > 0 && self.index_type == "sections" {
            // Section result
            let result = self.conn.query_row(
                r#"
                SELECT title, parent, page_start, page_end, codes, section_type
                FROM sections WHERE id = ?1
                "#,
                [rowid],
                |row| {
                    Ok(SearchResult {
                        title: row.get(0)?,
                        parent: row.get(1)?,
                        page_start: row.get::<_, i64>(2)? as usize,
                        page_end: row.get::<_, i64>(3)? as usize,
                        codes: parse_codes(&row.get::<_, String>(4)?),
                        section_type: row.get(5)?,
                        snippet,
                        rank,
                    })
                },
            );
            Ok(result.ok())
        } else {
            // Page result (negative rowid)
            let page_num = (-rowid) as usize;
            Ok(Some(SearchResult {
                title: format!("Page {}", page_num),
                parent: None,
                page_start: page_num,
                page_end: page_num,
                codes: Vec::new(),
                section_type: "page".to_string(),
                snippet,
                rank,
            }))
        }
    }
}

fn parse_codes(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}
