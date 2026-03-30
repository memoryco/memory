//! Diesel model types for memory database rows
//!
//! These are the DB-layer representations. They get converted to/from
//! the domain types in `crate::memory_core::*`.

use super::schema::*;
use diesel::prelude::*;

// ============================================================================
// MEMORIES
// ============================================================================

/// Queryable memory row (SELECT)
/// Field order MUST match schema.rs column order
#[allow(dead_code)] // Fields mapped from DB schema
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = memories)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct MemoryRow {
    pub id: String,
    pub content: String,
    pub energy: f32,
    pub state: String,
    pub confidence: f32,
    pub created_at: i64,
    pub last_accessed: i64,
    pub access_count: i64,
    pub tags: String, // JSON array
    pub embedding: Option<Vec<u8>>,
}

/// Insertable memory (INSERT/REPLACE)
#[derive(Debug, Clone, Insertable, AsChangeset)]
#[diesel(table_name = memories)]
pub struct NewMemory<'a> {
    pub id: &'a str,
    pub content: &'a str,
    pub energy: f32,
    pub state: &'a str,
    pub confidence: f32,
    pub created_at: i64,
    pub last_accessed: i64,
    pub access_count: i64,
    pub tags: String, // JSON array
    pub embedding: Option<Vec<u8>>,
}

// ============================================================================
// ASSOCIATIONS
// ============================================================================

/// Queryable association row (SELECT)
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = associations)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct AssociationRow {
    pub from_id: String,
    pub to_id: String,
    pub weight: f32,
    pub created_at: i64,
    pub last_activated: i64,
    pub co_activation_count: i64,
    pub ordinal: Option<i32>,
}

/// Insertable association (INSERT/REPLACE)
#[derive(Debug, Clone, Insertable, AsChangeset)]
#[diesel(table_name = associations)]
pub struct NewAssociation<'a> {
    pub from_id: &'a str,
    pub to_id: &'a str,
    pub weight: f32,
    pub created_at: i64,
    pub last_activated: i64,
    pub co_activation_count: i64,
    pub ordinal: Option<i32>,
}

// ============================================================================
// CONFIG (single-row JSON blob)
// ============================================================================

/// Queryable config row
#[allow(dead_code)] // Fields mapped from DB schema
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = config)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct ConfigRow {
    pub id: i32,
    pub data: String, // JSON
}

/// Insertable config
#[derive(Debug, Clone, Insertable, AsChangeset)]
#[diesel(table_name = config)]
pub struct NewConfig<'a> {
    pub id: i32,
    pub data: &'a str, // JSON
}

// ============================================================================
// IDENTITY (single-row JSON blob)
// ============================================================================

/// Queryable identity row
#[allow(dead_code)] // Fields mapped from DB schema
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = identity)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct IdentityRow {
    pub id: i32,
    pub data: String, // JSON
}

/// Insertable identity
#[derive(Debug, Clone, Insertable, AsChangeset)]
#[diesel(table_name = identity)]
pub struct NewIdentity<'a> {
    pub id: i32,
    pub data: &'a str, // JSON
}

// ============================================================================
// METADATA (key-value store)
// ============================================================================

/// Queryable metadata row
#[allow(dead_code)] // Fields mapped from DB schema
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = metadata)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct MetadataRow {
    pub key: String,
    pub value: String,
}

/// Insertable metadata
#[derive(Debug, Clone, Insertable, AsChangeset)]
#[diesel(table_name = metadata)]
pub struct NewMetadata<'a> {
    pub key: &'a str,
    pub value: &'a str,
}

// ============================================================================
// ACCESS LOG
// ============================================================================

/// Insertable access log entry
#[derive(Debug, Clone, Insertable)]
#[diesel(table_name = access_log)]
pub struct NewAccessLogEntry<'a> {
    pub timestamp: i64,
    pub query_text: &'a str,
    pub result_ids: &'a str,   // JSON array of UUID strings
    pub recalled_ids: &'a str, // JSON array of UUID strings
}

// ============================================================================
// CONVERSIONS: DB rows -> Domain types
// ============================================================================

use crate::memory_core::{Association, Memory, MemoryState};
use crate::storage::StorageError;

impl MemoryRow {
    /// Convert DB row to domain Memory
    pub fn into_memory(self) -> Result<Memory, StorageError> {
        let id = uuid::Uuid::parse_str(&self.id)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let state = match self.state.as_str() {
            "active" => MemoryState::Active,
            "dormant" => MemoryState::Dormant,
            "deep" => MemoryState::Deep,
            "archived" => MemoryState::Archived,
            _ => {
                return Err(StorageError::Serialization(format!(
                    "Unknown state: {}",
                    self.state
                )));
            }
        };

        let tags: Vec<String> = serde_json::from_str(&self.tags)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        Ok(Memory {
            id,
            content: self.content,
            energy: self.energy as f64,
            state,
            confidence: self.confidence as f64,
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            access_count: self.access_count as u64,
            tags,
            embedding: None, // Loaded separately when needed
        })
    }
}

impl AssociationRow {
    /// Convert DB row to domain Association
    pub fn into_association(self) -> Result<Association, StorageError> {
        let from = uuid::Uuid::parse_str(&self.from_id)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let to = uuid::Uuid::parse_str(&self.to_id)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        Ok(Association {
            from,
            to,
            weight: self.weight as f64,
            created_at: self.created_at,
            last_activated: self.last_activated,
            co_activation_count: self.co_activation_count as u64,
            ordinal: self.ordinal.map(|v| v as u32),
        })
    }
}

// ============================================================================
// EMBEDDING HELPERS
// ============================================================================

/// Serialize embedding to bytes
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize embedding from bytes
pub fn bytes_to_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// Get state string from MemoryState enum
pub fn state_to_str(state: MemoryState) -> &'static str {
    match state {
        MemoryState::Active => "active",
        MemoryState::Dormant => "dormant",
        MemoryState::Deep => "deep",
        MemoryState::Archived => "archived",
    }
}
