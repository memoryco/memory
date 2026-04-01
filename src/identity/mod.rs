//! Identity module - classification and management
//!
//! This module handles persistent identity storage and classification.
//! Identity items are stored in a flat table with typed rows.
//!
//! Identity is the bedrock layer of self - unlike episodic/semantic memories,
//! identity doesn't decay. It's not "remembered" - it just IS.

mod classifier;
pub mod storage;
mod store;
mod types;

pub use classifier::{IdentityField, classify};
pub use storage::{DieselIdentityStorage, IdentityItemType};
pub use store::{IdentityStore, ListedItem, MigrationResult};
#[allow(unused_imports)] // Public API surface — used by consumers of the crate
pub use types::{
    Antipattern, CommunicationStyle, Identity, IdentitySearchResults, Persona, Preference,
    Relationship, Value,
};
