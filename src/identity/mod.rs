//! Identity module - classification and management
//!
//! This module handles persistent identity storage and classification.
//! Identity items are stored in a flat table with typed rows.
//!
//! Identity is the bedrock layer of self - unlike episodic/semantic memories,
//! identity doesn't decay. It's not "remembered" - it just IS.

mod classifier;
mod types;
mod store;
pub mod storage;

pub use classifier::{classify, ClassificationResult, IdentityField};
pub use storage::{IdentityStorage, DieselIdentityStorage, IdentityItemType, IdentityItemRow};
pub use store::{IdentityStore, ListedItem, MigrationResult};
pub use types::{
    Identity, Persona, Value, Preference, Relationship, 
    Antipattern, CommunicationStyle, IdentitySearchResults,
};
