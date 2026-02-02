//! Diesel models for identity items table
//!
//! Row types for database operations. The IdentityItemType enum
//! represents all valid identity item types.

use diesel::prelude::*;
use super::schema::identity_items;
use std::fmt;
use std::str::FromStr;

/// All valid identity item types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityItemType {
    // Persona (singular fields)
    PersonaName,
    PersonaDescription,
    
    // List types
    Trait,
    Value,
    Preference,
    Relationship,
    Antipattern,
    Expertise,
    Instruction,
    Tone,
    Directive,
}

impl IdentityItemType {
    /// All item types as a slice (useful for iteration)
    pub fn all() -> &'static [IdentityItemType] {
        use IdentityItemType::*;
        &[
            PersonaName, PersonaDescription,
            Trait, Value, Preference, Relationship,
            Antipattern, Expertise, Instruction, Tone, Directive,
        ]
    }
}

impl fmt::Display for IdentityItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IdentityItemType::PersonaName => "persona_name",
            IdentityItemType::PersonaDescription => "persona_description",
            IdentityItemType::Trait => "trait",
            IdentityItemType::Value => "value",
            IdentityItemType::Preference => "preference",
            IdentityItemType::Relationship => "relationship",
            IdentityItemType::Antipattern => "antipattern",
            IdentityItemType::Expertise => "expertise",
            IdentityItemType::Instruction => "instruction",
            IdentityItemType::Tone => "tone",
            IdentityItemType::Directive => "directive",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for IdentityItemType {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "persona_name" => Ok(IdentityItemType::PersonaName),
            "persona_description" => Ok(IdentityItemType::PersonaDescription),
            "trait" => Ok(IdentityItemType::Trait),
            "value" => Ok(IdentityItemType::Value),
            "preference" => Ok(IdentityItemType::Preference),
            "relationship" => Ok(IdentityItemType::Relationship),
            "antipattern" => Ok(IdentityItemType::Antipattern),
            "expertise" => Ok(IdentityItemType::Expertise),
            "instruction" => Ok(IdentityItemType::Instruction),
            "tone" => Ok(IdentityItemType::Tone),
            "directive" => Ok(IdentityItemType::Directive),
            _ => Err(format!("Unknown identity item type: {}", s)),
        }
    }
}

// ==================
// ROW TYPES
// ==================

/// Row type for reading from the database
#[derive(Queryable, Selectable, Debug, Clone)]
#[diesel(table_name = identity_items)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct IdentityItemRow {
    pub id: String,
    pub item_type: String,
    pub content: String,
    pub secondary: Option<String>,
    pub tertiary: Option<String>,
    pub category: Option<String>,
    pub created_at: i64,
}

impl IdentityItemRow {
    /// Parse the item_type string into the enum
    pub fn parsed_type(&self) -> Result<IdentityItemType, String> {
        self.item_type.parse()
    }
}

/// Row type for inserting into the database
#[derive(Insertable)]
#[diesel(table_name = identity_items)]
pub struct NewIdentityItem<'a> {
    pub id: &'a str,
    pub item_type: &'a str,
    pub content: &'a str,
    pub secondary: Option<&'a str>,
    pub tertiary: Option<&'a str>,
    pub category: Option<&'a str>,
    pub created_at: i64,
}
