//! IdentityStore - The coordination layer for identity
//!
//! IdentityStore wraps storage and provides the high-level typed API
//! for managing identity items. The storage layer is dumb (just rows),
//! this layer knows about the semantics of each identity type.

use super::types::{
    Identity, Value, Preference, Relationship,
    Antipattern,
};
use super::storage::{
    IdentityStorage, IdentityItemType, IdentityItemRow, StorageResult,
};

/// A listed identity item with its ID for modification
#[derive(Debug, Clone)]
pub struct ListedItem {
    pub id: String,
    pub item_type: IdentityItemType,
    pub content: String,
    pub secondary: Option<String>,
    pub tertiary: Option<String>,
    pub category: Option<String>,
}

impl From<IdentityItemRow> for ListedItem {
    fn from(row: IdentityItemRow) -> Self {
        let item_type = row.parsed_type().unwrap_or(IdentityItemType::Instruction);
        Self {
            id: row.id,
            item_type,
            content: row.content,
            secondary: row.secondary,
            tertiary: row.tertiary,
            category: row.category,
        }
    }
}

/// IdentityStore - high-level typed API for identity management
///
/// Provides typed add methods for each identity type, generic remove,
/// list by type, and assembly of the full Identity struct.
pub struct IdentityStore {
    storage: Box<dyn IdentityStorage>,
}

impl IdentityStore {
    /// Create a new IdentityStore with the given storage backend
    pub fn new<S: IdentityStorage + 'static>(mut storage: S) -> StorageResult<Self> {
        storage.initialize()?;
        Ok(Self {
            storage: Box::new(storage),
        })
    }
    
    // ==================
    // GET / LIST / REMOVE
    // ==================
    
    /// Get the full assembled Identity
    /// This is what identity_get() returns - the nice structured view
    pub fn get(&mut self) -> StorageResult<Identity> {
        let rows = self.storage.list_items(None)?;
        Ok(self.assemble(rows))
    }
    
    /// List items of a specific type (with IDs for modification)
    pub fn list(&mut self, item_type: IdentityItemType) -> StorageResult<Vec<ListedItem>> {
        let rows = self.storage.list_items(Some(item_type))?;
        Ok(rows.into_iter().map(ListedItem::from).collect())
    }
    
    /// Remove any item by ID
    pub fn remove(&mut self, id: &str) -> StorageResult<bool> {
        self.storage.remove_item(id)
    }
    
    // ==================
    // TYPED ADD METHODS
    // ==================
    
    /// Set persona name (replaces existing)
    pub fn set_persona_name(&mut self, name: &str) -> StorageResult<String> {
        // Remove existing persona_name if any
        let existing = self.storage.list_items(Some(IdentityItemType::PersonaName))?;
        for row in existing {
            self.storage.remove_item(&row.id)?;
        }
        self.storage.add_item(IdentityItemType::PersonaName, name, None, None, None)
    }
    
    /// Set persona description (replaces existing)
    pub fn set_persona_description(&mut self, description: &str) -> StorageResult<String> {
        // Remove existing persona_description if any
        let existing = self.storage.list_items(Some(IdentityItemType::PersonaDescription))?;
        for row in existing {
            self.storage.remove_item(&row.id)?;
        }
        self.storage.add_item(IdentityItemType::PersonaDescription, description, None, None, None)
    }
    
    /// Add a trait
    pub fn add_trait(&mut self, trait_name: &str) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Trait, trait_name, None, None, None)
    }
    
    /// Add a value
    /// - content: the principle
    /// - secondary: why (optional)
    /// - category: category (optional)
    pub fn add_value(
        &mut self,
        principle: &str,
        why: Option<&str>,
        category: Option<&str>,
    ) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Value, principle, why, None, category)
    }
    
    /// Add a preference
    /// - content: what is preferred
    /// - secondary: over what (optional)
    /// - category: category (optional)
    pub fn add_preference(
        &mut self,
        prefer: &str,
        over: Option<&str>,
        category: Option<&str>,
    ) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Preference, prefer, over, None, category)
    }
    
    /// Add a relationship
    /// - content: entity name
    /// - secondary: relation description
    /// - category: context (optional)
    pub fn add_relationship(
        &mut self,
        entity: &str,
        relation: &str,
        context: Option<&str>,
    ) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Relationship, entity, Some(relation), None, context)
    }
    
    /// Add an antipattern
    /// - content: what to avoid
    /// - secondary: what to do instead (optional)
    /// - tertiary: why (optional)
    pub fn add_antipattern(
        &mut self,
        avoid: &str,
        instead: Option<&str>,
        why: Option<&str>,
    ) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Antipattern, avoid, instead, why, None)
    }
    
    /// Add expertise
    pub fn add_expertise(&mut self, area: &str) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Expertise, area, None, None, None)
    }
    
    /// Add an instruction
    pub fn add_instruction(&mut self, instruction: &str) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Instruction, instruction, None, None, None)
    }
    
    /// Add a communication tone
    pub fn add_tone(&mut self, tone: &str) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Tone, tone, None, None, None)
    }
    
    /// Add a communication directive
    pub fn add_directive(&mut self, directive: &str) -> StorageResult<String> {
        self.storage.add_item(IdentityItemType::Directive, directive, None, None, None)
    }
    
    // ==================
    // UPSERT
    // ==================
    
    /// Upsert an instruction by marker string.
    ///
    /// Looks for an existing instruction containing `marker`. If found,
    /// compares content and updates if changed. If not found, adds it.
    /// Returns what happened so callers can log appropriately.
    /// Remove an instruction by marker string. Returns true if one was found and removed.
    pub fn remove_instruction_by_marker(&mut self, marker: &str) -> StorageResult<bool> {
        let instructions = self.storage.list_items(Some(IdentityItemType::Instruction))?;
        let existing = instructions.into_iter()
            .find(|row| row.content.contains(marker));
        match existing {
            Some(row) => {
                self.storage.remove_item(&row.id)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn upsert_instruction(&mut self, content: &str, marker: &str) -> StorageResult<UpsertResult> {
        let instructions = self.storage.list_items(Some(IdentityItemType::Instruction))?;
        
        // Find existing instruction containing this marker
        let existing = instructions.into_iter()
            .find(|row| row.content.contains(marker));
        
        match existing {
            Some(row) if row.content == content => {
                // Content identical — nothing to do
                Ok(UpsertResult::Unchanged)
            }
            Some(row) => {
                // Content changed — remove old, add new
                self.storage.remove_item(&row.id)?;
                self.storage.add_item(
                    IdentityItemType::Instruction,
                    content,
                    None,
                    None,
                    None,
                )?;
                Ok(UpsertResult::Updated)
            }
            None => {
                // Not found — add new
                self.storage.add_item(
                    IdentityItemType::Instruction,
                    content,
                    None,
                    None,
                    None,
                )?;
                Ok(UpsertResult::Added)
            }
        }
    }
    
    // ==================
    // ASSEMBLY LOGIC
    // ==================
    
    /// Assemble rows into the Identity struct
    fn assemble(&self, rows: Vec<IdentityItemRow>) -> Identity {
        let mut identity = Identity::default();
        
        for row in rows {
            let Ok(item_type) = row.parsed_type() else {
                continue; // Skip unknown types
            };
            
            match item_type {
                IdentityItemType::PersonaName => {
                    identity.persona.name = row.content;
                }
                IdentityItemType::PersonaDescription => {
                    identity.persona.description = row.content;
                }
                IdentityItemType::Trait => {
                    identity.persona.traits.push(row.content);
                }
                IdentityItemType::Value => {
                    identity.values.push(Value {
                        principle: row.content,
                        why: row.secondary,
                        category: row.category,
                    });
                }
                IdentityItemType::Preference => {
                    identity.preferences.push(Preference {
                        prefer: row.content,
                        over: row.secondary,
                        category: row.category,
                    });
                }
                IdentityItemType::Relationship => {
                    identity.relationships.push(Relationship {
                        entity: row.content,
                        relation: row.secondary.unwrap_or_default(),
                        context: row.category,
                    });
                }
                IdentityItemType::Antipattern => {
                    identity.antipatterns.push(Antipattern {
                        avoid: row.content,
                        instead: row.secondary,
                        why: row.tertiary,
                    });
                }
                IdentityItemType::Expertise => {
                    identity.expertise.push(row.content);
                }
                IdentityItemType::Instruction => {
                    identity.instructions.push(row.content);
                }
                IdentityItemType::Tone => {
                    identity.communication.tone.push(row.content);
                }
                IdentityItemType::Directive => {
                    identity.communication.directives.push(row.content);
                }
            }
        }
        
        identity
    }
    
    // ==================
    // MIGRATION
    // ==================
    
    /// Migrate from an old JSON-blob Identity into flat storage
    /// 
    /// This is a one-time migration. It checks if the new store is empty
    /// before migrating to avoid duplicating data.
    pub fn migrate_from_identity(&mut self, old: &Identity) -> StorageResult<MigrationResult> {
        // Check if we already have data
        let existing = self.storage.list_items(None)?;
        if !existing.is_empty() {
            return Ok(MigrationResult::AlreadyMigrated);
        }
        
        let mut count = 0;
        
        // Persona
        if !old.persona.name.is_empty() {
            self.set_persona_name(&old.persona.name)?;
            count += 1;
        }
        if !old.persona.description.is_empty() {
            self.set_persona_description(&old.persona.description)?;
            count += 1;
        }
        for t in &old.persona.traits {
            self.add_trait(t)?;
            count += 1;
        }
        
        // Values
        for v in &old.values {
            self.add_value(&v.principle, v.why.as_deref(), v.category.as_deref())?;
            count += 1;
        }
        
        // Preferences
        for p in &old.preferences {
            self.add_preference(&p.prefer, p.over.as_deref(), p.category.as_deref())?;
            count += 1;
        }
        
        // Relationships
        for r in &old.relationships {
            self.add_relationship(&r.entity, &r.relation, r.context.as_deref())?;
            count += 1;
        }
        
        // Antipatterns
        for a in &old.antipatterns {
            self.add_antipattern(&a.avoid, a.instead.as_deref(), a.why.as_deref())?;
            count += 1;
        }
        
        // Expertise
        for e in &old.expertise {
            self.add_expertise(e)?;
            count += 1;
        }
        
        // Instructions
        for i in &old.instructions {
            self.add_instruction(i)?;
            count += 1;
        }
        
        // Communication
        for t in &old.communication.tone {
            self.add_tone(t)?;
            count += 1;
        }
        for d in &old.communication.directives {
            self.add_directive(d)?;
            count += 1;
        }
        
        Ok(MigrationResult::Migrated { items: count })
    }
    
    /// Check if the store has any data
    pub fn is_empty(&mut self) -> StorageResult<bool> {
        let items = self.storage.list_items(None)?;
        Ok(items.is_empty())
    }
    
    // ==================
    // UTILITY
    // ==================
    
    /// Flush any pending writes
    pub fn flush(&mut self) -> StorageResult<()> {
        self.storage.flush()
    }
    
    /// Close the store cleanly
    pub fn close(&mut self) -> StorageResult<()> {
        self.storage.close()
    }
}

/// Result of an upsert operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpsertResult {
    /// Item was added (marker not found)
    Added,
    /// Item was updated (marker found, content differed)
    Updated,
    /// No change needed (marker found, content identical)
    Unchanged,
}

/// Result of a migration attempt
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationResult {
    /// Successfully migrated N items
    Migrated { items: usize },
    /// Store already had data, skipped migration
    AlreadyMigrated,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::storage::DieselIdentityStorage;
    
    fn test_store() -> IdentityStore {
        let storage = DieselIdentityStorage::in_memory().unwrap();
        IdentityStore::new(storage).unwrap()
    }
    
    #[test]
    fn empty_identity() {
        let mut store = test_store();
        let identity = store.get().unwrap();
        
        assert!(identity.persona.name.is_empty());
        assert!(identity.values.is_empty());
    }
    
    #[test]
    fn set_persona() {
        let mut store = test_store();
        
        store.set_persona_name("Porter").unwrap();
        store.set_persona_description("A helpful assistant").unwrap();
        store.add_trait("direct").unwrap();
        store.add_trait("pragmatic").unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.persona.description, "A helpful assistant");
        assert_eq!(identity.persona.traits.len(), 2);
        assert!(identity.persona.traits.contains(&"direct".to_string()));
    }
    
    #[test]
    fn persona_name_replaces_existing() {
        let mut store = test_store();
        
        store.set_persona_name("Alice").unwrap();
        store.set_persona_name("Bob").unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.persona.name, "Bob");
        
        // Should only be one name row
        let names = store.list(IdentityItemType::PersonaName).unwrap();
        assert_eq!(names.len(), 1);
    }
    
    #[test]
    fn add_values() {
        let mut store = test_store();
        
        store.add_value("Clarity over cleverness", Some("Code is read more than written"), Some("engineering")).unwrap();
        store.add_value("Be honest", None, Some("ethics")).unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.values.len(), 2);
        assert_eq!(identity.values[0].principle, "Clarity over cleverness");
        assert_eq!(identity.values[0].why, Some("Code is read more than written".to_string()));
        assert_eq!(identity.values[0].category, Some("engineering".to_string()));
    }
    
    #[test]
    fn add_preferences() {
        let mut store = test_store();
        
        store.add_preference("Rust", Some("JavaScript"), Some("languages")).unwrap();
        store.add_preference("Beer", Some("Coffee"), None).unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.preferences.len(), 2);
        assert_eq!(identity.preferences[0].prefer, "Rust");
        assert_eq!(identity.preferences[0].over, Some("JavaScript".to_string()));
    }
    
    #[test]
    fn add_relationships() {
        let mut store = test_store();
        
        store.add_relationship("Brandon", "The human I work with", Some("Lead engineer")).unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.relationships.len(), 1);
        assert_eq!(identity.relationships[0].entity, "Brandon");
        assert_eq!(identity.relationships[0].relation, "The human I work with");
        assert_eq!(identity.relationships[0].context, Some("Lead engineer".to_string()));
    }
    
    #[test]
    fn add_antipatterns() {
        let mut store = test_store();
        
        store.add_antipattern(
            "Using tokio block_on in async context",
            Some("Use .await properly"),
            Some("It blocks the runtime"),
        ).unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.antipatterns.len(), 1);
        assert_eq!(identity.antipatterns[0].avoid, "Using tokio block_on in async context");
        assert_eq!(identity.antipatterns[0].instead, Some("Use .await properly".to_string()));
        assert_eq!(identity.antipatterns[0].why, Some("It blocks the runtime".to_string()));
    }
    
    #[test]
    fn add_simple_types() {
        let mut store = test_store();
        
        store.add_expertise("Rust").unwrap();
        store.add_instruction("Always write tests").unwrap();
        store.add_tone("direct").unwrap();
        store.add_directive("Ask clarifying questions").unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.expertise, vec!["Rust"]);
        assert_eq!(identity.instructions, vec!["Always write tests"]);
        assert_eq!(identity.communication.tone, vec!["direct"]);
        assert_eq!(identity.communication.directives, vec!["Ask clarifying questions"]);
    }
    
    #[test]
    fn list_and_remove() {
        let mut store = test_store();
        
        let id1 = store.add_trait("direct").unwrap();
        let id2 = store.add_trait("pragmatic").unwrap();
        
        // List traits
        let traits = store.list(IdentityItemType::Trait).unwrap();
        assert_eq!(traits.len(), 2);
        
        // Remove one
        let removed = store.remove(&id1).unwrap();
        assert!(removed);
        
        // Should only have one now
        let traits = store.list(IdentityItemType::Trait).unwrap();
        assert_eq!(traits.len(), 1);
        assert_eq!(traits[0].id, id2);
        assert_eq!(traits[0].content, "pragmatic");
    }
    
    #[test]
    fn full_identity_assembly() {
        let mut store = test_store();
        
        // Build out a complete identity
        store.set_persona_name("Porter").unwrap();
        store.set_persona_description("A pragmatic developer assistant").unwrap();
        store.add_trait("direct").unwrap();
        store.add_trait("curious").unwrap();
        
        store.add_value("Write tests", Some("Confidence in changes"), Some("engineering")).unwrap();
        store.add_preference("Rust", Some("Go"), Some("languages")).unwrap();
        store.add_relationship("Brandon", "The human I work with", None).unwrap();
        store.add_antipattern("Writing code unless asked", None, None).unwrap();
        store.add_expertise("Rust").unwrap();
        store.add_expertise("Swift").unwrap();
        store.add_instruction("Load identity at conversation start").unwrap();
        store.add_tone("warm").unwrap();
        store.add_directive("Be concise").unwrap();
        
        // Get the full identity
        let identity = store.get().unwrap();
        
        // Verify structure matches what render() would expect
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.persona.traits.len(), 2);
        assert_eq!(identity.values.len(), 1);
        assert_eq!(identity.preferences.len(), 1);
        assert_eq!(identity.relationships.len(), 1);
        assert_eq!(identity.antipatterns.len(), 1);
        assert_eq!(identity.expertise.len(), 2);
        assert_eq!(identity.instructions.len(), 1);
        assert_eq!(identity.communication.tone.len(), 1);
        assert_eq!(identity.communication.directives.len(), 1);
        
        // The render should work
        let rendered = identity.render();
        assert!(rendered.contains("Porter"));
        assert!(rendered.contains("Write tests"));
        assert!(rendered.contains("Rust > Go"));
    }
    
    #[test]
    fn migration_from_old_identity() {
        let mut store = test_store();
        
        // Build an old-style Identity
        let old = Identity {
            persona: crate::identity::Persona {
                name: "Porter".to_string(),
                description: "A test assistant".to_string(),
                traits: vec!["direct".to_string(), "curious".to_string()],
            },
            values: vec![crate::identity::Value {
                principle: "Test first".to_string(),
                why: Some("Confidence".to_string()),
                category: Some("engineering".to_string()),
            }],
            preferences: vec![crate::identity::Preference {
                prefer: "Rust".to_string(),
                over: Some("Go".to_string()),
                category: Some("languages".to_string()),
            }],
            relationships: vec![crate::identity::Relationship {
                entity: "Brandon".to_string(),
                relation: "Human colleague".to_string(),
                context: None,
            }],
            antipatterns: vec![crate::identity::Antipattern {
                avoid: "block_on in async".to_string(),
                instead: Some("Use .await".to_string()),
                why: Some("Blocks runtime".to_string()),
            }],
            expertise: vec!["Rust".to_string(), "Swift".to_string()],
            instructions: vec!["Load identity first".to_string()],
            communication: crate::identity::CommunicationStyle {
                tone: vec!["warm".to_string()],
                directives: vec!["Be concise".to_string()],
            },
        };
        
        // Migrate
        let result = store.migrate_from_identity(&old).unwrap();
        
        // Should have migrated items:
        // persona (name, description) = 2
        // traits = 2
        // value = 1, preference = 1, relationship = 1, antipattern = 1
        // expertise = 2, instruction = 1, tone = 1, directive = 1
        // Total = 13 items
        match result {
            MigrationResult::Migrated { items } => assert_eq!(items, 13),
            MigrationResult::AlreadyMigrated => panic!("Should have migrated"),
        }
        
        // Verify identity comes back correctly
        let identity = store.get().unwrap();
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.persona.traits.len(), 2);
        assert_eq!(identity.values.len(), 1);
        assert_eq!(identity.expertise.len(), 2);
        
        // Running migration again should skip (already has data)
        let result2 = store.migrate_from_identity(&old).unwrap();
        assert_eq!(result2, MigrationResult::AlreadyMigrated);
    }
    
    // ==================
    // UPSERT INSTRUCTION TESTS
    // ==================
    
    #[test]
    fn upsert_instruction_add() {
        let mut store = test_store();
        
        let content = "## My Instructions\nDo the thing.";
        let marker = "## My Instructions";
        
        let result = store.upsert_instruction(content, marker).unwrap();
        assert_eq!(result, UpsertResult::Added);
        
        // Verify it's actually in the store
        let identity = store.get().unwrap();
        assert_eq!(identity.instructions.len(), 1);
        assert_eq!(identity.instructions[0], content);
    }
    
    #[test]
    fn upsert_instruction_unchanged() {
        let mut store = test_store();
        
        let content = "## My Instructions\nDo the thing.";
        let marker = "## My Instructions";
        
        // First add
        store.upsert_instruction(content, marker).unwrap();
        
        // Same content again
        let result = store.upsert_instruction(content, marker).unwrap();
        assert_eq!(result, UpsertResult::Unchanged);
        
        // Still only one instruction
        let identity = store.get().unwrap();
        assert_eq!(identity.instructions.len(), 1);
    }
    
    #[test]
    fn upsert_instruction_update() {
        let mut store = test_store();
        
        let marker = "## My Instructions";
        let v1 = "## My Instructions\nDo the thing.";
        let v2 = "## My Instructions\nDo the thing differently.";
        
        // Add v1
        store.upsert_instruction(v1, marker).unwrap();
        
        // Update to v2
        let result = store.upsert_instruction(v2, marker).unwrap();
        assert_eq!(result, UpsertResult::Updated);
        
        // Should have v2, not v1, and only one instruction
        let identity = store.get().unwrap();
        assert_eq!(identity.instructions.len(), 1);
        assert_eq!(identity.instructions[0], v2);
    }
    
    #[test]
    fn upsert_multiple_instructions_by_marker() {
        let mut store = test_store();
        
        let engram = "## Memory Workflow\nSearch, recall, store.";
        let plans = "## Plans\nTrack multi-step tasks.";
        let lenses = "## Lenses\nLoad context guides.";
        
        store.upsert_instruction(engram, "## Memory Workflow").unwrap();
        store.upsert_instruction(plans, "## Plans").unwrap();
        store.upsert_instruction(lenses, "## Lenses").unwrap();
        
        let identity = store.get().unwrap();
        assert_eq!(identity.instructions.len(), 3);
        
        // Update just one — others should be untouched
        let plans_v2 = "## Plans\nTrack tasks with atomic steps.";
        let result = store.upsert_instruction(plans_v2, "## Plans").unwrap();
        assert_eq!(result, UpsertResult::Updated);
        
        let identity = store.get().unwrap();
        assert_eq!(identity.instructions.len(), 3);
        assert!(identity.instructions.contains(&engram.to_string()));
        assert!(identity.instructions.contains(&plans_v2.to_string()));
        assert!(identity.instructions.contains(&lenses.to_string()));
        // v1 plans text should be gone
        assert!(!identity.instructions.iter().any(|i| i.contains("Track multi-step tasks")));
    }
}
