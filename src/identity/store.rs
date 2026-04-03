//! IdentityStore - The coordination layer for identity
//!
//! IdentityStore wraps storage and provides the high-level typed API
//! for managing identity items. The storage layer is dumb (just rows),
//! this layer knows about the semantics of each identity type.

use super::storage::{IdentityItemRow, IdentityItemType, IdentityStorage, StorageResult};
use super::types::{Identity, Preference, Relationship, Rule, Value};

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
        let item_type = row.parsed_type().unwrap_or(IdentityItemType::Rule);
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
    /// Create a new IdentityStore with the given storage backend.
    /// Runs initialize() and then migrate_v1() to handle legacy data.
    pub fn new<S: IdentityStorage + 'static>(mut storage: S) -> StorageResult<Self> {
        storage.initialize()?;
        let mut store = Self {
            storage: Box::new(storage),
        };
        store.migrate_v1()?;
        Ok(store)
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
        let existing = self
            .storage
            .list_items(Some(IdentityItemType::PersonaName))?;
        for row in existing {
            self.storage.remove_item(&row.id)?;
        }
        self.storage
            .add_item(IdentityItemType::PersonaName, name, None, None, None)
    }

    /// Set persona description (replaces existing)
    pub fn set_persona_description(&mut self, description: &str) -> StorageResult<String> {
        // Remove existing persona_description if any
        let existing = self
            .storage
            .list_items(Some(IdentityItemType::PersonaDescription))?;
        for row in existing {
            self.storage.remove_item(&row.id)?;
        }
        self.storage.add_item(
            IdentityItemType::PersonaDescription,
            description,
            None,
            None,
            None,
        )
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
        self.storage
            .add_item(IdentityItemType::Value, principle, why, None, category)
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
        self.storage
            .add_item(IdentityItemType::Preference, prefer, over, None, category)
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
        self.storage.add_item(
            IdentityItemType::Relationship,
            entity,
            Some(relation),
            None,
            context,
        )
    }

    /// Add a rule
    /// - content: the rule itself ("do X" or "don't do X")
    /// - instead: what to do instead / additional context (optional)
    /// - why: why this rule matters (optional)
    /// - negative: true for "don't" rules, false for "do" rules
    pub fn add_rule(
        &mut self,
        content: &str,
        instead: Option<&str>,
        why: Option<&str>,
        negative: bool,
    ) -> StorageResult<String> {
        let category = if negative { Some("negative") } else { None };
        self.storage
            .add_item(IdentityItemType::Rule, content, instead, why, category)
    }

    // ==================
    // ASSEMBLY LOGIC
    // ==================

    /// Assemble rows into the Identity struct
    fn assemble(&self, rows: Vec<IdentityItemRow>) -> Identity {
        let mut identity = Identity::default();

        for row in rows {
            let Ok(item_type) = row.parsed_type() else {
                continue; // Skip unknown/legacy types
            };

            match item_type {
                IdentityItemType::PersonaName => {
                    identity.persona.name = row.content;
                }
                IdentityItemType::PersonaDescription => {
                    identity.persona.description = row.content;
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
                IdentityItemType::Rule => {
                    identity.rules.push(Rule {
                        content: row.content,
                        instead: row.secondary,
                        why: row.tertiary,
                        negative: row.category.as_deref() == Some("negative"),
                    });
                }
            }
        }

        identity
    }

    // ==================
    // MIGRATION
    // ==================

    /// Migrate legacy identity data from v0 (10 types) to v1 (5 types).
    ///
    /// This migration:
    /// 1. Absorbs trait, tone, expertise rows into persona_description
    /// 2. Converts directive and antipattern rows to rule type
    /// 3. Deletes the absorbed rows
    ///
    /// Transaction-wrapped: a crash mid-migration won't leave the database
    /// in a partially migrated state.
    /// Idempotent: returns early if no legacy rows exist.
    /// Generic: works for any user's data, no hardcoded IDs.
    pub fn migrate_v1(&mut self) -> StorageResult<MigrationResult> {
        let legacy_types = ["trait", "tone", "expertise", "directive", "antipattern"];

        // Check if migration is needed (outside transaction — read-only)
        let mut has_legacy = false;
        for lt in &legacy_types {
            let rows = self.storage.list_items_by_type_str(lt)?;
            if !rows.is_empty() {
                has_legacy = true;
                break;
            }
        }

        if !has_legacy {
            return Ok(MigrationResult::NotNeeded);
        }

        // Wrap all mutations in a transaction
        self.storage.begin_transaction()?;
        match self.run_v1_migration_body() {
            Ok(result) => {
                self.storage.commit_transaction()?;
                Ok(result)
            }
            Err(e) => {
                let _ = self.storage.rollback_transaction();
                Err(e)
            }
        }
    }

    /// Inner migration body, called within a transaction by `migrate_v1`.
    fn run_v1_migration_body(&mut self) -> StorageResult<MigrationResult> {
        // Read current persona description
        let desc_rows = self
            .storage
            .list_items(Some(IdentityItemType::PersonaDescription))?;
        let current_description = desc_rows
            .first()
            .map(|r| r.content.clone())
            .unwrap_or_default();

        // Read trait, tone, expertise rows to absorb into description
        let trait_rows = self.storage.list_items_by_type_str("trait")?;
        let tone_rows = self.storage.list_items_by_type_str("tone")?;
        let expertise_rows = self.storage.list_items_by_type_str("expertise")?;

        // Build enriched description
        let mut enriched = current_description.clone();

        if !trait_rows.is_empty() {
            let traits: Vec<String> = trait_rows.iter().map(|r| r.content.clone()).collect();
            let traits_str = traits.join(", ");
            if enriched.is_empty() {
                enriched = format!("Traits: {}", traits_str);
            } else {
                enriched = format!("{} Traits: {}.", enriched.trim_end_matches('.'), traits_str);
            }
        }

        if !tone_rows.is_empty() {
            let tones: Vec<String> = tone_rows.iter().map(|r| r.content.clone()).collect();
            let tones_str = tones.join(", ");
            enriched = format!("{} Tone: {}.", enriched.trim_end_matches('.'), tones_str);
        }

        if !expertise_rows.is_empty() {
            let areas: Vec<String> = expertise_rows.iter().map(|r| r.content.clone()).collect();
            let areas_str = areas.join(", ");
            enriched = format!(
                "{} Expertise: {}.",
                enriched.trim_end_matches('.'),
                areas_str
            );
        }

        // Update persona description if it changed
        if enriched != current_description {
            self.set_persona_description(&enriched)?;
        }

        // Convert directive -> rule (positive, no category change)
        let directives_updated = self.storage.update_item_type("directive", "rule")?;
        // Convert antipattern -> rule (negative, set category="negative")
        let antipatterns_updated =
            self.storage
                .update_item_type_with_category("antipattern", "rule", Some("negative"))?;

        // Delete absorbed rows
        let traits_deleted = self.storage.delete_items_by_type_str("trait")?;
        let tones_deleted = self.storage.delete_items_by_type_str("tone")?;
        let expertise_deleted = self.storage.delete_items_by_type_str("expertise")?;

        let total = directives_updated
            + antipatterns_updated
            + traits_deleted
            + tones_deleted
            + expertise_deleted;

        Ok(MigrationResult::Migrated { items: total })
    }

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

        // Rules
        for r in &old.rules {
            self.add_rule(&r.content, r.instead.as_deref(), r.why.as_deref(), r.negative)?;
            count += 1;
        }

        Ok(MigrationResult::Migrated { items: count })
    }

    /// Delete all items of a given raw type string.
    /// Used for cleaning up removed types (e.g., "instruction") that are
    /// no longer in the IdentityItemType enum.
    pub fn delete_items_by_type_str(&mut self, type_str: &str) -> StorageResult<usize> {
        self.storage.delete_items_by_type_str(type_str)
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

/// Result of a migration attempt
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationResult {
    /// Successfully migrated N items
    Migrated { items: usize },
    /// Store already had data, skipped migration (from migrate_from_identity)
    AlreadyMigrated,
    /// No legacy data found, migration not needed
    NotNeeded,
}

#[cfg(test)]
mod tests {
    use super::super::storage::DieselIdentityStorage;
    use super::*;
    use diesel::connection::SimpleConnection;

    fn test_store() -> IdentityStore {
        let storage = DieselIdentityStorage::in_memory().unwrap();
        IdentityStore::new(storage).unwrap()
    }

    /// Create an IdentityStore with legacy data pre-loaded for migration tests.
    /// We bypass the normal IdentityStore::new() to insert legacy rows first.
    fn store_with_legacy_data() -> IdentityStore {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Insert legacy rows directly using raw SQL to bypass enum validation
        storage
            .conn_mut()
            .batch_execute(
                r#"
            INSERT INTO identity_items (id, item_type, content, secondary, tertiary, category, created_at)
            VALUES
                ('pn01', 'persona_name', 'Porter', NULL, NULL, NULL, 1000),
                ('pd01', 'persona_description', 'A pragmatic developer', NULL, NULL, NULL, 1001),
                ('tr01', 'trait', 'direct', NULL, NULL, NULL, 1002),
                ('tr02', 'trait', 'curious', NULL, NULL, NULL, 1003),
                ('tn01', 'tone', 'warm', NULL, NULL, NULL, 1004),
                ('tn02', 'tone', 'snarky but constructive', NULL, NULL, NULL, 1005),
                ('ex01', 'expertise', 'Rust', NULL, NULL, NULL, 1006),
                ('ex02', 'expertise', 'Swift', NULL, NULL, NULL, 1007),
                ('di01', 'directive', 'Use full cargo path', NULL, NULL, NULL, 1008),
                ('di02', 'directive', 'Spawn agents with thorough prompts', NULL, NULL, NULL, 1009),
                ('ap01', 'antipattern', 'Using tokio', 'std::thread + channels', 'Too heavy', NULL, 1010),
                ('ap02', 'antipattern', 'Writing code unless asked', 'Discuss approach first', NULL, NULL, 1011),
                ('v001', 'value', 'Test everything', 'Confidence in changes', NULL, 'engineering', 1012),
                ('p001', 'preference', 'Rust', 'Go', NULL, 'languages', 1013),
                ('r001', 'relationship', 'Brandon', 'The human I work with', NULL, 'Lead engineer', 1014);
            "#,
            )
            .unwrap();

        // Now create the store — migrate_v1 will run during new()
        // But we already initialized, so we need a different approach.
        // We'll create the store without auto-migration, then migrate manually.
        let mut store = IdentityStore {
            storage: Box::new(storage),
        };
        // Run migration
        store.migrate_v1().unwrap();
        store
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
        store
            .set_persona_description("A helpful assistant")
            .unwrap();

        let identity = store.get().unwrap();
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.persona.description, "A helpful assistant");
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

        store
            .add_value(
                "Clarity over cleverness",
                Some("Code is read more than written"),
                Some("engineering"),
            )
            .unwrap();
        store.add_value("Be honest", None, Some("ethics")).unwrap();

        let identity = store.get().unwrap();
        assert_eq!(identity.values.len(), 2);
        assert_eq!(identity.values[0].principle, "Clarity over cleverness");
        assert_eq!(
            identity.values[0].why,
            Some("Code is read more than written".to_string())
        );
        assert_eq!(identity.values[0].category, Some("engineering".to_string()));
    }

    #[test]
    fn add_preferences() {
        let mut store = test_store();

        store
            .add_preference("Rust", Some("JavaScript"), Some("languages"))
            .unwrap();
        store.add_preference("Beer", Some("Coffee"), None).unwrap();

        let identity = store.get().unwrap();
        assert_eq!(identity.preferences.len(), 2);
        assert_eq!(identity.preferences[0].prefer, "Rust");
        assert_eq!(identity.preferences[0].over, Some("JavaScript".to_string()));
    }

    #[test]
    fn add_relationships() {
        let mut store = test_store();

        store
            .add_relationship("Brandon", "The human I work with", Some("Lead engineer"))
            .unwrap();

        let identity = store.get().unwrap();
        assert_eq!(identity.relationships.len(), 1);
        assert_eq!(identity.relationships[0].entity, "Brandon");
        assert_eq!(identity.relationships[0].relation, "The human I work with");
        assert_eq!(
            identity.relationships[0].context,
            Some("Lead engineer".to_string())
        );
    }

    #[test]
    fn add_rules() {
        let mut store = test_store();

        store
            .add_rule(
                "Don't use tokio block_on in async context",
                Some("Use .await properly"),
                Some("It blocks the runtime"),
                true,
            )
            .unwrap();

        let identity = store.get().unwrap();
        assert_eq!(identity.rules.len(), 1);
        assert_eq!(
            identity.rules[0].content,
            "Don't use tokio block_on in async context"
        );
        assert_eq!(
            identity.rules[0].instead,
            Some("Use .await properly".to_string())
        );
        assert_eq!(
            identity.rules[0].why,
            Some("It blocks the runtime".to_string())
        );
    }

    #[test]
    fn list_and_remove() {
        let mut store = test_store();

        let id1 = store.add_rule("Always run tests", None, None, false).unwrap();
        let id2 = store
            .add_rule("Don't skip linting", None, None, true)
            .unwrap();

        // List rules
        let rules = store.list(IdentityItemType::Rule).unwrap();
        assert_eq!(rules.len(), 2);

        // Remove one
        let removed = store.remove(&id1).unwrap();
        assert!(removed);

        // Should only have one now
        let rules = store.list(IdentityItemType::Rule).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, id2);
        assert_eq!(rules[0].content, "Don't skip linting");
    }

    #[test]
    fn full_identity_assembly() {
        let mut store = test_store();

        // Build out a complete identity
        store.set_persona_name("Porter").unwrap();
        store
            .set_persona_description("A pragmatic developer assistant")
            .unwrap();

        store
            .add_value(
                "Write tests",
                Some("Confidence in changes"),
                Some("engineering"),
            )
            .unwrap();
        store
            .add_preference("Rust", Some("Go"), Some("languages"))
            .unwrap();
        store
            .add_relationship("Brandon", "The human I work with", None)
            .unwrap();
        store
            .add_rule("Don't write code unless asked", None, None, true)
            .unwrap();
        store
            .add_rule("Always run tests before committing", None, None, false)
            .unwrap();

        // Get the full identity
        let identity = store.get().unwrap();

        // Verify structure matches what render() would expect
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.values.len(), 1);
        assert_eq!(identity.preferences.len(), 1);
        assert_eq!(identity.relationships.len(), 1);
        assert_eq!(identity.rules.len(), 2);

        // The render should work
        let rendered = identity.render();
        assert!(rendered.contains("Porter"));
        assert!(rendered.contains("Write tests"));
        assert!(rendered.contains("Rust > Go"));
        assert!(rendered.contains("<rules>"));
    }

    #[test]
    fn migration_from_old_identity() {
        let mut store = test_store();

        // Build an old-style Identity (now using the new struct shape)
        let old = Identity {
            persona: crate::identity::Persona {
                name: "Porter".to_string(),
                description: "A test assistant".to_string(),
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
            rules: vec![crate::identity::Rule {
                content: "Don't block_on in async".to_string(),
                instead: Some("Use .await".to_string()),
                why: Some("Blocks runtime".to_string()),
                negative: true,
            }],
        };

        // Migrate
        let result = store.migrate_from_identity(&old).unwrap();

        // Should have migrated items:
        // persona (name, description) = 2
        // value = 1, preference = 1, relationship = 1, rule = 1
        // Total = 6 items
        match result {
            MigrationResult::Migrated { items } => assert_eq!(items, 6),
            other => panic!("Expected Migrated, got {:?}", other),
        }

        // Verify identity comes back correctly
        let identity = store.get().unwrap();
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.values.len(), 1);
        assert_eq!(identity.rules.len(), 1);

        // Running migration again should skip (already has data)
        let result2 = store.migrate_from_identity(&old).unwrap();
        assert_eq!(result2, MigrationResult::AlreadyMigrated);
    }

    // ==================
    // V1 MIGRATION TESTS
    // ==================

    #[test]
    fn migrate_v1_converts_legacy_data() {
        let mut store = store_with_legacy_data();
        let identity = store.get().unwrap();

        // Persona name preserved
        assert_eq!(identity.persona.name, "Porter");

        // Description should include absorbed traits, tone, expertise
        assert!(
            identity.persona.description.contains("direct"),
            "Description should contain trait 'direct': {}",
            identity.persona.description
        );
        assert!(
            identity.persona.description.contains("curious"),
            "Description should contain trait 'curious'"
        );
        assert!(
            identity.persona.description.contains("warm"),
            "Description should contain tone 'warm'"
        );
        assert!(
            identity.persona.description.contains("snarky but constructive"),
            "Description should contain tone"
        );
        assert!(
            identity.persona.description.contains("Rust"),
            "Description should contain expertise 'Rust'"
        );
        assert!(
            identity.persona.description.contains("Swift"),
            "Description should contain expertise 'Swift'"
        );

        // Directives converted to rules (positive)
        let cargo_rule = identity
            .rules
            .iter()
            .find(|r| r.content.contains("cargo"))
            .expect("Directive should be converted to rule");
        assert!(
            !cargo_rule.negative,
            "Directives should become positive rules"
        );
        let agents_rule = identity
            .rules
            .iter()
            .find(|r| r.content.contains("agents"))
            .expect("Directive should be converted to rule");
        assert!(
            !agents_rule.negative,
            "Directives should become positive rules"
        );

        // Antipatterns converted to rules (negative)
        let tokio_rule = identity
            .rules
            .iter()
            .find(|r| r.content.contains("tokio"))
            .expect("Antipattern should be converted to rule");
        assert!(
            tokio_rule.negative,
            "Antipatterns should become negative rules"
        );
        let code_rule = identity
            .rules
            .iter()
            .find(|r| r.content.contains("Writing code unless asked"))
            .expect("Antipattern should be converted to rule");
        assert!(
            code_rule.negative,
            "Antipatterns should become negative rules"
        );

        // Values and preferences preserved
        assert_eq!(identity.values.len(), 1);
        assert_eq!(identity.preferences.len(), 1);
        assert_eq!(identity.relationships.len(), 1);

        // Total rules: 2 directives + 2 antipatterns = 4
        assert_eq!(identity.rules.len(), 4);
    }

    #[test]
    fn migrate_v1_is_idempotent() {
        let mut store = store_with_legacy_data();

        // First migration already happened in store_with_legacy_data()
        // Running again should be a no-op
        let result = store.migrate_v1().unwrap();
        assert_eq!(result, MigrationResult::NotNeeded);

        // Identity should be unchanged
        let identity = store.get().unwrap();
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.rules.len(), 4);
    }

    #[test]
    fn migrate_v1_empty_store_is_noop() {
        let mut store = test_store();

        // Already ran in new(), but let's be explicit
        let result = store.migrate_v1().unwrap();
        assert_eq!(result, MigrationResult::NotNeeded);
    }

    #[test]
    fn migrate_v1_already_migrated_store() {
        // A store with only new types should not trigger migration
        let mut store = test_store();
        store.set_persona_name("Porter").unwrap();
        store
            .set_persona_description("A pragmatic assistant")
            .unwrap();
        store
            .add_rule("Always run tests", None, None, false)
            .unwrap();
        store.add_value("Quality first", None, None).unwrap();

        let result = store.migrate_v1().unwrap();
        assert_eq!(result, MigrationResult::NotNeeded);
    }
}
