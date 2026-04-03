//! Identity - The bedrock layer of self
//!
//! Unlike episodic/semantic memories in the Substrate, identity doesn't decay.
//! It's not "remembered" - it just IS. The lens you see through, not the thing you look at.

use serde::{Deserialize, Serialize};

/// The core identity - who you are, not what you've experienced
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Identity {
    /// Core persona - the fundamental "I am"
    #[serde(default)]
    pub persona: Persona,

    /// Values - what matters, principles that guide behavior
    #[serde(default)]
    pub values: Vec<Value>,

    /// Preferences - likes, dislikes, stylistic choices
    #[serde(default)]
    pub preferences: Vec<Preference>,

    /// Key relationships - the important entities in your world
    #[serde(default)]
    pub relationships: Vec<Relationship>,

    /// Rules - behavioral constraints, both positive ("do X") and negative ("don't do X")
    #[serde(default)]
    pub rules: Vec<Rule>,
}

/// The fundamental "I am" - name, nature, essence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Persona {
    /// Name / identifier
    #[serde(default)]
    pub name: String,

    /// Description of nature, traits, tone, expertise — all in natural language
    #[serde(default)]
    pub description: String,
}

/// A value - something that matters, a principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    /// The value itself
    #[serde(default)]
    pub principle: String,

    /// Why it matters (optional context)
    #[serde(default)]
    pub why: Option<String>,

    /// Category (e.g., "engineering", "communication", "ethics")
    #[serde(default)]
    pub category: Option<String>,
}

impl Value {
    pub fn new(principle: impl Into<String>) -> Self {
        Self {
            principle: principle.into(),
            why: None,
            category: None,
        }
    }

    pub fn with_why(mut self, why: impl Into<String>) -> Self {
        self.why = Some(why.into());
        self
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}

/// A preference - something liked or disliked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    /// What is preferred
    #[serde(default)]
    pub prefer: String,

    /// Over what (optional comparison)
    #[serde(default)]
    pub over: Option<String>,

    /// Category
    #[serde(default)]
    pub category: Option<String>,
}

impl Preference {
    pub fn new(prefer: impl Into<String>) -> Self {
        Self {
            prefer: prefer.into(),
            over: None,
            category: None,
        }
    }

    pub fn over(mut self, alternative: impl Into<String>) -> Self {
        self.over = Some(alternative.into());
        self
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}

/// A relationship - a key entity and your connection to them
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Who/what
    #[serde(default)]
    pub entity: String,

    /// The nature of the relationship
    #[serde(default)]
    pub relation: String,

    /// Additional context
    #[serde(default)]
    pub context: Option<String>,
}

impl Relationship {
    pub fn new(entity: impl Into<String>, relation: impl Into<String>) -> Self {
        Self {
            entity: entity.into(),
            relation: relation.into(),
            context: None,
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// A rule - a behavioral constraint, either positive ("do X") or negative ("don't do X")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// The rule itself — either "do X" or "don't do X"
    #[serde(default)]
    pub content: String,

    /// What to do instead (for negative rules) or additional context
    #[serde(default)]
    pub instead: Option<String>,

    /// Why this rule matters — enables judgment in edge cases
    #[serde(default)]
    pub why: Option<String>,

    /// Whether this is a negative rule ("don't do X") vs positive ("do X")
    #[serde(default)]
    pub negative: bool,
}

impl Rule {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            instead: None,
            why: None,
            negative: false,
        }
    }

    pub fn negative(mut self) -> Self {
        self.negative = true;
        self
    }

    pub fn with_instead(mut self, instead: impl Into<String>) -> Self {
        self.instead = Some(instead.into());
        self
    }

    pub fn with_why(mut self, why: impl Into<String>) -> Self {
        self.why = Some(why.into());
        self
    }
}

/// Results from searching identity
#[derive(Debug)]
pub struct IdentitySearchResults<'a> {
    pub values: Vec<&'a Value>,
    pub preferences: Vec<&'a Preference>,
    pub relationships: Vec<&'a Relationship>,
    pub rules: Vec<&'a Rule>,
}

impl<'a> IdentitySearchResults<'a> {
    /// Check if any results were found
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
            && self.preferences.is_empty()
            && self.relationships.is_empty()
            && self.rules.is_empty()
    }

    /// Total count of all matches
    pub fn total_count(&self) -> usize {
        self.values.len()
            + self.preferences.len()
            + self.relationships.len()
            + self.rules.len()
    }
}

impl Identity {
    /// Create a new empty identity
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the persona
    pub fn with_persona(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.persona.name = name.into();
        self.persona.description = description.into();
        self
    }

    /// Add a value
    pub fn with_value(mut self, value: Value) -> Self {
        self.values.push(value);
        self
    }

    /// Add a preference
    pub fn with_preference(mut self, pref: Preference) -> Self {
        self.preferences.push(pref);
        self
    }

    /// Add a relationship
    pub fn with_relationship(mut self, rel: Relationship) -> Self {
        self.relationships.push(rel);
        self
    }

    /// Add a rule
    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    // ==================
    // DISCOVERY METHODS
    // ==================

    /// List all unique categories used in values
    pub fn list_value_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .values
            .iter()
            .filter_map(|v| v.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// List all unique categories used in preferences
    pub fn list_preference_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .preferences
            .iter()
            .filter_map(|p| p.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// List all unique categories (from both values and preferences)
    pub fn list_all_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .values
            .iter()
            .filter_map(|v| v.category.clone())
            .chain(self.preferences.iter().filter_map(|p| p.category.clone()))
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// List all entity names from relationships
    pub fn list_entities(&self) -> Vec<String> {
        self.relationships
            .iter()
            .map(|r| r.entity.clone())
            .collect()
    }

    // ==================
    // LOOKUP METHODS
    // ==================

    /// Get values by category
    pub fn values_in_category(&self, category: &str) -> Vec<&Value> {
        self.values
            .iter()
            .filter(|v| v.category.as_deref() == Some(category))
            .collect()
    }

    /// Get preferences by category
    pub fn preferences_in_category(&self, category: &str) -> Vec<&Preference> {
        self.preferences
            .iter()
            .filter(|p| p.category.as_deref() == Some(category))
            .collect()
    }

    /// Get relationship by entity name
    pub fn relationship_for(&self, entity: &str) -> Option<&Relationship> {
        self.relationships
            .iter()
            .find(|r| r.entity.eq_ignore_ascii_case(entity))
    }

    // ==================
    // SEARCH METHODS
    // ==================

    /// Search all identity content for a term (case-insensitive)
    /// Returns a summary of where matches were found
    pub fn search(&self, query: &str) -> IdentitySearchResults<'_> {
        let q = query.to_lowercase();

        let values: Vec<&Value> = self
            .values
            .iter()
            .filter(|v| {
                v.principle.to_lowercase().contains(&q)
                    || v.why
                        .as_ref()
                        .map(|w| w.to_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .collect();

        let preferences: Vec<&Preference> = self
            .preferences
            .iter()
            .filter(|p| {
                p.prefer.to_lowercase().contains(&q)
                    || p.over
                        .as_ref()
                        .map(|o| o.to_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .collect();

        let relationships: Vec<&Relationship> = self
            .relationships
            .iter()
            .filter(|r| {
                r.entity.to_lowercase().contains(&q)
                    || r.relation.to_lowercase().contains(&q)
                    || r.context
                        .as_ref()
                        .map(|c| c.to_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .collect();

        let rules: Vec<&Rule> = self
            .rules
            .iter()
            .filter(|r| {
                r.content.to_lowercase().contains(&q)
                    || r.why
                        .as_ref()
                        .map(|w| w.to_lowercase().contains(&q))
                        .unwrap_or(false)
                    || r.instead
                        .as_ref()
                        .map(|i| i.to_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .collect();

        IdentitySearchResults {
            values,
            preferences,
            relationships,
            rules,
        }
    }

    /// Render identity persona only.
    /// Used by identity_setup to show current state.
    pub fn render_persona(&self) -> String {
        let rendered = self.render();
        let trimmed = rendered.trim();
        if trimmed.is_empty() {
            "No identity currently configured".to_string()
        } else {
            trimmed.to_string()
        }
    }

    /// Render identity as a string (for context injection)
    ///
    /// Uses XML-tagged output with clean separation of concerns.
    /// Rules are split into Do/Don't sections per prompt optimization Technique 1
    /// (boundaries as precise as goals).
    pub fn render(&self) -> String {
        let mut out = String::new();

        // Persona
        if !self.persona.name.is_empty() || !self.persona.description.is_empty() {
            out.push_str("<persona>\n");
            let has_name = !self.persona.name.is_empty();
            let has_desc = !self.persona.description.is_empty();
            match (has_name, has_desc) {
                (true, true) => {
                    out.push_str(&format!(
                        "I am {}. {}\n",
                        self.persona.name, self.persona.description
                    ));
                }
                (true, false) => {
                    out.push_str(&format!("I am {}.\n", self.persona.name));
                }
                (false, true) => {
                    out.push_str(&format!("{}\n", self.persona.description));
                }
                _ => {}
            }
            out.push_str("</persona>\n\n");
        }

        // Values
        if !self.values.is_empty() {
            out.push_str("<values>\n");
            for v in &self.values {
                out.push_str(&format!("  • {}", v.principle));
                if let Some(why) = &v.why {
                    out.push_str(&format!("\n    ({})", why));
                }
                out.push('\n');
            }
            out.push_str("</values>\n\n");
        }

        // Preferences
        if !self.preferences.is_empty() {
            out.push_str("<preferences>\n");
            for p in &self.preferences {
                if let Some(over) = &p.over {
                    out.push_str(&format!("  • {} > {}\n", p.prefer, over));
                } else {
                    out.push_str(&format!("  • {}\n", p.prefer));
                }
            }
            out.push_str("</preferences>\n\n");
        }

        // Relationships
        if !self.relationships.is_empty() {
            out.push_str("<relationships>\n");
            for r in &self.relationships {
                out.push_str(&format!("  • {}: {}", r.entity, r.relation));
                if let Some(ctx) = &r.context {
                    out.push_str(&format!("\n    ({})", ctx));
                }
                out.push('\n');
            }
            out.push_str("</relationships>\n\n");
        }

        // Rules — split into Do/Don't
        if !self.rules.is_empty() {
            out.push_str("<rules>\n");

            let (dont_rules, do_rules): (Vec<&Rule>, Vec<&Rule>) =
                self.rules.iter().partition(|r| r.negative);

            if !do_rules.is_empty() {
                out.push_str("Do:\n");
                for r in &do_rules {
                    out.push_str(&format!("  • {}", r.content));
                    if let Some(instead) = &r.instead {
                        out.push_str(&format!(" — {}", instead));
                    }
                    if let Some(why) = &r.why {
                        out.push_str(&format!("\n    ({})", why));
                    }
                    out.push('\n');
                }
            }

            if !do_rules.is_empty() && !dont_rules.is_empty() {
                out.push('\n');
            }

            if !dont_rules.is_empty() {
                out.push_str("Don't:\n");
                for r in &dont_rules {
                    out.push_str(&format!("  \u{2717} {}", r.content));
                    if let Some(instead) = &r.instead {
                        out.push_str(&format!(" \u{2192} {}", instead));
                    }
                    if let Some(why) = &r.why {
                        out.push_str(&format!("\n    ({})", why));
                    }
                    out.push('\n');
                }
            }

            out.push_str("</rules>\n\n");
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_identity() -> Identity {
        Identity::new()
            .with_persona("Porter", "A pragmatic developer assistant")
            .with_value(Value::new("Write exhaustive tests").with_category("engineering"))
            .with_value(Value::new("KISS/DRY principles").with_category("engineering"))
            .with_value(Value::new("Be honest").with_category("ethics"))
            .with_preference(
                Preference::new("Rust")
                    .over("JavaScript")
                    .with_category("languages"),
            )
            .with_preference(
                Preference::new("Beer")
                    .over("Coffee")
                    .with_category("beverages"),
            )
            .with_relationship(
                Relationship::new("Brandon", "The human I work with")
                    .with_context("Lead engineer at Segment"),
            )
            .with_relationship(
                Relationship::new("Ingrid", "Brandon's fiancée").with_context("MSW student"),
            )
            .with_rule(
                Rule::new("Don't ask 'why do you want to do that?'")
                    .negative()
                    .with_instead("Just answer, then ask clarifying questions")
                    .with_why("Forces context dumps"),
            )
            .with_rule(Rule::new("Don't write code unless asked").negative())
            .with_rule(Rule::new("Use full cargo path for builds"))
    }

    #[test]
    fn build_identity_fluently() {
        let identity = build_test_identity();

        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.values.len(), 3);
        assert_eq!(identity.preferences.len(), 2);
        assert_eq!(identity.relationships.len(), 2);
        assert_eq!(identity.rules.len(), 3);
    }

    // =================
    // DISCOVERY TESTS
    // =================

    #[test]
    fn list_value_categories() {
        let identity = build_test_identity();
        let cats = identity.list_value_categories();

        assert_eq!(cats.len(), 2);
        assert!(cats.contains(&"engineering".to_string()));
        assert!(cats.contains(&"ethics".to_string()));
    }

    #[test]
    fn list_preference_categories() {
        let identity = build_test_identity();
        let cats = identity.list_preference_categories();

        assert_eq!(cats.len(), 2);
        assert!(cats.contains(&"languages".to_string()));
        assert!(cats.contains(&"beverages".to_string()));
    }

    #[test]
    fn list_all_categories() {
        let identity = build_test_identity();
        let cats = identity.list_all_categories();

        assert_eq!(cats.len(), 4);
        assert!(cats.contains(&"engineering".to_string()));
        assert!(cats.contains(&"ethics".to_string()));
        assert!(cats.contains(&"languages".to_string()));
        assert!(cats.contains(&"beverages".to_string()));
    }

    #[test]
    fn list_entities() {
        let identity = build_test_identity();
        let entities = identity.list_entities();

        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&"Brandon".to_string()));
        assert!(entities.contains(&"Ingrid".to_string()));
    }

    // =================
    // LOOKUP TESTS
    // =================

    #[test]
    fn filter_by_category() {
        let identity = build_test_identity();

        let eng_values = identity.values_in_category("engineering");
        assert_eq!(eng_values.len(), 2);

        let ethics_values = identity.values_in_category("ethics");
        assert_eq!(ethics_values.len(), 1);
    }

    #[test]
    fn relationship_for_entity() {
        let identity = build_test_identity();

        let rel = identity.relationship_for("Brandon");
        assert!(rel.is_some());
        assert_eq!(rel.unwrap().relation, "The human I work with");

        // Case insensitive
        let rel2 = identity.relationship_for("brandon");
        assert!(rel2.is_some());

        // Not found
        let rel3 = identity.relationship_for("Nobody");
        assert!(rel3.is_none());
    }

    // =================
    // SEARCH TESTS
    // =================

    #[test]
    fn search_finds_values() {
        let identity = build_test_identity();
        let results = identity.search("test");

        assert_eq!(results.values.len(), 1);
        assert!(results.values[0].principle.contains("test"));
    }

    #[test]
    fn search_finds_relationships() {
        let identity = build_test_identity();
        let results = identity.search("Segment");

        assert_eq!(results.relationships.len(), 1);
        assert_eq!(results.relationships[0].entity, "Brandon");
    }

    #[test]
    fn search_finds_rules() {
        let identity = build_test_identity();
        let results = identity.search("cargo");

        assert_eq!(results.rules.len(), 1);
        assert!(results.rules[0].content.contains("cargo"));
    }

    #[test]
    fn search_is_case_insensitive() {
        let identity = build_test_identity();

        let results1 = identity.search("rust");
        let results2 = identity.search("RUST");

        assert_eq!(results1.total_count(), results2.total_count());
        assert!(!results1.is_empty());
    }

    #[test]
    fn search_empty_for_no_match() {
        let identity = build_test_identity();
        let results = identity.search("xyzzy");

        assert!(results.is_empty());
        assert_eq!(results.total_count(), 0);
    }

    // =================
    // RENDER TESTS
    // =================

    #[test]
    fn render_produces_xml_output() {
        let identity = Identity::new()
            .with_persona("TestBot", "A test persona")
            .with_value(Value::new("Testing is good"))
            .with_preference(Preference::new("Green").over("Red"));

        let rendered = identity.render();

        assert!(rendered.contains("<persona>"));
        assert!(rendered.contains("</persona>"));
        assert!(rendered.contains("TestBot"));
        assert!(rendered.contains("<values>"));
        assert!(rendered.contains("Testing is good"));
        assert!(rendered.contains("<preferences>"));
        assert!(rendered.contains("Green > Red"));
    }

    #[test]
    fn render_rules_split_do_dont() {
        let identity = Identity::new()
            .with_persona("TestBot", "A test persona")
            .with_rule(Rule::new("Always run tests"))
            .with_rule(
                Rule::new("Don't use global state")
                    .negative()
                    .with_instead("Use dependency injection"),
            );

        let rendered = identity.render();

        assert!(rendered.contains("<rules>"));
        assert!(rendered.contains("Do:"));
        assert!(rendered.contains("Don't:"));
        assert!(rendered.contains("Always run tests"));
        assert!(rendered.contains("\u{2717} Don't use global state"));
        assert!(rendered.contains("\u{2192} Use dependency injection"));
        assert!(rendered.contains("</rules>"));
    }

    #[test]
    fn render_persona_includes_content() {
        let identity = Identity::new()
            .with_persona("Porter", "A pragmatic assistant");

        let rendered = identity.render_persona();

        assert!(
            rendered.contains("Porter"),
            "render_persona should include persona name"
        );
        assert!(
            rendered.contains("pragmatic"),
            "render_persona should include description"
        );
    }

    #[test]
    fn render_persona_empty_identity() {
        let identity = Identity::new();
        let rendered = identity.render_persona();

        assert_eq!(
            rendered, "No identity currently configured",
            "Empty identity should render as not configured"
        );
    }

    #[test]
    fn render_persona_description_only() {
        let mut identity = Identity::new();
        identity.persona.description = "A pragmatic assistant with deep Rust expertise".to_string();

        let rendered = identity.render();

        assert!(
            rendered.contains("<persona>"),
            "Description-only persona should render persona block"
        );
        assert!(
            rendered.contains("pragmatic assistant"),
            "Should include description text"
        );
        assert!(
            !rendered.contains("I am ."),
            "Should not render 'I am .' when name is empty"
        );

        let persona = identity.render_persona();
        assert_ne!(
            persona, "No identity currently configured",
            "Description-only should not say 'no identity configured'"
        );
    }

    #[test]
    fn render_persona_name_only() {
        let mut identity = Identity::new();
        identity.persona.name = "Porter".to_string();

        let rendered = identity.render();

        assert!(rendered.contains("<persona>"));
        assert!(rendered.contains("I am Porter."));
        assert!(!rendered.contains("I am Porter. \n"));
    }

    #[test]
    fn negative_rule_flag() {
        let positive = Rule::new("Always run tests");
        assert!(!positive.negative);

        let neg = Rule::new("Don't use tokio").negative();
        assert!(neg.negative);
    }
}
