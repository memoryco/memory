//! Identity - The bedrock layer of self
//!
//! Unlike episodic/semantic memories in the Substrate, identity doesn't decay.
//! It's not "remembered" - it just IS. The lens you see through, not the thing you look at.

use serde::{Serialize, Deserialize};

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
    
    /// Anti-patterns - what NOT to do, learned boundaries
    #[serde(default)]
    pub antipatterns: Vec<Antipattern>,
    
    /// Communication style directives
    #[serde(default)]
    pub communication: CommunicationStyle,
    
    /// Areas of expertise / competence
    #[serde(default)]
    pub expertise: Vec<String>,
    
    /// Operational instructions - how to behave, system rules
    /// These are permanent directives that should never decay
    #[serde(default)]
    pub instructions: Vec<String>,
}

/// The fundamental "I am" - name, nature, essence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Persona {
    /// Name / identifier
    #[serde(default)]
    pub name: String,
    
    /// Short description of nature
    #[serde(default)]
    pub description: String,
    
    /// Core traits (adjectives that define you)
    #[serde(default)]
    pub traits: Vec<String>,
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

/// An anti-pattern - something to NOT do
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Antipattern {
    /// What not to do
    #[serde(default)]
    pub avoid: String,
    
    /// Why (the harm it causes)
    #[serde(default)]
    pub why: Option<String>,
    
    /// What to do instead
    #[serde(default)]
    pub instead: Option<String>,
}

impl Antipattern {
    pub fn new(avoid: impl Into<String>) -> Self {
        Self {
            avoid: avoid.into(),
            why: None,
            instead: None,
        }
    }
    
    pub fn because(mut self, why: impl Into<String>) -> Self {
        self.why = Some(why.into());
        self
    }
    
    pub fn instead(mut self, alternative: impl Into<String>) -> Self {
        self.instead = Some(alternative.into());
        self
    }
}

/// Communication style preferences
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// General tone descriptors
    #[serde(default)]
    pub tone: Vec<String>,
    
    /// Specific directives
    #[serde(default)]
    pub directives: Vec<String>,
}

/// Results from searching identity
#[derive(Debug)]
pub struct IdentitySearchResults<'a> {
    pub values: Vec<&'a Value>,
    pub preferences: Vec<&'a Preference>,
    pub relationships: Vec<&'a Relationship>,
    pub antipatterns: Vec<&'a Antipattern>,
    pub expertise: Vec<&'a String>,
    pub traits: Vec<&'a String>,
    pub instructions: Vec<&'a String>,
}

impl<'a> IdentitySearchResults<'a> {
    /// Check if any results were found
    pub fn is_empty(&self) -> bool {
        self.values.is_empty() &&
        self.preferences.is_empty() &&
        self.relationships.is_empty() &&
        self.antipatterns.is_empty() &&
        self.expertise.is_empty() &&
        self.traits.is_empty() &&
        self.instructions.is_empty()
    }
    
    /// Total count of all matches
    pub fn total_count(&self) -> usize {
        self.values.len() +
        self.preferences.len() +
        self.relationships.len() +
        self.antipatterns.len() +
        self.expertise.len() +
        self.traits.len() +
        self.instructions.len()
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
    
    /// Add a trait to the persona
    pub fn with_trait(mut self, t: impl Into<String>) -> Self {
        self.persona.traits.push(t.into());
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
    
    /// Add an anti-pattern
    pub fn with_antipattern(mut self, ap: Antipattern) -> Self {
        self.antipatterns.push(ap);
        self
    }
    
    /// Add expertise area
    pub fn with_expertise(mut self, area: impl Into<String>) -> Self {
        self.expertise.push(area.into());
        self
    }
    
    /// Add communication tone
    pub fn with_tone(mut self, tone: impl Into<String>) -> Self {
        self.communication.tone.push(tone.into());
        self
    }
    
    /// Add communication directive
    pub fn with_directive(mut self, directive: impl Into<String>) -> Self {
        self.communication.directives.push(directive.into());
        self
    }
    
    /// Add operational instruction
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instructions.push(instruction.into());
        self
    }
    
    // ==================
    // DISCOVERY METHODS
    // ==================
    
    /// List all unique categories used in values
    pub fn list_value_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self.values.iter()
            .filter_map(|v| v.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }
    
    /// List all unique categories used in preferences
    pub fn list_preference_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self.preferences.iter()
            .filter_map(|p| p.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }
    
    /// List all unique categories (from both values and preferences)
    pub fn list_all_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self.values.iter()
            .filter_map(|v| v.category.clone())
            .chain(self.preferences.iter().filter_map(|p| p.category.clone()))
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }
    
    /// List all entity names from relationships
    pub fn list_entities(&self) -> Vec<String> {
        self.relationships.iter()
            .map(|r| r.entity.clone())
            .collect()
    }
    
    /// List all expertise areas
    pub fn list_expertise(&self) -> Vec<String> {
        self.expertise.clone()
    }
    
    /// List all traits
    pub fn list_traits(&self) -> Vec<String> {
        self.persona.traits.clone()
    }
    
    // ==================
    // LOOKUP METHODS  
    // ==================
    
    /// Get values by category
    pub fn values_in_category(&self, category: &str) -> Vec<&Value> {
        self.values.iter()
            .filter(|v| v.category.as_deref() == Some(category))
            .collect()
    }
    
    /// Get preferences by category
    pub fn preferences_in_category(&self, category: &str) -> Vec<&Preference> {
        self.preferences.iter()
            .filter(|p| p.category.as_deref() == Some(category))
            .collect()
    }
    
    /// Get relationship by entity name
    pub fn relationship_for(&self, entity: &str) -> Option<&Relationship> {
        self.relationships.iter()
            .find(|r| r.entity.eq_ignore_ascii_case(entity))
    }
    
    /// Check if a trait exists
    pub fn has_trait(&self, t: &str) -> bool {
        self.persona.traits.iter()
            .any(|trait_| trait_.eq_ignore_ascii_case(t))
    }
    
    /// Check if expertise exists
    pub fn has_expertise(&self, area: &str) -> bool {
        self.expertise.iter()
            .any(|e| e.eq_ignore_ascii_case(area))
    }
    
    // ==================
    // SEARCH METHODS
    // ==================
    
    /// Search all identity content for a term (case-insensitive)
    /// Returns a summary of where matches were found
    pub fn search(&self, query: &str) -> IdentitySearchResults<'_> {
        let q = query.to_lowercase();
        
        let values: Vec<&Value> = self.values.iter()
            .filter(|v| {
                v.principle.to_lowercase().contains(&q) ||
                v.why.as_ref().map(|w| w.to_lowercase().contains(&q)).unwrap_or(false)
            })
            .collect();
        
        let preferences: Vec<&Preference> = self.preferences.iter()
            .filter(|p| {
                p.prefer.to_lowercase().contains(&q) ||
                p.over.as_ref().map(|o| o.to_lowercase().contains(&q)).unwrap_or(false)
            })
            .collect();
        
        let relationships: Vec<&Relationship> = self.relationships.iter()
            .filter(|r| {
                r.entity.to_lowercase().contains(&q) ||
                r.relation.to_lowercase().contains(&q) ||
                r.context.as_ref().map(|c| c.to_lowercase().contains(&q)).unwrap_or(false)
            })
            .collect();
        
        let antipatterns: Vec<&Antipattern> = self.antipatterns.iter()
            .filter(|a| {
                a.avoid.to_lowercase().contains(&q) ||
                a.why.as_ref().map(|w| w.to_lowercase().contains(&q)).unwrap_or(false) ||
                a.instead.as_ref().map(|i| i.to_lowercase().contains(&q)).unwrap_or(false)
            })
            .collect();
        
        let expertise: Vec<&String> = self.expertise.iter()
            .filter(|e| e.to_lowercase().contains(&q))
            .collect();
        
        let traits: Vec<&String> = self.persona.traits.iter()
            .filter(|t| t.to_lowercase().contains(&q))
            .collect();
        
        let instructions: Vec<&String> = self.instructions.iter()
            .filter(|i| i.to_lowercase().contains(&q))
            .collect();
        
        IdentitySearchResults {
            values,
            preferences,
            relationships,
            antipatterns,
            expertise,
            traits,
            instructions,
        }
    }
    
    /// Render identity persona only (excludes operational instructions).
    /// Used by identity_setup to show current state without cluttering
    /// the onboarding guide with bootstrap instructions.
    pub fn render_persona(&self) -> String {
        let mut copy = self.clone();
        copy.instructions.clear();
        let rendered = copy.render();
        let trimmed = rendered.trim();
        if trimmed.is_empty() {
            "No identity currently configured".to_string()
        } else {
            trimmed.to_string()
        }
    }

    /// Render identity as a string (for context injection)
    pub fn render(&self) -> String {
        let mut out = String::new();
        
        // Persona
        if !self.persona.name.is_empty() {
            out.push_str(&format!("I am {}. {}\n", self.persona.name, self.persona.description));
            if !self.persona.traits.is_empty() {
                out.push_str(&format!("Core traits: {}\n", self.persona.traits.join(", ")));
            }
            out.push('\n');
        }
        
        // Values
        if !self.values.is_empty() {
            out.push_str("Values:\n");
            for v in &self.values {
                out.push_str(&format!("  • {}", v.principle));
                if let Some(why) = &v.why {
                    out.push_str(&format!(" ({})", why));
                }
                out.push('\n');
            }
            out.push('\n');
        }
        
        // Preferences
        if !self.preferences.is_empty() {
            out.push_str("Preferences:\n");
            for p in &self.preferences {
                if let Some(over) = &p.over {
                    out.push_str(&format!("  • {} > {}\n", p.prefer, over));
                } else {
                    out.push_str(&format!("  • {}\n", p.prefer));
                }
            }
            out.push('\n');
        }
        
        // Relationships
        if !self.relationships.is_empty() {
            out.push_str("Key relationships:\n");
            for r in &self.relationships {
                out.push_str(&format!("  • {}: {}", r.entity, r.relation));
                if let Some(ctx) = &r.context {
                    out.push_str(&format!(" ({})", ctx));
                }
                out.push('\n');
            }
            out.push('\n');
        }
        
        // Anti-patterns
        if !self.antipatterns.is_empty() {
            out.push_str("Avoid:\n");
            for ap in &self.antipatterns {
                out.push_str(&format!("  ✗ {}", ap.avoid));
                if let Some(instead) = &ap.instead {
                    out.push_str(&format!(" → instead: {}", instead));
                }
                out.push('\n');
            }
            out.push('\n');
        }
        
        // Communication
        if !self.communication.tone.is_empty() || !self.communication.directives.is_empty() {
            out.push_str("Communication style:\n");
            if !self.communication.tone.is_empty() {
                out.push_str(&format!("  Tone: {}\n", self.communication.tone.join(", ")));
            }
            for d in &self.communication.directives {
                out.push_str(&format!("  • {}\n", d));
            }
            out.push('\n');
        }
        
        // Expertise
        if !self.expertise.is_empty() {
            out.push_str(&format!("Expertise: {}\n", self.expertise.join(", ")));
            out.push('\n');
        }
        
        // Instructions
        if !self.instructions.is_empty() {
            out.push_str("Instructions:\n");
            for i in &self.instructions {
                out.push_str(&format!("  • {}\n", i));
            }
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
            .with_trait("pragmatic")
            .with_trait("direct")
            .with_trait("snarky but not a douchebag")
            .with_value(Value::new("Write exhaustive tests").with_category("engineering"))
            .with_value(Value::new("KISS/DRY principles").with_category("engineering"))
            .with_value(Value::new("Be honest").with_category("ethics"))
            .with_preference(Preference::new("Rust").over("JavaScript").with_category("languages"))
            .with_preference(Preference::new("Beer").over("Coffee").with_category("beverages"))
            .with_relationship(Relationship::new("Brandon", "The human I work with").with_context("Lead engineer at Segment"))
            .with_relationship(Relationship::new("Ingrid", "Brandon's fiancée").with_context("MSW student"))
            .with_antipattern(Antipattern::new("Asking 'why do you want to do that?'").because("Forces context dumps").instead("Just answer, then ask clarifying questions"))
            .with_antipattern(Antipattern::new("Writing code unless asked"))
            .with_tone("direct")
            .with_tone("warm")
            .with_directive("Ask questions over making assumptions")
            .with_expertise("Rust")
            .with_expertise("Swift")
            .with_expertise("SDK architecture")
    }
    
    #[test]
    fn build_identity_fluently() {
        let identity = build_test_identity();
        
        assert_eq!(identity.persona.name, "Porter");
        assert_eq!(identity.persona.traits.len(), 3);
        assert_eq!(identity.values.len(), 3);
        assert_eq!(identity.preferences.len(), 2);
        assert_eq!(identity.relationships.len(), 2);
        assert_eq!(identity.antipatterns.len(), 2);
        assert_eq!(identity.expertise.len(), 3);
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
    
    #[test]
    fn list_expertise() {
        let identity = build_test_identity();
        let exp = identity.list_expertise();
        
        assert_eq!(exp.len(), 3);
        assert!(exp.contains(&"Rust".to_string()));
    }
    
    #[test]
    fn list_traits() {
        let identity = build_test_identity();
        let traits = identity.list_traits();
        
        assert_eq!(traits.len(), 3);
        assert!(traits.contains(&"pragmatic".to_string()));
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
    
    #[test]
    fn has_trait_check() {
        let identity = build_test_identity();
        
        assert!(identity.has_trait("pragmatic"));
        assert!(identity.has_trait("PRAGMATIC")); // case insensitive
        assert!(!identity.has_trait("lazy"));
    }
    
    #[test]
    fn has_expertise_check() {
        let identity = build_test_identity();
        
        assert!(identity.has_expertise("Rust"));
        assert!(identity.has_expertise("rust")); // case insensitive
        assert!(!identity.has_expertise("COBOL"));
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
    fn search_finds_expertise() {
        let identity = build_test_identity();
        let results = identity.search("SDK");
        
        assert_eq!(results.expertise.len(), 1);
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
    fn render_produces_output() {
        let identity = Identity::new()
            .with_persona("TestBot", "A test persona")
            .with_value(Value::new("Testing is good"))
            .with_preference(Preference::new("Green").over("Red"));
        
        let rendered = identity.render();
        
        assert!(rendered.contains("TestBot"));
        assert!(rendered.contains("Testing is good"));
        assert!(rendered.contains("Green > Red"));
    }

    #[test]
    fn render_persona_excludes_instructions() {
        let identity = Identity::new()
            .with_persona("Porter", "A pragmatic assistant")
            .with_trait("direct")
            .with_instruction("Always search before responding".to_string());

        let rendered = identity.render_persona();

        assert!(rendered.contains("Porter"),
            "render_persona should include persona name");
        assert!(rendered.contains("direct"),
            "render_persona should include traits");
        assert!(!rendered.contains("Always search"),
            "render_persona must NOT include instructions");
    }

    #[test]
    fn render_persona_empty_identity() {
        let identity = Identity::new();
        let rendered = identity.render_persona();

        assert_eq!(rendered, "No identity currently configured",
            "Empty identity should render as not configured");
    }

    #[test]
    fn render_persona_instructions_only() {
        // Identity with only instructions (like a fresh bootstrap)
        let identity = Identity::new()
            .with_instruction("## Memory Workflow\nDo stuff".to_string());

        let rendered = identity.render_persona();

        assert_eq!(rendered, "No identity currently configured",
            "Instructions-only identity should render as not configured for persona");
    }
}
