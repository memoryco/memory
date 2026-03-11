//! Identity field classifier using semantic embeddings
//!
//! Uses k-NN style classification to determine which identity field
//! a piece of content most likely belongs to.

use crate::embedding::EmbeddingGenerator;
use std::sync::OnceLock;

/// The identity field types we can classify into
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityField {
    Value,
    Preference,
    Relationship,
    Expertise,
    Trait,
    Instruction,
    Antipattern,
}

impl IdentityField {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::Preference => "preference",
            Self::Relationship => "relationship",
            Self::Expertise => "expertise",
            Self::Trait => "trait",
            Self::Instruction => "instruction",
            Self::Antipattern => "antipattern",
        }
    }

    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "value" => Some(Self::Value),
            "preference" => Some(Self::Preference),
            "relationship" => Some(Self::Relationship),
            "expertise" => Some(Self::Expertise),
            "trait" => Some(Self::Trait),
            "instruction" => Some(Self::Instruction),
            "antipattern" => Some(Self::Antipattern),
            _ => None,
        }
    }

    /// All field types
    #[allow(dead_code)]
    pub fn all() -> &'static [IdentityField] {
        &[
            Self::Value,
            Self::Preference,
            Self::Relationship,
            Self::Expertise,
            Self::Trait,
            Self::Instruction,
            Self::Antipattern,
        ]
    }
}

impl std::fmt::Display for IdentityField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Example phrases for each field type
struct FieldExamples {
    field: IdentityField,
    examples: &'static [&'static str],
}

/// All the training examples
const FIELD_EXAMPLES: &[FieldExamples] = &[
    FieldExamples {
        field: IdentityField::Value,
        examples: &[
            "Ship working code over perfect code",
            "Be honest even when it's hard",
            "Test everything thoroughly",
            "Quality matters more than speed",
            "Transparency builds trust",
            "Always leave code better than you found it",
            "Simplicity over complexity",
            "User experience comes first",
        ],
    },
    FieldExamples {
        field: IdentityField::Preference,
        examples: &[
            "I prefer Rust over JavaScript",
            "I like beer better than coffee",
            "Tabs over spaces",
            "I prefer dark mode",
            "Morning meetings over afternoon meetings",
            "I like concise responses",
            "Prefer CLI tools over GUIs",
            "I favor functional programming",
        ],
    },
    FieldExamples {
        field: IdentityField::Relationship,
        examples: &[
            "Brandon is my colleague",
            "Ingrid is Brandon's fiancée",
            "Sarah is my manager",
            "I work with the Platform team",
            "John is my mentor",
            "The user is a senior developer",
            "Alice is a stakeholder on this project",
            "My partner's name is Chris",
        ],
    },
    FieldExamples {
        field: IdentityField::Expertise,
        examples: &[
            "I know Rust",
            "I specialize in SDK architecture",
            "Expert in distributed systems",
            "Proficient in Swift and Kotlin",
            "I have experience with machine learning",
            "Strong background in databases",
            "Skilled at API design",
            "I understand compiler internals",
        ],
    },
    FieldExamples {
        field: IdentityField::Trait,
        examples: &[
            "I am pragmatic",
            "I am direct",
            "I am thorough",
            "I'm sarcastic but friendly",
            "I am patient",
            "I'm detail-oriented",
            "I am curious",
            "I'm a perfectionist",
        ],
    },
    FieldExamples {
        field: IdentityField::Instruction,
        examples: &[
            "Always run tests before committing",
            "Check for null values before using them",
            "Use semantic versioning for releases",
            "Document all public APIs",
            "Always ask before deleting files",
            "Format code before submitting PRs",
            "Include error handling in all functions",
            "Run linters before pushing code",
        ],
    },
    FieldExamples {
        field: IdentityField::Antipattern,
        examples: &[
            "Don't ask 'why do you want that'",
            "Avoid premature optimization",
            "Never commit directly to main",
            "Don't use magic numbers",
            "Avoid global state",
            "Don't repeat yourself",
            "Never store passwords in plain text",
            "Avoid deeply nested callbacks",
        ],
    },
];

/// A labeled embedding for classification
struct LabeledEmbedding {
    field: IdentityField,
    embedding: Vec<f32>,
}

/// Cached embeddings for all examples
static EXAMPLE_EMBEDDINGS: OnceLock<Vec<LabeledEmbedding>> = OnceLock::new();

/// Classification result
#[derive(Debug)]
pub struct ClassificationResult {
    /// The predicted field type
    pub predicted: IdentityField,
    /// Confidence score for the prediction (0.0 - 1.0)
    pub confidence: f32,
    /// Scores for all field types
    pub scores: Vec<(IdentityField, f32)>,
}

impl ClassificationResult {
    /// Check if prediction matches the expected field
    #[allow(dead_code)]
    pub fn matches(&self, expected: IdentityField) -> bool {
        self.predicted == expected
    }

    /// Get a warning message if there's a mismatch
    pub fn mismatch_warning(&self, requested: IdentityField) -> Option<String> {
        if self.predicted == requested {
            return None;
        }

        // Find the score for the requested field
        let requested_score = self
            .scores
            .iter()
            .find(|(f, _)| *f == requested)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        // Only warn if there's a meaningful difference
        if self.confidence - requested_score < 0.1 {
            return None; // Close enough, don't be pedantic
        }

        Some(format!(
            "You specified '{}', but this looks more like a '{}' (confidence: {:.0}% vs {:.0}%). Proceeding anyway.",
            requested,
            self.predicted,
            self.confidence * 100.0,
            requested_score * 100.0,
        ))
    }
}

/// Initialize the example embeddings (call once at startup or lazily)
fn init_embeddings() -> Vec<LabeledEmbedding> {
    let generator = EmbeddingGenerator::new();
    let mut embeddings = Vec::new();

    for field_ex in FIELD_EXAMPLES {
        for example in field_ex.examples {
            if let Ok(emb) = generator.generate(example) {
                embeddings.push(LabeledEmbedding {
                    field: field_ex.field,
                    embedding: emb,
                });
            }
        }
    }

    embeddings
}

/// Get or initialize the cached embeddings
fn get_embeddings() -> &'static Vec<LabeledEmbedding> {
    EXAMPLE_EMBEDDINGS.get_or_init(init_embeddings)
}

/// Classify content into an identity field type
pub fn classify(content: &str) -> Option<ClassificationResult> {
    let generator = EmbeddingGenerator::new();
    let content_embedding = generator.generate(content).ok()?;

    let examples = get_embeddings();
    if examples.is_empty() {
        return None;
    }

    // Calculate similarity to each example
    let mut field_scores: std::collections::HashMap<IdentityField, Vec<f32>> =
        std::collections::HashMap::new();

    for labeled in examples {
        let sim = cosine_similarity(&content_embedding, &labeled.embedding);
        field_scores.entry(labeled.field).or_default().push(sim);
    }

    // Average scores per field
    let mut avg_scores: Vec<(IdentityField, f32)> = field_scores
        .into_iter()
        .map(|(field, scores)| {
            let avg = scores.iter().sum::<f32>() / scores.len() as f32;
            (field, avg)
        })
        .collect();

    // Sort by score descending
    avg_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (predicted, confidence) = avg_scores.first().cloned()?;

    Some(ClassificationResult {
        predicted,
        confidence,
        scores: avg_scores,
    })
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_from_str_works() {
        assert_eq!(IdentityField::from_str("value"), Some(IdentityField::Value));
        assert_eq!(IdentityField::from_str("VALUE"), Some(IdentityField::Value));
        assert_eq!(
            IdentityField::from_str("preference"),
            Some(IdentityField::Preference)
        );
        assert_eq!(IdentityField::from_str("unknown"), None);
    }

    #[test]
    fn classify_preference() {
        let result = classify("I like vim better than emacs").unwrap();
        assert_eq!(result.predicted, IdentityField::Preference);
    }

    #[test]
    fn classify_value() {
        let result = classify("Code quality is more important than shipping fast").unwrap();
        assert_eq!(result.predicted, IdentityField::Value);
    }

    #[test]
    fn classify_relationship() {
        let result = classify("Mike is my team lead").unwrap();
        assert_eq!(result.predicted, IdentityField::Relationship);
    }

    #[test]
    fn classify_expertise() {
        let result = classify("I'm proficient in Python and Go").unwrap();
        assert_eq!(result.predicted, IdentityField::Expertise);
    }

    #[test]
    fn classify_trait() {
        let result = classify("I am meticulous and careful").unwrap();
        assert_eq!(result.predicted, IdentityField::Trait);
    }

    #[test]
    fn classify_instruction() {
        let result = classify("Run the test suite before merging pull requests").unwrap();
        assert_eq!(result.predicted, IdentityField::Instruction);
    }

    #[test]
    fn classify_antipattern() {
        let result = classify("Avoid using global mutable state").unwrap();
        assert_eq!(result.predicted, IdentityField::Antipattern);
    }
}
