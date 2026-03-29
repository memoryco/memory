//! MCP tools for the engram memory system.
//!
//! Tools are organized by category:
//! - `engram` - Memory CRUD and graph operations
//! - `identity` - Persona, values, preferences management
//! - `config` - System configuration
//! - `lenses` - Task-specific context guides
//! - `reference` - Authoritative document search and citation

pub mod config;
pub mod date_resolve;
pub mod engram;
pub mod identity;
pub mod lenses;
pub mod open_dashboard;

pub mod reference;

use crate::engram::Engram;
use sml_mcps::{CallToolResult, Content};

// Re-export all tools for easy registration
pub use engram::{
    EngramAssociateTool,
    EngramAssociationsTool,
    // tools::engram (tool implementations)
    EngramCreateTool,
    EngramDeleteTool,
    EngramGetTool,
    EngramGraphTool,
    EngramRecallTool,
    EngramSearchTool,
    EngramStatsTool,
};

pub use identity::{
    IdentityAddAntipatternTool,
    IdentityAddDirectiveTool,
    IdentityAddExpertiseTool,
    IdentityAddInstructionTool,
    IdentityAddPreferenceTool,
    IdentityAddRelationshipTool,
    IdentityAddToneTool,
    IdentityAddTraitTool,
    IdentityAddValueTool,
    // Core identity tools
    IdentityGetTool,
    IdentityListTool,
    IdentityRemoveTool,
    IdentitySearchTool,
    IdentitySetPersonaDescriptionTool,
    // Typed add tools
    IdentitySetPersonaNameTool,
    IdentitySetupTool,
};

pub use config::{ConfigGetTool, ConfigSetTool};

pub use lenses::{LensesGetTool, LensesListTool};

pub use reference::{
    ReferenceCitationTool, ReferenceGetTool, ReferenceListTool, ReferenceSearchTool,
    ReferenceSectionsTool,
};

pub use date_resolve::DateResolveTool;
pub use open_dashboard::OpenDashboardTool;

// =============================================================================
// Common helpers
// =============================================================================

/// HTML template for graph visualization (embedded at compile time)
pub const GRAPH_TEMPLATE: &str = include_str!("../../templates/graph.html");

/// Create a simple text response
pub fn text_response(text: String) -> CallToolResult {
    CallToolResult::text(text)
}

/// Extract text content from a CallToolResult
pub fn extract_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format an engram for display
pub fn format_engram(e: &Engram) -> String {
    format!(
        "ID: {}\nContent: {}\nState: {} (energy: {:.2})\nAccess count: {}\nCreated: {}",
        e.id,
        e.content,
        e.state.emoji(),
        e.energy,
        e.access_count,
        e.created_at
    )
}

/// Truncate content for display
pub fn truncate_content(content: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }

    if content.len() <= max_len {
        content.to_string()
    } else {
        if max_len <= 3 {
            return ".".repeat(max_len);
        }

        let prefix_budget = max_len - 3;
        let mut end = prefix_budget.min(content.len());
        while end > 0 && !content.is_char_boundary(end) {
            end -= 1;
        }

        let prefix = content.get(..end).unwrap_or_default();
        format!("{}...", prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_content;

    #[test]
    fn truncate_content_unicode_boundary() {
        let input =
            "Step: Run full test suite — cargo test in /work/memoryco/memory, all tests must pass";
        let result = truncate_content(input, 30);
        assert_eq!(result, "Step: Run full test suite ...");
        assert!(result.len() <= 30);
    }

    #[test]
    fn truncate_content_tiny_max_len() {
        assert_eq!(truncate_content("abcdef", 0), "");
        assert_eq!(truncate_content("abcdef", 1), ".");
        assert_eq!(truncate_content("abcdef", 2), "..");
        assert_eq!(truncate_content("abcdef", 3), "...");
    }
}
