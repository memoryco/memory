//! MCP tools for the engram memory system.
//! 
//! Tools are organized by category:
//! - `engram` - Memory CRUD and graph operations
//! - `identity` - Persona, values, preferences management
//! - `config` - System configuration
//! - `lenses` - Task-specific context guides
//! - `reference` - Authoritative document search and citation

pub mod engram;
pub mod identity;
pub mod config;
pub mod lenses;
pub mod plans;
pub mod reference;

use crate::engram::Engram;
use sml_mcps::CallToolResult;

// Re-export all tools for easy registration
pub use engram::{  // tools::engram (tool implementations)
    EngramCreateTool,
    EngramRecallTool,
    EngramSearchTool,
    EngramGetTool,
    EngramDeleteTool,
    EngramAssociateTool,
    EngramStatsTool,
    EngramAssociationsTool,
    EngramGraphTool,
};

pub use identity::{
    IdentityAddTool,
    IdentityGetTool,
    IdentitySetTool,
    IdentitySearchTool,
    IdentityAddInstructionTool,
    IdentityRemoveInstructionTool,
};

pub use config::{
    ConfigGetTool,
    ConfigSetTool,
};

pub use lenses::{
    LensesListTool,
    LensesGetTool,
};

pub use reference::{
    ReferenceListTool,
    ReferenceSearchTool,
    ReferenceGetTool,
    ReferenceSectionsTool,
    ReferenceCitationTool,
};

pub use plans::{
    PlansListTool,
    PlanGetTool,
    PlanStartTool,
    PlanStopTool,
    StepAddTool,
    StepCompleteTool,
};

// =============================================================================
// Common helpers
// =============================================================================

/// HTML template for graph visualization (embedded at compile time)
pub const GRAPH_TEMPLATE: &str = include_str!("../../templates/graph.html");

/// Create a simple text response
pub fn text_response(text: String) -> CallToolResult {
    CallToolResult::text(text)
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
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len.saturating_sub(3)])
    }
}
