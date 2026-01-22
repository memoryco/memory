//! MCP tools for the engram memory system.
//! 
//! Tools are organized by category:
//! - `engram` - Memory CRUD and graph operations
//! - `identity` - Persona, values, preferences management
//! - `config` - System configuration

pub mod engram;
pub mod identity;
pub mod config;

use ::engram::Engram;
use sml_mcps::CallToolResult;
use std::path::PathBuf;

// Re-export all tools for easy registration
pub use engram::{
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
    IdentityGetTool,
    IdentitySetTool,
    IdentitySearchTool,
};

pub use config::{
    ConfigGetTool,
    ConfigSetTool,
};

// =============================================================================
// Common helpers
// =============================================================================

/// HTML template for graph visualization (embedded at compile time)
pub const GRAPH_TEMPLATE: &str = include_str!("../../templates/graph.html");

/// Get the output path for the graph HTML file
pub fn get_graph_output_path() -> PathBuf {
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."));
    data_dir.join("memory").join("graph.html")
}

/// Create a simple text response
pub fn text_response(text: String) -> CallToolResult {
    CallToolResult::text(text)
}

/// Format an engram for display
pub fn format_engram(e: &Engram) -> String {
    format!(
        "ID: {}\nContent: {}\nState: {} (energy: {:.2})\nTags: {:?}\nAccess count: {}\nCreated: {}",
        e.id,
        e.content,
        e.state.emoji(),
        e.energy,
        e.tags,
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
