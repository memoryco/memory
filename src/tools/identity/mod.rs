//! Identity tools - persona, values, preferences management

mod add_typed;
mod get;
mod list;
mod remove;
mod search;
// TODO: set.rs needs rework to clear + re-add items
// mod set;

// Core tools
pub use get::IdentityGetTool;
pub use search::IdentitySearchTool;
pub use list::IdentityListTool;
pub use remove::IdentityRemoveTool;

// Typed add tools
pub use add_typed::{
    IdentitySetPersonaNameTool,
    IdentitySetPersonaDescriptionTool,
    IdentityAddTraitTool,
    IdentityAddExpertiseTool,
    IdentityAddInstructionTool,  // v2 - uses new storage
    IdentityAddToneTool,
    IdentityAddDirectiveTool,
    IdentityAddValueTool,
    IdentityAddPreferenceTool,
    IdentityAddRelationshipTool,
    IdentityAddAntipatternTool,
};
