//! Identity tools - persona, values, preferences management

mod add_typed;
mod get;
mod list;
mod remove;
mod search;
mod setup;
// TODO: set.rs needs rework to clear + re-add items
// mod set;

// Core tools
pub use get::IdentityGetTool;
pub use list::IdentityListTool;
pub use remove::IdentityRemoveTool;
pub use search::IdentitySearchTool;
pub use setup::IdentitySetupTool;

// Typed add tools
pub use add_typed::{
    IdentityAddAntipatternTool,
    IdentityAddDirectiveTool,
    IdentityAddExpertiseTool,
    IdentityAddInstructionTool, // v2 - uses new storage
    IdentityAddPreferenceTool,
    IdentityAddRelationshipTool,
    IdentityAddToneTool,
    IdentityAddTraitTool,
    IdentityAddValueTool,
    IdentitySetPersonaDescriptionTool,
    IdentitySetPersonaNameTool,
};
