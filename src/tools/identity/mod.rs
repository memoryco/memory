//! Identity tools - persona, values, preferences management

mod add;
mod get;
mod set;
mod search;
mod add_instruction;
mod remove_instruction;

pub use add::IdentityAddTool;
pub use get::IdentityGetTool;
pub use set::IdentitySetTool;
pub use search::IdentitySearchTool;
pub use add_instruction::IdentityAddInstructionTool;
pub use remove_instruction::IdentityRemoveInstructionTool;
