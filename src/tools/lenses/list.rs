//! lenses_list - List available lenses

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, PromptDef, Tool, ToolEnv};

use crate::Context;
use crate::lenses::load_lenses;
use crate::tools::text_response;

pub struct LensesListTool;

impl Tool<Context> for LensesListTool {
    fn name(&self) -> &str {
        "lenses_list"
    }

    fn description(&self) -> &str {
        "List available lenses."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    fn execute(
        &self,
        _args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let lenses = load_lenses(&context.lenses_dir);

        if lenses.is_empty() {
            return Ok(text_response(format!(
                "No lenses found in {}",
                context.lenses_dir.display()
            )));
        }

        let mut output = format!("Available lenses ({}):\n\n", lenses.len());
        for lens in &lenses {
            let desc = lens
                .description()
                .map(|d| format!(" - {}", d))
                .unwrap_or_default();
            output.push_str(&format!("• {}{}\n", lens.name(), desc));
        }

        Ok(text_response(output))
    }
}
