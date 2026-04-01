//! instructions - Returns all operational instructions
//!
//! Concatenates hardcoded const instruction strings from each module,
//! plugin instructions from bootstrap.d/ TOML files, and per-source
//! citation instructions from loaded references.

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct InstructionsTool;

impl Tool<Context> for InstructionsTool {
    fn name(&self) -> &str {
        "instructions"
    }

    fn description(&self) -> &str {
        "Get all operational instructions. Call on first message \
         or when you need to review behavioral guidelines."
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
        let mut sections: Vec<String> = Vec::new();

        // 1. Hardcoded instruction consts from each module
        sections.push(crate::memory_core::bootstrap::INSTRUCTIONS.to_string());
        sections.push(crate::lenses::bootstrap::INSTRUCTIONS.to_string());
        sections.push(crate::reference::bootstrap::INSTRUCTIONS.to_string());

        // 2. Per-source citation instructions from loaded references
        {
            let refs = context.references.lock().unwrap();
            let citation_text = crate::reference::bootstrap::generate_citation_instructions(&refs);
            if !citation_text.is_empty() {
                sections.push(citation_text);
            }
        }

        // 3. Plugin instructions from bootstrap.d/ TOML files
        let plugin_instructions =
            crate::plugins::bootstrap::load_plugin_instructions(&context.memory_home);
        for instruction in plugin_instructions {
            sections.push(instruction);
        }

        let output = sections.join("\n\n---\n\n");
        Ok(text_response(output))
    }
}

#[cfg(test)]
mod tests {
    use crate::reference::ReferenceManager;
    use tempfile::TempDir;

    /// Helper: assemble instructions the same way the tool does, but without
    /// needing a full Context + ToolEnv.
    fn assemble_instructions(memory_home: &std::path::Path, refs: &ReferenceManager) -> String {
        let mut sections: Vec<String> = Vec::new();

        sections.push(crate::memory_core::bootstrap::INSTRUCTIONS.to_string());
        sections.push(crate::lenses::bootstrap::INSTRUCTIONS.to_string());
        sections.push(crate::reference::bootstrap::INSTRUCTIONS.to_string());

        let citation_text = crate::reference::bootstrap::generate_citation_instructions(refs);
        if !citation_text.is_empty() {
            sections.push(citation_text);
        }

        let plugin_instructions =
            crate::plugins::bootstrap::load_plugin_instructions(memory_home);
        for instruction in plugin_instructions {
            sections.push(instruction);
        }

        sections.join("\n\n---\n\n")
    }

    #[test]
    fn returns_hardcoded_instructions() {
        let tmp = TempDir::new().unwrap();
        let refs = ReferenceManager::new();
        let text = assemble_instructions(tmp.path(), &refs);

        assert!(text.contains("## Workflow"), "Missing memory instructions");
        assert!(text.contains("Lenses"), "Missing lenses instructions");
        assert!(text.contains("References"), "Missing reference instructions");
    }

    #[test]
    fn picks_up_plugin_content() {
        let tmp = TempDir::new().unwrap();
        let bootstrap_dir = tmp.path().join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(
            bootstrap_dir.join("testplugin.toml"),
            "[meta]\nname = \"testplugin\"\nversion = \"0.1.0\"\n\n[instructions]\ncontent = \"Test Plugin instructions for testing.\"",
        )
        .unwrap();

        let refs = ReferenceManager::new();
        let text = assemble_instructions(tmp.path(), &refs);

        assert!(
            text.contains("Test Plugin"),
            "Should include plugin instructions"
        );
    }

    #[test]
    fn skips_invalid_toml_gracefully() {
        let tmp = TempDir::new().unwrap();
        let bootstrap_dir = tmp.path().join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(bootstrap_dir.join("broken.toml"), "not valid toml {{{").unwrap();

        let refs = ReferenceManager::new();
        let text = assemble_instructions(tmp.path(), &refs);

        // Should still have the hardcoded instructions
        assert!(text.contains("## Workflow"));
    }

    #[test]
    fn works_with_no_bootstrap_dir() {
        let tmp = TempDir::new().unwrap();
        // Don't create bootstrap.d/
        let refs = ReferenceManager::new();
        let text = assemble_instructions(tmp.path(), &refs);

        assert!(text.contains("## Workflow"));
    }

    #[test]
    fn works_with_empty_bootstrap_dir() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("bootstrap.d")).unwrap();

        let refs = ReferenceManager::new();
        let text = assemble_instructions(tmp.path(), &refs);

        assert!(text.contains("## Workflow"));
        assert!(text.contains("Lenses"));
        assert!(text.contains("References"));
    }
}
