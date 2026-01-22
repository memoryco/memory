//! MemoryPlugin trait
//!
//! Defines the interface for memory system components (engram, identity, reference, lenses).
//! Each plugin registers its tools and provides instructions for the LLM.

use sml_mcps::Server;

/// A plugin that provides tools and instructions for the memory system.
///
/// Each plugin (engram, identity, reference, lenses) implements this trait
/// to register its tools with the server and provide LLM instructions.
pub trait MemoryPlugin<C>: Send + Sync {
    /// Plugin name (e.g., "engram", "identity", "reference", "lenses")
    fn name(&self) -> &'static str;

    /// Instructions for the LLM on how to use this plugin's tools.
    /// These get combined into the server's overall instructions.
    fn instructions(&self) -> &'static str;

    /// Register this plugin's tools with the server.
    fn register_tools(&self, server: &mut Server<C>);
}

/// Collects instructions from multiple plugins into a single string.
pub fn combine_instructions<C>(plugins: &[&dyn MemoryPlugin<C>]) -> String {
    let mut combined = String::new();
    
    for plugin in plugins {
        if !combined.is_empty() {
            combined.push_str("\n\n---\n\n");
        }
        combined.push_str(&format!("## {}\n\n", plugin.name()));
        combined.push_str(plugin.instructions());
    }
    
    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct MockContext;
    
    struct TestPlugin {
        name: &'static str,
        instructions: &'static str,
    }
    
    impl MemoryPlugin<MockContext> for TestPlugin {
        fn name(&self) -> &'static str {
            self.name
        }
        
        fn instructions(&self) -> &'static str {
            self.instructions
        }
        
        fn register_tools(&self, _server: &mut Server<MockContext>) {
            // No tools for test
        }
    }
    
    #[test]
    fn combine_single_plugin() {
        let plugin = TestPlugin {
            name: "test",
            instructions: "Do the thing.",
        };
        
        let plugins: Vec<&dyn MemoryPlugin<MockContext>> = vec![&plugin];
        let combined = combine_instructions(&plugins);
        
        assert!(combined.contains("## test"));
        assert!(combined.contains("Do the thing."));
    }
    
    #[test]
    fn combine_multiple_plugins() {
        let p1 = TestPlugin {
            name: "alpha",
            instructions: "Alpha instructions.",
        };
        let p2 = TestPlugin {
            name: "beta", 
            instructions: "Beta instructions.",
        };
        
        let plugins: Vec<&dyn MemoryPlugin<MockContext>> = vec![&p1, &p2];
        let combined = combine_instructions(&plugins);
        
        assert!(combined.contains("## alpha"));
        assert!(combined.contains("Alpha instructions."));
        assert!(combined.contains("---"));
        assert!(combined.contains("## beta"));
        assert!(combined.contains("Beta instructions."));
    }
}
