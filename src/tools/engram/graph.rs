//! engram_graph - Get full association graph for visualization

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};
use std::collections::HashSet;

use crate::Context;
use crate::tools::{text_response, truncate_content, GRAPH_TEMPLATE};

pub struct EngramGraphTool;

#[derive(Deserialize)]
struct Args {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    min_weight: Option<f64>,
}

impl Tool<Context> for EngramGraphTool {
    fn name(&self) -> &str {
        "engram_graph"
    }

    fn description(&self) -> &str {
        "Get the full association graph for visualization. Use format='html' to \
         generate an interactive D3.js visualization saved to a local file (returns \
         file path). Use 'json' for raw data or 'summary' for a text overview. \
         Use min_weight to filter weak associations."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["json", "summary", "html"],
                    "description": "Output format. 'html' for interactive visualization (recommended), 'json' for raw data, 'summary' for text overview. Default: summary"
                },
                "min_weight": {
                    "type": "number",
                    "description": "Minimum association weight to include (0.0-1.0). Default: 0.0 (all)"
                }
            }
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let format = args.format.as_deref().unwrap_or("summary");
        let min_weight = args.min_weight.unwrap_or(0.0);
        let mut brain = context.brain.lock().unwrap();
        
        // Lazy decay - check if interval elapsed, apply if so
        let _ = brain.apply_time_decay();

        // Collect all associations
        let all_assocs = brain.all_associations();
        let filtered: Vec<_> = all_assocs.iter()
            .filter(|a| a.weight >= min_weight)
            .collect();

        // Collect all node IDs that are part of associations
        let mut node_ids: HashSet<EngramId> = HashSet::new();
        for assoc in &filtered {
            node_ids.insert(assoc.from);
            node_ids.insert(assoc.to);
        }

        if format == "json" {
            // Build JSON structure for visualization
            let nodes: Vec<JsonValue> = node_ids.iter()
                .filter_map(|id| brain.get(id))
                .map(|e| json!({
                    "id": e.id.to_string(),
                    "label": truncate_content(&e.content, 30),
                    "energy": e.energy,
                    "state": e.state.emoji(),
                    "access_count": e.access_count
                }))
                .collect();

            let edges: Vec<JsonValue> = filtered.iter()
                .map(|a| json!({
                    "from": a.from.to_string(),
                    "to": a.to.to_string(),
                    "weight": a.weight,
                    "co_activations": a.co_activation_count,
                    "created_at": a.created_at,
                    "last_activated": a.last_activated
                }))
                .collect();

            let graph = json!({
                "nodes": nodes,
                "edges": edges,
                "meta": {
                    "total_nodes": nodes.len(),
                    "total_edges": edges.len(),
                    "min_weight_filter": min_weight
                }
            });

            Ok(text_response(serde_json::to_string_pretty(&graph).unwrap()))
        } else if format == "html" {
            // HTML visualization - generate file and return path
            let nodes: Vec<JsonValue> = node_ids.iter()
                .filter_map(|id| brain.get(id))
                .map(|e| json!({
                    "id": e.id.to_string(),
                    "label": truncate_content(&e.content, 30),
                    "energy": e.energy,
                    "state": e.state.emoji(),
                    "access_count": e.access_count
                }))
                .collect();

            let edges: Vec<JsonValue> = filtered.iter()
                .map(|a| json!({
                    "source": a.from.to_string(),
                    "target": a.to.to_string(),
                    "weight": a.weight,
                    "co_activations": a.co_activation_count
                }))
                .collect();

            let graph = json!({
                "nodes": nodes,
                "edges": edges
            });

            // Inject data into template
            let graph_json = serde_json::to_string(&graph)
                .map_err(|e| McpError::ToolError(format!("Failed to serialize graph: {}", e)))?;
            let html = GRAPH_TEMPLATE.replace("{{GRAPH_DATA}}", &graph_json);

            // Write to file
            let output_path = context.memory_home.join("graph.html");
            std::fs::write(&output_path, &html)
                .map_err(|e| McpError::ToolError(format!("Failed to write graph.html: {}", e)))?;

            Ok(text_response(format!(
                "Graph visualization generated!\n\n\
                 📊 {} nodes, {} edges\n\
                 📁 File: {}\n\n\
                 Open in browser: file://{}",
                nodes.len(),
                edges.len(),
                output_path.display(),
                output_path.display()
            )))
        } else {
            // Human-readable summary
            let mut output = String::new();
            output.push_str(&format!(
                "=== MEMORY GRAPH ===\nNodes: {} | Edges: {} | Min weight: {:.2}\n\n",
                node_ids.len(),
                filtered.len(),
                min_weight
            ));

            // Group edges by weight buckets
            let strong: Vec<_> = filtered.iter().filter(|a| a.weight >= 0.7).collect();
            let medium: Vec<_> = filtered.iter().filter(|a| a.weight >= 0.4 && a.weight < 0.7).collect();
            let weak: Vec<_> = filtered.iter().filter(|a| a.weight < 0.4).collect();

            output.push_str(&format!("Strong connections (≥0.7): {}\n", strong.len()));
            for a in strong.iter().take(10) {
                let from_label = brain.get(&a.from).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                let to_label = brain.get(&a.to).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                output.push_str(&format!("  {} → {} (w:{:.2}, co:{})\n", from_label, to_label, a.weight, a.co_activation_count));
            }
            if strong.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", strong.len() - 10));
            }

            output.push_str(&format!("\nMedium connections (0.4-0.7): {}\n", medium.len()));
            for a in medium.iter().take(10) {
                let from_label = brain.get(&a.from).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                let to_label = brain.get(&a.to).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                output.push_str(&format!("  {} → {} (w:{:.2}, co:{})\n", from_label, to_label, a.weight, a.co_activation_count));
            }
            if medium.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", medium.len() - 10));
            }

            output.push_str(&format!("\nWeak connections (<0.4): {}\n", weak.len()));
            for a in weak.iter().take(5) {
                let from_label = brain.get(&a.from).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                let to_label = brain.get(&a.to).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                output.push_str(&format!("  {} → {} (w:{:.2}, co:{})\n", from_label, to_label, a.weight, a.co_activation_count));
            }
            if weak.len() > 5 {
                output.push_str(&format!("  ... and {} more\n", weak.len() - 5));
            }

            // Hebbian learning indicator
            let hebbian_active: Vec<_> = filtered.iter().filter(|a| a.co_activation_count > 0).collect();
            output.push_str(&format!(
                "\n=== HEBBIAN LEARNING ===\nAssociations with co-activations: {}/{}\n",
                hebbian_active.len(),
                filtered.len()
            ));
            
            if !hebbian_active.is_empty() {
                let max_co = hebbian_active.iter().map(|a| a.co_activation_count).max().unwrap_or(0);
                let total_co: u64 = hebbian_active.iter().map(|a| a.co_activation_count).sum();
                output.push_str(&format!("Max co-activations: {}\n", max_co));
                output.push_str(&format!("Total co-activations: {}\n", total_co));
            }

            Ok(text_response(output))
        }
    }
}
