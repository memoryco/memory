//! MCP tools for the engram memory system.

use engram::{EngramId, Engram, Identity, SearchOptions, TagMatchMode};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sovran_mcp::server::server::{McpTool, McpToolEnvironment};
use sovran_mcp::types::{CallToolResponse, McpError, ToolResponseContent};

use crate::Context;

fn text_response(text: String) -> CallToolResponse {
    CallToolResponse {
        content: vec![ToolResponseContent::Text { text }],
        is_error: None,
        meta: None,
    }
}

fn format_engram(e: &Engram) -> String {
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

// =============================================================================
// engram_create - Create new memories (batch)
// =============================================================================

pub struct EngramCreateTool;

#[derive(Deserialize, Clone)]
struct MemoryInput {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct EngramCreateArgs {
    memories: Vec<MemoryInput>,
}

impl McpTool<Context> for EngramCreateTool {
    fn name(&self) -> &str {
        "engram_create"
    }

    fn description(&self) -> &str {
        "Create new memories. Accepts an array of memories to create in one call. \
         Memories start with full energy and decay over time without use."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "description": "Array of memories to create. Each item has 'content' and optional 'tags'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "tags": { "type": "array", "items": { "type": "string" } }
                        },
                        "required": ["content"]
                    }
                }
            },
            "required": ["memories"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramCreateArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();
        let mut output = String::new();
        let mut created_count = 0;

        for memory in &args.memories {
            let id = if memory.tags.is_empty() {
                brain.create(&memory.content)
            } else {
                brain.create_with_tags(&memory.content, memory.tags.clone())
            }.map_err(|e| McpError::Other(e.to_string()))?;

            created_count += 1;
            output.push_str(&format!(
                "ID: {}\nContent: {}\nTags: {:?}\n\n",
                id, memory.content, memory.tags
            ));
        }

        let header = format!("{} memories created.\n\n", created_count);
        Ok(text_response(format!("{}{}", header, output.trim())))
    }
}

// =============================================================================
// engram_recall - Actively recall memories (batch)
// =============================================================================

pub struct EngramRecallTool;

#[derive(Deserialize)]
struct EngramRecallArgs {
    ids: Vec<String>,
    #[serde(default)]
    strength: Option<f64>,
}

impl McpTool<Context> for EngramRecallTool {
    fn name(&self) -> &str {
        "engram_recall"
    }

    fn description(&self) -> &str {
        "Actively recall memories. Accepts an array of IDs to recall in one call. \
         Stimulates memories (increases energy), triggers Hebbian learning between \
         all recalled memories, and can resurrect archived memories. Memories \
         recalled together form associations automatically."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of UUIDs to recall. All memories will be linked via Hebbian learning."
                },
                "strength": {
                    "type": "number",
                    "description": "Stimulation strength (0.0-1.0). Default uses config value."
                }
            },
            "required": ["ids"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramRecallArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();
        let mut output = String::new();
        let mut recalled_count = 0;
        let mut not_found_count = 0;
        let mut total_affected = 0;

        for id_str in &args.ids {
            let id: EngramId = id_str.parse()
                .map_err(|e| McpError::InvalidArguments(format!("Invalid UUID '{}': {}", id_str, e)))?;

            let result = if let Some(s) = args.strength {
                brain.recall_with_strength(id, s)
            } else {
                brain.recall(id)
            }.map_err(|e| McpError::Other(e.to_string()))?;

            if !result.found() {
                not_found_count += 1;
                output.push_str(&format!("Memory {} not found.\n\n", id));
                continue;
            }

            recalled_count += 1;
            total_affected += result.affected_count();

            let engram = result.engram.as_ref().unwrap();
            output.push_str(&format_engram(engram));

            if result.resurrected {
                output.push_str(&format!(
                    "\n🔄 RESURRECTED from {:?}!",
                    result.previous_state.unwrap()
                ));
            }
            output.push_str("\n\n");
        }

        // Contextual reminder at TOP - prime the behavior before showing content
        let reminder = "💾 **REQUIRED:** If you explored files or learned ANY new facts, call engram_create BEFORE responding.\n---\n\n";
        
        let header = format!(
            "Recalled {} memories ({} not found). Affected {} total via Hebbian learning.\n\n",
            recalled_count, not_found_count, total_affected
        );

        Ok(text_response(format!("{}{}{}", reminder, header, output.trim())))
    }
}

// =============================================================================
// engram_search - Passively search memories (no side effects)
// =============================================================================

pub struct EngramSearchTool;

#[derive(Deserialize)]
struct EngramSearchArgs {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    tag_mode: Option<String>,
    #[serde(default)]
    include_deep: Option<bool>,
    #[serde(default)]
    include_archived: Option<bool>,
    #[serde(default)]
    limit: Option<usize>,
}

impl McpTool<Context> for EngramSearchTool {
    fn name(&self) -> &str {
        "engram_search"
    }

    fn description(&self) -> &str {
        "Search memories by content or tags. This is a passive operation - \
         it does NOT stimulate memories or trigger learning. Use engram_recall \
         when you actually want to use a memory."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in memory content."
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Filter by tags."
                },
                "tag_mode": {
                    "type": "string",
                    "enum": ["all", "any"],
                    "description": "Match all tags or any tag. Default: any"
                },
                "include_deep": {
                    "type": "boolean",
                    "description": "Include deep storage memories. Default: false"
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived memories. Default: false"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return."
                }
            }
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramSearchArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let brain = context.brain.lock().unwrap();

        let mut options = SearchOptions::default();
        if args.include_archived.unwrap_or(false) {
            options = options.include_all();
        } else if args.include_deep.unwrap_or(false) {
            options = options.include_deep();
        }
        if let Some(limit) = args.limit {
            options = options.with_limit(limit);
        }

        let results: Vec<&Engram> = if !args.tags.is_empty() {
            let tag_refs: Vec<&str> = args.tags.iter().map(|s| s.as_str()).collect();
            let mode = match args.tag_mode.as_deref() {
                Some("all") => TagMatchMode::All,
                _ => TagMatchMode::Any,
            };
            brain.search_by_tags(&tag_refs, mode)
        } else if let Some(query) = &args.query {
            brain.search_with_options(query, options)
        } else {
            // No query or tags - return recent/high-energy memories
            brain.searchable_engrams().take(args.limit.unwrap_or(10)).collect()
        };

        if results.is_empty() {
            return Ok(text_response("No memories found.".to_string()));
        }

        // Contextual reminder at TOP - prime the behavior before showing content
        let mut output = String::from(
            "⚡ **REQUIRED:** Call engram_recall on IDs you use. \n\
             💾 **REQUIRED:** Call engram_create if you learn ANY new facts this turn.\n\
             ---\n\n"
        );
        output.push_str(&format!("Found {} memories:\n\n", results.len()));
        for engram in results {
            output.push_str(&format_engram(engram));
            output.push_str("\n\n");
        }

        Ok(text_response(output))
    }
}

// =============================================================================
// engram_get - Get a specific memory by ID (no side effects)
// =============================================================================

pub struct EngramGetTool;

#[derive(Deserialize)]
struct EngramGetArgs {
    id: String,
}

impl McpTool<Context> for EngramGetTool {
    fn name(&self) -> &str {
        "engram_get"
    }

    fn description(&self) -> &str {
        "Get a memory by ID without stimulating it. Use engram_recall if you \
         want to actively use the memory."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the memory."
                }
            },
            "required": ["id"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramGetArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let id: EngramId = args.id.parse()
            .map_err(|e| McpError::InvalidArguments(format!("Invalid UUID: {}", e)))?;

        let brain = context.brain.lock().unwrap();

        match brain.get(&id) {
            Some(engram) => Ok(text_response(format_engram(engram))),
            None => Ok(text_response(format!("Memory {} not found.", id))),
        }
    }
}

// =============================================================================
// engram_delete - Delete memories permanently
// =============================================================================

pub struct EngramDeleteTool;

#[derive(Deserialize)]
struct EngramDeleteArgs {
    ids: Vec<String>,
}

impl McpTool<Context> for EngramDeleteTool {
    fn name(&self) -> &str {
        "engram_delete"
    }

    fn description(&self) -> &str {
        "Delete memories permanently. Accepts an array of IDs to delete. \
         Also removes all associations from/to deleted memories. Use for \
         housekeeping: removing duplicates, correcting mistakes, cleaning up. \
         IMPORTANT: Never delete memories unless the user explicitly asks you to."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of UUIDs to delete permanently."
                }
            },
            "required": ["ids"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramDeleteArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();
        let mut output = String::new();
        let mut deleted_count = 0;
        let mut not_found_count = 0;

        for id_str in &args.ids {
            let id: EngramId = id_str.parse()
                .map_err(|e| McpError::InvalidArguments(format!("Invalid UUID '{}': {}", id_str, e)))?;

            let existed = brain.delete(id)
                .map_err(|e| McpError::Other(e.to_string()))?;

            if existed {
                deleted_count += 1;
                output.push_str(&format!("Deleted: {}\n", id));
            } else {
                not_found_count += 1;
                output.push_str(&format!("Not found: {}\n", id));
            }
        }

        let header = format!(
            "Deleted {} memories ({} not found).\n\n",
            deleted_count, not_found_count
        );

        Ok(text_response(format!("{}{}", header, output.trim())))
    }
}

// =============================================================================
// engram_associate - Create an explicit association between memories
// =============================================================================

pub struct EngramAssociateTool;

#[derive(Deserialize)]
struct EngramAssociateArgs {
    from: String,
    to: String,
    #[serde(default)]
    weight: Option<f64>,
}

impl McpTool<Context> for EngramAssociateTool {
    fn name(&self) -> &str {
        "engram_associate"
    }

    fn description(&self) -> &str {
        "Create an explicit association between two memories. Associations also \
         form automatically through Hebbian learning when memories are recalled together."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "UUID of the source memory."
                },
                "to": {
                    "type": "string",
                    "description": "UUID of the target memory."
                },
                "weight": {
                    "type": "number",
                    "description": "Association strength (0.0-1.0). Default: 0.5"
                }
            },
            "required": ["from", "to"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramAssociateArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let from: EngramId = args.from.parse()
            .map_err(|e| McpError::InvalidArguments(format!("Invalid 'from' UUID: {}", e)))?;
        let to: EngramId = args.to.parse()
            .map_err(|e| McpError::InvalidArguments(format!("Invalid 'to' UUID: {}", e)))?;

        let weight = args.weight.unwrap_or(0.5);

        let mut brain = context.brain.lock().unwrap();

        brain.associate(from, to, weight)
            .map_err(|e| McpError::Other(e.to_string()))?;

        Ok(text_response(format!(
            "Association created: {} → {} (weight: {:.2})",
            from, to, weight
        )))
    }
}

// =============================================================================
// engram_stats - Get substrate statistics
// =============================================================================

pub struct EngramStatsTool;

impl McpTool<Context> for EngramStatsTool {
    fn name(&self) -> &str {
        "engram_stats"
    }

    fn description(&self) -> &str {
        "Get statistics about the memory substrate."
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let brain = context.brain.lock().unwrap();
        let stats = brain.stats();

        let output = format!(
            "Memory Substrate Statistics:\n\n\
             Total memories: {}\n\
             ✨ Active: {}\n\
             💤 Dormant: {}\n\
             🌊 Deep: {}\n\
             🧊 Archived: {}\n\n\
             Total associations: {}\n\
             Average energy: {:.2}",
            stats.total_engrams,
            stats.active_engrams,
            stats.dormant_engrams,
            stats.deep_engrams,
            stats.archived_engrams,
            stats.total_associations,
            stats.average_energy
        );

        Ok(text_response(output))
    }
}

// =============================================================================
// identity_get - Get the current identity
// =============================================================================

pub struct IdentityGetTool;

impl McpTool<Context> for IdentityGetTool {
    fn name(&self) -> &str {
        "identity_get"
    }

    fn description(&self) -> &str {
        "Get the current identity (persona, values, preferences, relationships). \
         Identity never decays - it's who you ARE, not what you remember."
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let brain = context.brain.lock().unwrap();
        let identity = brain.identity();

        if identity.persona.name.is_empty() && identity.values.is_empty() && identity.instructions.is_empty() {
            return Ok(text_response("No identity configured yet.".to_string()));
        }

        Ok(text_response(identity.render()))
    }
}

// =============================================================================
// identity_set - Set the identity from JSON
// =============================================================================

pub struct IdentitySetTool;

#[derive(Deserialize)]
struct IdentitySetArgs {
    identity: Identity,
}

impl McpTool<Context> for IdentitySetTool {
    fn name(&self) -> &str {
        "identity_set"
    }

    fn description(&self) -> &str {
        "Set the identity from a JSON object. This replaces the entire identity."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "identity": {
                    "type": "object",
                    "description": "The identity object with persona, values, preferences, etc.",
                    "properties": {
                        "persona": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "description": { "type": "string" },
                                "traits": { "type": "array", "items": { "type": "string" } }
                            }
                        },
                        "values": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "principle": { "type": "string" },
                                    "why": { "type": "string" },
                                    "category": { "type": "string" }
                                }
                            }
                        },
                        "preferences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prefer": { "type": "string" },
                                    "over": { "type": "string" },
                                    "category": { "type": "string" }
                                }
                            }
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": { "type": "string" },
                                    "relation": { "type": "string" },
                                    "context": { "type": "string" }
                                }
                            }
                        },
                        "antipatterns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "avoid": { "type": "string" },
                                    "why": { "type": "string" },
                                    "instead": { "type": "string" }
                                }
                            }
                        },
                        "communication": {
                            "type": "object",
                            "properties": {
                                "tone": { "type": "array", "items": { "type": "string" } },
                                "directives": { "type": "array", "items": { "type": "string" } }
                            }
                        },
                        "expertise": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                }
            },
            "required": ["identity"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: IdentitySetArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();

        brain.set_identity(args.identity.clone())
            .map_err(|e| McpError::Other(e.to_string()))?;

        Ok(text_response(format!(
            "Identity set.\n\n{}",
            args.identity.render()
        )))
    }
}

// =============================================================================
// identity_search - Search identity content
// =============================================================================

pub struct IdentitySearchTool;

#[derive(Deserialize)]
struct IdentitySearchArgs {
    query: String,
}

impl McpTool<Context> for IdentitySearchTool {
    fn name(&self) -> &str {
        "identity_search"
    }

    fn description(&self) -> &str {
        "Search identity for matching values, preferences, relationships, etc."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in identity."
                }
            },
            "required": ["query"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: IdentitySearchArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let brain = context.brain.lock().unwrap();
        let results = brain.identity().search(&args.query);

        if results.is_empty() {
            return Ok(text_response(format!(
                "No identity content matching '{}'",
                args.query
            )));
        }

        let mut output = format!(
            "Found {} matches for '{}':\n\n",
            results.total_count(),
            args.query
        );

        if !results.values.is_empty() {
            output.push_str("Values:\n");
            for v in results.values {
                output.push_str(&format!("  • {}\n", v.principle));
            }
            output.push('\n');
        }

        if !results.preferences.is_empty() {
            output.push_str("Preferences:\n");
            for p in results.preferences {
                if let Some(over) = &p.over {
                    output.push_str(&format!("  • {} > {}\n", p.prefer, over));
                } else {
                    output.push_str(&format!("  • {}\n", p.prefer));
                }
            }
            output.push('\n');
        }

        if !results.relationships.is_empty() {
            output.push_str("Relationships:\n");
            for r in results.relationships {
                output.push_str(&format!("  • {}: {}\n", r.entity, r.relation));
            }
            output.push('\n');
        }

        if !results.antipatterns.is_empty() {
            output.push_str("Antipatterns:\n");
            for a in results.antipatterns {
                output.push_str(&format!("  ✗ {}\n", a.avoid));
            }
            output.push('\n');
        }

        if !results.expertise.is_empty() {
            let exp: Vec<&str> = results.expertise.iter().map(|s| s.as_str()).collect();
            output.push_str(&format!("Expertise: {}\n", exp.join(", ")));
        }

        if !results.traits.is_empty() {
            let traits: Vec<&str> = results.traits.iter().map(|s| s.as_str()).collect();
            output.push_str(&format!("Traits: {}\n", traits.join(", ")));
        }

        if !results.instructions.is_empty() {
            output.push_str("Instructions:\n");
            for i in results.instructions {
                output.push_str(&format!("  • {}\n", i));
            }
        }

        Ok(text_response(output))
    }
}

// =============================================================================
// config_get - Get current configuration
// =============================================================================

pub struct ConfigGetTool;

impl McpTool<Context> for ConfigGetTool {
    fn name(&self) -> &str {
        "config_get"
    }

    fn description(&self) -> &str {
        "Get the current memory system configuration."
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let brain = context.brain.lock().unwrap();
        let config = brain.config();

        let output = format!(
            "Memory Configuration:\n\n\
             decay_rate_per_day: {:.2} ({}% energy loss per day of non-use)\n\
             decay_interval_hours: {:.1} (minimum hours between decay checks)\n\
             propagation_damping: {:.2} (signal reduction to neighbors)\n\
             hebbian_learning_rate: {:.2} (association strength on co-access)\n\
             recall_strength: {:.2} (energy boost when recalling)",
            config.decay_rate_per_day,
            config.decay_rate_per_day * 100.0,
            config.decay_interval_hours,
            config.propagation_damping,
            config.hebbian_learning_rate,
            config.recall_strength
        );

        Ok(text_response(output))
    }
}

// =============================================================================
// config_set - Update a configuration value
// =============================================================================

pub struct ConfigSetTool;

#[derive(Deserialize)]
struct ConfigSetArgs {
    key: String,
    value: f64,
}

impl McpTool<Context> for ConfigSetTool {
    fn name(&self) -> &str {
        "config_set"
    }

    fn description(&self) -> &str {
        "Update a configuration value. Keys: decay_rate_per_day, decay_interval_hours, \
         propagation_damping, hebbian_learning_rate, recall_strength"
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": [
                        "decay_rate_per_day",
                        "decay_interval_hours",
                        "propagation_damping",
                        "hebbian_learning_rate",
                        "recall_strength"
                    ],
                    "description": "Configuration key to update."
                },
                "value": {
                    "type": "number",
                    "description": "New value."
                }
            },
            "required": ["key", "value"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: ConfigSetArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();

        let updated = brain.configure(&args.key, args.value)
            .map_err(|e| McpError::Other(e.to_string()))?;

        if updated {
            Ok(text_response(format!(
                "Configuration updated: {} = {}",
                args.key, args.value
            )))
        } else {
            Ok(text_response(format!(
                "Unknown configuration key: {}",
                args.key
            )))
        }
    }
}

// =============================================================================
// engram_associations - Inspect associations for a single memory
// =============================================================================

pub struct EngramAssociationsTool;

#[derive(Deserialize)]
struct EngramAssociationsArgs {
    id: String,
    #[serde(default)]
    direction: Option<String>,
}

impl McpTool<Context> for EngramAssociationsTool {
    fn name(&self) -> &str {
        "engram_associations"
    }

    fn description(&self) -> &str {
        "Inspect associations for a specific memory. Shows detailed info including \
         weights, co-activation counts (Hebbian learning indicator), and timestamps. \
         Use direction to see outbound, inbound, or both."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the memory to inspect."
                },
                "direction": {
                    "type": "string",
                    "enum": ["outbound", "inbound", "both"],
                    "description": "Which associations to show. Default: outbound"
                }
            },
            "required": ["id"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramAssociationsArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let id: EngramId = args.id.parse()
            .map_err(|e| McpError::InvalidArguments(format!("Invalid UUID: {}", e)))?;

        let direction = args.direction.as_deref().unwrap_or("outbound");
        let brain = context.brain.lock().unwrap();

        // Get the memory itself for context
        let memory_info = match brain.get(&id) {
            Some(e) => format!("Memory: {} (energy: {:.2})\n", truncate_content(&e.content, 60), e.energy),
            None => return Ok(text_response(format!("Memory {} not found.", id))),
        };

        let mut output = memory_info;
        output.push_str(&format!("Direction: {}\n\n", direction));

        // Outbound associations
        if direction == "outbound" || direction == "both" {
            output.push_str("=== OUTBOUND (this memory → others) ===\n");
            match brain.associations_from(&id) {
                Some(assocs) if !assocs.is_empty() => {
                    let mut sorted: Vec<_> = assocs.iter().collect();
                    sorted.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
                    
                    for assoc in sorted {
                        let target_preview = brain.get(&assoc.to)
                            .map(|e| truncate_content(&e.content, 40))
                            .unwrap_or_else(|| "[deleted]".to_string());
                        
                        output.push_str(&format!(
                            "  → {} (weight: {:.3}, co-activations: {})\n    {}\n",
                            assoc.to,
                            assoc.weight,
                            assoc.co_activation_count,
                            target_preview
                        ));
                    }
                }
                _ => output.push_str("  (none)\n"),
            }
            output.push('\n');
        }

        // Inbound associations
        if direction == "inbound" || direction == "both" {
            output.push_str("=== INBOUND (others → this memory) ===\n");
            match brain.associations_to(&id) {
                Some(source_ids) if !source_ids.is_empty() => {
                    for source_id in source_ids {
                        // Get the association details from the source
                        let assoc_info = brain.associations_from(source_id)
                            .and_then(|assocs| assocs.iter().find(|a| a.to == id));
                        
                        let source_preview = brain.get(source_id)
                            .map(|e| truncate_content(&e.content, 40))
                            .unwrap_or_else(|| "[deleted]".to_string());
                        
                        if let Some(assoc) = assoc_info {
                            output.push_str(&format!(
                                "  ← {} (weight: {:.3}, co-activations: {})\n    {}\n",
                                source_id,
                                assoc.weight,
                                assoc.co_activation_count,
                                source_preview
                            ));
                        } else {
                            output.push_str(&format!(
                                "  ← {}\n    {}\n",
                                source_id,
                                source_preview
                            ));
                        }
                    }
                }
                _ => output.push_str("  (none)\n"),
            }
        }

        Ok(text_response(output))
    }
}

// =============================================================================
// engram_graph - Get full association graph for visualization
// =============================================================================

pub struct EngramGraphTool;

#[derive(Deserialize)]
struct EngramGraphArgs {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    min_weight: Option<f64>,
}

impl McpTool<Context> for EngramGraphTool {
    fn name(&self) -> &str {
        "engram_graph"
    }

    fn description(&self) -> &str {
        "Get the full association graph for visualization. Returns nodes (memories) \
         and edges (associations) with all metadata. Use min_weight to filter weak \
         associations. Results are best rendered visually as an SVG graph showing \
         nodes as circles and edges as weighted connections - strongly recommended \
         to generate an SVG artifact from the JSON output."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["json", "summary"],
                    "description": "Output format. 'json' for raw data, 'summary' for human-readable. Default: summary"
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: EngramGraphArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let format = args.format.as_deref().unwrap_or("summary");
        let min_weight = args.min_weight.unwrap_or(0.0);
        let brain = context.brain.lock().unwrap();

        // Collect all associations
        let all_assocs = brain.all_associations();
        let filtered: Vec<_> = all_assocs.iter()
            .filter(|a| a.weight >= min_weight)
            .collect();

        // Collect all node IDs that are part of associations
        let mut node_ids: std::collections::HashSet<EngramId> = std::collections::HashSet::new();
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
                    "tags": e.tags,
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
                output.push_str(&format!("  {} → {} (w:{:.2}, co:{})", from_label, to_label, a.weight, a.co_activation_count));
                output.push('\n');
            }
            if strong.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", strong.len() - 10));
            }

            output.push_str(&format!("\nMedium connections (0.4-0.7): {}\n", medium.len()));
            for a in medium.iter().take(10) {
                let from_label = brain.get(&a.from).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                let to_label = brain.get(&a.to).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                output.push_str(&format!("  {} → {} (w:{:.2}, co:{})", from_label, to_label, a.weight, a.co_activation_count));
                output.push('\n');
            }
            if medium.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", medium.len() - 10));
            }

            output.push_str(&format!("\nWeak connections (<0.4): {}\n", weak.len()));
            for a in weak.iter().take(5) {
                let from_label = brain.get(&a.from).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                let to_label = brain.get(&a.to).map(|e| truncate_content(&e.content, 25)).unwrap_or_default();
                output.push_str(&format!("  {} → {} (w:{:.2}, co:{})", from_label, to_label, a.weight, a.co_activation_count));
                output.push('\n');
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

/// Helper to truncate content for display
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len.saturating_sub(3)])
    }
}
