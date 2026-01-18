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
// engram_create - Create a new memory
// =============================================================================

pub struct EngramCreateTool;

#[derive(Deserialize)]
struct EngramCreateArgs {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
}

impl McpTool<Context> for EngramCreateTool {
    fn name(&self) -> &str {
        "engram_create"
    }

    fn description(&self) -> &str {
        "Create a new memory. Memories start with full energy and decay over time without use."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content."
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional tags for categorization."
                }
            },
            "required": ["content"]
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

        let id = if args.tags.is_empty() {
            brain.create(&args.content)
        } else {
            brain.create_with_tags(&args.content, args.tags.clone())
        }.map_err(|e| McpError::Other(e.to_string()))?;

        Ok(text_response(format!(
            "Memory created.\nID: {}\nContent: {}\nTags: {:?}",
            id, args.content, args.tags
        )))
    }
}

// =============================================================================
// engram_recall - Actively recall a memory (stimulates it)
// =============================================================================

pub struct EngramRecallTool;

#[derive(Deserialize)]
struct EngramRecallArgs {
    id: String,
    #[serde(default)]
    strength: Option<f64>,
}

impl McpTool<Context> for EngramRecallTool {
    fn name(&self) -> &str {
        "engram_recall"
    }

    fn description(&self) -> &str {
        "Actively recall a memory. This stimulates the memory (increases energy), \
         triggers Hebbian learning with other recent recalls, and can resurrect \
         archived memories. Use this when referencing a memory in conversation."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the memory to recall."
                },
                "strength": {
                    "type": "number",
                    "description": "Stimulation strength (0.0-1.0). Default uses config value."
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
        let args: EngramRecallArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let id: EngramId = args.id.parse()
            .map_err(|e| McpError::InvalidArguments(format!("Invalid UUID: {}", e)))?;

        let mut brain = context.brain.lock().unwrap();

        let result = if let Some(strength) = args.strength {
            brain.recall_with_strength(id, strength)
        } else {
            brain.recall(id)
        }.map_err(|e| McpError::Other(e.to_string()))?;

        if !result.found() {
            return Ok(text_response(format!("Memory {} not found.", id)));
        }

        let engram = result.engram.as_ref().unwrap();
        let mut output = format_engram(engram);

        if result.resurrected {
            output.push_str(&format!(
                "\n\n🔄 RESURRECTED from {:?}!",
                result.previous_state.unwrap()
            ));
        }

        output.push_str(&format!(
            "\n\nAffected {} memories (propagation + Hebbian learning)",
            result.affected_count()
        ));

        Ok(text_response(output))
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

        let mut output = format!("Found {} memories:\n\n", results.len());
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
