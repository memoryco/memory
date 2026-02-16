//! Dashboard HTTP server.
//!
//! Serves a web dashboard on `127.0.0.1:4243` showing memory, identity,
//! references, and the association graph. Runs on a background daemon thread,
//! completely independent of the MCP stdio transport.

use crate::engram::Brain;
use crate::identity::IdentityStore;
use crate::reference::{self, ReferenceManager};
use serde_json::{json, Value as JsonValue};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const DASHBOARD_HTML: &str = include_str!("dashboard.html");
const BIND_ADDR: &str = "127.0.0.1:4243";

/// Shared state for the dashboard thread — backed by the same Arcs as the MCP
/// server so edits in the dashboard are immediately visible via MCP and vice-versa.
#[allow(dead_code)]
struct DashboardState {
    brain: Arc<Mutex<Brain>>,
    identity: Arc<Mutex<IdentityStore>>,
    references: Arc<Mutex<ReferenceManager>>,
    memory_home: PathBuf,
    references_dir: PathBuf,
}

/// Start the dashboard HTTP server on a background daemon thread.
///
/// Accepts the same shared `Arc<Mutex<>>` instances used by the MCP server so
/// that edits through the dashboard are immediately visible to tools and vice-versa.
pub fn start_dashboard(
    brain: Arc<Mutex<Brain>>,
    identity: Arc<Mutex<IdentityStore>>,
    references: Arc<Mutex<ReferenceManager>>,
    memory_home: &Path,
) {
    let memory_home = memory_home.to_path_buf();
    let references_dir = memory_home.join("references");

    let server = match tiny_http::Server::http(BIND_ADDR) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Dashboard: Failed to bind {}: {}", BIND_ADDR, e);
            return;
        }
    };

    eprintln!("Dashboard: http://{}/", BIND_ADDR);

    std::thread::Builder::new()
        .name("dashboard".into())
        .spawn(move || {
            let state = Arc::new(DashboardState {
                brain,
                identity,
                references,
                memory_home,
                references_dir,
            });

            for request in server.incoming_requests() {
                let url = request.url().to_string();
                let method = request.method().to_string();

                // Handle CORS preflight
                if method == "OPTIONS" {
                    let response = tiny_http::Response::from_string("")
                        .with_header(cors_header())
                        .with_header(cors_methods_header())
                        .with_header(cors_headers_header());
                    let _ = request.respond(response);
                    continue;
                }

                let response = route(&method, &url, request, &state);
                // response already sent inside route()
                let _ = response;
            }
        })
        .expect("Failed to spawn dashboard thread");
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

fn route(
    method: &str,
    url: &str,
    mut request: tiny_http::Request,
    state: &Arc<DashboardState>,
) {
    // Strip query string for path matching, but keep it for parameter parsing
    let (path, query) = match url.find('?') {
        Some(i) => (&url[..i], &url[i + 1..]),
        None => (url, ""),
    };

    let response = match (method, path) {
        // Static
        ("GET", "/") => html_response(DASHBOARD_HTML),

        // Identity
        ("GET", "/api/identity") => handle_get_identity(state),
        ("POST", "/api/identity/persona/name") => {
            handle_set_persona_name(&mut request, state)
        }
        ("POST", "/api/identity/persona/description") => {
            handle_set_persona_description(&mut request, state)
        }
        ("POST", "/api/identity/add") => handle_identity_add(&mut request, state),

        // References
        ("GET", "/api/references") => handle_list_references(state),
        ("POST", "/api/references/upload") => handle_upload_reference(&mut request, state),

        // Engrams
        ("GET", "/api/engrams") => handle_list_engrams(query, state),

        // Graph
        ("GET", "/api/graph") => handle_graph(query, state),

        // Dynamic DELETE routes
        _ if method == "DELETE" => route_delete(path, state),

        _ => json_response(404, r#"{"error":"Not found"}"#),
    };

    let _ = request.respond(response);
}

fn route_delete(
    path: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    match segments.as_slice() {
        // DELETE /api/identity/:id
        ["api", "identity", id] => handle_identity_remove(id, state),
        // DELETE /api/engrams/:id
        ["api", "engrams", id] => handle_delete_engram(id, state),
        // DELETE /api/references/:name
        ["api", "references", name] => handle_delete_reference(name, state),
        _ => json_response(404, r#"{"error":"Not found"}"#),
    }
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn json_response(status: u16, body: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(
            "Content-Type: application/json"
                .parse::<tiny_http::Header>()
                .unwrap(),
        )
        .with_header(cors_header())
}

fn html_response(body: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body.to_string())
        .with_header(
            "Content-Type: text/html; charset=utf-8"
                .parse::<tiny_http::Header>()
                .unwrap(),
        )
        .with_header(cors_header())
}

fn json_ok(value: &JsonValue) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    json_response(
        200,
        &serde_json::to_string(value).unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e)),
    )
}

fn cors_header() -> tiny_http::Header {
    "Access-Control-Allow-Origin: *"
        .parse::<tiny_http::Header>()
        .unwrap()
}

fn cors_methods_header() -> tiny_http::Header {
    "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS"
        .parse::<tiny_http::Header>()
        .unwrap()
}

fn cors_headers_header() -> tiny_http::Header {
    "Access-Control-Allow-Headers: Content-Type"
        .parse::<tiny_http::Header>()
        .unwrap()
}

/// Parse query string parameters into key=value pairs.
fn parse_query(query: &str) -> Vec<(&str, &str)> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?;
            let value = parts.next().unwrap_or("");
            Some((key, value))
        })
        .collect()
}

fn query_param<'a>(params: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    params.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// Read the full request body as a string.
fn read_body(request: &mut tiny_http::Request) -> Result<String, String> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .map_err(|e| format!("Failed to read body: {}", e))?;
    Ok(body)
}

/// Read the full request body as bytes.
fn read_body_bytes(request: &mut tiny_http::Request) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut body)
        .map_err(|e| format!("Failed to read body: {}", e))?;
    Ok(body)
}

// ---------------------------------------------------------------------------
// Identity handlers
// ---------------------------------------------------------------------------

fn handle_get_identity(
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut identity = state.identity.lock().unwrap();

    // Get structured identity
    let id = match identity.get() {
        Ok(id) => id,
        Err(e) => return json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    // Also get raw items with IDs so the dashboard can delete them
    use crate::identity::storage::IdentityItemType;
    let all_types = IdentityItemType::all();
    let mut items: Vec<JsonValue> = Vec::new();
    for item_type in all_types {
        if let Ok(listed) = identity.list(*item_type) {
            for item in listed {
                items.push(json!({
                    "id": item.id,
                    "type": item.item_type.to_string(),
                    "content": item.content,
                    "secondary": item.secondary,
                    "tertiary": item.tertiary,
                    "category": item.category,
                }));
            }
        }
    }

    let mut result = serde_json::to_value(&id).unwrap_or(json!({}));
    result["_items"] = json!(items);

    json_ok(&result)
}

fn handle_set_persona_name(
    request: &mut tiny_http::Request,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = match read_body(request) {
        Ok(b) => b,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let parsed: JsonValue = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let value = match parsed.get("value").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return json_response(400, r#"{"error":"Missing 'value' field"}"#),
    };

    let mut identity = state.identity.lock().unwrap();
    match identity.set_persona_name(value) {
        Ok(id) => json_ok(&json!({"id": id})),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

fn handle_set_persona_description(
    request: &mut tiny_http::Request,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = match read_body(request) {
        Ok(b) => b,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let parsed: JsonValue = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let value = match parsed.get("value").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return json_response(400, r#"{"error":"Missing 'value' field"}"#),
    };

    let mut identity = state.identity.lock().unwrap();
    match identity.set_persona_description(value) {
        Ok(id) => json_ok(&json!({"id": id})),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

fn handle_identity_add(
    request: &mut tiny_http::Request,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = match read_body(request) {
        Ok(b) => b,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let parsed: JsonValue = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let item_type = match parsed.get("type").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return json_response(400, r#"{"error":"Missing 'type' field"}"#),
    };

    let content = match parsed.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return json_response(400, r#"{"error":"Missing 'content' field"}"#),
    };

    let secondary = parsed.get("secondary").and_then(|v| v.as_str());
    let tertiary = parsed.get("tertiary").and_then(|v| v.as_str());
    let category = parsed.get("category").and_then(|v| v.as_str());

    let mut identity = state.identity.lock().unwrap();

    let result = match item_type {
        "trait" => identity.add_trait(content),
        "value" => identity.add_value(content, secondary, category),
        "preference" => identity.add_preference(content, secondary, category),
        "relationship" => {
            let relation = secondary.unwrap_or("");
            identity.add_relationship(content, relation, category)
        }
        "antipattern" => identity.add_antipattern(content, secondary, tertiary),
        "tone" => identity.add_tone(content),
        "directive" => identity.add_directive(content),
        "expertise" => identity.add_expertise(content),
        "instruction" => identity.add_instruction(content),
        _ => return json_response(400, &format!(r#"{{"error":"Unknown type: {}"}}"#, item_type)),
    };

    match result {
        Ok(id) => json_ok(&json!({"id": id})),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

fn handle_identity_remove(
    id: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut identity = state.identity.lock().unwrap();
    match identity.remove(id) {
        Ok(true) => json_ok(&json!({"removed": id})),
        Ok(false) => json_response(404, &format!(r#"{{"error":"Item '{}' not found"}}"#, id)),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

// ---------------------------------------------------------------------------
// References handlers
// ---------------------------------------------------------------------------

fn handle_list_references(
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let refs = state.references.lock().unwrap();
    let sources: Vec<&str> = refs.sources();
    let result: Vec<JsonValue> = sources
        .iter()
        .map(|name| {
            let title = refs
                .get_meta(name)
                .and_then(|m| m.citation.as_ref())
                .map(|c| c.title.as_str())
                .unwrap_or(*name);
            json!({
                "name": name,
                "title": title,
            })
        })
        .collect();
    json_ok(&json!(result))
}

fn handle_upload_reference(
    request: &mut tiny_http::Request,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    // Get Content-Type header
    let content_type = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();

    let boundary = match extract_boundary(&content_type) {
        Some(b) => b,
        None => {
            return json_response(
                400,
                r#"{"error":"Missing multipart boundary in Content-Type"}"#,
            )
        }
    };

    let body = match read_body_bytes(request) {
        Ok(b) => b,
        Err(e) => return json_response(400, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let file = match parse_multipart(&body, &boundary) {
        Some(f) => f,
        None => return json_response(400, r#"{"error":"No file found in multipart body"}"#),
    };

    // Write to a temp file, then sanitize_and_copy
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(&file.filename);

    if let Err(e) = std::fs::write(&temp_path, &file.data) {
        return json_response(500, &format!(r#"{{"error":"Failed to write temp file: {}"}}"#, e));
    }

    let dest = match reference::sanitize_and_copy(&temp_path, &state.references_dir) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_file(&temp_path);
            return json_response(400, &format!(r#"{{"error":"{}"}}"#, e));
        }
    };

    let _ = std::fs::remove_file(&temp_path);

    // Reload references and bootstrap citation instructions
    let upload_result = {
        let mut refs = state.references.lock().unwrap();
        refs.add_source(&dest).map(|name| name.to_string())
    };

    match upload_result {
        Ok(name) => {
            // Bootstrap citation instructions into identity so new references
            // are immediately usable without restarting the server.
            let refs = state.references.lock().unwrap();
            let mut identity = state.identity.lock().unwrap();
            if let Err(e) = crate::reference::bootstrap::bootstrap(&mut identity, &refs) {
                eprintln!("Dashboard: Warning: reference bootstrap failed: {}", e);
            }
            json_ok(&json!({"uploaded": name}))
        }
        Err(e) => json_response(500, &format!(r#"{{"error":"Failed to index: {}"}}"#, e)),
    }
}

fn handle_delete_reference(
    name: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    // URL-decode the name
    let decoded_name = url_decode(name);

    // Find and delete the PDF file
    let pdf_path = state.references_dir.join(format!("{}.pdf", decoded_name));
    if !pdf_path.exists() {
        return json_response(
            404,
            &format!(r#"{{"error":"Reference '{}' not found"}}"#, decoded_name),
        );
    }

    if let Err(e) = std::fs::remove_file(&pdf_path) {
        return json_response(500, &format!(r#"{{"error":"Failed to delete: {}"}}"#, e));
    }

    // Also remove the index db and idx files if they exist
    let index_path = state.references_dir.join(format!("{}.db", decoded_name));
    let _ = std::fs::remove_file(&index_path);
    let idx_path = state.references_dir.join(format!("{}.idx", decoded_name));
    let _ = std::fs::remove_file(&idx_path);

    // Reload references
    let mut refs = state.references.lock().unwrap();
    *refs = ReferenceManager::new();
    let _ = refs.load_directory(&state.references_dir);

    // TODO: Clean up orphaned citation instructions from identity when a reference
    // is deleted. Would need to find and remove the instruction whose marker matches
    // `reference:<decoded_name>`. For now, stale citation instructions persist until
    // the next server restart when bootstrap reconciles them.

    json_ok(&json!({"deleted": decoded_name}))
}

// ---------------------------------------------------------------------------
// Engram handlers
// ---------------------------------------------------------------------------

fn handle_list_engrams(
    query: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let params = parse_query(query);
    let search_query = query_param(&params, "q");
    let limit: usize = query_param(&params, "limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let include_archived = query_param(&params, "include_archived")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    let include_deep = query_param(&params, "include_deep")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let mut brain = state.brain.lock().unwrap();

    // Sync cross-process writes
    let _ = brain.sync_from_storage();

    if let Some(q) = search_query {
        if q.is_empty() {
            return list_recent_engrams(&brain, limit, include_archived, include_deep);
        }

        // Try semantic search first
        let embedding_gen = crate::embedding::EmbeddingGenerator::new();
        match embedding_gen.generate(q) {
            Ok(query_embedding) => {
                match brain.find_similar_by_embedding(&query_embedding, limit, 0.3) {
                    Ok(results) => {
                        let engrams: Vec<JsonValue> = results
                            .iter()
                            .filter_map(|r| {
                                brain.get(&r.id).map(|e| engram_to_json(e, Some(r.score)))
                            })
                            .collect();
                        json_ok(&json!({
                            "engrams": engrams,
                            "search": q,
                            "method": "semantic"
                        }))
                    }
                    Err(_) => {
                        // Fall back to text search
                        let results = brain.search(q);
                        let engrams: Vec<JsonValue> =
                            results.iter().take(limit).map(|e| engram_to_json(e, None)).collect();
                        json_ok(&json!({
                            "engrams": engrams,
                            "search": q,
                            "method": "text"
                        }))
                    }
                }
            }
            Err(_) => {
                // Fall back to text search
                let results = brain.search(q);
                let engrams: Vec<JsonValue> =
                    results.iter().take(limit).map(|e| engram_to_json(e, None)).collect();
                json_ok(&json!({
                    "engrams": engrams,
                    "search": q,
                    "method": "text"
                }))
            }
        }
    } else {
        list_recent_engrams(&brain, limit, include_archived, include_deep)
    }
}

fn list_recent_engrams(
    brain: &Brain,
    limit: usize,
    include_archived: bool,
    include_deep: bool,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut engrams: Vec<&crate::engram::Engram> = if include_archived {
        brain.all_engrams().collect()
    } else if include_deep {
        brain
            .all_engrams()
            .filter(|e| !e.is_archived())
            .collect()
    } else {
        brain.searchable_engrams().collect()
    };

    // Sort by last_accessed descending (most recent first)
    engrams.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));

    let result: Vec<JsonValue> = engrams
        .iter()
        .take(limit)
        .map(|e| engram_to_json(e, None))
        .collect();

    let stats = brain.stats();
    json_ok(&json!({
        "engrams": result,
        "stats": {
            "total": stats.total_engrams,
            "active": stats.active_engrams,
            "dormant": stats.dormant_engrams,
            "deep": stats.deep_engrams,
            "archived": stats.archived_engrams,
            "associations": stats.total_associations,
            "average_energy": stats.average_energy,
        }
    }))
}

fn engram_to_json(e: &crate::engram::Engram, score: Option<f32>) -> JsonValue {
    let mut obj = json!({
        "id": e.id.to_string(),
        "content": e.content,
        "energy": e.energy,
        "state": format!("{:?}", e.state),
        "state_emoji": e.state.emoji(),
        "confidence": e.confidence,
        "created_at": e.created_at,
        "last_accessed": e.last_accessed,
        "access_count": e.access_count,
        "tags": e.tags,
    });
    if let Some(s) = score {
        obj["score"] = json!(s);
    }
    obj
}

fn handle_delete_engram(
    id: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let engram_id = match uuid::Uuid::parse_str(id) {
        Ok(id) => id,
        Err(e) => return json_response(400, &format!(r#"{{"error":"Invalid UUID: {}"}}"#, e)),
    };

    let mut brain = state.brain.lock().unwrap();
    match brain.delete(engram_id) {
        Ok(true) => json_ok(&json!({"deleted": id})),
        Ok(false) => json_response(404, &format!(r#"{{"error":"Engram '{}' not found"}}"#, id)),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

// ---------------------------------------------------------------------------
// Graph handler
// ---------------------------------------------------------------------------

fn handle_graph(
    query: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let params = parse_query(query);
    let min_weight: f64 = query_param(&params, "min_weight")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let mut brain = state.brain.lock().unwrap();
    let _ = brain.sync_from_storage();

    let all_assocs = brain.all_associations();
    let filtered: Vec<_> = all_assocs.iter().filter(|a| a.weight >= min_weight).collect();

    // Collect node IDs
    let mut node_ids: HashSet<uuid::Uuid> = HashSet::new();
    for assoc in &filtered {
        node_ids.insert(assoc.from);
        node_ids.insert(assoc.to);
    }

    let nodes: Vec<JsonValue> = node_ids
        .iter()
        .filter_map(|id| brain.get(id))
        .map(|e| {
            json!({
                "id": e.id.to_string(),
                "label": truncate_content(&e.content, 30),
                "energy": e.energy,
                "state": e.state.emoji(),
                "access_count": e.access_count
            })
        })
        .collect();

    let edges: Vec<JsonValue> = filtered
        .iter()
        .map(|a| {
            json!({
                "from": a.from.to_string(),
                "to": a.to.to_string(),
                "weight": a.weight,
                "co_activations": a.co_activation_count,
                "created_at": a.created_at,
                "last_activated": a.last_activated
            })
        })
        .collect();

    json_ok(&json!({
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "total_nodes": nodes.len(),
            "total_edges": edges.len(),
            "min_weight_filter": min_weight
        }
    }))
}

// ---------------------------------------------------------------------------
// Multipart parsing
// ---------------------------------------------------------------------------

/// Extracted file from a multipart form body.
#[derive(Debug)]
struct MultipartFile {
    filename: String,
    data: Vec<u8>,
}

/// Extract the boundary string from a Content-Type header.
fn extract_boundary(content_type: &str) -> Option<String> {
    content_type
        .split(';')
        .map(|s| s.trim())
        .find(|s| s.starts_with("boundary="))
        .map(|s| s["boundary=".len()..].trim_matches('"').to_string())
}

/// Parse a multipart/form-data body to extract the first file part.
fn parse_multipart(body: &[u8], boundary: &str) -> Option<MultipartFile> {
    let delimiter = format!("--{}", boundary);
    let _end_delimiter = format!("--{}--", boundary);

    // We need to work with raw bytes because file content may not be UTF-8
    let delim_bytes = delimiter.as_bytes();

    // Find all boundary positions
    let mut positions = Vec::new();
    let mut start = 0;
    while start < body.len() {
        if let Some(pos) = find_bytes(&body[start..], delim_bytes) {
            positions.push(start + pos);
            start = start + pos + delim_bytes.len();
        } else {
            break;
        }
    }

    if positions.len() < 2 {
        return None;
    }

    // Extract the content between first and second boundary
    for i in 0..positions.len() - 1 {
        let part_start = positions[i] + delim_bytes.len();
        let part_end = positions[i + 1];

        let part = &body[part_start..part_end];

        // Skip leading \r\n
        let part = if part.starts_with(b"\r\n") {
            &part[2..]
        } else {
            part
        };

        // Find the blank line separating headers from body (\r\n\r\n)
        let header_end = match find_bytes(part, b"\r\n\r\n") {
            Some(pos) => pos,
            None => continue,
        };

        let headers = &part[..header_end];
        let file_data = &part[header_end + 4..]; // skip \r\n\r\n

        // Strip trailing \r\n before the next boundary
        let file_data = if file_data.ends_with(b"\r\n") {
            &file_data[..file_data.len() - 2]
        } else {
            file_data
        };

        // Parse headers to find filename
        let headers_str = String::from_utf8_lossy(headers);
        if let Some(filename) = extract_filename(&headers_str) {
            return Some(MultipartFile {
                filename,
                data: file_data.to_vec(),
            });
        }
    }

    None
}

/// Find a byte sequence within a larger byte slice.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Extract filename from Content-Disposition header value.
fn extract_filename(headers: &str) -> Option<String> {
    for line in headers.lines() {
        if line
            .to_lowercase()
            .contains("content-disposition")
            && line.contains("filename=")
        {
            // Find filename="..." or filename=...
            let idx = line.find("filename=")?;
            let rest = &line[idx + "filename=".len()..];
            let filename = if rest.starts_with('"') {
                // Quoted filename
                let end = rest[1..].find('"')?;
                &rest[1..1 + end]
            } else {
                // Unquoted — up to next ; or end
                rest.split(';').next().unwrap_or("").trim()
            };
            if !filename.is_empty() {
                return Some(filename.to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// URL decoding
// ---------------------------------------------------------------------------

/// Simple URL percent-decoding.
fn url_decode(input: &str) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut chars = input.bytes();

    while let Some(b) = chars.next() {
        match b {
            b'%' => {
                let hi = chars.next();
                let lo = chars.next();
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    if let (Some(h), Some(l)) = (hex_val(hi), hex_val(lo)) {
                        bytes.push(h << 4 | l);
                        continue;
                    }
                    bytes.push(b'%');
                    bytes.push(hi);
                    bytes.push(lo);
                } else {
                    bytes.push(b'%');
                    if let Some(hi) = hi {
                        bytes.push(hi);
                    }
                }
            }
            b'+' => bytes.push(b' '),
            _ => bytes.push(b),
        }
    }

    String::from_utf8_lossy(&bytes).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Truncate content for display.
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len.saturating_sub(3)])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Multipart parsing ──────────────────────────────────────────────────

    #[test]
    fn extract_boundary_basic() {
        let ct = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW";
        let b = extract_boundary(ct).unwrap();
        assert_eq!(b, "----WebKitFormBoundary7MA4YWxkTrZu0gW");
    }

    #[test]
    fn extract_boundary_quoted() {
        let ct = r#"multipart/form-data; boundary="my-boundary""#;
        let b = extract_boundary(ct).unwrap();
        assert_eq!(b, "my-boundary");
    }

    #[test]
    fn extract_boundary_missing() {
        let ct = "application/json";
        assert!(extract_boundary(ct).is_none());
    }

    #[test]
    fn parse_multipart_basic() {
        let boundary = "----boundary";
        let body = format!(
            "------boundary\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.pdf\"\r\n\
             Content-Type: application/pdf\r\n\
             \r\n\
             %PDF-1.4 fake content\r\n\
             ------boundary--\r\n"
        );

        let file = parse_multipart(body.as_bytes(), boundary).unwrap();
        assert_eq!(file.filename, "test.pdf");
        assert_eq!(file.data, b"%PDF-1.4 fake content");
    }

    #[test]
    fn parse_multipart_no_file() {
        let boundary = "----boundary";
        let body = format!(
            "------boundary\r\n\
             Content-Disposition: form-data; name=\"text\"\r\n\
             \r\n\
             just some text\r\n\
             ------boundary--\r\n"
        );

        assert!(parse_multipart(body.as_bytes(), boundary).is_none());
    }

    #[test]
    fn parse_multipart_multiple_parts() {
        let boundary = "----boundary";
        let body = format!(
            "------boundary\r\n\
             Content-Disposition: form-data; name=\"text\"\r\n\
             \r\n\
             some text\r\n\
             ------boundary\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"report.pdf\"\r\n\
             Content-Type: application/pdf\r\n\
             \r\n\
             %PDF-1.7 content here\r\n\
             ------boundary--\r\n"
        );

        let file = parse_multipart(body.as_bytes(), boundary).unwrap();
        assert_eq!(file.filename, "report.pdf");
        assert_eq!(file.data, b"%PDF-1.7 content here");
    }

    // ── Extract filename ───────────────────────────────────────────────────

    #[test]
    fn extract_filename_quoted() {
        let headers = "Content-Disposition: form-data; name=\"file\"; filename=\"my doc.pdf\"";
        assert_eq!(extract_filename(headers), Some("my doc.pdf".to_string()));
    }

    #[test]
    fn extract_filename_unquoted() {
        let headers = "Content-Disposition: form-data; name=\"file\"; filename=simple.pdf";
        assert_eq!(extract_filename(headers), Some("simple.pdf".to_string()));
    }

    #[test]
    fn extract_filename_no_filename() {
        let headers = "Content-Disposition: form-data; name=\"text\"";
        assert!(extract_filename(headers).is_none());
    }

    // ── Route matching ─────────────────────────────────────────────────────

    #[test]
    fn route_delete_identity() {
        let segments: Vec<&str> = "api/identity/abc-123"
            .split('/')
            .collect();
        match segments.as_slice() {
            ["api", "identity", id] => assert_eq!(*id, "abc-123"),
            _ => panic!("Should match identity delete"),
        }
    }

    #[test]
    fn route_delete_engram() {
        let segments: Vec<&str> = "api/engrams/550e8400-e29b-41d4-a716-446655440000"
            .split('/')
            .collect();
        match segments.as_slice() {
            ["api", "engrams", id] => {
                assert_eq!(*id, "550e8400-e29b-41d4-a716-446655440000");
            }
            _ => panic!("Should match engram delete"),
        }
    }

    #[test]
    fn route_delete_reference() {
        let segments: Vec<&str> = "api/references/dsm5tr"
            .split('/')
            .collect();
        match segments.as_slice() {
            ["api", "references", name] => assert_eq!(*name, "dsm5tr"),
            _ => panic!("Should match reference delete"),
        }
    }

    // ── Query parsing ──────────────────────────────────────────────────────

    #[test]
    fn parse_query_basic() {
        let params = parse_query("q=hello&limit=10");
        assert_eq!(query_param(&params, "q"), Some("hello"));
        assert_eq!(query_param(&params, "limit"), Some("10"));
        assert_eq!(query_param(&params, "missing"), None);
    }

    #[test]
    fn parse_query_empty() {
        let params = parse_query("");
        assert!(params.is_empty());
    }

    #[test]
    fn parse_query_no_value() {
        let params = parse_query("key=");
        assert_eq!(query_param(&params, "key"), Some(""));
    }

    // ── URL decoding ───────────────────────────────────────────────────────

    #[test]
    fn url_decode_basic() {
        assert_eq!(url_decode("hello%20world"), "hello world");
    }

    #[test]
    fn url_decode_plus() {
        assert_eq!(url_decode("hello+world"), "hello world");
    }

    #[test]
    fn url_decode_passthrough() {
        assert_eq!(url_decode("simple"), "simple");
    }

    // ── Find bytes ─────────────────────────────────────────────────────────

    #[test]
    fn find_bytes_basic() {
        assert_eq!(find_bytes(b"hello world", b"world"), Some(6));
        assert_eq!(find_bytes(b"hello world", b"missing"), None);
        assert_eq!(find_bytes(b"hello world", b"hello"), Some(0));
    }

    #[test]
    fn find_bytes_empty_needle() {
        assert_eq!(find_bytes(b"hello", b""), None);
    }

    // ── Truncate ───────────────────────────────────────────────────────────

    #[test]
    fn truncate_content_short() {
        assert_eq!(truncate_content("hello", 10), "hello");
    }

    #[test]
    fn truncate_content_long() {
        let result = truncate_content("this is a long string", 10);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 10);
    }
}
