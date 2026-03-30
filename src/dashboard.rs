//! Dashboard HTTP server.
//!
//! Serves a web dashboard on `127.0.0.1:4242` (or the port set via
//! `MEMORYCO_DASHBOARD_PORT`) showing memory, identity, references, and
//! the association graph. Runs on a background daemon thread, completely
//! independent of the MCP stdio transport.

use crate::memory_core::Brain;
use crate::identity::IdentityStore;
use crate::reference::{self, ReferenceManager};
use serde_json::{Value as JsonValue, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use memoryco_design as design;
use std::sync::LazyLock;

const DASHBOARD_HTML: &str = include_str!("dashboard.html");

static PAGE_HTML: LazyLock<String> = LazyLock::new(|| design::compose(DASHBOARD_HTML));
pub(crate) const BIND_HOST: &str = "127.0.0.1";
pub(crate) const BIND_PORT: u16 = 4242;

/// Parse a port string, returning `None` on invalid or out-of-range values.
fn parse_port(s: &str) -> Option<u16> {
    s.parse().ok()
}

/// Resolve the dashboard port: `MEMORYCO_DASHBOARD_PORT` env var, or [`BIND_PORT`].
pub(crate) fn resolve_dashboard_port() -> u16 {
    std::env::var("MEMORYCO_DASHBOARD_PORT")
        .ok()
        .and_then(|p| parse_port(&p))
        .unwrap_or(BIND_PORT)
}

/// Shared state for the dashboard thread — backed by the same Arcs as the MCP
/// server so edits in the dashboard are immediately visible via MCP and vice-versa.
#[allow(dead_code)]
struct DashboardState {
    brain: Arc<RwLock<Brain>>,
    identity: Arc<Mutex<IdentityStore>>,
    references: Arc<Mutex<ReferenceManager>>,
    memory_home: PathBuf,
    references_dir: PathBuf,
}

/// Start the dashboard HTTP server on a background daemon thread.
///
/// Accepts the same shared `Arc<RwLock<>>` / `Arc<Mutex<>>` instances used by
/// the MCP server so that edits through the dashboard are immediately visible
/// to tools and vice-versa.
pub fn start_dashboard(
    brain: Arc<RwLock<Brain>>,
    identity: Arc<Mutex<IdentityStore>>,
    references: Arc<Mutex<ReferenceManager>>,
    memory_home: &Path,
) {
    let memory_home = memory_home.to_path_buf();
    let references_dir = memory_home.join("references");

    let port = resolve_dashboard_port();
    let addr = format!("{}:{}", BIND_HOST, port);
    let server = match tiny_http::Server::http(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Dashboard: Failed to bind {}: {}", addr, e);
            return;
        }
    };

    eprintln!("Dashboard: http://{}:{}/", BIND_HOST, port);

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

                route(&method, &url, request, &state);
            }
        })
        .expect("Failed to spawn dashboard thread");
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

fn route(method: &str, url: &str, mut request: tiny_http::Request, state: &Arc<DashboardState>) {
    // Strip query string for path matching, but keep it for parameter parsing
    let (path, query) = url.split_once('?').unwrap_or((url, ""));

    let response = match (method, path) {
        // Static
        ("GET", "/") => html_response(&PAGE_HTML),

        // Identity
        ("GET", "/api/identity") => handle_get_identity(state),
        ("POST", "/api/identity/persona/name") => handle_set_persona_name(&mut request, state),
        ("POST", "/api/identity/persona/description") => {
            handle_set_persona_description(&mut request, state)
        }
        ("POST", "/api/identity/add") => handle_identity_add(&mut request, state),

        // References
        ("GET", "/api/references") => handle_list_references(state),
        ("POST", "/api/references/upload") => handle_upload_reference(&mut request, state),

        // Lenses
        ("GET", "/api/lenses") => handle_list_lenses(state),
        ("POST", "/api/lenses") => handle_save_lens(&mut request, state),
        ("POST", "/api/lenses/upload") => handle_upload_lens(&mut request, state),

        // Updates
        ("GET", "/api/updates") => handle_check_updates(),

        // Memorys
        ("GET", "/api/memories") => handle_list_memories(query, state),

        // Graph
        ("GET", "/api/graph") => handle_graph(query, state),

        // Dynamic GET routes
        _ if method == "GET" => route_get(path, state),

        // Dynamic POST routes
        _ if method == "POST" => route_post(path, state),

        // Dynamic DELETE routes
        _ if method == "DELETE" => route_delete(path, state),

        _ => json_response(404, r#"{"error":"Not found"}"#),
    };

    let _ = request.respond(response);
}

fn route_post(
    path: &str,
    _state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    match segments.as_slice() {
        // POST /api/updates/:binary/stage
        ["api", "updates", binary, "stage"] => handle_stage_update(binary),
        // POST /api/updates/stage-all
        ["api", "updates", "stage-all"] => handle_stage_all_updates(),
        _ => json_response(404, r#"{"error":"Not found"}"#),
    }
}

fn route_delete(
    path: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    match segments.as_slice() {
        // DELETE /api/identity/:id
        ["api", "identity", id] => handle_identity_remove(id, state),
        // DELETE /api/memories/:id
        ["api", "memories", id] => handle_delete_memory(id, state),
        // DELETE /api/references/:name
        ["api", "references", name] => handle_delete_reference(name, state),
        // DELETE /api/lenses/:name
        ["api", "lenses", name] => handle_delete_lens(name, state),
        _ => json_response(404, r#"{"error":"Not found"}"#),
    }
}

fn route_get(
    path: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    match segments.as_slice() {
        // GET /api/memories/:id/associations
        ["api", "memories", id, "associations"] => handle_memory_associations(id, state),
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

/// Parse query string parameters into key=value pairs, URL-decoding values.
fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?;
            let value = parts.next().unwrap_or("");
            Some((key.to_string(), url_decode(value)))
        })
        .collect()
}

fn query_param<'a>(params: &'a [(String, String)], key: &str) -> Option<&'a str> {
    params.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
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
    result["version"] = json!(env!("CARGO_PKG_VERSION"));
    result["name"] = json!("memory");

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
        _ => {
            return json_response(
                400,
                &format!(r#"{{"error":"Unknown type: {}"}}"#, item_type),
            );
        }
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
            );
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
        return json_response(
            500,
            &format!(r#"{{"error":"Failed to write temp file: {}"}}"#, e),
        );
    }

    let dest = match reference::sanitize_and_copy(&temp_path, &state.references_dir) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_file(&temp_path);
            let msg = friendly_error(&e.to_string());
            return json_response(400, &format!(r#"{{"error":"{}"}}"#, msg));
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
        Err(e) => {
            let msg = friendly_error(&e.to_string());
            json_response(500, &format!(r#"{{"error":"{}"}}"#, msg))
        }
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
// Lenses handlers
// ---------------------------------------------------------------------------

/// Sanitize a lens name to only allow alphanumeric chars, hyphens, and underscores.
fn sanitize_lens_name(name: &str) -> Option<String> {
    let clean: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if clean.is_empty() { None } else { Some(clean) }
}

fn handle_list_lenses(
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let lenses_dir = state.memory_home.join("lenses");
    if !lenses_dir.exists() {
        return json_ok(&json!([]));
    }

    let entries = match std::fs::read_dir(&lenses_dir) {
        Ok(e) => e,
        Err(e) => return json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    };

    let mut lenses: Vec<JsonValue> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let description = content
            .lines()
            .next()
            .filter(|line| line.starts_with('#'))
            .map(|line| line.trim_start_matches('#').trim().to_string())
            .unwrap_or_default();

        lenses.push(json!({
            "name": name,
            "description": description,
            "content": content,
        }));
    }

    json_ok(&json!(lenses))
}

fn handle_save_lens(
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

    let raw_name = match parsed.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return json_response(400, r#"{"error":"Missing 'name' field"}"#),
    };

    let name = match sanitize_lens_name(raw_name) {
        Some(n) => n,
        None => return json_response(400, r#"{"error":"Invalid lens name"}"#),
    };

    let content = match parsed.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return json_response(400, r#"{"error":"Missing 'content' field"}"#),
    };

    let lenses_dir = state.memory_home.join("lenses");
    if let Err(e) = std::fs::create_dir_all(&lenses_dir) {
        return json_response(500, &format!(r#"{{"error":"{}"}}"#, e));
    }

    let path = lenses_dir.join(format!("{}.md", name));
    if let Err(e) = std::fs::write(&path, content) {
        return json_response(500, &format!(r#"{{"error":"{}"}}"#, e));
    }

    json_ok(&json!({"saved": name}))
}

fn handle_upload_lens(
    request: &mut tiny_http::Request,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
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
            );
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

    // Only accept .md files
    if !file.filename.to_lowercase().ends_with(".md") {
        return json_response(400, r#"{"error":"Only .md files are accepted"}"#);
    }

    // Derive lens name from filename
    let raw_name = file
        .filename
        .strip_suffix(".md")
        .or_else(|| file.filename.strip_suffix(".MD"))
        .unwrap_or(&file.filename);

    let name = match sanitize_lens_name(raw_name) {
        Some(n) => n,
        None => return json_response(400, r#"{"error":"Invalid lens filename"}"#),
    };

    let lenses_dir = state.memory_home.join("lenses");
    if let Err(e) = std::fs::create_dir_all(&lenses_dir) {
        return json_response(500, &format!(r#"{{"error":"{}"}}"#, e));
    }

    let dest = lenses_dir.join(format!("{}.md", name));
    if let Err(e) = std::fs::write(&dest, &file.data) {
        return json_response(500, &format!(r#"{{"error":"{}"}}"#, e));
    }

    json_ok(&json!({"uploaded": name}))
}

fn handle_delete_lens(
    name: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let decoded_name = url_decode(name);

    let sanitized = match sanitize_lens_name(&decoded_name) {
        Some(n) => n,
        None => return json_response(400, r#"{"error":"Invalid lens name"}"#),
    };

    let lenses_dir = state.memory_home.join("lenses");
    let path = lenses_dir.join(format!("{}.md", sanitized));

    if !path.exists() {
        return json_response(
            404,
            &format!(r#"{{"error":"Lens '{}' not found"}}"#, sanitized),
        );
    }

    if let Err(e) = std::fs::remove_file(&path) {
        return json_response(500, &format!(r#"{{"error":"Failed to delete: {}"}}"#, e));
    }

    json_ok(&json!({"deleted": sanitized}))
}

// ---------------------------------------------------------------------------
// Memory handlers
// ---------------------------------------------------------------------------

fn handle_list_memories(
    query: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let params = parse_query(query);
    let search_query = query_param(&params, "q");
    let limit: usize = query_param(&params, "limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let offset: usize = query_param(&params, "offset")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let include_archived = query_param(&params, "include_archived")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    let include_deep = query_param(&params, "include_deep")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    let state_filter = query_param(&params, "state");

    let mut brain = state.brain.write().unwrap();

    // Sync cross-process writes
    let _ = brain.sync_from_storage();

    if let Some(q) = search_query {
        if q.is_empty() {
            return list_recent_memories(
                &brain,
                limit,
                offset,
                include_archived,
                include_deep,
                state_filter,
            );
        }

        // Direct lookup if the query is a UUID
        if is_uuid(q) {
            let memories: Vec<JsonValue> = uuid::Uuid::parse_str(q)
                .ok()
                .and_then(|id| brain.get(&id))
                .into_iter()
                .map(|e| memory_to_json(e, None))
                .collect();
            let memories = filter_json_memories_by_state(memories, state_filter);
            return json_ok(&json!({
                "memories": memories,
                "search": q,
                "method": "id"
            }));
        }

        // Try semantic search first
        let (min_score_config, hybrid_enabled) = {
            let config = brain.config();
            (config.search_min_score, config.hybrid_search_enabled)
        };
        // Dashboard floor: higher than MCP pipeline since we lack LLM reranking to filter noise
        let min_score = (min_score_config as f32).max(0.5);

        let embedding_gen = crate::embedding::EmbeddingGenerator::new();
        match embedding_gen.generate(q) {
            Ok(query_embedding) => {
                match brain.find_similar_by_embedding(&query_embedding, limit, min_score) {
                    Ok(vector_results) => {
                        // If hybrid search is enabled, merge with BM25
                        let results = if hybrid_enabled {
                            let bm25_results = brain.keyword_search(q, limit).unwrap_or_default();
                            if bm25_results.is_empty() {
                                vector_results
                            } else {
                                use crate::memory_core::storage::rrf;
                                rrf::reciprocal_rank_fusion(
                                    &[&vector_results, &bm25_results],
                                    rrf::DEFAULT_K,
                                )
                            }
                        } else {
                            vector_results
                        };
                        let memories: Vec<JsonValue> = results
                            .iter()
                            .filter_map(|r| {
                                brain.get(&r.id).map(|e| memory_to_json(e, Some(r.score)))
                            })
                            .collect();
                        let memories = filter_json_memories_by_state(memories, state_filter);
                        json_ok(&json!({
                            "memories": memories,
                            "search": q,
                            "method": "semantic"
                        }))
                    }
                    Err(_) => {
                        // Fall back to text search
                        let results = brain.search(q);
                        let memories: Vec<JsonValue> = results
                            .iter()
                            .take(limit)
                            .map(|e| memory_to_json(e, None))
                            .collect();
                        let memories = filter_json_memories_by_state(memories, state_filter);
                        json_ok(&json!({
                            "memories": memories,
                            "search": q,
                            "method": "text"
                        }))
                    }
                }
            }
            Err(_) => {
                // Fall back to text search
                let results = brain.search(q);
                let memories: Vec<JsonValue> = results
                    .iter()
                    .take(limit)
                    .map(|e| memory_to_json(e, None))
                    .collect();
                let memories = filter_json_memories_by_state(memories, state_filter);
                json_ok(&json!({
                    "memories": memories,
                    "search": q,
                    "method": "text"
                }))
            }
        }
    } else {
        list_recent_memories(
            &brain,
            limit,
            offset,
            include_archived,
            include_deep,
            state_filter,
        )
    }
}

fn list_recent_memories(
    brain: &Brain,
    limit: usize,
    offset: usize,
    include_archived: bool,
    include_deep: bool,
    state_filter: Option<&str>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut memories: Vec<&crate::memory_core::Memory> = if let Some(state_str) = state_filter {
        if let Some(target_state) = parse_memory_state(state_str) {
            brain
                .all_memories()
                .filter(|e| e.state == target_state)
                .collect()
        } else {
            Vec::new()
        }
    } else if include_archived {
        brain.all_memories().collect()
    } else if include_deep {
        brain.all_memories().filter(|e| !e.is_archived()).collect()
    } else {
        brain.searchable_memories().collect()
    };

    // Sort by last_accessed descending (most recent first)
    memories.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));

    let total_filtered = memories.len();

    let result: Vec<JsonValue> = memories
        .iter()
        .skip(offset)
        .take(limit)
        .map(|e| memory_to_json(e, None))
        .collect();

    let stats = brain.stats();
    json_ok(&json!({
        "memories": result,
        "total_filtered": total_filtered,
        "has_more": offset + result.len() < total_filtered,
        "offset": offset,
        "limit": limit,
        "stats": {
            "total": stats.total_memories,
            "active": stats.active_memories,
            "dormant": stats.dormant_memories,
            "deep": stats.deep_memories,
            "archived": stats.archived_memories,
            "associations": stats.total_associations,
            "average_energy": stats.average_energy,
        }
    }))
}

fn memory_to_json(e: &crate::memory_core::Memory, score: Option<f32>) -> JsonValue {
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

/// Filter a Vec of JSON memories by state. The "state" field in JSON is the
/// Debug format like "Active", "Dormant", etc., so we case-insensitive match.
fn filter_json_memories_by_state(
    memories: Vec<JsonValue>,
    state_filter: Option<&str>,
) -> Vec<JsonValue> {
    if let Some(state_str) = state_filter {
        memories
            .into_iter()
            .filter(|e| {
                e.get("state")
                    .and_then(|s| s.as_str())
                    .map(|s| s.eq_ignore_ascii_case(state_str))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        memories
    }
}

fn handle_delete_memory(
    id: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let memory_id = match uuid::Uuid::parse_str(id) {
        Ok(id) => id,
        Err(e) => return json_response(400, &format!(r#"{{"error":"Invalid UUID: {}"}}"#, e)),
    };

    let mut brain = state.brain.write().unwrap();
    match brain.delete(memory_id) {
        Ok(true) => json_ok(&json!({"deleted": id})),
        Ok(false) => json_response(404, &format!(r#"{{"error":"Memory '{}' not found"}}"#, id)),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

fn handle_memory_associations(
    id: &str,
    state: &Arc<DashboardState>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let memory_id = match uuid::Uuid::parse_str(id) {
        Ok(id) => id,
        Err(e) => return json_response(400, &format!(r#"{{"error":"Invalid UUID: {}"}}"#, e)),
    };

    let mut brain = state.brain.write().unwrap();
    let _ = brain.sync_from_storage();

    let all_assocs = brain.all_associations();
    let related: Vec<JsonValue> = all_assocs
        .iter()
        .filter(|a| a.from == memory_id || a.to == memory_id)
        .filter_map(|a| {
            let other_id = if a.from == memory_id { a.to } else { a.from };
            let direction = if a.from == memory_id { "outbound" } else { "inbound" };
            brain.get(&other_id).map(|e| {
                json!({
                    "id": other_id.to_string(),
                    "content": e.content.clone(),
                    "direction": direction,
                    "weight": a.weight,
                    "energy": e.energy,
                    "state": format!("{:?}", e.state),
                    "state_emoji": e.state.emoji(),
                })
            })
        })
        .collect();

    json_ok(&json!({ "associations": related }))
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

    let mut brain = state.brain.write().unwrap();
    let _ = brain.sync_from_storage();

    let all_assocs = brain.all_associations();
    let filtered: Vec<_> = all_assocs
        .iter()
        .filter(|a| a.weight >= min_weight)
        .collect();

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
                "content": e.content,
                "energy": e.energy,
                "state": e.state.emoji(),
                "access_count": e.access_count,
                "tags": e.tags
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
// Update handlers
// ---------------------------------------------------------------------------

/// GET /api/updates — check all installed MemoryCo binaries for updates.
fn handle_check_updates() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let updater = memoryco_updater::Updater::new();
    match updater.check_all() {
        Ok(checks) => match serde_json::to_string(&checks) {
            Ok(body) => json_response(200, &body),
            Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
        },
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

/// POST /api/updates/:binary/stage — stage an update for a specific binary.
fn handle_stage_update(binary: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let updater = memoryco_updater::Updater::new();
    match updater.stage(binary) {
        Ok(result) => json_ok(&json!({
            "binary": result.binary,
            "previous_version": result.previous_version,
            "new_version": result.new_version,
            "staged": true,
        })),
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

/// POST /api/updates/stage-all — stage updates for all binaries with newer versions.
fn handle_stage_all_updates() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let updater = memoryco_updater::Updater::new();
    match updater.stage_all() {
        Ok(results) => {
            let staged: Vec<JsonValue> = results
                .iter()
                .map(|r| {
                    json!({
                        "binary": r.binary,
                        "previous_version": r.previous_version,
                        "new_version": r.new_version,
                    })
                })
                .collect();
            json_ok(&json!({ "staged": staged }))
        }
        Err(e) => json_response(500, &format!(r#"{{"error":"{}"}}"#, e)),
    }
}

// ---------------------------------------------------------------------------
// Friendly error mapping
// ---------------------------------------------------------------------------

/// Map technical reference-upload error messages to user-friendly messages.
fn friendly_error(technical: &str) -> String {
    if technical.contains("not a PDF file (missing .pdf extension)") {
        return "This file doesn't appear to be a PDF".to_string();
    }
    if technical.contains("invalid PDF: missing %PDF- magic bytes") {
        return "This file isn't a valid PDF — it may be corrupted or not actually a PDF"
            .to_string();
    }
    if technical.contains("panicked") {
        return "This PDF couldn't be read — it may be damaged or use unsupported features"
            .to_string();
    }
    if technical.contains("encryption scheme that is not supported") {
        return "This PDF is encrypted and can't be read".to_string();
    }
    if technical.contains("No extractable text") {
        return "This PDF contains only images — text search isn't available for it".to_string();
    }
    technical.to_string()
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
        .and_then(|s| s.strip_prefix("boundary="))
        .map(|s| s.trim_matches('"').to_string())
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
        if line.to_lowercase().contains("content-disposition") && line.contains("filename=") {
            // Find filename="..." or filename=...
            let (_, rest) = line.split_once("filename=")?;
            let filename = if let Some(stripped) = rest.strip_prefix('"') {
                // Quoted filename
                stripped.split_once('"').map(|(name, _)| name)?
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

/// Check if a string looks like a UUID (8-4-4-4-12 hex format).
fn is_uuid(s: &str) -> bool {
    if s.len() != 36 {
        return false;
    }
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match i {
            8 | 13 | 18 | 23 => {
                if b != b'-' {
                    return false;
                }
            }
            _ => {
                if !b.is_ascii_hexdigit() {
                    return false;
                }
            }
        }
    }
    true
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse a state string (from query param) into a MemoryState.
fn parse_memory_state(s: &str) -> Option<crate::memory_core::MemoryState> {
    match s.to_lowercase().as_str() {
        "active" => Some(crate::memory_core::MemoryState::Active),
        "dormant" => Some(crate::memory_core::MemoryState::Dormant),
        "deep" => Some(crate::memory_core::MemoryState::Deep),
        "archived" => Some(crate::memory_core::MemoryState::Archived),
        _ => None,
    }
}

/// Truncate content for display.
fn truncate_content(content: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }

    if content.len() <= max_len {
        content.to_string()
    } else {
        if max_len <= 3 {
            return ".".repeat(max_len);
        }

        let prefix_budget = max_len - 3;
        let mut end = prefix_budget.min(content.len());
        while end > 0 && !content.is_char_boundary(end) {
            end -= 1;
        }

        let prefix = content.get(..end).unwrap_or_default();
        format!("{}...", prefix)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Friendly error mapping ────────────────────────────────────────────

    #[test]
    fn friendly_error_not_a_pdf() {
        let msg = friendly_error("not a PDF file (missing .pdf extension)");
        assert_eq!(msg, "This file doesn't appear to be a PDF");
    }

    #[test]
    fn friendly_error_invalid_magic_bytes() {
        let msg = friendly_error("invalid PDF: missing %PDF- magic bytes");
        assert_eq!(
            msg,
            "This file isn't a valid PDF — it may be corrupted or not actually a PDF"
        );
    }

    #[test]
    fn friendly_error_panicked() {
        let msg = friendly_error("pdf-extract panicked: something went wrong");
        assert_eq!(
            msg,
            "This PDF couldn't be read — it may be damaged or use unsupported features"
        );
    }

    #[test]
    fn friendly_error_panicked_sanitize() {
        let msg = friendly_error("sanitize_filename panicked: unexpected input");
        assert_eq!(
            msg,
            "This PDF couldn't be read — it may be damaged or use unsupported features"
        );
    }

    #[test]
    fn friendly_error_panicked_indexer() {
        let msg = friendly_error("indexer panicked: thread error");
        assert_eq!(
            msg,
            "This PDF couldn't be read — it may be damaged or use unsupported features"
        );
    }

    #[test]
    fn friendly_error_encryption() {
        let msg =
            friendly_error("pdf-extract: encryption scheme that is not supported by this library");
        assert_eq!(msg, "This PDF is encrypted and can't be read");
    }

    #[test]
    fn friendly_error_no_extractable_text() {
        let msg = friendly_error("No extractable text (PDF may be scanned images without OCR)");
        assert_eq!(
            msg,
            "This PDF contains only images — text search isn't available for it"
        );
    }

    #[test]
    fn friendly_error_passthrough_unknown() {
        let msg = friendly_error("some unknown error happened");
        assert_eq!(msg, "some unknown error happened");
    }

    #[test]
    fn friendly_error_passthrough_io() {
        let msg = friendly_error("IO error: permission denied");
        assert_eq!(msg, "IO error: permission denied");
    }

    #[test]
    fn friendly_error_not_pdf_embedded_in_longer_message() {
        // The error might be wrapped in a larger message
        let msg = friendly_error("Validation failed: not a PDF file (missing .pdf extension)");
        assert_eq!(msg, "This file doesn't appear to be a PDF");
    }

    #[test]
    fn friendly_error_empty_string() {
        let msg = friendly_error("");
        assert_eq!(msg, "");
    }

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
        let segments: Vec<&str> = "api/identity/abc-123".split('/').collect();
        match segments.as_slice() {
            ["api", "identity", id] => assert_eq!(*id, "abc-123"),
            _ => panic!("Should match identity delete"),
        }
    }

    #[test]
    fn route_delete_memory() {
        let segments: Vec<&str> = "api/memories/550e8400-e29b-41d4-a716-446655440000"
            .split('/')
            .collect();
        match segments.as_slice() {
            ["api", "memories", id] => {
                assert_eq!(*id, "550e8400-e29b-41d4-a716-446655440000");
            }
            _ => panic!("Should match memory delete"),
        }
    }

    #[test]
    fn route_delete_reference() {
        let segments: Vec<&str> = "api/references/dsm5tr".split('/').collect();
        match segments.as_slice() {
            ["api", "references", name] => assert_eq!(*name, "dsm5tr"),
            _ => panic!("Should match reference delete"),
        }
    }

    #[test]
    fn route_post_stage_update() {
        let segments: Vec<&str> = "api/updates/memoryco_fs/stage".split('/').collect();
        match segments.as_slice() {
            ["api", "updates", binary, "stage"] => assert_eq!(*binary, "memoryco_fs"),
            _ => panic!("Should match stage update"),
        }
    }

    #[test]
    fn route_post_stage_all() {
        let segments: Vec<&str> = "api/updates/stage-all".split('/').collect();
        match segments.as_slice() {
            ["api", "updates", "stage-all"] => {} // matches
            _ => panic!("Should match stage-all"),
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

    #[test]
    fn parse_query_url_encoded() {
        let params = parse_query("q=hello%20world&limit=10");
        assert_eq!(query_param(&params, "q"), Some("hello world"));
        assert_eq!(query_param(&params, "limit"), Some("10"));
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

    #[test]
    fn truncate_content_unicode_boundary() {
        let input =
            "Step: Run full test suite — cargo test in /work/memoryco/memory, all tests must pass";
        let result = truncate_content(input, 30);
        assert_eq!(result, "Step: Run full test suite ...");
        assert!(result.len() <= 30);
    }

    #[test]
    fn truncate_content_tiny_max_len() {
        assert_eq!(truncate_content("abcdef", 0), "");
        assert_eq!(truncate_content("abcdef", 1), ".");
        assert_eq!(truncate_content("abcdef", 2), "..");
        assert_eq!(truncate_content("abcdef", 3), "...");
    }

    // ── Lens name sanitization ────────────────────────────────────────────

    #[test]
    fn sanitize_lens_name_basic() {
        assert_eq!(
            sanitize_lens_name("humanizer"),
            Some("humanizer".to_string())
        );
        assert_eq!(
            sanitize_lens_name("codereview"),
            Some("codereview".to_string())
        );
    }

    #[test]
    fn sanitize_lens_name_strips_bad_chars() {
        assert_eq!(
            sanitize_lens_name("../../../etc/passwd"),
            Some("etcpasswd".to_string())
        );
        assert_eq!(sanitize_lens_name("my lens!"), Some("mylens".to_string()));
        assert_eq!(
            sanitize_lens_name("foo bar/baz"),
            Some("foobarbaz".to_string())
        );
    }

    #[test]
    fn sanitize_lens_name_empty_after_strip() {
        assert_eq!(sanitize_lens_name("../../../"), None);
        assert_eq!(sanitize_lens_name("..."), None);
        assert_eq!(sanitize_lens_name("   "), None);
        assert_eq!(sanitize_lens_name(""), None);
    }

    #[test]
    fn sanitize_lens_name_preserves_hyphens_underscores() {
        assert_eq!(
            sanitize_lens_name("my-lens_v2"),
            Some("my-lens_v2".to_string())
        );
        assert_eq!(sanitize_lens_name("a-b_c"), Some("a-b_c".to_string()));
    }

    #[test]
    fn route_delete_lens() {
        let segments: Vec<&str> = "api/lenses/humanizer".split('/').collect();
        match segments.as_slice() {
            ["api", "lenses", name] => assert_eq!(*name, "humanizer"),
            _ => panic!("Should match lens delete"),
        }
    }

    // ── Port resolution ────────────────────────────────────────────────

    #[test]
    fn parse_port_valid() {
        assert_eq!(parse_port("4242"), Some(4242));
        assert_eq!(parse_port("9999"), Some(9999));
        assert_eq!(parse_port("0"), Some(0));
        assert_eq!(parse_port("65535"), Some(65535));
    }

    #[test]
    fn parse_port_invalid() {
        assert_eq!(parse_port("not_a_number"), None);
        assert_eq!(parse_port(""), None);
        assert_eq!(parse_port("-1"), None);
        // 99999 exceeds u16::MAX (65535)
        assert_eq!(parse_port("99999"), None);
    }

    #[test]
    fn bind_port_constant() {
        assert_eq!(BIND_PORT, 4242);
        assert_eq!(BIND_HOST, "127.0.0.1");
    }
}
