# memory TODO

## Unified Dashboard

### Vision

One dashboard at port 4242. When memoryco is running, it becomes the hub for all
companion tools. Each tool (memoryco_fs, memoryco_agents) owns its own UI and HTTP
server, but memoryco's dashboard surfaces them as tabs via iframe.

If memoryco isn't running, each tool falls back to its own standalone dashboard on
its own port.

### How it works

**Plugin registration:**
- Each plugin already writes a manifest to `~/.memoryco/bootstrap.d/{plugin}.toml`
- Add a `port` field to each manifest so memory knows where to find them:
  ```toml
  [meta]
  name = "filesystem"
  version = "1.0.0"
  port = 4244
  ```
- Memory's dashboard reads `bootstrap.d/` on load, discovers registered plugins
  and their ports, and renders a tab per plugin

**Iframing:**
- Memory dashboard adds a tab for each discovered plugin
- Tab content = `<iframe src="http://127.0.0.1:{port}">` 
- Each tool owns its UI entirely — memory just provides the chrome

**`open_dashboard` in each plugin (fs, agents):**
- Probe 4242 for memory's dashboard signature (`/api/identity`)
- If found: open `http://127.0.0.1:4242?tab={plugin_name}`
- If not found: fall back to own port (`http://127.0.0.1:{own_port}`)
- Memory's dashboard reads the `?tab=` param on load and activates the right tab

**Standalone mode (no memory):**
- memoryco_fs runs its dashboard on 4244
- memoryco_agents runs its dashboard on 4243
- Each tool's `open_dashboard` probe fails, falls back cleanly — no regression

### What needs to happen

**memory:**
- [ ] Read `bootstrap.d/` manifests for `port` field on dashboard startup
- [ ] Render a tab per discovered plugin in the dashboard UI
- [ ] Iframe each plugin's dashboard in its tab
- [ ] Handle `?tab={name}` URL param to activate a specific tab on load
- [ ] Add `/api/plugins` endpoint listing registered plugins + ports (optional, nice to have)

**memoryco_fs:**
- [ ] Add `port = 4244` to manifest written by `registry.rs`
- [ ] Update `open_dashboard` tool to probe 4242 first, open `?tab=filesystem` if found

**memoryco_agents:**
- [ ] Add `port = 4243` to manifest written by `registry.rs`
- [ ] Update `open_dashboard` tool to probe 4242 first, open `?tab=agents` if found
