# Memory

A cognitive AI memory system powered by [memory](../../rust/memory).

## Features

- **Identity Layer** - Permanent bedrock: persona, values, preferences, relationships. Never decays.
- **Substrate Layer** - Episodic/semantic memories with organic decay and Hebbian learning.
- **Associative Learning** - Memories that are recalled together become linked together.
- **Resurrection** - Archived memories can be brought back through strong stimulation.

## Installation

```bash
cargo build --release
```

Binary: `target/release/memory`

## Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/memoryco/memory/target/release/memory"
    }
  }
}
```

## Bootstrap Prompt

Add this to your Claude custom instructions or user preferences:

```markdown
You have access to a cognitive memory system via MCP tools (`memory:*`).

At conversation start, bootstrap by loading your identity and operational context:

```
memory:identity_get
memory:memory_search(tags: ["operational"])
```

**Identity** is your permanent self - persona, values, preferences, relationships. It never decays.

**Memorys** are memories that can strengthen or fade based on use:
- Use `memory_recall` when actively referencing a memory (strengthens it, triggers learning)
- Use `memory_search` for passive lookups (no side effects)
- Use `memory_create` to store new facts/events

Follow what you find in your identity and operational memories.
```

## Tools

### Identity Tools

| Tool | Description |
|------|-------------|
| `identity_get` | Get your full identity (persona, values, preferences, etc.) |
| `identity_set` | Replace your identity from JSON |
| `identity_search` | Search identity for specific content |

### Memory Tools

| Tool | Description |
|------|-------------|
| `memory_create` | Create a new memory with optional tags |
| `memory_recall` | **Active** recall - stimulates memory, triggers Hebbian learning, can resurrect |
| `memory_search` | **Passive** search - no side effects, just lookup |
| `memory_get` | Get by ID without stimulation |
| `memory_associate` | Create explicit link between memories |
| `memory_stats` | Get substrate statistics |

### Config Tools

| Tool | Description |
|------|-------------|
| `config_get` | View current decay/learning configuration |
| `config_set` | Update configuration values |

## Memory States

Memories flow through energy states based on use:

```
✨ Active (≥0.3)  →  💤 Dormant (0.1-0.3)  →  🌊 Deep (0.02-0.1)  →  🧊 Archived (<0.02)
```

- **Active/Dormant**: Searchable, participates in learning
- **Deep**: Decays 75% slower, still retrievable
- **Archived**: Frozen, can be resurrected with strong recall

Memories never delete - they just sink deeper.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_DB` | Database path | `~/Library/Application Support/memory/brain.db` |

## Architecture

```
┌─────────────────────────────────────┐
│ MCP Server (memory)                 │
├─────────────────────────────────────┤
│ memory::Brain                       │
│  ├─ Identity (permanent)            │
│  ├─ Substrate (organic decay)       │
│  └─ SqliteStorage (persistence)     │
└─────────────────────────────────────┘
```
