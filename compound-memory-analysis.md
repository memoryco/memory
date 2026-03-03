# Compound Memory Analysis Report

**Database:** `brain.db`
**Analysis date:** 2026-03-03 09:17
**Total engrams:** 1514

## 1. Basic Statistics

| Metric | Characters | Words |
|--------|-----------|-------|
| Average | 305 | 39 |
| Median | 283 | 35 |
| Max | 1306 | 178 |

## 2. Sentence Count Distribution

| Sentences | Count | Percentage | Bar |
|-----------|-------|------------|-----|
| 1 | 326 | 21.5% | ██████████ |
| 2 | 285 | 18.8% | █████████ |
| 3-4 | 624 | 41.2% | ████████████████████ |
| 5-6 | 209 | 13.8% | ██████ |
| 7+ | 70 | 4.6% | ██ |

**3+ sentence memories: 903 (59.6%)** — these are candidates for splitting.

## 3. Character Length Distribution

| Length | Count | Percentage | Bar |
|--------|-------|------------|-----|
| <100 | 52 | 3.4% | █ |
| 100-200 | 331 | 21.9% | ██████████ |
| 200-400 | 804 | 53.1% | ██████████████████████████ |
| 400-600 | 262 | 17.3% | ████████ |
| 600+ | 65 | 4.3% | ██ |

**400+ char memories: 327 (21.6%)** — likely compound.

## 4. Compound Score Distribution

The compound score combines: sentence count, numbered items, semicolons, bullets, 
compound connectors ('also', 'additionally'), colon-lists, length penalty, 
technical term density, and line breaks.

| Score Range | Count | Percentage | Bar |
|-------------|-------|------------|-----|
| 0 (atomic) | 424 | 28.0% | ██████████████ |
| 0.1-2 (low) | 699 | 46.2% | ███████████████████████ |
| 2.1-5 (moderate) | 293 | 19.4% | █████████ |
| 5.1-10 (high) | 86 | 5.7% | ██ |
| 10+ (severe) | 12 | 0.8% |  |

**Moderate+ compound score: 391 (25.8%)** — strong candidates for decomposition.

## 5. Top 15 Worst Offenders

| # | ID (short) | Preview | Sent | Chars | Score | Flags |
|---|-----------|---------|------|-------|-------|-------|
| 1 | `02db1a78` | Code review of agents crate (2026-02-18, second pass). Prior issues all fixed. New code: agent_types_get.rs, agent_types | 12 | 1306 | 19.6 | numbered_items=5, connectors=2, sentences=12 |
| 2 | `dd21fdf5` | All 9 code review issues fixed in agents crate (2026-02-18): 1) state::agents_dir() now returns Result<PathBuf, StateErr | 10 | 653 | 17.0 | numbered_items=9, sentences=10, long=653 |
| 3 | `268341d9` | Agents crate replication requirements (2026-02-14): New 'agents' crate should replicate filesystem-mcp patterns. Root st | 10 | 1063 | 16.2 | numbered_items=8, sentences=10, long=1063 |
| 4 | `28abd2b4` | Code review of agents crate (2026-02-18, second pass): All prior issues fixed. New findings: 1) reader_thread sends SIGT | 10 | 923 | 15.8 | numbered_items=8, sentences=10, long=923 |
| 5 | `2fd642fb` | agents crate code review (2026-02-18 round 2): Previous issues all fixed — state.rs agents_dir() now returns Result, das | 7 | 1002 | 15.0 | numbered_items=8, connectors=1, sentences=7 |
| 6 | `78ccd047` | Code review of agents crate completed (2026-02-18). Key findings: 1) state.rs agents_dir() panics on missing home dir in | 9 | 716 | 14.8 | numbered_items=7, connectors=1, sentences=9 |
| 7 | `89766eb7` | DLNA test app confirmed working with WiiM Pro Receiver. Key findings: (1) Seek must happen while transport is PLAYING -  | 7 | 690 | 13.5 | numbered_items=6, connectors=1, sentences=7 |
| 8 | `afdace16` | exports/mobile.rs has 1,277 lines of C ABI FFI code. Key findings: (1) null C string handling via `parse_c_str()` - safe | 6 | 728 | 11.5 | numbered_items=4, connectors=1, sentences=6 |
| 9 | `384b3814` | Code review by Ethel Slapwhistle (2026-02-19): First typed agent spawn worked. Review found: (1) CRITICAL: state::agents | 7 | 583 | 11.4 | numbered_items=5, sentences=7, long=583 |
| 10 | `fbbce78c` | Round 4 fixes completed by Deborah Thunderpants (2026-02-19): 212 tests passing. Major changes: (1) truncate_str made pu | 8 | 558 | 11.0 | numbered_items=6, sentences=8, long=558 |
| 11 | `e6924f93` | Segment flush integration test pattern: 1) Create SegmentConfiguration with write_key, 2) Set file_storage_at() with Tra | 3 | 617 | 10.5 | numbered_items=8, sentences=3, long=617 |
| 12 | `a960fd85` | LOCOMO benchmark harness for MemoryCo: fully built (2025-03-01). Location: /Users/bsneed/work/memoryco/bench/locomo/. 8  | 11 | 759 | 10.2 | sentences=11, long=759, tech_terms=13 |
| 13 | `c1a4cff9` | filesystem-mcp semantic index internals: SemanticIndex::open() finds git root or uses given path, creates .ai-index/.git | 4 | 713 | 9.6 | numbered_items=7, sentences=4, long=713 |
| 14 | `0817f3df` | filesystem-mcp/src/main.rs bootstrap flow: (1) CLI arg parsing with --help/-h, (2) Config::from_args_and_env() loads all | 1 | 529 | 9.4 | numbered_items=7, long=529, tech_terms=6 |
| 15 | `a3939dc2` | Stereotype v10 mockup approved: Brandon said 'looking clean as hell'. Layout: 352x425 logical (704x849 @2x retina). Vide | 11 | 479 | 9.1 | numbered_items=3, sentences=11, long=479 |

## 6. Energy & Access Correlation

Comparing **atomic** (1-2 sentences AND <200 chars) vs **compound** (5+ sentences OR 400+ chars):

| Metric | Atomic | Compound | Delta |
|--------|--------|----------|-------|
| Count | 339 | 423 | — |
| Avg Energy | 0.348 | 0.510 | +0.162 |
| Avg Access Count | 1.4 | 1.0 | -0.3 |

Compound memories have **46.6% higher energy** than atomic ones.
Atomic memories are accessed **1.3x more** — they're more findable and useful.

## 7. Time Trend

| Month | Count | Avg Sentences | Avg Chars | Avg Score |
|-------|-------|---------------|-----------|-----------|
| 2026-01 | 292 | 2.3 | 233 | 1.0 |
| 2026-02 | 1062 | 3.4 | 326 | 1.8 |
| 2026-03 | 160 | 2.5 | 300 | 1.6 |

## Summary & Recommendations

- **1514** total memories analyzed
- **60%** have 3+ sentences (should be 1-2)
- **22%** are 400+ characters (likely compound)
- **26%** score moderate+ on compound heuristics
- Average content length is **305 chars / 39 words** (ideal: <150 chars)

### Recommended Actions

1. **Automated splitting**: Build a decomposition pass that splits compound memories into atomic facts
2. **Creation guard**: Add validation at `engram_create` to reject or warn on multi-sentence inputs
3. **Priority targets**: Start with the 10+ compound score memories — these are the worst offenders
4. **Estimated work**: ~391 memories need decomposition, likely yielding 1173+ atomic memories