# Code Intelligence MCP Tools Comparison

## codebase-memory-mcp vs code-review-graph vs semcode

Date: 2026-07-15

> Academic reference: [*Codebase-Memory: Tree-Sitter-Based Knowledge Graphs for LLM Code Exploration via MCP*](https://arxiv.org/abs/2603.27277), arXiv:2603.27277. PDF: https://arxiv.org/pdf/2603.27277

---

## 1. Tool Overview

### codebase-memory-mcp (DeusData)

- **Repo**: https://github.com/DeusData/codebase-memory-mcp
- **Language**: Pure C (single static binary, zero dependencies)
- **License**: MIT
- **Stars**: ~31.5K (most popular of the three)
- **Latest**: v0.9.0 (2026-07-08)
- **Primary use case**: High-performance code intelligence knowledge graph for AI coding agents. Indexes codebases into a persistent SQLite-backed graph of functions, classes, call chains, HTTP routes, and cross-service links. Designed to reduce token consumption by ~120x vs file-by-file exploration.

### code-review-graph (CRG)

- **Repo**: Community MCP server
- **Language**: Python
- **Storage**: SQLite + graph algorithms (Leiden community detection, betweenness centrality)
- **Primary use case**: Code review intelligence -- maps changes to affected functions, execution flows, communities, and test coverage gaps. Produces risk-scored, priority-ordered review guidance. Also generates architecture overviews, wiki pages, and refactoring suggestions.

### semcode (sem)

- **Repo**: Community MCP server
- **Language**: Python
- **Storage**: Tree-sitter AST + SQLite FTS + optional vector embeddings
- **Primary use case**: Semantic code search at the function level, git history analysis, lore.kernel.org email archive search, and commit similarity search. Heavily oriented toward Linux kernel development workflows.

---

## 2. Academic Validation (arXiv:2603.27277)

The codebase-memory-mcp project has a peer-reviewed preprint with quantitative evaluation:

- **Evaluation scope**: 31 real-world repositories, 66 languages (paper scope; current release has expanded to 158)
- **Answer quality**: 83% vs 92% for a file-exploration agent (9% gap, but at dramatically lower cost)
- **Token efficiency**: 10x fewer tokens, 2.1x fewer tool calls vs file-by-file exploration
- **Graph-native queries**: For hub detection and caller ranking, matches or exceeds the file explorer on 19 of 31 languages
- **Architecture**: Multi-phase pipeline with parallel worker pools, call-graph traversal, impact analysis, community discovery
- **Key tradeoff**: 9% quality gap (83% vs 92%) for 10x token savings. The gap is concentrated in queries requiring deep semantic understanding that graph structure alone cannot answer -- precisely where Hybrid LSP (added post-paper) aims to close it.

Neither code-review-graph nor semcode have published academic evaluations.

---

## 3. Feature-by-Feature Comparison

### 3.1 Core Purpose

| Dimension | codebase-memory-mcp | code-review-graph | semcode |
|-----------|--------------------|--------------------|---------|
| Primary mission | General-purpose code knowledge graph for AI agents | Code review intelligence + architecture analysis | Semantic code search + kernel dev workflow |
| Target user | Any AI coding agent user | Code reviewers doing PR/diff analysis | Kernel developers, git history researchers |
| Scope | Whole-codebase indexing + exploration | Change-focused review + community detection | Function-level search + git/lore search |

### 3.2 Indexing Approach

| Dimension | codebase-memory-mcp | code-review-graph | semcode |
|-----------|--------------------|--------------------|---------|
| Parser | tree-sitter (158 languages, vendored grammars compiled in) | tree-sitter (Python, Go, TS, JS, supported languages) | tree-sitter (multi-language) |
| Semantic layer | Hybrid LSP (10 languages: Python, TS/JS, PHP, C#, Go, C/C++, Java, Kotlin, Rust) | None (graph algorithms only) | Optional vector embeddings (sentence-transformers, OpenAI, Gemini, Minimax) |
| Graph model | Nodes (Function, Class, File, Route, Resource) + Edges (CALLS, IMPORTS, HTTP_CALLS, DATA_FLOWS, SIMILAR_TO, etc.) | Nodes (Function, Class, File, Test) + Edges (calls, imports, inherits) + Leiden communities + execution flows | Function/Class/Test nodes + caller/callee/import edges |
| Incremental indexing | Yes (background watcher, git-based change detection) | Yes (incremental_update mode, diff-based) | Yes (branch-based, git ref comparison) |
| Index speed | Extreme: Linux kernel in 3 min, Django in ~6s | Moderate: depends on repo size | Moderate: depends on repo size + optional embedding pass |
| Storage | SQLite (in-memory during index, dumped to disk) | SQLite (persistent) | SQLite (persistent) |

### 3.3 Query Capabilities

| Capability | codebase-memory-mcp | code-review-graph | semcode |
|------------|--------------------|--------------------|---------|
| Structural search | Yes (regex name patterns, label filters, degree filters, file scoping) | Yes (semantic_search_nodes, query_graph with callers_of/callees_of/imports_of/etc.) | Yes (smart_search, smart_outline, smart_unfold) |
| Semantic/vec search | Yes (bundled nomic-embed-code, 768d int8, 11-signal scoring, no API key) | Yes (embed_graph_tool with sentence-transformers/OpenAI/Gemini/Minimax) | Yes (vgrep_functions for vector similarity) |
| Full-text search | Yes (BM25 via SQLite FTS5 with camelCase/snake_case tokenizer) | Yes (FTS5 index) | Yes (grep_functions with regex on function bodies) |
| Call graph tracing | Yes (trace_path, BFS up to depth 5, import-aware, type-inferred) | Yes (query_graph: callers_of, callees_of; get_affected_flows; traverse_graph) | Yes (find_callers, find_calls, find_callchain) |
| Architecture overview | Yes (get_architecture: languages, packages, entry points, routes, hotspots, layers, clusters) | Yes (get_architecture_overview_tool with community-based analysis) | No dedicated tool |
| Community detection | Yes (Louvain community detection) | Yes (Leiden algorithm, community listing, cohesion metrics) | No |
| Dead code detection | Yes (zero-caller functions, entry-point exclusion) | Yes (refactor_tool mode=dead_code) | No |
| Change impact analysis | Yes (detect_changes: git diff to affected symbols + risk classification) | Yes (detect_changes_tool: risk-scored, priority-ordered; get_impact_radius; get_affected_flows) | Yes (diff_functions: extract functions from unified diff) |
| Cypher-like queries | Yes (MATCH/RETURN syntax against the graph) | No (uses predefined query patterns) | No |
| Cross-service linking | Yes (HTTP, gRPC, GraphQL, tRPC, pub/sub channels) | No | No |
| Data-flow tracing | Yes (DATA_FLOWS edges with arg-to-param + field-access chains) | No | No |
| Clone detection | Yes (SIMILAR_TO via MinHash+LSH) | No | No |
| Git history analysis | No | No (focused on current-state analysis) | Yes (find_commit, list_branches, compare_branches, vcommit_similar_commits) |
| Email/lore search | No | No | Yes (lore_search, vlore_similar_emails -- kernel mailing list search) |
| Refactoring support | No dedicated tool | Yes (refactor_tool: rename preview, dead code, suggestions) | No |
| Wiki generation | No | Yes (generate_wiki_tool from community structure) | No |
| Architecture Decision Records | Yes (manage_adr persists decisions) | No | No |
| Graph visualization | Yes (3D UI at localhost:9749, optional) | No | No |

### 3.4 Language Support

| Dimension | codebase-memory-mcp | code-review-graph | semcode |
|-----------|--------------------|--------------------|---------|
| Total languages | 158 (vendored tree-sitter grammars) | Limited (Python, Go, TS/JS primarily) | Multiple via tree-sitter |
| Deep type resolution | 10 languages (Python, TS/JS, PHP, C#, Go, C/C++, Java, Kotlin, Rust, Perl) | None | None |
| Semantic analysis quality | High for Hybrid LSP languages; syntactic-only for the rest | Graph-level (community/flow), not type-level | Function-level structural |

### 3.5 MCP Integration Quality

| Dimension | codebase-memory-mcp | code-review-graph | semcode |
|-----------|--------------------|--------------------|---------|
| MCP tools count | 15 | ~30+ | ~25+ |
| Agent auto-setup | Yes (43 agents auto-detected and configured) | No (manual MCP config) | No (manual MCP config) |
| Auto-index on session start | Yes (configurable) | No (manual build_or_update_graph) | No (manual or background) |
| Install experience | One-line curl, single binary | pip install, Python dependencies | pip install, Python dependencies + optional embedding models |
| Platform support | macOS/Linux/Windows (static binaries) | Any Python environment | Any Python environment |
| Token efficiency claims | ~120x reduction vs file-by-file (benchmarked) | Not benchmarked, but graph queries are inherently compact | Not benchmarked |
| Pre-tool hooks | Yes (context injection for agent awareness) | No | No |

### 3.6 Unique Strengths

#### codebase-memory-mcp

1. **Performance**: By far the fastest indexer. Linux kernel in 3 minutes is unmatched. RAM-first pipeline with LZ4 compression.
2. **Zero dependencies**: Single static C binary. No Python, no Docker, no API keys. Easiest to install and maintain.
3. **Broadest language coverage**: 158 languages with vendored grammars. No external tree-sitter installation needed.
4. **Hybrid LSP**: Semantic type resolution for 10 major languages, embedded in the binary. This is a significant differentiator -- it resolves dotted chains like `user.profile.display_name()` that pure AST tools miss.
5. **Cross-service intelligence**: HTTP/gRPC/GraphQL route matching, pub/sub channel detection, async queue dispatch. No other tool does this.
6. **Bundled embeddings**: nomic-embed-code baked into the binary. Semantic search with zero setup, zero API cost.
7. **Cypher queries**: Expressive graph query language for complex multi-hop patterns.
8. **3D visualization**: Interactive graph UI for exploring architecture visually.
9. **Team collaboration**: Share compressed graph snapshots via git.
10. **ADR management**: Persist architectural decisions alongside the graph.

#### code-review-graph

1. **Review-focused design**: Purpose-built for code review workflows. Every tool serves the review pipeline.
2. **Risk scoring**: Automatically scores changes by blast radius, test coverage gaps, and community boundaries.
3. **Execution flow analysis**: Traces call chains from entry points through changed code. Unique capability.
4. **Leiden community detection**: Discovers functional modules and detects cross-community coupling surprises.
5. **Bridge/hub node analysis**: Identifies architectural chokepoints (betweenness centrality) and hotspots (degree).
6. **Wiki generation**: Auto-generates documentation from community structure.
7. **Refactoring intelligence**: Rename preview with blast-radius analysis, dead code detection, move suggestions.
8. **Surprising connections**: Finds unexpected cross-community coupling that signals design issues.
9. **Knowledge gaps**: Identifies isolated nodes, thin communities, untested hotspots.
10. **Minimal context entry point**: `get_minimal_context_tool` returns ~100 tokens for any task, suggesting next tools.

#### semcode

1. **Git history depth**: commit search, branch comparison, author filtering, path filtering, regex matching on diffs. No other tool matches this.
2. **lore.kernel.org integration**: Search kernel mailing lists, find email threads for commits, semantic email search. Unique for kernel developers.
3. **Vector-based function search**: `vgrep_functions` finds functions by meaning ("memory allocation function") not just name.
4. **Vector-based commit search**: `vcommit_similar_commits` finds commits similar to a description.
5. **Structural code reading**: `smart_outline` (file symbols without bodies), `smart_unfold` (expand one symbol) -- very token-efficient.
6. **Commit similarity**: Find similar commits across history for pattern recognition.

---

## 4. Weaknesses and Limitations

### codebase-memory-mcp

- **Closed-source binary**: While the README says "full source is here," the primary distribution is a precompiled C binary. Auditing requires building from source. Trust model depends on signed releases.
- **No git history analysis**: Focused on current-state graphs. Cannot search commits, compare branches, or analyze authorship.
- **No kernel-specific tooling**: No mailing list search, no checkpatch integration, no lore.kernel.org support.
- **No refactoring preview**: Can detect dead code but cannot preview rename blast radius.
- **No review-specific pipeline**: General-purpose exploration, not change-focused review guidance.
- **Young project**: v0.9.0, 248 open issues, breaking changes possible before 1.0.

### code-review-graph

- **Performance**: Python-based, slower indexing than C. Not benchmarked against large codebases like the Linux kernel.
- **Language coverage**: Fewer supported languages than codebase-memory-mcp. Primarily Python, Go, TS/JS.
- **No semantic type resolution**: Graph algorithms detect communities and flows, but cannot resolve `user.profile.display_name()` chains.
- **No cross-service linking**: Cannot match HTTP routes to call sites across services.
- **No bundled embeddings**: Requires separate sentence-transformers or API key setup for semantic search.
- **No visualization**: No built-in graph UI.
- **Review-centric**: Less useful for pure exploration tasks outside of a review context.

### semcode

- **Narrow scope**: Git history + function search + kernel lore. Not a general code intelligence platform.
- **No community detection**: Cannot discover functional modules or detect cross-community coupling.
- **No architecture overview**: No high-level view of codebase structure.
- **No cross-service analysis**: No HTTP/gRPC route matching.
- **No call graph depth**: Basic caller/callee but no BFS traversal or flow analysis.
- **Kernel-oriented**: lore.kernel.org and commit similarity are niche features. Less useful for non-kernel projects.
- **Embedding-dependent**: Best features (vcommit, vgrep) require vector index, which needs sentence-transformers or API keys.

---

## 5. Scenario-Based Recommendations

### Scenario 1: General AI Coding Agent (exploring a new codebase)

**Recommended: codebase-memory-mcp**

Reason: 158-language coverage, extreme indexing speed, bundled embeddings, Cypher queries, zero dependencies. One install, one "Index this project" command, and the agent has full structural awareness. The 120x token savings claim is credible given the architecture.

### Scenario 2: Code Review / PR Analysis

**Recommended: code-review-graph**

Reason: Purpose-built for review workflows. Risk scoring, execution flow tracing, blast radius analysis, community boundary detection, and knowledge gap identification are exactly what a reviewer needs. The `detect_changes_tool` + `get_review_context_tool` pipeline is unmatched for change-focused analysis.

### Scenario 3: Linux Kernel Development

**Recommended: semcode (primary) + codebase-memory-mcp (secondary)**

Reason: semcode's `lore_search`, `dig` (commit-to-email lookup), `vcommit_similar_commits`, and git history tools are purpose-built for kernel workflows. codebase-memory-mcp adds structural exploration of the massive kernel codebase (indexed in 3 minutes). code-review-graph adds little value here since kernel review uses checkpatch + mailing list, not GitHub PRs.

### Scenario 4: Microservice / Multi-Repo Architecture

**Recommended: codebase-memory-mcp**

Reason: Cross-service HTTP/gRPC/GraphQL linking, cross-repo `CROSS_*` edges, multi-galaxy 3D visualization, and infrastructure-as-code indexing (Dockerfiles, K8s manifests) are unique to this tool. No other tool can trace an HTTP call from one service to the handler in another.

### Scenario 5: Large-Scale Refactoring

**Recommended: codebase-memory-mcp + code-review-graph**

Reason: codebase-memory-mcp provides dead code detection, clone detection (SIMILAR_TO), and semantic search across the full graph. code-review-graph adds rename preview with blast radius, move suggestions, and community-aware refactoring guidance. Use codebase-memory-mcp for discovery, code-review-graph for planning.

### Scenario 6: Token Budget is Tight (maximize context efficiency)

**Recommended: codebase-memory-mcp**

Reason: The bundled embeddings + Cypher queries + 15 MCP tools are designed specifically for token-minimal exploration. The `get_minimal_context` pattern (100 tokens for task orientation) is built into the architecture. No external dependencies means no context wasted on tool setup.

### Scenario 7: Pure Git History Research

**Recommended: semcode**

Reason: `find_commit` with author/subject/path/regex filtering, `compare_branches`, `vcommit_similar_commits` with semantic search, and the full commit metadata pipeline. No other tool has comparable git history depth.

---

## 6. Complementary Usage (All Three Together)

The three tools are more complementary than competitive:

1. **codebase-memory-mcp** = the structural backbone. Index the codebase once, get fast graph queries, cross-service linking, and semantic search. Use for exploration, architecture understanding, and impact analysis.

2. **code-review-graph** = the review brain. When a PR or diff arrives, use CRG to score risk, trace affected flows, detect community boundary violations, and generate review guidance. Its community detection and bridge-node analysis catch architectural issues that structural queries miss.

3. **semcode** = the history lens. When you need to understand WHY code exists (git history), WHO to ask (lore.emails), or find SIMILAR patterns (commit similarity), semcode is the only tool that does this.

**Optimal stack**: codebase-memory-mcp as the always-on index, code-review-graph activated for review sessions, semcode activated for kernel/history work.

---

## 7. Decision Matrix (Quick Reference)

| If you need... | Use... | Why... |
|----------------|--------|--------|
| Fastest codebase indexing | codebase-memory-mcp | C binary, Linux kernel in 3 min |
| Broadest language support | codebase-memory-mcp | 158 languages, 10 with type resolution |
| Zero-dependency install | codebase-memory-mcp | Single static binary |
| Code review risk scoring | code-review-graph | Purpose-built review pipeline |
| Community/module detection | code-review-graph | Leiden algorithm + cohesion metrics |
| Execution flow tracing | code-review-graph | Entry-point-to-change call chains |
| Git commit history search | semcode | Author, subject, path, regex, semantic |
| Kernel mailing list search | semcode | lore.kernel.org integration |
| Cross-service architecture | codebase-memory-mcp | HTTP/gRPC/GraphQL/pub-sub linking |
| Semantic code search (no API) | codebase-memory-mcp | Bundled nomic-embed-code embeddings |
| Rename blast radius preview | code-review-graph | refactor_tool with apply_refactor |
| Clone/duplicate detection | codebase-memory-mcp | MinHash+LSH SIMILAR_TO edges |
| 3D graph visualization | codebase-memory-mcp | Built-in UI at localhost:9749 |
| Architecture Decision Records | codebase-memory-mcp | manage_adr tool |
| Wiki/docs generation | code-review-graph | generate_wiki from communities |
