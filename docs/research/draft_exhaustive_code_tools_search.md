# Exhaustive Code Intelligence MCP Tools Search Report

**Date**: 2026-07-15
**Search scope**: Exa, Firecrawl, GitHub Topics, GitHub Search (code-intelligence, code-graph, codebase-indexing, mcp-server)
**Already known**: 17 tools (see user's list)

---

## SECTION A: NEWLY DISCOVERED TOOLS (not in the original 17)

### Tier 1 -- High-impact, actively maintained, significant differentiation

#### 1. gortex (zzet/gortex)
- **Repo**: https://github.com/zzet/gortex
- **Stars**: Unknown (created 2026-04-06, very active -- pushed today)
- **Language**: Unknown (likely Rust given description)
- **License**: Unknown
- **What it does**: High-performance code-intelligence engine supporting **257 languages**, multi-repository support, graph-based, with CLI + MCP Server + API. Claims 50x token reduction.
- **Unique vs known tools**: Widest language support (257 langs). Multi-repo federation. Discord community.
- **Solves our problems?**: Language breadth is unmatched. Could be interesting for polyglot repos.
- **Maturity**: New (3 months), but extremely active (pushed hours ago). Discord community suggests real users.

#### 2. CoreGraph (simplecore-inc/coregraph)
- **Repo**: https://github.com/simplecore-inc/coregraph
- **Stars**: Unknown (created 2026-06-07)
- **Language**: Rust
- **License**: MIT
- **What it does**: Rust CLI that combines **tree-sitter + stack-graphs** for cross-file name resolution. Confidence-scored edges (0.0-1.0). MCP + LSP + HTTP + IPC. Background daemon architecture.
- **Unique vs known tools**: Uses **stack-graphs** (GitHub's cross-file name resolution) alongside tree-sitter. Every edge has a confidence score and trust model. Cross-file inconsistency detection (enum mismatches, API path drift).
- **Solves our problems?**: stack-graphs is the closest thing to compiler-grade cross-file resolution without a compiler. Confidence scoring is unique.
- **Maturity**: Very new (5 weeks), but well-documented, MIT licensed.

#### 3. GitLab Orbit Knowledge Graph (gitlabhq/orbit-knowledge-graph)
- **Repo**: https://github.com/gitlabhq/orbit-knowledge-graph
- **Stars**: 28
- **Language**: Rust (91.9%)
- **License**: Custom (NOASSERTION)
- **What it does**: GitLab's official code intelligence platform. Two modes: **Orbit Local** (single binary, DuckDB, code-only graph from any local repo) and **Orbit Remote** (ClickHouse-backed, indexes full SDLC + code). MCP + REST + gRPC. Indexes 11+ languages.
- **Unique vs known tools**: Backed by GitLab. The local mode is a standalone CLI that needs no GitLab account. Uses rust-analyzer for Rust indexing. Uses a YAML ontology for graph schema. Production-proven at GitLab scale.
- **Solves our problems?**: The local binary (`orbit index`) could work on any repo. Uses tree-sitter + language-specific parsers. Bash/Shell support is limited (function defs + source imports only, no cross-file call resolution).
- **Maturity**: 28 stars, 40 contributors, actively developed. Replaced the older `gitlab-org/rust/knowledge-graph`.

#### 4. sdsrss/code-graph-mcp
- **Repo**: https://github.com/sdsrss/code-graph-mcp
- **Stars**: Unknown (created 2026-03-10)
- **Language**: Rust
- **License**: Unknown
- **What it does**: AST knowledge graph MCP server. 12 tools including `project_map`, `semantic_code_search` (hybrid BM25 + vector + RRF), `get_call_graph`, `trace_http_chain`, `impact_analysis`, `find_dead_code`. Uses sqlite-vec for vector search, Candle for local embeddings.
- **Unique vs known tools**: HTTP route tracing is unusual. Merkle tree incremental indexing. Context compressor with token estimation. CLI + MCP dual interface.
- **Solves our problems?**: Good general-purpose option. HTTP route tracing could be useful for web service repos.
- **Maturity**: 4 months old, Rust-based.

#### 5. Infigraph (GudmundrLLC/infigraph)
- **Repo**: https://github.com/GudmundrLLC/infigraph (also johnintuit/infigraph fork)
- **Stars**: Unknown (created recently)
- **Language**: Unknown
- **License**: Unknown
- **What it does**: AST-powered code intelligence engine. **62 languages**, Cypher queries, hybrid semantic search, cross-file call resolution, SCIP-enriched edges. Edge types include CALLS, INHERITS, IMPLEMENTS, BRIDGE_TO (FFI/JNI/gRPC/WASM), CALLS_SERVICE (cross-service HTTP).
- **Unique vs known tools**: **SCIP integration** for enrichment. Cross-language bridge detection (FFI, JNI, gRPC, COM, WASM). Cross-service HTTP call detection. "Learned resolution" -- records successful cross-file resolutions to improve accuracy.
- **Solves our problems?**: SCIP integration is novel. Bridge/service detection could be valuable for microservice architectures.
- **Maturity**: New, fork exists suggesting early development.

#### 6. simplecore-inc/coregraph -- already listed above

#### 7. Ataraxy-Labs/sem
- **Repo**: https://github.com/Ataraxy-Labs/sem
- **Stars**: Unknown (created 2026-02-05, very active)
- **Language**: Rust (likely)
- **License**: Unknown
- **What it does**: **Semantic version control** -- entity-level diffs, blame, and impact analysis on top of git. 28 languages via tree-sitter. Built for coding agents.
- **Unique vs known tools**: Not an MCP server per se -- it's a **git-aware semantic diff tool**. Entity-level (not line-level) diffs. Companion project `weave` does entity-level merge conflict resolution (95% reduction vs line-based).
- **Solves our problems?**: Could complement any code graph tool with semantic git operations.

#### 8. Goldziher/basemind
- **Repo**: https://github.com/Goldziher/basemind
- **Stars**: Unknown (created 2024-06-01, oldest tool found, very active)
- **Language**: Rust
- **License**: Unknown
- **What it does**: Full AI context layer: tree-sitter code-map, document RAG, shared memory, multi-agent comms, web crawl, git history + blame. **300+ languages**, 10+ agent harnesses, pure Rust.
- **Unique vs known tools**: Broadest scope -- combines code intelligence with document RAG, web crawl, and multi-agent communication. 300+ language support.
- **Solves our problems?**: More than just code intelligence -- it's a full context layer. Could be overkill if you only need code graphs.
- **Maturity**: 2+ years old, actively maintained.

### Tier 2 -- Noteworthy, mid-maturity

#### 9. bartolli/codanna
- **Repo**: https://github.com/bartolli/codanna
- **Stars**: Unknown (created 2025-07-24, active)
- **Language**: Rust (likely)
- **What it does**: Local code intelligence MCP server and CLI for AI coding agents.

#### 10. GlitterKill/sdl-mcp (Symbol Delta Ledger)
- **Repo**: https://github.com/GlitterKill/sdl-mcp
- **Stars**: Unknown (created 2026-02-08, very active)
- **Language**: Unknown
- **What it does**: Policy-centered context budget layer for coding agents. Symbol-graph intelligence + precision tools. Turns codebases into compact, high-signal context.

#### 11. GlitterKill/scip-io
- **Repo**: https://github.com/GlitterKill/scip-io
- **Stars**: 4
- **Language**: Rust
- **License**: MIT
- **What it does**: **SCIP Index Orchestrator** -- detects languages, downloads SCIP indexers, runs them, merges results into a single `index.scip`. Orchestrates 11 languages across 9 indexers.
- **Unique**: The only tool that orchestrates multiple SCIP indexers into one merged index. Deterministic merging.
- **Solves our problems?**: Directly answers "Is there a tool that does SCIP-based indexing for multiple languages?" YES -- this is it.

#### 12. Cranot/roam-code
- **Repo**: https://github.com/Cranot/roam-code
- **Stars**: Unknown (created 2026-02-09, very active)
- **Language**: Unknown
- **What it does**: SQLite code graph, 28 languages, **238 CLI commands, 224 MCP tools**, change-safety gates, audit evidence, zero API keys.
- **Unique**: Sheer tool count (224 MCP tools) is the highest of any tool found.

#### 13. g-tiwari/mcp-codebase-intelligence
- **Repo**: https://github.com/g-tiwari/mcp-codebase-intelligence
- **Stars**: Unknown
- **Language**: Unknown
- **What it does**: 18 tools, 8 languages, tree-sitter + LSP for TS/JS. `semantic_diff` (git ref-based), `analyze_change_impact`, `architecture_diagram` (mermaid), `query_codebase` (natural language).
- **Unique**: LSP-powered go-to-definition for TS/JS alongside tree-sitter for other languages. Mermaid diagram generation.

#### 14. ProfessioneIT/lsp-mcp-server
- **Repo**: https://github.com/ProfessioneIT/lsp-mcp-server
- **Stars**: Unknown (created 2026-01-26)
- **Language**: Unknown
- **What it does**: **Bridges Claude Code to any LSP server**. Acts as a generic MCP-to-LSP proxy.
- **Unique**: Not a code intelligence tool itself -- it's a **universal adapter** that lets any LSP server be consumed via MCP. Could give access to clangd, rust-analyzer, etc. through MCP.
- **Solves our problems?**: Could solve kernel C depth by proxying clangd through MCP.

#### 15. sjkim1127/Architect_MCP
- **Repo**: https://github.com/sjkim1127/Architect_MCP
- **Stars**: Unknown (created 2026-03-13)
- **Language**: Rust
- **What it does**: Professional-grade static analysis. Tree-sitter + Rayon parallel processing. 11+ languages. Architecture governance (layer boundaries, circular dependency rules). SSE for cloud deployment.
- **Unique**: Architectural governance (enforce layer boundaries via JSON rules). External coupling analysis. Outbound call mapping (HTTP/gRPC/DB).

#### 16. jasondk/clangaroo
- **Repo**: https://github.com/jasondk/clangaroo
- **Stars**: Unknown (created 2025-06-25)
- **Language**: Unknown
- **What it does**: **Fast C++ code intelligence for LLMs via MCP**. Uses clangd.
- **Solves our problems?**: C++/C focused. Could be relevant for kernel C if combined with compile_commands.json.

#### 17. 2015xli/clangd-graph-rag
- **Repo**: https://github.com/2015xli/clangd-graph-rag
- **Stars**: Unknown (created 2025-09-27)
- **Language**: Unknown
- **What it does**: **Source code graph RAG for C/C++ based on clang/clangd**.
- **Solves our problems?**: Another clangd-based option for C/C++ code intelligence.

#### 18. MarcelRoozekrans/roslyn-codelens-mcp
- **Repo**: https://github.com/MarcelRoozekrans/roslyn-codelens-mcp
- **Stars**: Unknown (created 2026-03-06, very active)
- **Language**: C#/.NET
- **What it does**: **Roslyn-based MCP server for .NET/C#** -- 57 tools for navigation, call graphs, diagnostics, code fixes, test intelligence, DI graphs, IL/external-assembly inspection.
- **Unique**: Compiler-grade .NET intelligence via Roslyn (the C# compiler itself). DI graph extraction.

#### 19. abdulmunimjemal/codescope-mcp
- **Repo**: https://github.com/abdulmunimjemal/codescope-mcp
- **Stars**: Unknown (created 2026-06-01)
- **Language**: Unknown
- **What it does**: Local-first codebase knowledge-graph MCP. Watch-first architecture, 21 languages. Claims "faster, leaner & more accurate than the incumbent" (codebase-memory-mcp).

#### 20. anvia-hq/lexa
- **Repo**: https://github.com/anvia-hq/lexa
- **Stars**: Unknown (created 2026-06-02, active)
- **Language**: Rust (likely)
- **What it does**: Fast local code intelligence. Portable, queryable graph. MCP server.

#### 21. CodeBendKit/codeseek
- **Repo**: https://github.com/CodeBendKit/codeseek
- **Stars**: Unknown (created 2026-06-03, active)
- **Language**: Rust
- **What it does**: Call graphs + hybrid semantic search (Dense + Sparse + RRF + Reranker) across 7 languages. MCP tools for Claude Code and Codex CLI.
- **Unique**: Explicit reranker in the search pipeline.

#### 22. 1337Xcode/Cortex
- **Repo**: https://github.com/1337Xcode/Cortex
- **Stars**: Unknown (created 2026-05-18)
- **Language**: Rust
- **What it does**: Local code intelligence. Call graph, MCP. 29 languages, 32 tools, single Rust binary.

#### 23. Anandb71/arbor
- **Repo**: https://github.com/Anandb71/arbor
- **Stars**: Unknown (created 2026-01-04, very active)
- **Language**: Rust (likely)
- **What it does**: Graph-native code intelligence that **replaces embedding-based RAG with deterministic program understanding**.
- **Unique**: Explicitly anti-embedding -- deterministic graph traversal only.

#### 24. aovestdipaperino/tokensave
- **Repo**: https://github.com/aovestdipaperino/tokensave
- **Stars**: Unknown (created 2026-02-26, very active)
- **Language**: Rust (likely)
- **What it does**: 40+ tools, 30+ languages, 9 agent integrations. Pre-indexed semantic knowledge graphs. 100% local.

#### 25. blackwell-systems/knowing
- **Repo**: https://github.com/blackwell-systems/knowing
- **Stars**: Unknown (created 2026-05-15, active)
- **Language**: Unknown
- **What it does**: Permanent code intelligence layer. Content-addressed graph with **Merkle proofs**. 25 extractors, 23 MCP tools. "Gets smarter with use."
- **Unique**: Merkle proof verification of code knowledge. Content-addressed storage.

#### 26. kage-core/Kage
- **Repo**: https://github.com/kage-core/Kage
- **Stars**: Unknown (created 2025-07-19, active)
- **Language**: Unknown
- **What it does**: Persistent, verified memory for coding agents. Every memory checked against actual code. Lives in repo as plain files, shared via git. No account, no DB.
- **Unique**: Git-native storage of code knowledge. Verification against live code.

#### 27. supermodeltools/mcp
- **Repo**: https://github.com/supermodeltools/mcp
- **Stars**: Unknown (created 2025-12-24)
- **Language**: Unknown
- **What it does**: Call graphs, dependency graphs, dead code detection, blast radius analysis. Claims 40% token savings.

#### 28. ShiftinBits/constellation-mcp
- **Repo**: https://github.com/ShiftinBits/constellation-mcp
- **Stars**: Unknown (created 2025-09-17, active)
- **Language**: Unknown
- **What it does**: Code Intelligence Platform for AI Coding Assistants.

#### 29. lambda-alpha-labs/Graphenium
- **Repo**: https://github.com/lambda-alpha-labs/Graphenium
- **Stars**: Unknown (created 2026-06-24, active)
- **Language**: Unknown
- **What it does**: Pre-flight linter and architecture gate for AI agents. Uses **tree-sitter + Stack Graphs + Datalog**. Mechanically blocks structural drift, layering bypasses, and scope creep on virtual ASTs before code changes land.
- **Unique**: Stack Graphs + Datalog combination. Pre-flight (blocks changes, not just reports).

#### 30. fallow-rs/fallow
- **Repo**: https://github.com/fallow-rs/fallow
- **Stars**: Unknown (created 2026-03-17, very active)
- **Language**: Rust
- **What it does**: TypeScript/JavaScript focused. Free static analysis: unused code, duplication, circular deps, complexity hotspots, architecture boundaries, design-system drift. Optional paid runtime layer.

### Tier 3 -- Code review / quality focused (not pure code intelligence, but overlapping)

#### 31. religa/multi_mcp -- 32 stars, multi-model code review orchestration
#### 32. Lyx3314844-03/static-analysis-mcp -- 50+ tools, 28 languages, AI-powered review
#### 33. JCHETAN26/CodeAudit-MCP -- Semgrep-powered polyglot review
#### 34. tehprof/quality-guardian-mcp -- Aggregates Semgrep + PHPStan + Knip + PHPMetrics + Deptrac
#### 35. mauriziomocci/mcp-code-review -- GitHub PRs + GitLab MRs review
#### 36. thangtn83/mcp_pr_review -- Skill-based multi-language PR review

### Tier 4 -- Semantic search only (no graph/call analysis)

#### 37. smallthinkingmachines/semantic-code-mcp -- ONNX nomic-embed-code, LanceDB, cross-encoder reranking
#### 38. ceaksan/mcp-code-search -- LanceDB + sentence-transformers, 20+ langs
#### 39. axkirillov/semantic-search-mcp -- Qdrant + OpenAI embeddings, 25+ langs
#### 40. theunreal/codebase-mcp -- ChromaDB, remote MCP, multi-repo
#### 41. vrppaul/semantic-code-mcp -- LanceDB, FastMCP
#### 42. itseasy21/mcp-codebase-index -- Qdrant + Gemini/OpenAI/Ollama

---

## SECTION B: ANSWERS TO SPECIFIC QUESTIONS

### Q1: Is there a tool that does SCIP-based indexing for multiple languages (not just C/C++)?

**YES -- two answers:**

1. **SCIP itself** (scip-code/scip, 654 stars, Apache-2.0) is the protocol. Per-language indexers already exist:
   - scip-java: Java, Scala, Kotlin
   - scip-typescript: TypeScript, JavaScript
   - rust-analyzer: Rust
   - scip-clang: C++, C
   - scip-ruby: Ruby
   - scip-python: Python
   - scip-dotnet: C#, VB
   - scip-dart: Dart
   - scip-php: PHP

2. **SCIP-IO** (GlitterKill/scip-io, 4 stars, MIT, Rust) is the **orchestrator** that detects languages, downloads indexers, runs them, and merges results into a single `index.scip`. Currently orchestrates 11 languages across 9 indexers. Deterministic merging. This is the missing glue.

3. **Infigraph** (GudmundrLLC/infigraph) uses SCIP for edge enrichment alongside its own tree-sitter parsing.

### Q2: Is there a tool that uses LLM-assisted call graph resolution?

**Research exists but no production MCP tool yet:**

- Academic papers: **CALLME** (OpenReview) combines LLMs with static analysis to resolve dynamic property accesses. **SEA** (Semantic-Enhanced Analysis) uses LLMs for indirect call analysis. Several papers from FSE 2025 and Springer investigate LLM-augmented call graphs.
- **codebase-memory-mcp** (DeusData) claims "LSP-style hybrid type resolution" inspired by tsserver/gopls/rust-analyzer embedded in its C binary -- not LLM-assisted but compiler-inspired.
- **Infigraph** has "learned resolution" that records successful cross-file call resolutions to improve accuracy on subsequent indexes -- a form of learned (but not LLM-based) resolution.
- **No production MCP tool currently uses LLM calls to resolve dynamic dispatch/calls at index time.** This remains an unsolved gap.

### Q3: Is there a unified "code intelligence as a service" platform that we could self-host?

**YES -- GitLab Orbit:**

- **GitLab Orbit** (gitlabhq/orbit-knowledge-graph) is the closest. It has:
  - **Orbit Local**: Single Rust binary, DuckDB, indexes any local repo, no GitLab account needed
  - **Orbit Remote**: ClickHouse-backed, indexes full SDLC + code, requires GitLab.com group
  - MCP + REST + gRPC interfaces
  - Production-proven at GitLab scale
  - Open source

- **Sourcegraph** is the original code intelligence platform but is not MCP-native and is heavier to self-host.

- **codegraph-ai/CodeGraph** (Rust, 45 tools, VS Code extension) is a strong self-hosted option with MCP + LSP dual protocol support.

### Q4: Are there any Rust-native code intelligence tools beyond code2graph?

**YES -- many.** The Rust ecosystem is now dominant for code intelligence tools:

| Tool | Language | Key Differentiator |
|------|----------|-------------------|
| codebase-memory-mcp | C | 158 langs, bundled embeddings |
| gortex | Rust? | 257 langs, multi-repo |
| narsil-mcp | Rust | 90 tools, 32 langs |
| codegraph-ai/CodeGraph | Rust | 45 tools, 38 langs, VS Code, MCP+LSP |
| CoreGraph | Rust | tree-sitter + stack-graphs, confidence scoring |
| GitLab Orbit | Rust | Enterprise-grade, DuckDB/ClickHouse |
| code-index-mcp | Rust | BSL/1C support, federation, zstd compression |
| codanna | Rust | Local MCP server |
| codeseek | Rust | Dense+Sparse+RRF+Reranker search |
| basemind | Rust | 300+ langs, full context layer |
| fallow | Rust | TS/JS focused, runtime analysis |
| Architect_MCP | Rust | Architecture governance |
| Cortex | Rust | 29 langs, 32 tools |
| stria | Rust | Zero-config, grammar-free, 4.7MB binary |
| tokensave | Rust | 40+ tools, 30+ langs |

### Q5: Shell cross-file call problem -- any solutions?

**Very limited. No production-grade solution exists.**

- **shell-call-graph** (codeberg.org/rak/shell-call-graph): Single-file shell script call graph. Does NOT follow `source`/`.` commands. Only works within one file.
- **callGraph** (koknat/callGraph): Multi-language (bash, perl, python, TCL, etc.) regex-based call graph. Line-by-line regex, not AST. No cross-file resolution.
- **GitLab Orbit**: Bash/Shell supports function definitions and `source`/`.` sourced-script imports only. Cross-file call resolution is explicitly "out of scope."
- **ops-codegraph-tool** (optave): Has "Bash rules" but details unclear.
- **No tool resolves bash function calls across sourced files** (e.g., `source lib.sh; my_function` where `my_function` is defined in `lib.sh`). This remains an unsolved problem because bash has no static import semantics -- `source` can be conditional, paths can be variables, etc.

---

## SECTION C: TOOLS THAT SOLVE SPECIFIC UNSOLVED PROBLEMS

### Cross-file shell calls
**UNSOLVED.** No tool handles this. The closest approaches are:
- GitLab Orbit (source/`.` imports only)
- bash `FUNCNAME`/`BASH_SOURCE`/`BASH_LINENO` runtime tracing (not static)

### Kernel C depth
**Partially solved by:**
- **scip-clang** (Sourcegraph): Compiler-grade C/C++ via clang, requires `compile_commands.json`
- **clangaroo** (jasondk): C++ focused, uses clangd
- **clangd-graph-rag** (2015xli): C/C++ graph RAG based on clang/clangd
- **lsp-mcp-server** (ProfessioneIT): Generic MCP-to-LSP proxy -- could proxy clangd
- **KGraph** (already known): scip-clang based

### Large repo performance
**Best options:**
- **codebase-memory-mcp**: Claims Linux kernel (28M LOC) in 3 minutes, sub-ms queries
- **gortex**: Claims 50x token reduction, 257 languages
- **GitLab Orbit**: Production-proven at GitLab scale (ClickHouse backend)
- **CoreGraph**: Background daemon architecture, precomputed graph

---

## SECTION D: SUMMARY OF FINDINGS

### Total tools found: ~42 new tools + 17 already known = ~59 total

### Top recommendations for further investigation:

1. **CoreGraph** (simplecore-inc/coregraph) -- stack-graphs + tree-sitter + confidence scoring. Most technically interesting new find.
2. **GitLab Orbit Local** (gitlabhq/orbit-knowledge-graph) -- production-proven, standalone binary, Rust.
3. **SCIP-IO** (GlitterKill/scip-io) -- the missing SCIP multi-language orchestrator.
4. **gortex** (zzet/gortex) -- 257 languages, most breadth of any tool.
5. **lsp-mcp-server** (ProfessioneIT) -- universal MCP-to-LSP adapter, could proxy clangd for kernel C.
6. **Graphenium** (lambda-alpha-labs) -- Stack Graphs + Datalog for pre-flight architecture gates.
7. **basemind** (Goldziher/basemind) -- broadest scope (code + docs + web + memory), 300+ langs, 2+ years old.

### Key gap that remains:
- **No tool uses LLM-assisted call graph resolution** at index time (only academic papers)
- **No tool solves cross-file shell script call resolution** (bash `source` across files)
- **No single tool combines SCIP-precision with tree-sitter-breadth** -- SCIP-IO gets closest by orchestrating both
