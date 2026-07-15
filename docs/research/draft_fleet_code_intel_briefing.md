# Code Intelligence MCP 工具调研报告 — 舰队长决策简报

日期：2026-07-15
调研人：ashare 主 session（三轮搜索，Exa + Firecrawl + Context7 + GitHub）
范围：代码智能 MCP 服务器生态全景 + 自建可行性

---

## 执行摘要

1. **code-review-graph (CRG) 有 3 个不可忽视的痛点**，其中 Shell 跨文件调用和内核 C 外部 API 解析严重影响日常工作流。
2. **三轮搜索发现 ~59 个代码智能工具**，其中 5 个关键新发现改变了架构方案：CoreGraph（stack-graphs）、SCIP-IO、GitLab Orbit Local、lsp-mcp-server/clangd、shuck_semantic。
3. **推荐混合方案 C**：先组合现有工具（1 天见效），再渐进自建 Rust 统一服务器（8 周）。Shell 问题已有解（shuck_semantic，95%+ 准确率）。

---

## 一、问题陈述：CRG 痛点

| 痛点 | 严重度 | 影响 | 根因 |
|------|--------|------|------|
| Shell/Bash 跨文件调用返回 0 | HIGH | surflare-watchdog 等 shell 项目无法分析 | tree-sitter 无法解析 `source` 链 |
| 内核 C 外部 API callees=0 | HIGH | kernel networking 开发无调用图 | printk/register_kprobe 不在 repo 里 |
| 超大 repo 3.8M chars 崩溃 | MEDIUM | 大项目架构概览不可用 | 输出为巨型文本而非结构化 JSON |
| 小 repo 不值得建图 | LOW | 日常小脚本分析无意义 | Grep 够用 |

---

## 二、调研规模

| 轮次 | 工具 | 方法 | 新发现 |
|------|------|------|--------|
| 第一轮 | 17 | GitHub 搜索 + README | codebase-memory-mcp, KGraph, sverklo 等 |
| 第二轮 | 42 | Exa + Firecrawl + Context7 | CoreGraph, SCIP-IO, GitLab Orbit, gortex 等 |
| 第三轮 | 深入 | 针对 Shell 问题专项 | shuck_semantic（Rust crate，已解决 source 链） |
| **合计** | **~59** | | |

---

## 三、5 个关键新发现

| 工具 | 类型 | 核心价值 | 对我们的意义 |
|------|------|---------|------------|
| **CoreGraph** | Rust, MIT | tree-sitter + stack-graphs（GitHub 跨文件解析），边置信度评分 | 不需要编译环境就能获得接近编译器级的跨文件解析 |
| **SCIP-IO** | Rust, MIT | SCIP 多语言编排器（11 语言，9 个 indexer） | 不需要自己写 SCIP 编排，免费获得 11 语言 SCIP 支持 |
| **GitLab Orbit Local** | Rust | 生产级单二进制，DuckDB，28 stars，40 contributors | 现成的生产级方案，可能不需要自建 |
| **lsp-mcp-server** | TS | 通用 MCP-to-LSP 代理 | 零代码获得 clangd 内核 C 深度 |
| **shuck_semantic** | Rust | Shell source 链解析（95%+），WorkspaceCallIndex + CrossFileCall | Shell 问题已有解，只需加图导出层 |

---

## 四、方案对比

| | 方案 A：自建 | 方案 B：组合现有 | 方案 C：混合（推荐） |
|---|---|---|---|
| 描述 | Rust 核心 + stack-graphs + SCIP-IO + shuck_semantic | Orbit Local + lsp-mcp-server/clangd + semcode + sverklo | 先 B（立即见效）再 A（渐进统一） |
| 工作量 | 4-6 周 | 1 天 | 8 周 |
| 第 1 天可用 | 否 | 是 | 是 |
| 统一图 | 是 | 否（4 个独立索引） | 是（最终） |
| Shell 解决 | 是（shuck_semantic） | 否 | 是（Phase 2） |
| 内核 C 解决 | 是（SCIP-IO + clangd） | 是（lsp-mcp-server/clangd） | 是 |
| 维护成本 | 高（自建代码） | 低（上游维护） | 中（渐进迁移） |
| 技术风险 | 中（code2graph pre-0.1） | 低 | 低（分阶段验证） |

---

## 五、推荐方案：混合 C

### 时间线

| 阶段 | 周 | 做什么 | 交付物 |
|------|---|--------|--------|
| Phase 0 | 1 | 安装 Orbit Local + lsp-mcp-server/clangd + semcode + sverklo | 4 个 MCP server 配置，立即可用 |
| Phase 1 | 2-4 | Rust 核心 + tree-sitter + stack-graphs + SQLite | 统一代码图服务器（15 个 MCP 工具） |
| Phase 2 | 5-6 | 集成 shuck_semantic（Shell）+ SCIP-IO（多语言 SCIP） | Shell 跨文件 + SCIP 深度 |
| Phase 3 | 7-8 | 迁移：从 Orbit 到自建服务器，保留 sverklo/semcode 作为插件 | 统一图 + 部分插件 |

### 架构（Phase 1 完成后）

MCP Server (rmcp, Rust binary)
  Query Engine: search | trace | impact | review | community | flow
  Graph Store: SQLite + FTS5 + embeddings
  Analysis: leiden-rs (社区检测) | petgraph (PageRank/BFS)
  Parsing Layer (pluggable):
    Tier 1: tree-sitter + stack-graphs (158 语言，跨文件解析)
    Tier 2: SCIP-IO (opt-in, 11 语言 SCIP)
    Tier 3: shuck_semantic (Shell source 链, 95%+)
    Tier 4: clangd via lsp-mcp-server (opt-in, C/C++ 内核深度)

---

## 六、技术栈

| 组件 | 选择 | 理由 |
|------|------|------|
| 语言 | Rust | 性能（CBM 3 分钟索引内核），单二进制分发，生态成熟 |
| 解析器 | tree-sitter + stack-graphs | 158 语言 + 跨文件解析，不需要编译环境 |
| Shell 解析 | shuck_semantic | Rust crate，source 链解析 95%+，WorkspaceCallIndex |
| 内核 C | SCIP-IO + lsp-mcp-server/clangd | 双保险：SCIP 深度 + LSP 代理 |
| 存储 | SQLite + FTS5 | 所有成功工具都用 SQLite，零配置，WAL 并发 |
| 嵌入 | ort (ONNX) + all-MiniLM-L6-v2 | 捆绑在二进制里，无 API key，384d |
| 社区检测 | leiden-rs | Rust 原生，petgraph 适配，rayon 并行 |
| MCP SDK | rmcp | 官方 Rust SDK，proc macro 注册 |

---

## 七、风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| stack-graphs 不够成熟 | MEDIUM | HIGH | Phase 0 先用 Orbit Local 验证需求 |
| Shell 动态路径无法解析 | HIGH | LOW | shuck_semantic 正确标记为 unresolved，不误报 |
| 性能目标未达（内核 3 分钟） | MEDIUM | MEDIUM | 分析 CBM 的 C 实现热路径，必要时 FFI |
| 范围蔓延 | HIGH | HIGH | Phase 1 严格限制 15 个工具 |
| Orbit Local 够用不想自建 | LOW | LOW | 这是好事，节省 6 周工作量 |

---

## 八、未解决问题

| 问题 | 现状 | 前景 |
|------|------|------|
| Shell 动态路径（eval, $()） | 只有运行时追踪能解决 | Phase 3 可选：bash -x + 沙箱 |
| LLM 辅助调用图解析 | 只有学术论文（CALLME, SEA） | 2026 内无生产工具预期 |
| Python/JS 动态 import | tree-sitter 级别 | stack-graphs 可能改善 |
| 跨工具统一索引 | 无标准 | 我们的 SQLite 图 schema 就是答案 |

---

## 九、决策点（需要舰队长裁决）

1. 是否先试 Orbit Local？ — 建议是。1 天安装，如果满足 80% 需求可以推迟自建。
2. 是否投入 8 周自建？ — 如果 Orbit 不够用，推荐方案 C。
3. Shell 优先级？ — shuck_semantic 已解决核心问题，可以作为 Phase 2 的低成本增量。
4. 内核 C 方案？ — lsp-mcp-server/clangd 是零成本方案，SCIP-IO 是深度方案，两者可并存。

---

## 附录：调研报告索引

| 报告 | 路径 | 内容 |
|------|------|------|
| 三工具对比 | /tmp/draft_codebase_tools_comparison.md | CRG vs CBM vs semcode |
| 编排方案 | /tmp/draft_mcp_code_tools_orchestration.md | 配置方法 + 替代工具清单 |
| 深度分析 | /tmp/draft_deep_code_tools_analysis.md | 6 个工具 vs CRG 痛点逐项对比 |
| 自建可行性 | /tmp/draft_custom_code_intel_feasibility.md | 架构设计 + 技术栈 + 阶段规划（含 Appendix B） |
| 穷举搜索 | /tmp/draft_exhaustive_code_tools_search.md | 59 个工具完整清单 |
| Shell 专项 | /tmp/draft_shell_crossfile_research.md | 跨文件调用解析方案 + shuck_semantic 分析 |
