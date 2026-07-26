# L1 Review: risk layer state machine (shadow-first)

Date: 2026-07-26
Branch: fix/risk-state-machine
Files: 7 changed (940 insertions, 52 deletions)

## Review evidence

### Code-review-graph analysis
- Overall risk score: 0.40
- 0 confirmed findings
- 20 test gaps (pipeline/report integration functions, not unit-testable)
- Highest risk: RiskStateContext (0.4), test_log_and_query (0.4)

### Test results
- 32 new tests (test_risk_state.py): ALL PASSED
- 46 existing tests (test_risk.py): ALL PASSED (no regression)
- Bug-injection: remove lockdown -> oscillation test FAILS (proves lockdown works)

### Static checks
- py_compile: all files OK
- ruff: all checks passed
- Non-ASCII gate: clean (only intentional Chinese/emoji in report template)

### Forge review (deepseek-direct, 3 passes, 36752 tokens)
- Job: git mode, FORGE_ALLOW_NO_BASELINE=1
- Verdict: FAIL (11 findings: 1 confirmed, 7 uncertain, 3 dismissed)
- ALL findings are about CLAUDE.md symlink (false positive)
  - forge误判: worktree symlink 指向主仓库 CLAUDE.md，不是自引用
  - worktree 路径: .worktrees/risk-state-machine/CLAUDE.md
  - 目标路径: /home/houminxi/code/ashare-lab/CLAUDE.md (主仓库)
  - 不同文件，非循环引用
- L1 代码改动无 confirmed findings
- Advisory: taint on report.py:620-635 (pre-existing, not from this change)
- Runtime: smoke 0/1 surfaces verified (no Python files in mutation diff)

## Findings

No confirmed code defects in L1 changes.
CLAUDE.md symlink finding is a false positive (worktree -> main repo, not self-referential).

## Advisory items (non-blocking)

1. pipeline.py functions (_step4_load_state, _step9_risk_checks) lack unit tests
   - Integration-tested through full pipeline
   - Not blocking: these are orchestrator functions

2. report.py functions (gather_report_data, format_chinese_report) lack unit tests
   - Pre-existing gap, not introduced by this change
   - Not blocking: these are template functions

3. Taint advisory on report.py:620-635 (pre-existing)
   - Tainted data flows to network sink
   - Not introduced by this change
