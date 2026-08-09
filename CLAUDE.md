# ashare-lab

Global rules in ~/CLAUDE.md apply. This file adds project-specific rules only -- no state, no architecture descriptions (those live in project memory).


## Checkpoint Pipeline

Inherits ~/CLAUDE.md "GSD Checkpoint Pipeline" verbatim. Project-specific additions only:

- CP1b model panel: gm (L3 runtime), ds (L1 docs), mm (L0 surface), mimo (L0.5 signatures). When gm fails (VPN exit CN), proceed with 3 models, retry gm next round.
- CP3 forge backend: mimo-pro (gate.yaml, stream: true, timeout_s: 3600) or deepseek fallback.
- Non-code changes (docs/config/chore) skip to CP5 only.


## Machine Roles

Three machines, three distinct roles:

- **Local dev machine**: write code, run forge review, unit tests. NOT for archiving -- reboots unpredictably, no persistent storage guarantee.
- **GPU (admin@192.168.100.11, Windows)**: multi-tenant training host.
    Repo and all data live on **H: drive** (`H:\ashare-lab\`, `H:\.qlib\`).
    **All files on win-gpu must go to H: drive.** Never create files on C:\Users\admin\Desktop\, C:\Users\admin\AppData\, or any C: drive path. Temp files go to `H:\tmp\`. This applies to subagents too — include "禁止在桌面或C盘创建文件" in all subagent prompts that SSH to win-gpu.
    C: drive has NO ashare artifacts -- cleared during Phase 6 deploy.
    Python 3.12 is global (`C:\...\Python312\python.exe`), no .venv.
  - **ashare TRA training** (this project): `run_*.py` + `*.log` are
    temporary; cleanup after local verification (see Results Collection).
  - **harness forge-bot training** (other project, per harness
    PLAN.md): `experiments/<dir>/models/*.pt` and
    `predictions/*.parquet` are long-lived; **never clean these**.
  - Never edit code here -- it is a push/run/collect target, not a
    dev environment. All edits happen locally in a worktree, then
    push via tar/scp.
  - SSH key: X500 ed25519 authorized via `C:\ProgramData\ssh\administrators_authorized_keys`
    (Windows OpenSSH admin user rule -- NOT `~/.ssh/authorized_keys`).
- **X500 (x500:~/code/ashare-lab)**: permanent archive AND Phase 3+ execution host. Code, models, predictions, and results all land here permanently. Phase 3/4/5 paper engine runs on X500; test results are collected from X500.

X500 and GPU machine details in project memory `reference_gpu_machine.md`.


## Dev-First Workflow

Full flow for any code change:

1. Local worktree edit
2. Forge review + anti-ai-audit + non-ASCII check
3. Commit to main, push to x500 remote: `git push x500 main`
4. If GPU training needed: tar/scp worktree to GPU -> run -> scp results back -> local verify -> clean GPU temp files
5. Sync verified results + data artifacts to X500 (see X500 Sync below)

- Push from the worktree, never the main tree.
- GPU run scripts (`run_*.py`) are operational and not committed.
- Never clean GPU temp files before local verification confirms results are correct.
- Never clean the canonical experiment directory -- only ad-hoc scripts and logs.
- Cleanup policy is project-specific:
  - **ashare temp files** (`run_*.py`, `*.log`): clean after local verify.
  - **harness forge-bot output** (`models/*.pt`, `predictions/*.parquet`):
    preserve permanently; do not clean on ashare runs.
  - When in doubt, check which project the file belongs to before
    deleting -- ashare and harness live on the same machine.


## Results Collection

1. scp results back from GPU (predictions, verdict JSONs, validation JSONs, logs)
2. Verify locally (compare against expected IC/gate values, check file integrity)
3. Clean ashare temp files only (delete `run_*.py`, `*.log`).
   Preserve canonical experiment dir AND any harness forge-bot
   output (`models/*.pt`, `predictions/*.parquet`).
4. Sync verified results to X500 (see X500 Sync)

Order is mandatory: collect -> verify -> clean GPU -> sync X500. Never reverse.


## X500 Sync (mandatory after every verified result)

X500 is the only permanent store. Local machine reboots unpredictably -- a result that only exists locally is not archived. After local verification passes:

```bash
# Code
git push x500 main

# Gitignored data (models, predictions, results)
scp -r results/ x500:code/ashare-lab/
scp experiments/<dir>/models/*.pt x500:code/ashare-lab/experiments/<dir>/models/
scp experiments/<dir>/predictions/*.parquet x500:code/ashare-lab/experiments/<dir>/predictions/
```

Phase 3+ paper engine runs on X500. Test results for Phase 3/4/5 are collected from X500, not local.


## 60/40 Blend Lock

The 60/40 TRA/nTRA blend ratio is LOCKED. Do not sweep weights, add parameters, or introduce trainable blend coefficients. Live OOS decides whether to keep it.


## Fleet Zone Discipline (harness-fleet org-charter 2026-07-07)

ashare group sovereign territory (Zone C): paper.db, models, pipeline, sentinel, gpu-win BOX operations. No other group edits Zone C; cross-zone needs are contracts via fleet.

**H0b ownership**: ashare EXECUTES + OPERATES qwen serving on gpu-win. Zero ashare-lab code edits during observation window. Config edits via mhou_workspace worktree only.

**gpu-win GPU split (user policy 2026-07-25)**: static dual-GPU partition -- GPU 0 (RTX 3080) is RESERVED for local models (llama-server serving), ashare training runs pinned to GPU 1 (RTX 3060 Ti, `CUDA_VISIBLE_DEVICES=1`). Training no longer contends with serving; do NOT stop llama-server for training. Legacy: `batch_experiment.py` may still acquire `H:\gpu.lock` (stale-lock TTL, mtime-based expiry) from the pre-split era.

**X500 timer namespace** (reserved): ashare-wol 17:30, data-update 17:45, pipeline 18:00, dsa-sentinel 18:50, chenditc 21:00, backup 23:00 (CST). Harness dispatcher gets separate namespace per J2.

Fleet law binding: S1/S2/S2b, known-answer validation, pre-registration, zone discipline, R4 agent contract, Golden Rules 1-6. Full ref: memory `reference_fleet_assignments.md`.

**Coordinated session**: trinity-router (`~/code/trinity-router/`) is part of ashare group (assigned 2026-07-07). Provides LLM benchmark infrastructure (probe_common, worker_pool, HumanEval+/DebugBench/MBPP/LCB datasets, 8-model baseline). Current task: H0b W1 qwen3.6-27B evaluation.


## Sync Rule

Any change to this CLAUDE.md or to ashare-lab project memory must be followed by a scan of ~/CLAUDE.md and ~/code/ashare-lab/.planning/ memory files for consistency. If a rule here contradicts or duplicates a global rule, resolve it: project-specific overrides go here, universal rules stay global, and duplicates are removed from the less-specific location.


## Post-Commit Bookkeeping Checklist (mandatory, non-negotiable)

Every commit that changes code or closes a task MUST complete ALL of these before reporting done. Phase 3 review history showed repeated failures: GSD files forgotten, memory stale, counters drifting from evidence. This is a HARD GATE.

1. **GSD .planning/ sync**: update STATE.md (frontmatter counters + prose), write/update the plan SUMMARY.md in the phase directory (NOT repo root), update UAT/VERIFICATION if verify-work was invoked.
2. **Memory sync (both locations)**:
   - Project: `~/.claude/projects/-home-houminxi-code-ashare-lab/memory/MEMORY.md`
   - Global: `~/.claude/projects/-home-houminxi/memory/project_ashare_lab.md`
   - Both must agree on HEAD, commit count, plan count, test count. Partial update = contradiction the next session inherits.
3. **HEAD + commit count**: derive from git (`git rev-list --count <base>..HEAD`), never increment manually.
4. **Briefing sync**: update or mark stale.
5. **Non-ASCII check**: all changed files including CLAUDE.md and memory.

Origin: main session flagged the same "memory stale / GSD counters wrong / SUMMARY missing" pattern 4 times across 3 review rounds in Phase 3.


## Pipeline Data Quality Rules (2026-07-13 incident)

Incidence evidence: project memory `feedback_pipeline_incident_20260713.md`.

- **qlib $close is normalized (IPO day=1.0), not actual CNY.** Always divide by `$factor`: `actual_price = $close / $factor`. The `adjusted` flag in `ctx.prices` marks prices that have been divided. This is the #1 rule — violating it causes 30x NAV inflation.
- **Three data quality gates before settle are mandatory.** (1) Data completeness: halt if qlib calendar missing expected trading days. (2) Price sanity: halt if >10% prices look normalized (`factor < 0.1` without `adjusted` flag). (3) Report quality: skip WeChat report if NAV changed >20% or normalized prices detected. All three implemented in `pipeline.py`.
- **Never blame a data source without proof.** Subagent investigation proved chenditc was correct — the bug was in pipeline code. Run the actual query before accusing.
- **SSH environment != interactive environment.** Windows SSH sessions only have `py.exe`, not `python`. Verify commands in the actual execution context.
- **No report > wrong report.** Skip WeChat report when data quality is suspect. The user received "+3647% return" because the pipeline sent reports with garbage data.
- **`_is_normalized_price()` is the single source of truth.** Checks `adjusted` flag first (if True, price is real). Otherwise checks `factor < 0.1`. Used in both `_gate_price_sanity` and `_step13_report`.


## End-to-End Test Gate (2026-07-17 P0 incident)

Incidence evidence: `bump_suspension_carry_days` was added to ledger.py and called in pipeline.py:1091, but pipeline.py never imported it. All unit tests passed (mocks masked the missing import). Pipeline crashed on first real run with `NameError`.

- **Unit tests passing ≠ pipeline works.** Mocks hide missing imports, dead code, and schema mismatches. An import that is mocked in tests but missing in production code will never fail in tests.
- **End-to-end test is mandatory for logic-bearing changes.** After any change to pipeline.py/engine.py/risk.py/ledger.py imports or function signatures, run `python3 -m ashare_lab.cli paper run-all` with real data on X500. This is a HARD GATE -- do not skip even for "obvious" one-line fixes.
- **Import smoke test is NOT sufficient.** `python3 -c "from ashare_lab.paper.pipeline import run_daily"` only verifies the import chain resolves. It does NOT verify the function bodies execute without NameError on missing imports that are called deeper in the code path.
- **New function checklist**: when adding a function to any module, verify: (1) it is imported in every file that calls it, (2) it is mocked in every test file that exercises the calling code path, (3) it is actually called (not dead code).
- **Forge review is required for all logic-bearing changes.** 3 rounds minimum, fix all confirmed findings before committing. This applies even to "trivial" wiring changes like adding a function call to an aggregator.

## GPU Script Testing Safety (2026-07-22)

Incident: subagent testing `gpu-switch.bat ollama` killed llama-server, breaking trinity-router's inference session.

- **Never test `ollama` or `training-3080` targets during automated runs.** These targets stop llama-server, which is shared with trinity-router on GPU 0. Only test `training` (stops OllamaService only) and `llama-server` (restores after training).
- **Safe test sequence**: `gpu-switch training` → verify → `gpu-switch llama-server` → verify. Never call `gpu-switch ollama` from a subagent or automated script.
- **OllamaService is manual-start only.** It has `CUDA_VISIBLE_DEVICES=0` pinned. Starting it while llama-server is running will conflict. If OllamaService needs to run, stop llama-server first — but this is a human decision, not an automated one.
- **Include this rule in all subagent prompts that SSH to win-gpu.** Add: "只测试 training 和 llama-server 目标，禁止测试 ollama 和 training-3080"


## Commit Message Hygiene (2026-07-19)

Never use severity labels (P0/P1/P2), review vocabulary (Layer N, finding #N),
or plan/task IDs in commit messages. Git readers cannot see our review context.
Use direct impact descriptions instead:
  BAD:  "P0: fix NAV gate crash alert suppression"
  GOOD: "fix NAV gate suppressing crash alerts on real market crashes"
  BAD:  "add Layer 2/3 tests for hedge volume"
  GOOD: "add isfinite guard and prefetch ordering tests for hedge volume"


## Vendor Docs Before Vendor APIs (2026-08-09 QMT probe saga)

Incident: an undocumented 8-arg passorder call shape (signal mode, next-bar
conversion) produced total order silence and was misdiagnosed as a
regulatory permission wall; days of probing before the official manual
(quickTrade semantics) overturned it.

- **When unsure about any vendor API (QMT/XTQuant, broker endpoints,
  PTrade), read the authoritative docs FIRST; never trust community
  snippets or model memory for call signatures.** Sources in order:
  the vendor knowledge base (dict.thinktrader.net/innerApi), the
  client-bundled API sources (<install>/bin.x64/Lib/site-packages/xtquant),
  and broker documents under docs/.
- Broker/compliance documents from the account manager are filed in
  docs/ (e.g. the 2025-06-03 programmatic-trading commitment letter
  template).
