# ashare-lab

Global rules in ~/CLAUDE.md apply. This file adds project-specific rules only -- no state, no architecture descriptions (those live in project memory).


## Review Gate

Every code change -- source, test, config with behavioral impact -- passes this pipeline before commit:

1. Forge three-cycle review (9 passes minimum, zero findings exit)
2. /anti-ai-audit on all changed files (code comments, docstrings, commit message)
3. Non-ASCII check on every changed file

A commit that skips step 2 is incomplete even if forge passes. AI-smell in comments, docstrings, and log messages is as real as a bug -- future readers cannot tell intent from template.


## Plan Review Pipeline

Every phase plan set (PLAN.md files) must pass this pipeline before execution. Exit criteria: 0B/0H/0M/0L across ALL stages (internal + external).

1. **Internal adversarial review**: gsd-plan-checker until 0B/0H/0M/0L
2. **External 4-model cross-review**: aicc dispatch to gm (Gemini), ds (DeepSeek), mm (MiniMax), mimo (MiMo) in parallel. Usage in global memory `reference_aicc_tool.md`. Model profiles in `reference_aicc_model_review_profiles.md`.
3. **Triage external results**:
   - If ALL 4 models return 0B/0H/0M/0L → present plans to user for human review → EXIT
   - If ANY findings → consolidate cross-model results, fix all B+H+M+L → step 4
4. **Internal adversarial re-review**: gsd-plan-checker on fixed plans, verify fixes did not introduce new issues, must pass 0B/0H/0M/0L
5. **Loop**: return to step 2 (external re-review on fixed plans). Repeat until step 3 reaches the clean exit.

The 4-model panel covers L0 (mm surface), L0.5 (mimo signatures), L1 (ds docs), L2 (kimi integration), L3 (gm runtime). A plan set that skips external cross-review has only one model's blind spots checked.

When gm fails (VPN exit CN), proceed with 3 models. Retry gm in the next round. gm's L3 runtime findings are the highest-severity unique catches -- do not declare convergence without at least one successful gm round.


## Machine Roles

Three machines, three distinct roles:

- **Local dev machine**: write code, run forge review, unit tests. NOT for archiving -- reboots unpredictably, no persistent storage guarantee.
- **GPU (admin@192.168.100.11, Windows)**: multi-tenant training host.
    Repo and all data live on **H: drive** (`H:\ashare-lab\`, `H:\.qlib\`).
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
