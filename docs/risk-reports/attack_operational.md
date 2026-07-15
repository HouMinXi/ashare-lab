# ashare-lab Operational Attack Surface Analysis

**Date**: 2026-07-15 ~23:00 CST
**Ground truth**: verified against actual X500 state (ssh houminxi@192.168.100.10)

## Verified System State

| Component | Actual State | Source |
|-----------|-------------|--------|
| Timers | Both enabled; pipeline ran 18:00, finished 18:05 SUCCESS | `systemctl --user status ashare-pipeline.service` |
| Model | latest.pt -> w10.pt (trained 2026-07-01, 14.5 days old) | `ls -la models/latest.pt` |
| Predictions | 2026-07-15.parquet exists; meta.json: model_age_days=23.9, ic=null | `cat predictions/2026-07-15.meta.json` |
| Working DB | `/code/ashare-lab/paper.db` = 192KB, 2026-07-15 settled, NAV=280,971 | `sqlite3 paper.db` |
| Dead DB | `/code/ashare-lab/data/paper.db` = 0 bytes (since 2026-07-10) | `ls -la data/paper.db` |
| Data stamp | `2026-07-15T17:45:14+08:00` (15 min before pipeline) | `cat ~/.cache/ashare-data-update.stamp` |
| IC history | File missing (`data/ic_history.tsv` not found) | `ls` |
| Retrain sentinel | Missing (never retrained) | `ls data/last_retrain_date` |
| GPU lock | Not present | `ls /tmp/ashare-gpu.lock` |
| Pipeline protection | No flock, no `Before=`, no `Conflicts=` in service unit | `systemctl --user cat ashare-pipeline.service` |

---

## Attack 1: GPU Down at 18:00 (WOL Fails)

**Mechanism**: GPU machine (192.168.100.11) unreachable. WOL packet sent but machine does not wake.

**Current handling** [VERIFIED from code]:
1. `ashare-pipeline.sh:188` -- first attempt, WOL + ping poll 200s + SSH poll 300s
2. `ashare-pipeline.sh:192-197` -- if fail, sleep 1800s (30 min), retry
3. `ashare-pipeline.sh:200-212` -- if both fail, `USE_STALE=1`, find latest `.parquet` in predictions/
4. `ashare-pipeline.sh:211` -- alert sent via `alert.py` (iLink WeChat)
5. `pipeline.py:1502-1503` -- report skipped when `ASHARE_USE_STALE=1`
6. `pipeline.py:1544` -- pipeline_run recorded as status="stale"

**Failure mode**: Pipeline runs to completion on stale predictions. No freshness check on the stale file's age.

**Detection delay**: Alert sent at ~19:05 (30 min retry + GPU timeout). User sees WeChat alert.

**Gap -- no staleness threshold**: [VERIFIED] `ashare-pipeline.sh:202` uses `find | sort | tail -1` which picks the most recent parquet regardless of age. If the last GPU success was 90 days ago, the pipeline trades on 90-day-old predictions with zero additional warning beyond the initial "gpu_inference" alert. The alert message (`alert.py`) does not include the stale file's date.

**Severity**: MEDIUM. Detection is fast (alert fires), but the system proceeds to trade on stale signals without a staleness cutoff. Over time, as model age grows, prediction quality degrades silently.

**Recommendation**: Add a staleness threshold (e.g., 7 days) in `ashare-pipeline.sh` after line 202. If the stale file is older than threshold, exit with alert rather than trade on ancient predictions.

---

## Attack 2: Data Stale (qlib Data Not Updated)

**Mechanism**: chenditc/qlib data feed fails to update overnight. Pipeline runs on yesterday's (or older) data.

**Current handling** [VERIFIED from code]:
1. `ashare-pipeline.sh:47-54` -- data stamp check; warns if >7200s old, but **continues anyway**
2. `pipeline.py:534-537` -- if qlib calendar already covers trade_date, skip refresh
3. `pipeline.py:539-552` -- `daily_refresh()` returns stale=1 -> records "skipped_stale", returns 1
4. `pipeline.py:556-576` -- `_gate_data_completeness()` checks for missing trading days between last run and today

**Failure mode**: Two paths diverge:
- Path A: qlib calendar already has the trade_date (data was partially updated) -> pipeline proceeds with potentially incomplete data. `_gate_data_completeness` only checks calendar presence, not data quality.
- Path B: qlib calendar missing the trade_date -> `_step3_data_update` returns 1, pipeline exits with rc=1. No alert fires (only `pipeline_rc != 0` triggers alert at line 245; rc=1 is skipped).

**Detection delay**: Path A: 24+ hours (next day when data diverges). Path B: no alert fires; user must check logs manually.

**Gap -- rc=1 silent exit**: [VERIFIED] `ashare-pipeline.sh:238-240` handles rc=0 with stale=1 but rc=1 (data stale) falls through to line 244 which only fires alert when `PIPELINE_RC -ne 0`. rc=1 IS non-zero, so the alert DOES fire. Correction: alert fires for rc=1 at line 245. However, the alert message is generic ("pipeline" failure), not specifically "data stale."

**Gap -- data stamp is advisory only**: [VERIFIED] `ashare-pipeline.sh:49` warns but proceeds. The stamp is not a gate.

**Severity**: LOW-MEDIUM. The completeness gate catches the worst case (calendar missing). Partial data updates are the real risk but harder to exploit.

---

## Attack 3: Database Corruption (paper.db)

**Mechanism**: paper.db becomes corrupted (0 bytes, truncated, or SQLite integrity error).

**Current handling** [VERIFIED from code]:
- **No integrity check anywhere.** `pipeline.py:491` calls `get_connection(db_path)` then `init_schema(conn)`. For a 0-byte file, SQLite creates a fresh empty database with the schema. No data loss detection.
- `pipeline.py:1494` -- `hot_backup()` overwrites the daily backup AFTER the pipeline runs.
- Backups exist in `backups/` directory, last 5 days retained.

**Actual state** [VERIFIED]: Two DB files exist:
- `/code/ashare-lab/paper.db` (192KB, working, 2026-07-15 settled) -- this is the one the pipeline uses
- `/code/ashare-lab/data/paper.db` (0 bytes since 2026-07-10) -- dead file, config says `db_path: paper.db` which resolves relative to PROJECT_ROOT

**Failure mode if working DB corrupts**:
1. Pipeline opens 0-byte file -> `init_schema` creates empty tables
2. `_step2_idempotency`: `is_day_settled` returns False (empty runs table) -> proceeds
3. `_step4_load_state`: `get_latest_positions` returns {} (empty), `get_latest_cash` returns initial_cash (300K)
4. Pipeline generates signals as if starting fresh with full capital and no positions
5. `_step12_backup_and_finalize`: `hot_backup` overwrites today's backup with the empty DB
6. All historical NAV, positions, trades, orders -- gone. Backups from previous days survive but today's is destroyed.

**Detection delay**: Next pipeline run shows NAV=300,000 (initial) instead of ~281,000. Report quality gate (`pipeline.py:1509-1514`) catches >20% NAV change and skips report. But the backup is already overwritten.

**Severity**: HIGH. Silent data loss. The report gate catches the symptom but the backup is already corrupted. No pre-backup integrity check exists.

**Recommendation**: Add `PRAGMA integrity_check` before `hot_backup`. Keep at least one "golden" backup that is never overwritten (e.g., weekly archive).

---

## Attack 4: Network Partition (X500 Loses Internet)

**Mechanism**: X500 loses internet connectivity during pipeline execution (after step 1, before or during step 5).

**Current handling** [VERIFIED from code]:
- `pipeline.py:370-405` -- baostock subprocess: 30s timeout, returns zeros on failure
- `pipeline.py:694-728` -- qlib D.features subprocess: 90s timeout, empty dict on failure
- `pipeline.py:846-865` -- industry subprocess: 180s timeout, partial data on failure
- `pipeline.py:297-318` -- eastmoney IPO: 10s timeout, returns [] on failure

**Failure mode**:
- Prices from qlib are LOCAL (reads from `~/.qlib/qlib_data/cn_data`) -- unaffected by network.
- Baostock benchmarks: return 0.0 for both csi300 and csi1000. Report shows 0% benchmark return.
- Industry map: partial or empty. Concentration risk check disabled.
- IPO calendar: empty. IPO subscriptions silently skipped.
- Report delivery: hermes-gateway is localhost:8642 -- works if local. iLink is external -- fails.

**Detection delay**: Report delivery fails -> user notices missing WeChat message next morning. If hermes-gateway works locally, report is sent but with degraded data (0 benchmarks, missing industry).

**Gap -- no benchmark staleness detection**: [VERIFIED] `_fetch_benchmark_closes` returns `{"csi300": 0.0, "csi1000": 0.0}` on failure. The pipeline and report treat 0.0 as a valid benchmark value. No warning logged for zero benchmarks.

**Severity**: LOW-MEDIUM. Core pipeline (settle, signals) works on local qlib data. Report quality degrades but does not produce dangerous trades. IPO subscriptions are silently lost (no retry mechanism).

---

## Attack 5: Double Execution (Timer Fires Twice)

**Mechanism**: systemd timer fires twice (clock glitch, `Persistent=true` catchup + manual start, or systemd bug).

**Current handling** [VERIFIED from code]:
- **No flock in ashare-pipeline.sh.** The GPU lock (`/tmp/ashare-gpu.lock`) only guards GPU inference, not the pipeline itself.
- **No `Conflicts=` or `Before=` in the service unit.** Nothing prevents two instances.
- `pipeline.py:503-504` -- `is_day_settled()` check. If first pipeline already settled, second exits rc=0.
- SQLite write lock prevents simultaneous writes but not logical races.

**Failure mode -- race window**:
1. Pipeline A starts at 18:00, enters `_step8_settle` (takes ~60s)
2. Pipeline B starts at 18:00:01, passes `_step2_idempotency` (day not yet settled)
3. Both run `_step5_fetch_prices_and_universe` simultaneously (read-only, safe)
4. Both enter `_step8_settle` -- SQLite write lock serializes them
5. Pipeline A settles, commits. Pipeline B settles again on the same data.
6. Result: duplicate trades recorded, positions double-counted, NAV inflated.

**Actual mitigation**: [VERIFIED] The `is_day_settled` check at `pipeline.py:503` is the ONLY guard. If the first pipeline settles before the second reaches step 2, the second exits cleanly. The race window is the time between step 2 and step 8 (~2-3 minutes of data fetching).

**Detection delay**: Next day when NAV shows anomalous values. Report gate (>20% NAV change) might catch it.

**Severity**: MEDIUM. The race window is real but narrow (2-3 min). SQLite serialization prevents DB corruption but not logical duplication.

**Recommendation**: Add `flock -n /tmp/ashare-pipeline.lock` at the top of `ashare-pipeline.sh`, or add `Conflicts=ashare-pipeline.service` to the timer unit (prevents systemd from starting a second instance).

---

## Attack 6: meta.json Missing (Stale Detection Broken)

**Mechanism**: meta.json file is deleted or not produced alongside the prediction parquet.

**Current handling** [VERIFIED from code]:
- `pipeline.py:1241-1279` -- meta.json is read for provenance logging, model_age warning (>7 days), and IC=nan detection. All are informational.
- `pipeline.py:1273-1279` -- if meta.json missing: `logger.debug("No meta.json found...")`. Pipeline continues.
- `pipeline.py:1723-1785` -- `append_ic_history` reads meta.json for IC tracking. If missing, silently skips.
- `ashare-pipeline.sh:182` -- SCP meta.json is `|| true` (non-fatal).

**Failure mode**: Pipeline runs to completion with no provenance logging. Model age warning suppressed. IC history not updated (retraining gate disabled). Report shows no model staleness annotation.

**Actual state** [VERIFIED]: Today's meta.json exists: `{"model": "latest.pt", "model_age_days": 23.9, "ic": null}`. The IC is already null (TRA mathematical limitation), so the IC history file (`data/ic_history.tsv`) is missing anyway -- the retraining gate has never fired.

**Severity**: LOW. meta.json is purely observational. No trading decisions depend on it. The real staleness risk is already present (model is 14.5 days old, IC always null) regardless of meta.json presence.

**Gap -- retraining gate is dead**: [VERIFIED] `ashare-retrain.sh` reads `data/ic_history.tsv` which does not exist. The retrain script exits at line 43 (`No IC history file`). Even if IC history existed, IC is always null (TRA limitation), which `ashare-retrain.sh:51-54` counts as "low" -- so the retrain gate would trigger immediately on 3 consecutive entries. But since the file never gets created, the entire event-driven retraining mechanism is non-functional.

---

## Summary Matrix

| # | Scenario | Handled? | Detection Delay | Severity | Key Gap |
|---|----------|----------|-----------------|----------|---------|
| 1 | GPU down at 18:00 | Partial | ~90 min (alert) | MEDIUM | No staleness threshold on stale predictions |
| 2 | Data stale | Partial | Next day or alert | LOW-MED | Data stamp is advisory; rc=1 alert is generic |
| 3 | DB corruption | NO | Next pipeline run | HIGH | No integrity check; backup overwritten blindly |
| 4 | Network partition | Partial | Report missing | LOW-MED | Zero benchmarks treated as valid; IPO silently lost |
| 5 | Double execution | Weak | Next day | MEDIUM | No flock; race window 2-3 min before settle |
| 6 | meta.json missing | N/A | N/A | LOW | Observational only; retraining gate already dead |

## Top 3 Recommendations

1. **DB integrity gate** (Attack 3): Run `PRAGMA integrity_check` before `hot_backup`. Keep one immutable golden backup per week. This is the only scenario with silent, unrecoverable data loss.

2. **Pipeline mutex** (Attack 5): Add `flock -n /tmp/ashare-pipeline.lock` at the top of `ashare-pipeline.sh`. One line, eliminates the race entirely.

3. **Staleness threshold** (Attack 1): After selecting the stale prediction file, check its date against the trade date. If older than N days (e.g., 7), exit with a specific alert rather than trade on ancient predictions.
