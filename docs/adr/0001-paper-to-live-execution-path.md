# ADR-0001: Paper-to-live execution path

**Date**: 2026-06-28
**Status**: accepted
**Deciders**: Minxi Hou

## Context

ashare-lab has a complete paper-trading pipeline (Phases 1-5): data ingestion,
model research (60/40 TRA/nTRA blend on CSI1000), fill simulation with A-share
rules, WeChat daily reports, and LLM sentiment veto. But the system cannot
make real money because three capabilities are missing: automated daily
predictions, unattended deployment, and broker order execution.

The user trades on 国信证券, which only runs on Windows. The compute
infrastructure is split: X500 (Linux, 7x24, runs the pipeline brain) and a
GPU Windows box (RTX 3080 20GB, runs model training/inference). There is no
Linux-native broker API for 国信.

The current roadmap (Phase 6: Deploy + 30-day observation) addresses unattended
operation but stops at paper trading. This ADR extends the roadmap to cover the
full path from "validated paper" to "live execution generating real P&L."

## Decision

We adopt a three-phase path from paper to live, each gated by the previous:

### Phase 6: Deploy + 30-day Paper Observation (existing roadmap, no change)

Unattended daily pipeline on X500 via systemd timer. Graduation gate:
>=30 trading days, >=95% pipeline success, zero manual edits. This proves
the system runs reliably before any automation of predictions or orders.

### Phase 7: Automated Prediction Production (NEW -- roadmap addition)

Daily model inference on the GPU Windows box, triggered by Windows Task
Scheduler after market close. Produces `predictions/{date}.parquet` and
pushes to X500 via SCP/rsync. X500 pipeline reads the parquet as-is
(Phase 3 already supports this).

Scope:
- Windows batch script or Python wrapper calling `research/predict.py`
- Scheduled task: 15:30 CST (30 min after close, data settled)
- Push to X500: `scp predictions/{date}.parquet x500:code/ashare-lab/predictions/`
- Health check: if prediction file missing by 16:00, alert via WeChat
- Model staleness gate: retrain every 6 months per the pre-registered
  K=3 live discriminator (decision_phase21_model_lock in project memory)

### Phase 8: Broker Bridge -- X500 to QMT (NEW -- roadmap addition)

Split-brain architecture: strategy on Linux (X500), execution on Windows
(GPU box + QMT). X500 generates a daily order list (JSON), Windows agent
reads it and submits orders via QMT Python API at market open.

Scope:
- X500 writes `orders/{date}.json` (side, symbol, qty, price_limit)
- Windows QMT agent polls for new order files, submits at 09:30 open
- Execution report written back: `fills/{date}.json`
- X500 reconciles paper fills vs real fills in a new `reconciliation` table
- Kill switch: if paper NAV drawdown > 10% from peak, halt all real orders
- Regulatory: QMT requires sufficient account balance for programmatic
  trading (verify current threshold before implementation)

## Alternatives Considered

### Alternative A: Manual follow (WeChat signals, human places orders)
- **Pros**: Zero engineering cost, works today
- **Cons**: Human latency (miss opening auction), emotion overrides signals,
  defeats the "unattended systematic" core value
- **Why not**: The whole point of the project is to remove human discretion
  from execution. Manual follow is the current state, not the target state.

### Alternative B: Full Windows migration (move everything to GPU box)
- **Pros**: Single machine, no split-brain, direct QMT access
- **Cons**: Windows is not a reliable 7x24 server (updates, reboots),
  loses X500's Linux stability, GPU box is multi-tenant (harness forge-bot)
- **Why not**: Reliability regression. X500 was chosen specifically for
  unattended operation.

### Alternative C: Cloud Windows VPS (rent a Windows VM in China)
- **Pros**: 7x24 Windows, dedicated, no local hardware dependency
- **Cons**: Monthly cost ($5-15), network latency to US for management,
  regulatory gray area (remote programmatic trading), another machine to
  maintain
- **Why not**: Adds cost and complexity when the GPU box already has Windows
  and a local network connection to X500. Revisit only if the GPU box proves
  unreliable for the execution role.

### Alternative D: Skip Phase 6, go straight to live
- **Pros**: Faster to "making money"
- **Cons**: Automates an unproven system, model BORDER_PASS (60% vs 50%
  threshold) with p=0.661 is not statistically significant, no evidence the
  sentiment veto adds value in practice
- **Why not**: The graduation gate exists for a reason. Deploying unproven
  alpha into real money is the canonical way retail quants lose money.

## Consequences

### Positive
- Clear validation gate before real money (Phase 6 graduation)
- Split-brain leverages each machine's strength (Linux reliability + Windows broker)
- Incremental: each phase can be paused if results disappoint
- Retraining cadence (6-month) with pre-registered revert criteria (K=3)
  prevents model staleness without over-fitting to noise

### Negative
- Split-brain adds operational complexity (two machines must coordinate)
- GPU box becomes a critical path for both predictions AND order execution
- QMT API is not open-source; documentation quality varies
- 6-month retraining requires GPU availability (shared with forge-bot)

### Risks
- **Model alpha decays before Phase 8 ships**: Mitigate by monitoring paper
  performance in Phase 6. If alpha is zero after 30 days, stop before Phase 7.
- **QMT API changes or broker revokes access**: Mitigate by keeping manual
  follow as permanent fallback (WeChat report already provides the signals).
- **GPU box offline during market hours**: Mitigate by pre-generating
  predictions at 15:30 (previous day's close). Orders are ready before
  market open. GPU only needs to be online for 30 min/day.
- **Reconciliation drift (paper vs real fills differ)**: Expected -- real
  fills have slippage, partial fills, and timing differences. The
  reconciliation table in Phase 8 tracks this systematically. Alert if
  drift exceeds 2% of daily turnover.

## Roadmap Impact

Current ROADMAP.md needs two new phases appended:

```
Phase 7: Prediction Automation
  Goal: Daily model inference on GPU box, push to X500
  Depends on: Phase 6 graduation gate PASS
  Requirements: PRED-01..03

Phase 8: Broker Bridge
  Goal: X500 order list -> QMT -> real execution
  Depends on: Phase 7 stable (>=30 days automated predictions)
  Requirements: EXEC-01..05
```

Phase 6 is unchanged. The existing "Out of Scope" line in PROJECT.md
("Live trading / real orders -- requires broker reporting + QMT, deferred")
becomes Phase 8's scope, no longer deferred.

## Immediate Next Steps

1. Complete Phase 5 closeout (push x500, update STATE/ROADMAP)
2. Execute Phase 6 as planned (deploy + 30-day observation)
3. During Phase 6 observation period: research QMT API capabilities and
   programmatic trading requirements (account balance, application process,
   API documentation)
4. After Phase 6 graduation: plan Phase 7 (predict automation)
