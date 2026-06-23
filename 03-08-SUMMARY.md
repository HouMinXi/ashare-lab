---
phase: 03-paper-engine
plan: 08
subsystem: research/predict
tags: [prediction, producer, tra-inference, blend]
dependency_graph:
  requires: [blend.py, smoke_test.py, train.py, config.py]
  provides: [predict_for_date, predictions/*.parquet]
  affects: [paper-engine-signal-reader]
tech_stack:
  added: []
  patterns: [deferred-imports, sys-modules-test-stubs]
key_files:
  created:
    - ashare_lab/research/predict.py
    - tests/research/test_predict.py
  modified:
    - ashare_lab/research/blend.py
decisions:
  - "sys.modules injection for test isolation: qlib.contrib.data.dataset imports torch at module level; monkeypatch.setattr triggers the real import chain. Solution: pre-populate sys.modules with fake ModuleType stubs before any test runs, so deferred imports inside predict_for_date resolve to mocks without touching real qlib/torch files."
  - "model._writer = None set unconditionally (not guarded by hasattr) per spike findings and plan I-5b"
metrics:
  duration_seconds: 302
  completed: "2026-06-23T14:49:12Z"
  tasks_completed: 1
  tasks_total: 1
  files_created: 2
  files_modified: 1
  tests_passed: 14
---

# Phase 03 Plan 08: Prediction Producer Summary

Daily prediction producer via TRA inference + locked 60/40 blend -> parquet

## What Was Built

`ashare_lab/research/predict.py` with `predict_for_date(trade_date, model_path=None)`:

1. **Window selection**: maps trade_date to the walk-forward window whose test period contains it (backfill -> w{k}.pt), or to latest.pt for dates past the last window. Pre-first-window dates raise ValueError; missing .pt files raise FileNotFoundError.

2. **TRA inference**: deferred imports (torch, qlib, Alpha158, MTSDatasetH) inside the function body. Mirrors train.py's Alpha158 construction with two spike-confirmed shims:
   - `model._writer = None` (unconditional) -- qlib drops the TensorBoard SummaryWriter when pickling TRAModel
   - DropnaLabel omitted from learn_processors -- keeps the live date whose forward-return label is NaN

3. **60/40 blend**: reuses `blend_tra_ntra(pred, window)` from blend.py. No tra_weight override. Blend exceptions propagate (no raw-TRA fallback file written).

4. **Output**: predictions/{trade_date}.parquet (instrument str, score float) + {trade_date}.meta.json provenance sidecar. Non-finite scores dropped before write; empty-after-filter raises ValueError.

5. **blend.py docstring updated** (R-24): documents that the producer does NOT fall back to raw TRA on blend failure.

## Test Coverage

14 unit tests, all mocked (no torch/qlib runtime required):

| Test Class | Count | Coverage |
|-----------|-------|----------|
| TestPredictForDate | 5 | parquet write, columns, row count, writer shim, blend args, meta.json |
| TestWindowSelection | 5 | W1 backfill, W2 backfill, future->latest.pt, pre-first ValueError, missing .pt FileNotFoundError |
| TestMissingDate | 1 | no trade_date rows -> ValueError |
| TestNonFiniteDrop | 2 | NaN dropped + finite kept; all-NaN -> ValueError |
| TestBlendExceptionPropagates | 1 | RuntimeError propagates, no file written |

Test isolation approach: sys.modules injection of fake torch/qlib ModuleType stubs (autouse fixture). This avoids the monkeypatch.setattr import-chain problem where qlib.contrib.data.dataset imports torch at module level.

## Real-Path Validation

The 2026-06-22 GPU spike (`run_spike_tra_liveinfer.py`) already exercised the exact mechanism this module implements: torch.load + _writer shim + omit-DropnaLabel + MTSDatasetH predict on a live date. Results confirmed: live date survived (control dropped it, treatment kept it), 300/300 finite scores, scores label-independent. The unit tests mock the model; the spike is the real-path smoke test.

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Commit | Message | Files |
|--------|---------|-------|
| 20dfba6 | research/predict: add daily prediction producer | predict.py, test_predict.py, blend.py |

## Known Stubs

None.
