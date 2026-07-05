# GPU Training Optimization Design Doc

Date: 2026-07-05
Status: IMPLEMENTED (AMP + pin_memory + batch_size passthrough)

## Critical Finding: bf16 .numpy() crash in TRA assign_data

TRAModel.train_epoch calls `dataset.assign_data(index, L)` every batch.
MTSDatasetH.assign_data does `vals.detach().cpu().numpy()`. Under
autocast(bf16), tensor L is bf16 -- PyTorch .numpy() has no bf16 dtype,
raises TypeError. Fix applied: patch assign_data to cast .float() first.
ALSTM is clean (uses DataLoader, no assign_data path).

## Ground Truth (from qlib source code analysis)

### MTSDatasetH (qlib/contrib/data/dataset.py)

- NOT a torch.utils.data.Dataset -- custom Python iterable, yields dicts directly
- No DataLoader wrapper -- no num_workers, no pin_memory, no prefetching
- batch_size=-1 (current): daily mode, one trading day = all stocks per batch
- batch_size>0: sample mode, N individual time-series slices per batch
- memory_mode="sample": per-sample memory array, shape (n_samples, num_states)
- assign_data(index, L): writes routing state back to dataset each step = CPU-GPU sync point

### TRAModel (qlib/contrib/model/pytorch_tra.py)

- fit() iterates MTSDatasetH directly: `for batch in tqdm(data_set)`
- Zero AMP/FP16: all float32
- Zero DataLoader parallelism: single-threaded data loading
- Zero torch.compile: no JIT compilation
- Bottleneck = data transfer + CPU-GPU sync, not GPU compute

## Optimization 1: AMP Mixed Precision

Wrap model.fit() in torch.amp.autocast(device_type="cuda", dtype=bfloat16).
bf16 chosen over fp16: same exponent range as fp32, no GradScaler needed.
Do NOT modify qlib source. Wrapper at train.py call site.

### Gray Areas

| # | Question | Risk | Resolution |
|---|----------|------|------------|
| G1 | Sinkhorn iterations in bf16 stable? | Numerical divergence | Run 1 window, compare IC to fp32 baseline |
| G2 | bf16 + TRA dual-optimizer (pretrain+main)? | Gradient corruption | Check loss curve shape |
| G3 | Windows WDDM supports AMP? | Silent fp32 fallback | torch.cuda.get_device_capability() |

Expected speedup: 20-40% on forward/backward. ~15 lines wrapper. No qlib mod.

## Optimization 2: Tensor Pin Memory

After MTSDatasetH yields CPU tensors, pin them before .to(device).
Applied via dynamic subclass patching on __iter__.

### Gray Areas

| # | Question | Risk | Resolution |
|---|----------|------|------------|
| G4 | pin_memory on Windows WDDM? | May no-op | Test on small tensor |
| G5 | assign_data writes back to CPU after pinning? | Memory conflict | Test with verify |

Expected speedup: 10-20%. ~10 lines wrapper. Low risk.

## Optimization 3: batch_size Tuning (MODEL CHANGE, not optimization)

### Verdict

transport_daily and transport_sample have different Sinkhorn normalization
scopes. Changing batch_size from negative to positive is a MODEL CHANGE,
not a speedup. Sample-mode TRA produces a different model with different
routing behavior.

Exposed as config passthrough (batch_size in yaml). New candidate
matrix_f_tra360_sample.yaml treats sample-mode TRA as a separate experiment.

## Priority (revised after gray-area scan)

| Priority | Optimization | Speedup | Risk | Action |
|----------|-------------|---------|------|--------|
| 1 | AMP (bf16) | 20-40% | Medium (G1-G3 testable) | Implemented |
| 2 | Pin memory | 10-20% | Low (G4-G5 trivial) | Implemented |
| 3 | batch_size | Unknown | HIGH (G6-G7 = model change) | Separate candidate |

## Verification Protocol

Before deploying to Phase 1 batch:
1. Run W1 with optimization on candidate C (TRA, the target)
2. Compare IC, excess, maxdd to Phase 0 baseline
3. Tolerance: IC delta < 0.002, excess delta < 1pp
4. Outside tolerance = diagnose, do not deploy
