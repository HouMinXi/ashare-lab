# Hedge Sleeve Data + Rules Verification Memo

Phase 2.4 Plan 01 deliverable. Research-only; no code, no backtest.

## 1. Candidate Instruments

### Primary: CASH (zero PFIC, zero paperwork)

The PFIC constraint (MASTER section 1b LEAK-3) makes CASH the default
de-risk instrument for a US tax resident. "Cash" in this context means:

| Instrument | T+0/T+1 | Fees | PFIC? | Notes |
|-----------|---------|------|-------|-------|
| Bank deposit (settlement account) | T+0 | 0 | No | Default. ~1.5% annualized demand deposit rate. Zero friction. |
| GC001 (204001, 1-day reverse repo) | T+0 settle, T+1 available | ~0.001%/txn | No | Not a fund; not PFIC. ~1.8-2.5% annualized (2024-2025). Must re-roll daily. |
| GC007 (204007, 7-day reverse repo) | T+0 settle, T+7 available | ~0.005%/txn | No | ~2.0-2.5%. Less friction than GC001 but locks capital. |

Decision: bank deposit is simplest. Reverse repo adds ~0.5%/yr over deposit
but requires daily rolling (GC001) or locks capital (GC007). For a sleeve
that activates infrequently (only in de-risk/halt states), bank deposit wins
on simplicity. Reverse repo is an optimization for later.

### Evaluated but deprioritized: CN-domiciled ETFs

| Instrument | Code | T+0/T+1 | Annual fee | PFIC? | After-tax edge vs cash |
|-----------|------|---------|------------|-------|----------------------|
| Bond ETF (10yr treasury) | 511010 | T+0 | ~0.15% | YES | ~1-2% gross, minus PFIC accounting cost. Marginal at sleeve scale. |
| Gold ETF | 518880 | T+0 | ~0.50% | YES | Correlated to global gold; ~8%/yr 2023-2025 bull. PFIC Form 8621 per fund. |
| Broad index ETF (CSI300) | 510300 | T+0 | ~0.50% | YES | Adds equity beta, opposite of de-risk intent. Rejected. |

PFIC accounting cost: Form 8621 per fund per year, even for a $500
position. At current sleeve scale (<50K CNY), the paperwork cost
exceeds the edge. The accountant's mark-to-market election (MASTER 1b)
can simplify but does not eliminate filing. **Cash wins by default.**

A CN ETF enters the sleeve only if its after-tax edge survives PFIC
accounting -- that is the accountant's call, not an engineering decision.

### Gold exposure (if wanted): US-side

Gold lives in a US brokerage account (GLD, IAU) if the user wants it.
Not PFIC. Not part of the A-share sleeve. Out of scope for this memo.

## 2. Data Availability (akshare / qlib pipeline)

| Data | Source | Verified | Notes |
|------|--------|----------|-------|
| ETF daily OHLCV | `ak.fund_etf_hist_em(symbol, period='daily')` | YES (518880, 511010 both returned 5 rows, 2021-01-04 to 2021-01-08, with columns: date/open/close/high/low/volume/amount/amplitude/change_pct/change/turnover) | Sufficient for correlation analysis. |
| CSI1000 daily | Already in qlib pipeline (`$close` on `SH000852`) | YES (used in all 10 walk-forward windows) | Benchmark for sleeve vs no-sleeve comparison. |
| Reverse repo rates | `ak.bond_repo_zh_tick()` or similar | NOT VERIFIED (Z66 network down during verification) | Needed only if reverse repo enters the sleeve. Deferred. |

**Data gap for correlation analysis**: ETF data comes from akshare (eastmoney),
not from the qlib binary store. The correlation computation must align dates
between the qlib CSI1000 returns and akshare ETF returns. Use trading calendar
intersection.

## 3. Trading Rules (VERIFIED facts marked; unverified marked COMMON)

- A-share ETFs trade T+0 (buy and sell same day) [COMMON, stable rule].
  This is unlike stocks (T+1). The sleeve can rebalance intraday.
- Minimum lot: 100 shares for all ETFs [COMMON].
- Commission: broker-dependent, typically 0.01-0.03% per trade [COMMON].
  No stamp tax on ETFs [COMMON].
- Settlement: T+1 for cash settlement (same as stocks) [COMMON].
- Reverse repo: GC001 settlement T+0 (usable next day), GC007 T+7 [COMMON].
  Minimum 100K CNY for GC001 on Shanghai (10K on Shenzhen) [VERIFY current].
- Cash deposit rate: ~1.5% demand, ~2-3% 7-day notice [COMMON, varies by bank].

## 4. Cash-Sleeve Baseline Definition

The "cash sleeve" is: a fixed fraction of total portfolio held as **bank
deposit** (the settlement account balance). No instrument to buy/sell;
it is simply NOT deploying that fraction into equity.

Mechanism in the existing engine:
- `initial_cash` in baseline.yaml = 300000 (fully deployed by topk logic)
- A cash sleeve of X% means topk deploys only (100-X)% of NAV into equities
- Implementation: adjust `capital_for_equity = nav * (1 - sleeve_pct)`
  in the engine, or equivalently reduce topk lots by the sleeve fraction.

This requires zero new instruments, zero new data feeds, zero PFIC paperwork.

## 5. What Remains for Plan 02

- Correlation table: CSI1000 daily returns vs a synthetic "cash" return
  (constant ~1.5%/252 per day) across all 10 optionA windows. This is
  trivially -1.0 corr with equity (cash always positive, equity volatile)
  but the point is to measure the BLEND's behavior: what does 90/10 or
  85/15 equity/cash do to per-window excess and drawdown?
- Sleeve backtest DESIGN with epsilon pre-registration draft.
- Both delivered in Plan 02 doc.

## Sources

- PFIC rules: IRC Section 1291-1298, IRS Form 8621 instructions [COMMON]
- ETF trading rules: SSE/SZSE trading rules chapter on ETFs [COMMON]
- akshare verification: `fund_etf_hist_em` API call 2026-07-05 (5 rows confirmed)
- Reverse repo mechanics: SSE GC001/GC007 product specifications [COMMON]
- MASTER architecture section 1b LEAK-3 (PFIC constraint)
