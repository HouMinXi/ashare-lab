# Phase 8a: Semi-Auto Bridge Design

Status: DESIGN ONLY. No execution, no deploy. Gated on Phase 6 graduation.

## 1. Architecture

X500 = brain. Windows/iQuant = dumb executor. Strategy NEVER lives inside
iQuant (ADR-0001). The bridge is a reporting + reconciliation layer, not
an execution engine.

```
X500 pipeline (18:00 CST)
  -> orders/{date}.json (new)
  -> QQ order card ~09:15 CST (evening US)
  -> user keys 1-3 orders into iQuant manually
  -> fill-entry CLI (new)
  -> reconciliation table (new)
  -> drift metric in daily report
```

## 2. orders/{date}.json Emission

Added as pipeline step 15 (after step 14 record_pipeline_run).

Schema:
```json
{
  "trade_date": "2026-07-10",
  "generated_at": "2026-07-09T18:05:00+08:00",
  "orders": [
    {
      "symbol": "SH600519",
      "name": "贵州茅台",
      "side": "buy",
      "target_qty": 100,
      "price_hint": null,
      "reason": "topk entry"
    }
  ],
  "portfolio_nav": 305000.0,
  "cash_available": 25000.0
}
```

File path: `~/code/ashare-lab/orders/{date}.json` on X500.
Retention: 90 days (older files auto-cleaned).

## 3. QQ Order Card Format

Sent through the X500 alert-bridge (POST :8377/alert) and then
`hermes send --to qqbot`, the same path ashare_lab/bridge.py already uses
for graduation and risk alerts. A separate message after the daily report,
only on days with non-empty orders. iLink / WeChat is not used.

```
[ashare-lab] 明日交易卡 2026-07-10

买 贵州茅台(600519) 100股
卖 中国平安(601318) 100股

现金 25,000 | 净值 305,000
```

Timing: generated at pipeline time (~18:00 CST = ~06:00 US ET).
User sees it in the morning (US evening) before market open (09:30 CST).

## 4. Fill-Entry CLI

```bash
paper fill 2026-07-10 SH600519 buy 100 25.50
paper fill 2026-07-10 SH601318 sell 100 48.20
```

Each call:
1. Looks up the matching pending order in `orders` table.
2. Records a trade with the real fill price.
3. Updates order status to 'filled'.
4. Computes real commission/stamp/transfer_fee from fill price and qty.

No batch mode needed (topk15/n_drop1 turnover is 1-3 orders/day).

## 5. Reconciliation Table DDL

```sql
CREATE TABLE IF NOT EXISTS reconciliation (
    trade_date   TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    paper_qty    INTEGER NOT NULL,
    paper_price  REAL,
    real_qty     INTEGER,
    real_price   REAL,
    slippage_bps REAL,
    status       TEXT NOT NULL
                 CHECK(status IN ('matched','partial','missed','extra')),
    PRIMARY KEY (trade_date, symbol, side)
);
```

Populated daily by comparing orders/{date}.json against filled trades.
`slippage_bps = (real_price - paper_price) / paper_price * 10000` for
buys (negative = favorable); reverse sign for sells.

## 6. Drift Metric Definition

Drift = |paper_nav - real_nav| / paper_nav

Drift is judged against daily turnover, not against NAV: the bridge is
healthy when the paper-versus-real difference stays under 2% of that day's
traded value for 30 consecutive trading days. Against NAV on a 50K account
2% is 1,000 CNY, which is looser than a day that trades a few thousand.
Tracked as a daily column in the reconciliation table or a separate
drift_history table.

Stop line (decided 2026-10-02, capital-governor-draft.md section 6): the
real account is flattened when losses reach 15% of invested principal, which
is 7,500 CNY on a 50K PILOT. The line is read from the real fills recorded
by `paper fill`, not from paper.db. The older rule here, halt when the paper
NAV drawdown exceeds 10%, is withdrawn: the paper book crossed 10% on
2026-07-27 and 15% on 2026-07-30, so a paper-linked line would have fired on
the real account for reasons the real account never experienced.

## 7. Skip-Day Rule

If the user does not enter fills for a trading day:
- No real trades happen (no-trade day).
- Paper portfolio continues as normal.
- Drift accounting treats the missed day as tracking noise.
- Reconciliation row: status='missed' for all orders that day.
- No penalty, no catch-up. T+1 pipeline produces fresh orders.

## 8. IPO Live Loop (per IPO design doc section 6)

### 8a. Morning Card

Added to the order card: "today's subscribable IPOs" with code, board,
issue price, EV verdict from ipo.py, ceiling lots for current market value.
Only shown on days with eligible IPOs (most days: none).

### 8b. Win Check

T+1 evening: pipeline queries eastmoney IPO win results (existing
`_fetch_ipo_calendar` path). On win: payment reminder in the daily report.

Cash-buffer rule for governor: keep >= 1 lot's cost in liquid cash
when IPO subscriptions are active. This is a governor input, not an
engine constraint.

### 8c. Sell Reminder

Listing-day card with board-specific sell rule (ipo.py already encodes:
main = first non-limit-up day, STAR/ChiNext = day 5).

### 8d. Ledger Tag (ipo_harvest)

**V1 verification result**: IPO P&L is NOT currently separable from alpha
P&L. Evidence: `trades` table has no tag/source column (ledger.py:48-59).
IPO positions enter via `_step11_ipo_processing` (pipeline.py:1056) which
directly mutates `ctx.current_positions` and `ctx.cash`.

Required migration:
1. Add `source TEXT NOT NULL DEFAULT 'alpha'` column to `orders` table.
   IPO orders get `source='ipo_harvest'`; alpha orders get `source='alpha'`.
2. Add same column to `trades` table (inherited from order).
3. Monthly report: sum P&L by source to show alpha vs harvest separately.

This is a schema migration (DEPLOY-GATED to Phase 8a execution, not now).

## 9. Broker Alternatives (see companion doc)

Criteria from project memory (reference_iquant_broker):
- QMT or PTrade API access
- Overseas-friendly (US-based user)
- Lower bar than Guosen's 500K/C4/6mo
- Competitive commission
- Stability / reputation

Research results in docs/broker-alternatives.md.

## 10. Gates

- Phase 6 graduation: gate passed 2026-08-12 (30 of 30). Signature still outstanding
- Stop line armed: 15% of invested principal, read from real fills
- Governor Phase 10 doc signed
- Bridge health: drift < 2% for 30 days before Phase 8b consideration

## 11. Flatten Delivery (open question, QQ channel)

The bridge is semi-automatic. X500 can tell the user to flatten; it cannot
sell. When the stop line trips:

1. X500 sends a QQ card: "STOP LINE HIT: flatten all positions in iQuant".
   Lists each held symbol and quantity.
2. The user sells manually in iQuant and enters each fill with `paper fill`.
3. Until every position shows filled in the reconciliation table, the
   pipeline keeps resending the card once per trading morning.

Open, needs the user's answer before signing: how long after the card is
acceptable before the flatten is done, and what happens if the card is not
acted on. Options are accept the delay, or add a second reminder at a fixed
interval. Nothing here is an automatic sell. Until this is answered, the
stop line is a reminder, not a guard, and the capital draft should not call
it automatic.
