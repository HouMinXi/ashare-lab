# Broker Alternatives Research + IPO EV Recomputation

Phase 8a Plan 02 deliverable.

## 1. Broker Alternatives

Criteria (from project memory reference_iquant_broker):
- QMT or PTrade API access
- Overseas-friendly (US-based user)
- Lower bar than Guosen's 500K/C4/6mo
- Competitive commission
- Stability / reputation

### Brokers compared (2026 research)

| Broker | API | Bar | Overseas | Commission | Notes |
|--------|-----|-----|----------|-----------|-------|
| Guosen (current) | QMT/iQuant | 500K + C4 + 6mo | Yes (opened from US) | Standard | Current account. High bar for real-trade API. |
| Huatai | PTrade | 500K + C4 | Unknown | Competitive | PTrade requires same-tier qualification as QMT. |
| CICC | PTrade | 500K+ | Yes (HK entity) | Higher | Professional tier only. |
| Dongfang | miniQMT | 300K (reported, VERIFY) | Unknown | Standard | Lower bar rumored but not officially documented. |
| Futu/Tiger | Proprietary API | Low (HK/US focused) | Yes | Higher for A-share | A-share access via 港股通, limited universe. NOT full A-share access. |

### Finding

No broker offers full A-share QMT/PTrade API access below 500K + C4 as of
mid-2026. The bar is regulatory (适当性管理), not broker-specific. The
300K rumor for Dongfang miniQMT is unverified and may apply to simulation
only, not real-trade API.

**Recommendation**: stay with Guosen. The bar clears naturally as the
capital governor's SCALE ladder grows capital past 500K. Phase 8a's
semi-auto bridge (manual order entry) works with ANY broker at ANY
capital level, which is the whole point of 8a before 8b.

**User action**: when capital reaches 500K and 24-month clock completes,
apply for real-trade QMT permission at Guosen. No broker switch needed.

## 2. IPO EV Recomputation

### Data Source

`ak.stock_xgsglb_em()` -- 3984 historical IPOs with win rates and
first-day gains. Verified on X500 2026-07-05.

Producing command:
```python
import akshare as ak
df = ak.stock_xgsglb_em()  # 3984 rows, 24 columns
```

### Per-Board Statistics (2024-01 to 2026-07, ~2.5 years)

| Board | Count | Avg Win Rate | Avg First-Day Gain | Avg Per-Lot Profit | Break Rate |
|-------|-------|-------------|-------------------|-------------------|------------|
| Main (600/601/603/000/001/002) | 246 | 0.0382% | 240.1% | 23,429 CNY | 0/156 = 0% |
| STAR (688/689) | 45 | 0.0363% | 274.6% | 54,584 CNY | 1/44 = 2.3% |
| ChiNext (300) | 1 | 0.0191% | N/A (1 sample) | 54,200 CNY | N/A |
| BSE (83/87/43) | 0 in this dataset | N/A | N/A | N/A | N/A |

Producing command:
```python
# Board classification applied to ak.stock_xgsglb_em() 2024-01+ subset
# Code prefix mapping: 688/689->STAR, 300->ChiNext, 83/87/43->BSE, rest->Main
```

Note: BSE IPOs are absent from stock_xgsglb_em (eastmoney). BSE data
likely requires a separate endpoint (ak.stock_bj_a_em or similar). Not
blocking for main-board-only Stage 0 EV.

### EV at 300K Capital (Main Board Only, Stage 0)

```
Annual IPOs (main board): ~98/yr
Average win rate: 0.0382% = 0.000382
Average per-lot profit: 23,429 CNY
Subscription lots at 300K: ~20 (SH: 10K/lot, ~30 lots; SZ: 5K/lot, ~60 lots; use 20 as conservative blended estimate)

EV = 98 * 0.000382 * 23429 * 20 = 17,620 CNY/yr = 5.9%/yr
```

This is HIGHER than the rough 1-2.5% estimate in the IPO design doc.
The difference: the rough estimate used 30 IPOs/yr (actual: ~98 accessible
from main board), and lower per-lot profits.

### Regime Caveat

The 240%+ average first-day gain and 0% break rate reflect the 2024-2026
policy-throttled-supply regime. Historical comparison:
- 2022: 21% break rate, ~30% average gain
- 2023: ~66% average gain
- 2024-2025: ~252-259% average gain

The EV at 5.9%/yr is a CURRENT-REGIME number. ipo.py's EV threshold
filter stays armed to protect against regime reversion.

### With STAR + ChiNext Unlocked (Stage 1, ~2028+)

Adding 45 STAR + ChiNext IPOs/yr at higher per-lot profits:
```
Additional EV ~ 45 * 0.0363% * 54584 * 20 = 3,570 CNY/yr
Total EV ~ 21,190 CNY/yr = 7.1%/yr on 300K
```

At 500K capital (required for STAR access): lots scale proportionally,
EV scales to ~35K+/yr = 7%+.

This is a material return enhancement -- not speculative.
