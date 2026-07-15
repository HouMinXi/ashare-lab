# QMT Application Guide for Guojin Securities (国金证券)

**Account:** 8890836643
**APP:** 佣金宝 (Yongjinbao)
**Date:** 2026-07-15

---

## IMPORTANT UPDATE (2026-07-06)

miniQMT has been **fully suspended** as of July 6, 2026. Existing users can
still use it for now, but service will be phased out. Apply for **full QMT**
instead -- it includes a "极简模式" (mini mode) that works similarly.

---

## Part 1: Pre-Application Checklist

Before opening the app, verify you meet ALL of these:

| Requirement | Threshold | Your Status |
|---|---|---|
| Account | Guojin normal A-share account, no freeze/dormancy | Verify |
| Assets | >= 100,000 RMB (stocks + cash, system auto-verifies) | Verify |
| Risk level | C4 (积极型/Aggressive) or C5 (激进型/Aggressive+) | See below |
| Trading history | >= 6 months A-share experience (recommended, not strict) | You have since 2026-06 |
| Python skills | Not required for approval, but needed for actual use | You have this |

### Risk Assessment (风险测评) -- Getting to C4

This is the #1 reason applications get rejected. If your current level is C3
or below, re-take the test in the app BEFORE applying.

**Path in 佣金宝:** 我的 -> 风险测评 (or 业务办理 -> 风险测评)

**How to answer to reach C4 (积极型):**

| Question Category | Recommended Answer |
|---|---|
| Investment experience | 3+ years (even if less, count any fund/stock experience) |
| Annual income | Select the higher brackets (50万+) |
| Acceptable max loss | 30% or higher |
| Investment goal | Capital appreciation (资产增值), not preservation |
| Risk tolerance | Can accept significant short-term fluctuations |
| Knowledge of derivatives | "Understand" or "familiar with" options, futures |
| Investment horizon | 3+ years |
| Portfolio composition | Mostly equities / high-risk products |

If C3 is the max you can get, contact a customer manager -- they may be able
to approve at C3 for certain cases.

---

## Part 2: Step-by-Step Application in 佣金宝 APP

### Step 1: Navigate to the Application

```
佣金宝 APP -> 我的 (bottom right) -> 我的业务 -> 权限开通
-> PTrade/QMT/底仓增强权限
```

### Step 2: Select Permission Type

You will see options:
- **QMT 普通交易权限** -- SELECT THIS ONE (full QMT, includes mini mode)
- PTrade 普通交易权限 -- optional, cloud-based, drag-and-drop
- LDP 极速柜台 -- requires 100万+ assets, skip for now

**Recommendation:** Select QMT. PTrade is optional (it is simpler but less
powerful for Python-based automated trading).

### Step 3: Fill in Your Email

The system asks for a **valid email address**. This is critical -- your
download link, login credentials, and activation notice will be sent here.

- Use your primary email that you check regularly
- Double-check for typos
- This cannot be easily changed later

### Step 4: Sign Agreements

You will need to read and electronically sign:
1. 《量化交易风险揭示书》 (Quantitative Trading Risk Disclosure)
2. 《程序化交易协议》 (Programmatic Trading Agreement)
3. 《程序化交易承诺书》 (Programmatic Trading Commitment Letter)

Just read and accept all of them. These are standard regulatory requirements.

### Step 5: Submit and Wait

- System auto-verifies your assets and risk level
- Typical approval time: **minutes to 1 business day**
- Notifications via: APP push, SMS, and email
- If rejected: most common reason is risk level < C4 or assets < 100K

---

## Part 3: 程序化交易报备 (Programmatic Trading Registration)

**This is a SEPARATE step after approval.** It is a regulatory requirement
for anyone using automated/programmatic trading. You must complete this
before doing real automated trading.

### Path in 佣金宝:

```
佣金宝 APP -> 业务办理 -> 问卷调查 -> 程序化交易问卷
```

### Typical Questions and Recommended Answers:

#### Q: Which platform do you use?
**Answer:** 国金证券智能策略交易终端（QMT）

#### Q: Strategy type (策略类型)?
**Answer options typically include:**
- 日内回转 (Intraday reversal / T+0)
- 趋势跟踪 (Trend following)
- 多因子选股 (Multi-factor stock selection)
- 网格交易 (Grid trading)
- 套利策略 (Arbitrage)

**Recommended for your case:** Select "趋势跟踪" (trend following) or
"多因子选股" (multi-factor selection). These are common, well-understood
strategy types that won't trigger extra scrutiny.

If asked to describe: "Python-based quantitative strategy for A-share
market, using technical indicators for entry/exit decisions."

#### Q: Expected maximum order frequency (预期最高下单频率)?
**Answer:** Fill in a moderate number.
- Recommended: **50-200 orders per day** for a retail individual
- Do NOT claim high-frequency (per-second) -- that triggers extra
  regulatory requirements
- "中低频" (medium-low frequency) is the safe zone

#### Q: Expected maximum cancellation ratio (预期最大撤单比例)?
**Answer:** Fill in a moderate number.
- Recommended: **15%-30%**
- Too high (>50%) flags you as potentially disruptive
- Too low (<5%) is unrealistic
- 20% is a safe, realistic number

#### Q: Applicable market (适用市场)?
**Answer:** 沪深A股 (Shanghai & Shenzhen A-shares)

#### Q: Strategy logic description (策略逻辑说明)?
**Recommended text:**
```
基于Python编写的量化择时策略，使用均线、MACD等技术指标
判断买卖信号，主要交易沪深A股标的，持仓周期1-5个交易日，
设有止损止盈和仓位管理规则。
```

#### Q: Risk control measures (风控规则)?
**Recommended text:**
```
单笔交易不超过总资金5%，单日最大亏损不超过总资金2%，
设置止损线-5%，策略运行异常时自动停止交易。
```

---

## Part 4: After Approval -- Setup

### Download and Install

1. Check your email for the download link (subject: "国金证券QMT系统已开通")
2. Download the client -- **use the link from the email, NOT the official
   website** (国金 uses a custom build)
3. Install to a **non-C: drive, English-only path** (e.g., `D:\QMT`)
   - Chinese characters in the path WILL cause Python library failures
4. System requirements: Windows 10 64-bit, 4-core CPU, 8GB+ RAM
5. First launch will prompt to download Python libraries -- accept and wait

### Login

- Username: your securities account number (资金账号) = **8890836643**
- Password: your trading password (交易密码)

### miniQMT Mode

When logging in, look for "独立交易" or "极简模式" checkbox -- this switches
to miniQMT mode, which is the lightweight version suitable for calling via
Python API (xtquant library).

### Python Setup (for miniQMT / API usage)

```bash
pip install xtquant
```

Or let the QMT client download the Python libraries automatically on first
launch.

---

## Part 5: Common Pitfalls and How to Avoid Them

### Pitfall 1: Risk Level Too Low
- **Problem:** Default risk assessment is often C3, application rejected
- **Fix:** Re-take the risk assessment BEFORE applying, answer aggressively
- **Path:** 我的 -> 风险测评

### Pitfall 2: No Customer Manager Relationship
- **Problem:** Without a customer manager, the QMT option may not appear,
  or the threshold may be higher (50万 instead of 10万)
- **Fix:** Contact Guojin customer service first to establish a manager
  relationship, then apply

### Pitfall 3: Installing to Chinese Path
- **Problem:** `D:\国金QMT\` causes Python library load failures
- **Fix:** Always use pure English path: `D:\GJQMT\`

### Pitfall 4: Confusing QMT and miniQMT
- **Problem:** Applying for the wrong version
- **Fix:** Apply for "QMT" -- it includes miniQMT as a mode within it.
  miniQMT as a standalone product has been suspended (2026-07-06).

### Pitfall 5: Skipping 程序化交易报备
- **Problem:** Without completing the registration questionnaire, automated
  trading may be flagged or blocked by compliance
- **Fix:** Complete the 程序化交易问卷 after approval, before running
  any automated strategy

### Pitfall 6: Using Generic QMT Download
- **Problem:** The version on QMT's official website does not work with
  Guojin's servers
- **Fix:** ONLY use the download link from Guojin's approval email

### Pitfall 7: Wrong Email Address
- **Problem:** Typo in email = never receive download link or credentials
- **Fix:** Triple-check your email before submitting

---

## Part 6: Tips for Faster Approval

1. **Establish customer manager relationship FIRST.** Call Guojin or use
   佣金宝 online chat to get assigned a manager. Applications through a
   manager channel are faster and can negotiate lower thresholds.

2. **Ensure assets are in the account BEFORE applying.** The system checks
   in real-time. Transfer funds and wait for them to settle.

3. **Complete risk assessment to C4+ before applying.** Don't waste an
   application round on a failed risk check.

4. **Use the customer manager's dedicated link/QR code** if they provide
   one. This is faster than self-service in the app.

5. **Apply during trading hours (9:00-15:00 on weekdays).** Some
   verification steps only work during market hours.

6. **Have your account number and trading password ready.** You will need
   them immediately after approval for login.

7. **Negotiate commission rate** at the same time. QMT users can often get
   lower rates (万1 or below). Ask your manager before signing.

---

## Part 7: Your Specific Situation Notes

- **Account:** 8890836643 (already opened)
- **US-based:** Use your CN-Shanghai reverse proxy for the 佣金宝 APP
- **Trading experience:** Since 2026-06 (~1 month). The "6 months" requirement
  is recommended, not strict. If asked, you may need to wait or have the
  customer manager vouch for you.
- **Python experience:** Advantage -- mention this if asked about technical
  capability
- **Goal:** Automated trading via XTQuant API (Phase 8 broker bridge)

### Action Items:

1. [ ] Check current risk level in 佣金宝 app
2. [ ] If below C4, re-take risk assessment
3. [ ] Ensure account has >= 100K RMB
4. [ ] Contact Guojin to establish customer manager relationship
5. [ ] Apply for QMT permission via 佣金宝
6. [ ] Wait for email with download link
7. [ ] Install QMT on a Windows machine (non-C: drive, English path)
8. [ ] Complete 程序化交易问卷 (after approval)
9. [ ] Test with XTQuant API

---

## Sources

- 国金证券官方系统信息: https://www.gjzq.com.cn/main/company-overview/business-sysinfo.html
- QMT开通流程指南: https://licai.jiantou8.com/user/guide_view_3338859.html
- 2026最新开通全攻略: https://licai.cofool.com/user/guide_view_3369934.html
- 程序化交易报备填写教程: https://licai.cofool.com/user/guide_view_3418554.html
- miniQMT开通指南: https://licai.jiantou8.com/user/guide_view_3375085.html
- 各券商QMT门槛汇总: https://www.xuntou.net/forum.php?mod=viewthread&page=1&tid=232
- miniQMT.com (note: miniQMT suspended 2026-07-06): https://www.miniqmt.com/
