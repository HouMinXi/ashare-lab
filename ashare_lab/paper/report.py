"""Report generation and delivery module."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import struct
import subprocess
import time
from dataclasses import dataclass, field
import sqlite3

from ashare_lab.paper.ledger import insert_report

logger = logging.getLogger(__name__)

# Section separator (Unicode box-drawing U+2501)
_SEP = "━━"

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_ID = "bot"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0
ITEM_TEXT = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2

@dataclass(frozen=True)
class ReportData:
    trade_date: str
    total_nav: float
    cash: float
    daily_return_pct: float
    daily_pnl: float
    cumulative_return_pct: float
    max_drawdown_pct: float
    cash_ratio: float
    benchmark_csi1000: float | None
    benchmark_return_pct: float
    trades: list[dict]
    positions: list[dict]
    pending_orders: list[dict]
    trade_count: int
    risk_status: dict
    industry_distribution: dict[str, int]
    hedge_active: bool = False
    hedge_dd: float = 0.0
    hedge_equity_pct: float = 1.0
    hedge_allocations: dict[str, float] = field(default_factory=dict)


def gather_report_data(
    conn: sqlite3.Connection,
    trade_date: str,
    stock_names: dict[str, str],
    industry_map: dict[str, str],
    config: dict,
) -> ReportData:
    # Query nav
    row_today = conn.execute(
        "SELECT total_nav, cash, benchmark_csi1000 FROM nav WHERE trade_date = ?", (trade_date,)
    ).fetchone()
    if not row_today:
        raise ValueError(f"No NAV data for {trade_date}")
    
    total_nav = float(row_today["total_nav"])
    cash = float(row_today["cash"])
    today_bench = row_today["benchmark_csi1000"]
    today_bench = float(today_bench) if today_bench is not None else None

    # First row for cumulative
    row_first = conn.execute(
        "SELECT total_nav FROM nav ORDER BY trade_date ASC LIMIT 1"
    ).fetchone()
    first_nav = float(row_first["total_nav"]) if row_first else total_nav
    cumulative_return_pct = (total_nav - first_nav) / first_nav * 100.0 if first_nav > 0 else 0.0

    # Previous row
    row_prev = conn.execute(
        "SELECT total_nav, benchmark_csi1000 FROM nav WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1",
        (trade_date,)
    ).fetchone()
    
    prev_nav = float(row_prev["total_nav"]) if row_prev else total_nav
    daily_return_pct = (total_nav - prev_nav) / prev_nav * 100.0 if prev_nav > 0 else 0.0
    daily_pnl = total_nav - prev_nav
    
    prev_bench = row_prev["benchmark_csi1000"] if row_prev else None
    if prev_bench is not None and today_bench is not None and float(prev_bench) > 0:
        benchmark_return_pct = (today_bench - float(prev_bench)) / float(prev_bench) * 100.0
    else:
        benchmark_return_pct = 0.0
        
    cash_ratio = cash / total_nav * 100.0 if total_nav > 0 else 100.0

    # max drawdown
    navs = conn.execute("SELECT total_nav FROM nav WHERE trade_date <= ?", (trade_date,)).fetchall()
    max_dd = 0.0
    peak = 0.0
    for r in navs:
        v = float(r["total_nav"])
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    max_drawdown_pct = max_dd

    # Trades
    trades_rows = conn.execute(
        "SELECT symbol, side, fill_qty, fill_price, commission, stamp, transfer_fee "
        "FROM trades WHERE trade_date = ?", (trade_date,)
    ).fetchall()
    trades = []
    for r in trades_rows:
        sym = r["symbol"]
        fee = float(r["commission"]) + float(r["stamp"]) + float(r["transfer_fee"])
        trades.append({
            "symbol": sym,
            "name": stock_names.get(sym, ""),
            "side": r["side"],
            "qty": int(r["fill_qty"]),
            "price": float(r["fill_price"]),
            "commission": float(r["commission"]),
            "stamp": float(r["stamp"]),
            "transfer_fee": float(r["transfer_fee"]),
            "total_fee": fee
        })
    trade_count = len(trades)

    # Positions
    pos_rows = conn.execute(
        "SELECT symbol, qty, market_value, avg_cost FROM positions WHERE trade_date = ?", (trade_date,)
    ).fetchall()
    positions = []
    ind_dist: dict[str, int] = {}
    for r in pos_rows:
        sym = r["symbol"]
        qty = int(r["qty"])
        if qty <= 0:
            continue
        mv = float(r["market_value"])
        avg = float(r["avg_cost"])
        pnl = mv - qty * avg
        w = mv / total_nav * 100.0 if total_nav > 0 else 0.0
        positions.append({
            "symbol": sym,
            "name": stock_names.get(sym, ""),
            "qty": qty,
            "market_value": mv,
            "unrealized_pnl": pnl,
            "weight": w
        })
        ind = industry_map.get(sym, "Unknown")
        ind_dist[ind] = ind_dist.get(ind, 0) + 1

    # Daily change per stock: compare per-share price vs previous trading day.
    # qlib $change is T+1 delayed (chenditc updates overnight), so we use
    # positions table which has same-day market_value from pipeline prices.
    # F2 note: market_value = close * qty. If stock is suspended, close=0
    # and market_value=0, but qty > 0. Division by zero is prevented by
    # old_mv > 0 guard. Pipeline always sets market_value = close * qty.
    # F4 note: adjustfactor handled by step6 before positions are written.
    # Per-share price comparison remains correct across factor changes:
    # step6 adjusts qty (split) and market_value (dividend) simultaneously,
    # so market_value/qty = adjusted close price for both days.
    daily_changes: dict[str, float] = {}
    daily_pnls: dict[str, float] = {}
    if positions:
        prev_rows = conn.execute(
            "SELECT symbol, market_value, qty FROM positions "
            "WHERE trade_date = (SELECT MAX(trade_date) FROM positions WHERE trade_date < ?)",
            (trade_date,),
        ).fetchall()
        prev_mv: dict[str, tuple[float, int]] = {}
        for r in prev_rows:
            prev_mv[r["symbol"]] = (float(r["market_value"]), int(r["qty"]))
        for p in positions:
            sym = p["symbol"]
            if sym in prev_mv and p["qty"] > 0:
                old_mv, old_qty = prev_mv[sym]
                if old_qty > 0 and old_mv > 0:
                    today_price = p["market_value"] / p["qty"]
                    prev_price = old_mv / old_qty
                    daily_changes[sym] = today_price / prev_price - 1  # fractional, e.g. 0.05 = 5%
                    daily_pnls[sym] = (today_price - prev_price) * p["qty"]

    for p in positions:
        p["daily_change_pct"] = daily_changes.get(p["symbol"], 0.0)
        # F3 note: initial position (no prev day) shows 0 change.
        # This is correct behavior - no reference price to compare.
        p["daily_pnl"] = daily_pnls.get(p["symbol"], 0.0)

    # Pending orders
    order_rows = conn.execute(
        "SELECT symbol, side, target_qty FROM orders "
        "WHERE status IN ('pending', 'carry') AND created_run_date = ?", (trade_date,)
    ).fetchall()
    pending_orders = []
    for r in order_rows:
        sym = r["symbol"]
        pending_orders.append({
            "symbol": sym,
            "name": stock_names.get(sym, ""),
            "side": r["side"],
            "qty": int(r["target_qty"]),
        })
        
    # Sell order count (all sells, not just forced)
    sell_order_count = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE created_run_date = ? AND side = 'sell'",
        (trade_date,)
    ).fetchone()[0]

    # Cooldown count
    cooldown_count = conn.execute("SELECT COUNT(*) FROM cooldowns").fetchone()[0]
    
    # Risk state (from state machine, not recomputed)
    row_state = conn.execute("SELECT value FROM paper_state WHERE key = 'risk_state'").fetchone()
    risk_state_str = row_state["value"] if row_state else "normal"
    row_soft = conn.execute("SELECT value FROM paper_state WHERE key = 'is_soft_reduced'").fetchone()
    is_soft_reduced = row_soft and row_soft["value"] == "true"
    row_lockdown = conn.execute("SELECT value FROM paper_state WHERE key = 'lockdown_enter_date'").fetchone()
    lockdown_enter_date = row_lockdown["value"] if row_lockdown else None

    # Shadow log (if available)
    shadow_row = conn.execute(
        "SELECT old_flags_json, shadow_state, would_do_json FROM risk_shadow_log WHERE trade_date = ?",
        (trade_date,),
    ).fetchone()
    shadow_log = None
    if shadow_row:
        import json
        shadow_log = {
            "old_flags": json.loads(shadow_row["old_flags_json"]),
            "shadow_state": shadow_row["shadow_state"],
            "would_do": json.loads(shadow_row["would_do_json"]),
        }

    # buying_halted: read from shadow old_flags (authoritative for current run)
    if shadow_log:
        old_flags = shadow_log["old_flags"]
        buying_halted = any(old_flags.get(k) for k in (
            "drawdown_halted", "daily_loss_halted", "regime_halted", "staleness_halted",
        ))
    else:
        # Fallback: recompute (legacy runs without shadow log)
        current_drawdown_pct = 0.0
        if peak > 0:
            current_drawdown_pct = (peak - total_nav) / peak * 100.0
        risk_cfg = config["paper"]["risk"]
        drawdown_halted = current_drawdown_pct > risk_cfg["drawdown_hard"] * 100.0
        daily_loss_halted = daily_return_pct <= -risk_cfg["daily_loss"] * 100.0
        regime_days = risk_cfg.get("market_regime_days", 10)
        c_rows = conn.execute(
            "SELECT benchmark_csi1000 FROM nav WHERE trade_date <= ? AND benchmark_csi1000 IS NOT NULL ORDER BY trade_date DESC LIMIT ?",
            (trade_date, regime_days + 1)
        ).fetchall()
        regime_halted = False
        if len(c_rows) == regime_days + 1:
            closes = [float(r[0]) for r in c_rows]
            closes.reverse()
            from ashare_lab.paper.risk import check_market_regime
            regime_halted = check_market_regime(closes, risk_cfg["market_regime_decline"], regime_days)
        buying_halted = drawdown_halted or daily_loss_halted or regime_halted

    # Hedge sleeve state
    hedge_row = conn.execute(
        "SELECT active, drawdown_pct, equity_target_pct, leg_json "
        "FROM hedge_state WHERE trade_date = ?",
        (trade_date,),
    ).fetchone()
    hedge_active = False
    hedge_dd = 0.0
    hedge_equity_pct = 1.0
    hedge_allocations: dict[str, float] = {}
    if hedge_row and hedge_row["active"]:
        hedge_active = True
        hedge_dd = hedge_row["drawdown_pct"]
        hedge_equity_pct = hedge_row["equity_target_pct"]
        hedge_allocations = json.loads(hedge_row["leg_json"])

    return ReportData(
        trade_date=trade_date,
        total_nav=total_nav,
        cash=cash,
        daily_return_pct=daily_return_pct,
        daily_pnl=daily_pnl,
        cumulative_return_pct=cumulative_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        cash_ratio=cash_ratio,
        benchmark_csi1000=today_bench,
        benchmark_return_pct=benchmark_return_pct,
        trades=trades,
        positions=positions,
        pending_orders=pending_orders,
        trade_count=trade_count,
        risk_status={
            "buying_halted": buying_halted,
            "is_soft_reduced": is_soft_reduced,
            "risk_state": risk_state_str,
            "lockdown_enter_date": lockdown_enter_date,
            "sell_order_count": sell_order_count,
            "cooldown_count": cooldown_count,
            "drawdown_halted": shadow_log["old_flags"]["drawdown_halted"] if shadow_log else None,
            "regime_halted": shadow_log["old_flags"]["regime_halted"] if shadow_log else None,
            "shadow_log": shadow_log,
        },
        industry_distribution=ind_dist,
        hedge_active=hedge_active,
        hedge_dd=hedge_dd,
        hedge_equity_pct=hedge_equity_pct,
        hedge_allocations=hedge_allocations,
    )


def _display_name(name: str, symbol: str) -> str:
    """Format as '中曼石油(603619)' or fallback to raw symbol."""
    if name:
        return f"{name}({symbol[2:]})"
    return symbol


# Maximum positions shown before collapsing to summary.
_TOP_N = 5


def format_chinese_report(
    report_data: ReportData,
    sentiment_section: str | None = None,
    ic_nan: bool = False,
) -> str:
    """Format report as Chinese mobile-first template."""
    rd = report_data
    lines: list[str] = []

    # Header
    lines.append(f"\U0001f4ca ashare-lab {rd.trade_date}")
    lines.append("")

    # NAV + daily P&L in yuan
    excess = rd.daily_return_pct - rd.benchmark_return_pct
    lines.append(
        f"\U0001f4b0 净值 {rd.total_nav:,.2f} "
        f"({rd.daily_return_pct:+.2f}% / {rd.daily_pnl:+,.0f}元)"
    )
    lines.append(
        f"\U0001f4c8 累计 {rd.cumulative_return_pct:+.2f}% | "
        f"回撤 {rd.max_drawdown_pct:.2f}% | "
        f"现金 {rd.cash_ratio:.1f}%"
    )
    lines.append(f"\U0001f3af 超额 vs CSI1000 {excess:+.2f}%")

    # Trades with amount
    if rd.trades:
        lines.append("")
        lines.append(f"{_SEP} 今日交易 {_SEP}")
        for t in rd.trades:
            name = _display_name(t.get("name", ""), t["symbol"])
            amount = t["qty"] * t["price"]
            side_icon = "\U0001f7e2 买" if t["side"] == "buy" else "\U0001f534 卖"
            lines.append(f"{side_icon} {name} {t['qty']}股 @{t['price']:.2f} ({amount:,.0f}元)")

    # Positions: top N by abs(daily_change), with P&L yuan
    # F12 note: sort by abs(daily_change) to show biggest movers regardless
    # of direction. Intentional design change (user requested "涨跌TOP").
    if rd.positions:
        all_zero = all(p.get("daily_change_pct", 0.0) == 0.0 for p in rd.positions)
        if all_zero:
            sorted_pos = sorted(rd.positions, key=lambda p: p.get("weight", 0.0), reverse=True)
        else:
            sorted_pos = sorted(
                rd.positions,
                key=lambda p: abs(p.get("daily_change_pct", 0.0)),
                reverse=True,
            )

        display_pos = sorted_pos
        title = f"涨跌TOP ({len(sorted_pos)}只)"

        lines.append("")
        lines.append(f"{_SEP} {title} {_SEP}")
        for p in display_pos:
            name = _display_name(p.get("name", ""), p["symbol"])
            change = p.get("daily_change_pct", 0.0)
            pnl = p.get("daily_pnl", 0.0)
            display_change = change * 100.0
            if change > 0:
                lines.append(
                    f"\U0001f4c8 {name} {p['weight']:.1f}% "
                    f"+{display_change:.2f}% +{pnl:,.0f}元"
                )
            elif change < 0:
                lines.append(
                    f"\U0001f4c9 {name} {p['weight']:.1f}% "
                    f"{display_change:.2f}% {pnl:,.0f}元"
                )
            else:
                lines.append(f"{name} {p['weight']:.1f}%")


    # Pending orders
    if rd.pending_orders:
        lines.append("")
        lines.append(f"{_SEP} 明日计划 {_SEP}")
        for o in rd.pending_orders:
            name = _display_name(o.get("name", ""), o["symbol"])
            side_icon = "\U0001f7e2 拟买" if o["side"] == "buy" else "\U0001f534 拟卖"
            lines.append(f"{side_icon} {name} {o['qty']}股")

    # Risk status
    lines.append("")
    rs = rd.risk_status
    any_risk = (
        rs["buying_halted"]
        or rs["is_soft_reduced"]
        or rs.get("drawdown_halted", False)
        or rs.get("regime_halted", False)
    )
    if any_risk:
        warnings = []
        if rs.get("drawdown_halted"):
            warnings.append("回撤暂停")
        if rs.get("regime_halted"):
            warnings.append("市场放缓")
        if rs["is_soft_reduced"]:
            warnings.append("软减仓")
        if rs["buying_halted"] and not warnings:
            warnings.append("买入暂停")
        lines.append(f"⚠️ {' | '.join(warnings)}")
    else:
        lines.append("✅ 风控正常")

    risk_state = rs.get("risk_state", "normal")
    if risk_state != "normal":
        state_labels = {
            "soft_reduced": "软减仓态",
            "buy_halt": "暂停买入态",
            "liquidated": "清仓锁定态",
        }
        label = state_labels.get(risk_state, risk_state)
        lockdown_date = rs.get("lockdown_enter_date")
        if lockdown_date:
            lines.append(f"🔒 风险状态: {label} (锁定起始: {lockdown_date})")
        else:
            lines.append(f"🔒 风险状态: {label}")

    if ic_nan:
        lines.append("⚠️ 模型自评指标不可用 (IC=nan)")

    if rd.hedge_active:
        lines.append(
            f"🛡 对冲: DD {rd.hedge_dd:.1%} → "
            f"权益 {rd.hedge_equity_pct:.0%} / "
            f"对冲 {1 - rd.hedge_equity_pct:.0%}"
        )

    # Sentiment section (optional, appended when available)
    if sentiment_section is not None:
        lines.append("")
        lines.append(sentiment_section)

    return "\n".join(lines)


def _format_sentiment_section(conn: sqlite3.Connection, trade_date: str, config: dict) -> str | None:
    """Build D7 sentiment section from veto events logged during pipeline run."""
    try:
        rows = conn.execute(
            "SELECT layer, target, score, detail FROM sentiment_events "
            "WHERE event_type='veto' AND trade_date=? ORDER BY layer, target",
            (trade_date,),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return "sentiment: all clear"

    lines = ["sentiment vetoes:"]
    for r in rows:
        lines.append(f"  {r[1]} | {r[0]} | score {r[2]}")
    return "\n".join(lines)


def _get_secret(key: str) -> str:
    try:
        r = subprocess.run(["pass", "show", key], capture_output=True, text=True, check=True, timeout=5)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Failed to get secret for %s: %s", key, e)
        raise


def split_report_text(text: str, max_length: int = 4000) -> list[str]:
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    paras = text.split("\n\n")
    current = []
    curr_len = 0
    buf_sep = "\n\n"
    
    for p in paras:
        plen = len(p)
        if curr_len + plen + 2 > max_length and current:
            chunks.append(buf_sep.join(current))
            current = []
            curr_len = 0
            buf_sep = "\n\n"
            
        if plen > max_length:
            buf_sep = "\n"
            lines = p.split("\n")
            for line in lines:
                llen = len(line)
                if curr_len + llen + 1 > max_length and current:
                    chunks.append(buf_sep.join(current))
                    current = []
                    curr_len = 0
                if llen > max_length:
                    # chunk strictly
                    for i in range(0, llen, max_length):
                        chunks.append(line[i:i+max_length])
                else:
                    current.append(line)
                    curr_len += llen + 1
            if current:
                chunks.append(buf_sep.join(current))
                current = []
                curr_len = 0
            buf_sep = "\n\n"
        else:
            current.append(p)
            curr_len += plen + 2
            
    if current:
        chunks.append(buf_sep.join(current))
        
    return chunks


def _random_wechat_uin() -> str:
    val = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(val).encode()).decode()


def _ilink_headers(token: str, body: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
        "Authorization": f"Bearer {token}",
    }


async def send_text_ilink(session, token: str, chat_id: str, text: str, timeout: int = 15) -> dict:
    import uuid
    import aiohttp
    
    payload = {
        "msg": {
            "from_user_id": "",
            "to_user_id": chat_id,
            "client_id": str(uuid.uuid4()),
            "message_type": MSG_TYPE_BOT,
            "message_state": MSG_STATE_FINISH,
            "item_list": [
                {
                    "type": ITEM_TEXT,
                    "text_item": {"text": text}
                }
            ]
        },
        "base_info": {
            "channel_version": CHANNEL_VERSION
        }
    }
    
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    headers = _ilink_headers(token, body)
    
    async with session.post(
        f"{ILINK_BASE_URL}/{EP_SEND_MESSAGE}",
        data=body.encode("utf-8"),
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        if resp.status != 200:
            resp_body = await resp.text()
            raise RuntimeError(f"iLink HTTP {resp.status}: {resp_body[:200]}")
        result = await resp.json(content_type=None)
        ret_code = result.get("ret", 0)
        if ret_code != 0:
            raise RuntimeError(f"iLink API ret={ret_code} (token expired or invalid)")
        return result


def _is_dry_run() -> bool:
    """Check if dry-run mode is enabled via ASHARE_DRY_RUN environment variable.

    Returns True only for explicit truthy values: "1", "true", "yes" (case-insensitive).
    Empty string, "0", "false", "no" return False.
    """
    val = os.environ.get("ASHARE_DRY_RUN", "").strip().lower()
    return val in ("1", "true", "yes")


def send_pushplus(token: str, title: str, content: str, timeout: int = 15) -> bool:
    if _is_dry_run():
        logger.info("[dry-run] pushplus send skipped: %s", title)
        return True
    try:
        import requests
        resp = requests.post(
            "http://www.pushplus.plus/send",
            json={"token": token, "title": title, "content": content},
            timeout=timeout
        )
        return resp.json().get("code") == 200
    except Exception as e:
        logger.warning("pushplus send failed: %s", e)
        return False


def send_serverchan(key: str, title: str, content: str, timeout: int = 15) -> bool:
    if _is_dry_run():
        logger.info("[dry-run] serverchan send skipped: %s", title)
        return True
    try:
        import requests
        resp = requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            data={"text": title, "desp": content},
            timeout=timeout
        )
        return resp.json().get("code") == 0
    except Exception as e:
        logger.warning("serverchan send failed: %s", e)
        return False


# Registry of fallback push services. Each entry: (pass_key, func_name).
# deliver_report resolves func_name via globals() at call time so that
# unittest.mock.patch on the module-level name takes effect.
_FALLBACK_PUSH = {
    "pushplus": ("ashare/pushplus-token", "send_pushplus"),
    "serverchan": ("ashare/serverchan-key", "send_serverchan"),
}


async def _send_via_ilink(chunks: list[str], chat_id: str, token: str,
                          timeout: int = 15, delay: float = 0.3,
                          sent_upto: list[int] | None = None) -> None:
    """Send report chunks directly via iLink API (same path as DSA sentinel).

    *sent_upto* is a single-element list ``[n]`` that tracks the index of the
    next chunk to send.  It is updated **after** each successful delivery so
    that callers can resume from the right position on retry.  When *None*,
    defaults to ``[0]`` (send everything).
    """
    if sent_upto is None:
        sent_upto = [0]
    import aiohttp
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as session:
        for i in range(sent_upto[0], len(chunks)):
            await send_text_ilink(session, token, chat_id, chunks[i], timeout=timeout)
            sent_upto[0] = i + 1
            logger.info("[deliver] iLink chunk %d/%d sent", i + 1, len(chunks))
            if i < len(chunks) - 1:
                await asyncio.sleep(delay)


def deliver_report(conn: sqlite3.Connection, trade_date: str, mode: str, report_text: str, config: dict, *, force: bool = False) -> str:
    logger.info("[deliver] start for %s (mode=%s, text_len=%d, force=%s)", trade_date, mode, len(report_text), force)
    insert_report(conn, trade_date, mode, report_text, None, "pending")
    conn.commit()

    # Dedup: skip if already sent for this (trade_date, mode).
    # Covers both primary and fallback paths. force=True bypasses.
    if not force:
        already = conn.execute(
            "SELECT COUNT(*) FROM reports WHERE trade_date=? AND mode=? AND delivery_status='sent'",
            (trade_date, mode),
        ).fetchone()[0]
        if already > 0:
            logger.info("[deliver] already sent for %s/%s, skipping", trade_date, mode)
            conn.execute(
                "UPDATE reports SET delivery_status='skipped_dedup' "
                "WHERE trade_date=? AND mode=? AND delivery_status='pending'",
                (trade_date, mode),
            )
            conn.commit()
            return "skipped"

    rcfg = config["paper"].get("report", {})
    channel = rcfg.get("delivery_channel", "alert_bridge")
    max_len = rcfg.get("ilink_max_message_length", 4000)

    chunks = split_report_text(report_text, max_len)
    logger.info("[deliver] split into %d chunks (max_len=%d, channel=%s)", len(chunks), max_len, channel)

    # --- Primary path: alert-bridge (default) or iLink (legacy) ---
    if channel == "alert_bridge":
        if _deliver_via_bridge(chunks, trade_date, conn, mode, report_text):
            conn.execute(
                "UPDATE reports SET delivery_status='sent', delivered_via='alert_bridge' "
                "WHERE trade_date=? AND mode=? AND delivery_status='pending'",
                (trade_date, mode),
            )
            conn.commit()
            return "sent"
    elif channel == "ilink":
        if _deliver_via_ilink_legacy(chunks, trade_date, conn, mode, report_text, rcfg):
            conn.execute(
                "UPDATE reports SET delivery_status='sent', delivered_via='ilink' "
                "WHERE trade_date=? AND mode=? AND delivery_status='pending'",
                (trade_date, mode),
            )
            conn.commit()
            return "sent"
    else:
        logger.warning("[deliver] unknown delivery_channel '%s', falling through", channel)

    # --- Fallback: serverchan / pushplus registry ---
    fb_name = rcfg.get("fallback_service", "serverchan")

    logger.info("[deliver] trying fallback=%s", fb_name)
    fb_entry = _FALLBACK_PUSH.get(fb_name)
    if fb_entry:
        pass_key, fn_name = fb_entry
        send_fn = globals()[fn_name]
        try:
            fb_token = _get_secret(pass_key)
            if send_fn(fb_token, f"A股日报 {trade_date}", report_text, rcfg.get("fallback_timeout", 15)):
                conn.execute(
                    "UPDATE reports SET delivery_status='sent', delivered_via=? "
                    "WHERE trade_date=? AND mode=? AND delivery_status='pending'",
                    (fb_name, trade_date, mode),
                )
                conn.commit()
                logger.info("[deliver] fallback %s succeeded", fb_name)
                return "sent"
        except Exception as e:
            logger.warning("[deliver] fallback %s failed: %s", fb_name, e)
    else:
        logger.warning("unknown fallback_service '%s', skipping fallback", fb_name)

    conn.execute(
        "UPDATE reports SET delivery_status='failed' "
        "WHERE trade_date=? AND mode=? AND delivery_status='pending'",
        (trade_date, mode),
    )
    conn.commit()
    return "failed"


def _deliver_via_bridge(chunks: list[str], trade_date: str, conn, mode: str, report_text: str) -> bool:
    """Send report chunks via alert-bridge. Returns True if all chunks sent."""
    from ashare_lab.bridge import send_bridge_alert  # noqa: PLC0415

    title_base = f"A股日报 {trade_date}"
    for i, chunk in enumerate(chunks):
        title = title_base if len(chunks) == 1 else f"{title_base} ({i+1}/{len(chunks)})"
        if not send_bridge_alert(title, chunk):
            logger.warning("[deliver] bridge chunk %d/%d failed", i + 1, len(chunks))
            return False
        logger.info("[deliver] bridge chunk %d/%d sent", i + 1, len(chunks))

    logger.info("[deliver] all %d chunks sent via alert-bridge", len(chunks))
    return True


def _deliver_via_ilink_legacy(chunks: list[str], trade_date: str, conn, mode: str, report_text: str, rcfg: dict) -> bool:
    """Send report chunks via iLink (legacy channel). Returns True if all chunks sent."""
    try:
        wx_token = _get_secret("ashare/weixin-token")
        wx_chat_id = _get_secret("ashare/weixin-chat-id")
        logger.info("[deliver] iLink secrets loaded, chat_id=%s",
                    wx_chat_id[:6] + "..." if len(wx_chat_id) > 6 else wx_chat_id)
    except Exception as exc:
        logger.warning("[deliver] iLink secret load failed: %s", exc)
        return False

    delay = rcfg.get("ilink_chunk_delay", 0.3)
    ilink_timeout = rcfg.get("ilink_timeout", 15)

    sent_upto = [0]
    for attempt in range(2):
        try:
            coro = _send_via_ilink(chunks, wx_chat_id, wx_token,
                                   timeout=ilink_timeout, delay=delay,
                                   sent_upto=sent_upto)
            try:
                asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                asyncio.run(coro)
            logger.info("[deliver] all %d chunks sent via iLink (attempt %d)", len(chunks), attempt + 1)
            return True
        except Exception as e:
            logger.warning("[deliver] iLink attempt %d failed at chunk %d: %s",
                           attempt + 1, sent_upto[0], e)
            if attempt == 0:
                time.sleep(5)

    return False


def generate_and_send_report(trade_date: str, conn: sqlite3.Connection, config: dict, dry_run: bool = False, *, force: bool = False) -> int:
    nav_check = conn.execute("SELECT 1 FROM nav WHERE trade_date=?", (trade_date,)).fetchone()
    if not nav_check:
        logger.warning("No data for %s", trade_date)
        return 1

    syms_rows = conn.execute(
        "SELECT DISTINCT symbol FROM positions WHERE trade_date=? "
        "UNION SELECT DISTINCT symbol FROM trades WHERE trade_date=? "
        "UNION SELECT DISTINCT symbol FROM orders WHERE created_run_date=? AND status IN ('pending','carry')",
        (trade_date, trade_date, trade_date)
    ).fetchall()
    symbols = {r[0] for r in syms_rows}

    from ashare_lab.paper.pipeline import _load_stock_names_cache, _load_industry_cache
    stock_names = _load_stock_names_cache(symbols) or {}
    industry_map = _load_industry_cache(trade_date, symbols) or {}

    report_data = gather_report_data(conn, trade_date, stock_names, industry_map, config)

    # sentiment section (fail-open: old DBs may lack table)
    sentiment_text = None
    try:
        sentiment_text = _format_sentiment_section(conn, trade_date, config)
    except Exception:
        pass

    mode = "chinese"
    ic_nan = config.get("_ic_nan", False)
    report_text = format_chinese_report(report_data, sentiment_section=sentiment_text, ic_nan=ic_nan)

    if dry_run:
        insert_report(conn, trade_date, mode, report_text, None, "dry_run")
        conn.commit()
        print(report_text)
        return 0

    status = deliver_report(conn, trade_date, mode, report_text, config, force=force)
    return 0 if status in ("sent", "skipped") else 1
