"""Report generation and delivery module."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import struct
import subprocess
import time
from dataclasses import dataclass
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

    # Daily change per stock from qlib bin data
    daily_changes: dict[str, float] = {}
    if positions:
        try:
            from qlib.data import D
            pos_syms = [p["symbol"] for p in positions]
            df = D.features(
                instruments=pos_syms,
                fields=["$change"],
                start_time=trade_date,
                end_time=trade_date,
            )
            if df is not None and not df.empty:
                for idx_tuple, row in df.iterrows():
                    sym = idx_tuple[0] if isinstance(idx_tuple, tuple) else str(idx_tuple)
                    val = row.iloc[0]
                    if val == val:  # not NaN
                        daily_changes[sym] = float(val)
        except Exception as exc:
            logger.warning("qlib daily change query failed: %s", exc)

    for p in positions:
        p["daily_change_pct"] = daily_changes.get(p["symbol"], 0.0)

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
    
    # is_soft_reduced
    row_soft = conn.execute("SELECT value FROM paper_state WHERE key = 'is_soft_reduced'").fetchone()
    is_soft_reduced = row_soft and row_soft["value"] == "true"

    # buying_halted
    current_drawdown_pct = 0.0
    if peak > 0:
        current_drawdown_pct = (peak - total_nav) / peak * 100.0
    
    risk_cfg = config["paper"]["risk"]
    drawdown_halted = current_drawdown_pct > risk_cfg["drawdown_hard"] * 100.0
    daily_loss_halted = daily_return_pct <= -risk_cfg["daily_loss"] * 100.0
    
    # regime halted
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

    return ReportData(
        trade_date=trade_date,
        total_nav=total_nav,
        cash=cash,
        daily_return_pct=daily_return_pct,
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
            "sell_order_count": sell_order_count,
            "cooldown_count": cooldown_count,
            "drawdown_halted": drawdown_halted,
            "regime_halted": regime_halted,
        },
        industry_distribution=ind_dist
    )


def format_chinese_report(
    report_data: ReportData,
    sentiment_section: str | None = None,
) -> str:
    """Format report as Chinese mobile-first template."""
    rd = report_data
    lines: list[str] = []

    # Header
    lines.append(f"\U0001f4ca ashare-lab {rd.trade_date}")
    lines.append("")

    # NAV + metrics
    excess = rd.daily_return_pct - rd.benchmark_return_pct
    lines.append(f"\U0001f4b0 净值 {rd.total_nav:,.2f} ({rd.daily_return_pct:+.2f}%)")
    lines.append(
        f"\U0001f4c8 累计 {rd.cumulative_return_pct:+.2f}% | "
        f"回撤 {rd.max_drawdown_pct:.2f}% | "
        f"现金 {rd.cash_ratio:.1f}%"
    )
    lines.append(f"\U0001f3af 超额 vs CSI1000 {excess:+.2f}%")

    # Trades
    if rd.trades:
        lines.append("")
        lines.append(f"{_SEP} 今日交易 {_SEP}")
        for t in rd.trades:
            name = t["name"] or t["symbol"]
            if t["side"] == "buy":
                lines.append(f"\U0001f7e2 买 {name} {t['qty']}股 @{t['price']:.2f}")
            else:
                lines.append(f"\U0001f534 卖 {name} {t['qty']}股 @{t['price']:.2f}")

    # Positions sorted by daily change desc
    if rd.positions:
        sorted_pos = sorted(rd.positions, key=lambda p: p.get("daily_change_pct", 0.0), reverse=True)
        lines.append("")
        lines.append(f"{_SEP} 持仓分布 ({len(sorted_pos)}只) {_SEP}")
        for p in sorted_pos:
            name = p["name"] or p["symbol"]
            change = p.get("daily_change_pct", 0.0)
            # $change from qlib is fractional return (e.g. 0.05 = 5%); multiply by 100 for display
            display_change = change * 100.0
            if change > 0:
                indicator = f"\U0001f53a+{display_change:.2f}%"
            elif change < 0:
                indicator = f"\U0001f53b{display_change:.2f}%"
            else:
                indicator = f" {display_change:.2f}%"
            lines.append(f"{name}  {p['weight']:.1f}% {indicator}")

    # Pending orders
    if rd.pending_orders:
        lines.append("")
        lines.append(f"{_SEP} 明日计划 {_SEP}")
        for o in rd.pending_orders:
            name = o["name"] or o["symbol"]
            if o["side"] == "buy":
                lines.append(f"\U0001f7e2 拟买 {name} {o['qty']}股")
            else:
                lines.append(f"\U0001f534 拟卖 {name} {o['qty']}股")

    # Risk status
    lines.append("")
    rs = rd.risk_status
    any_risk = rs["buying_halted"] or rs["is_soft_reduced"] or rs.get("drawdown_halted", False) or rs.get("regime_halted", False)
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


def send_pushplus(token: str, title: str, content: str, timeout: int = 15) -> bool:
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


def deliver_report(conn: sqlite3.Connection, trade_date: str, mode: str, report_text: str, config: dict) -> str:
    insert_report(conn, trade_date, mode, report_text, None, "pending")
    conn.commit()

    try:
        wx_token = _get_secret("ashare/weixin-token")
        # weixin-account-id: reserved for context_token path (T3 runtime
        # verification on X500). Not needed for token-less bot send where
        # from_user_id="" per hermes weixin.py:402.
        wx_chat_id = _get_secret("ashare/weixin-chat-id")
    except Exception:
        insert_report(conn, trade_date, mode, report_text, None, "failed")
        conn.commit()
        logger.warning("Failed to load iLink secrets, aborting delivery")
        return "failed"

    rcfg = config["paper"].get("report", {})
    max_len = rcfg.get("ilink_max_message_length", 4000)
    delay = rcfg.get("ilink_chunk_delay", 0.3)
    ilink_timeout = rcfg.get("ilink_timeout", 15)
    
    chunks = split_report_text(report_text, max_len)
    sent_upto = 0

    async def _send_chunks() -> None:
        nonlocal sent_upto
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for i in range(sent_upto, len(chunks)):
                await send_text_ilink(session, wx_token, wx_chat_id, chunks[i], ilink_timeout)
                sent_upto = i + 1
                if i < len(chunks) - 1:
                    await asyncio.sleep(delay)

    try:
        asyncio.run(_send_chunks())
        insert_report(conn, trade_date, mode, report_text, "ilink", "sent")
        conn.commit()
        return "sent"
    except Exception as e:
        logger.warning("iLink first attempt failed (sent %d/%d): %s", sent_upto, len(chunks), e)
        time.sleep(5)
        try:
            asyncio.run(_send_chunks())
            insert_report(conn, trade_date, mode, report_text, "ilink", "sent")
            conn.commit()
            return "sent"
        except Exception as e2:
            logger.warning("iLink second attempt failed: %s", e2)

    fb_name = rcfg.get("fallback_service", "serverchan")
    fb_entry = _FALLBACK_PUSH.get(fb_name)
    if fb_entry:
        pass_key, fn_name = fb_entry
        send_fn = globals()[fn_name]
        try:
            fb_token = _get_secret(pass_key)
            if send_fn(fb_token, f"A股日报 {trade_date}", report_text, rcfg.get("fallback_timeout", 15)):
                insert_report(conn, trade_date, mode, report_text, fb_name, "sent")
                conn.commit()
                return "sent"
        except Exception as e:
            logger.warning("%s fallback failed: %s", fb_name, e)
    else:
        logger.warning("unknown fallback_service '%s', skipping fallback", fb_name)

    insert_report(conn, trade_date, mode, report_text, None, "failed")
    conn.commit()
    return "failed"


def generate_and_send_report(trade_date: str, conn: sqlite3.Connection, config: dict, dry_run: bool = False) -> int:
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
    report_text = format_chinese_report(report_data, sentiment_section=sentiment_text)

    if dry_run:
        insert_report(conn, trade_date, mode, report_text, None, "dry_run")
        conn.commit()
        print(report_text)
        return 0

    status = deliver_report(conn, trade_date, mode, report_text, config)
    return 0 if status == "sent" else 1
