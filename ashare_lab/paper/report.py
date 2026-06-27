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
from datetime import datetime, timezone
import sqlite3

from ashare_lab.paper.ledger import insert_report

logger = logging.getLogger(__name__)

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
        
    # Forced sell count
    forced_sells = conn.execute(
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
            "forced_sell_count": forced_sells,
            "cooldown_count": cooldown_count,
            "drawdown_halted": drawdown_halted,
            "regime_halted": regime_halted,
        },
        industry_distribution=ind_dist
    )


def determine_report_mode(report_data: ReportData, triggers_config: dict) -> str:
    if report_data.daily_return_pct <= -triggers_config["daily_loss_pct"] * 100.0:
        return "detailed"
    if report_data.trade_count >= triggers_config["min_trade_count"]:
        return "detailed"
    rs = report_data.risk_status
    if rs["buying_halted"] or rs["is_soft_reduced"] or rs["forced_sell_count"] > 0:
        return "detailed"
    return "simple"


def format_simple_report(report_data: ReportData) -> str:
    sections = []
    
    # S1
    sections.append(f"{report_data.trade_date} | {report_data.trade_count} trades | NAV {report_data.total_nav:,.2f}")
    
    # S2
    s2 = (
        f"NAV: {report_data.total_nav:,.2f} | daily: {report_data.daily_return_pct:+.2f}%\n"
        f"cumulative: {report_data.cumulative_return_pct:+.2f}% | drawdown: {report_data.max_drawdown_pct:.2f}%\n"
        f"cash: {report_data.cash_ratio:.1f}% | vs CSI1000: {report_data.benchmark_return_pct:+.2f}%"
    )
    sections.append(s2)
    
    # S3
    if report_data.trade_count == 0:
        sections.append("no trades today")
    else:
        s3_lines = []
        for t in report_data.trades:
            name = f"{t['symbol']} {t['name']}".strip()
            s3_lines.append(f"{name} | {t['side']} | {t['qty']} shares | {t['price']:.2f} | fees {t['total_fee']:.2f}")
        sections.append("\n".join(s3_lines))
        
    # S4
    if not report_data.pending_orders:
        sections.append("no pending orders")
    else:
        s4_lines = []
        for o in report_data.pending_orders:
            name = f"{o['symbol']} {o['name']}".strip()
            s4_lines.append(f"{name} | {o['side']} | {o['qty']} shares")
        sections.append("\n".join(s4_lines))
        
    return "\n\n".join(sections)


def format_detailed_report(report_data: ReportData, llm_summary: str | None) -> str:
    sections = []
    
    if llm_summary is not None:
        sections.append(llm_summary)
        
    s2 = (
        f"NAV: {report_data.total_nav:,.2f} | daily: {report_data.daily_return_pct:+.2f}%\n"
        f"cumulative: {report_data.cumulative_return_pct:+.2f}% | drawdown: {report_data.max_drawdown_pct:.2f}%\n"
        f"cash: {report_data.cash_ratio:.1f}% | vs CSI1000: {report_data.benchmark_return_pct:+.2f}%"
    )
    sections.append(s2)
    
    if report_data.trade_count == 0:
        sections.append("no trades today")
    else:
        s3_lines = []
        for t in report_data.trades:
            name = f"{t['symbol']} {t['name']}".strip()
            s3_lines.append(f"{name} | {t['side']} | {t['qty']} shares | {t['price']:.2f} | fees {t['total_fee']:.2f}")
        sections.append("\n".join(s3_lines))
        
    if not report_data.positions:
        sections.append("no positions")
    else:
        s4_lines = []
        for p in report_data.positions:
            name = f"{p['symbol']} {p['name']}".strip()
            s4_lines.append(f"{name} | {p['qty']} | {p['market_value']:,.2f} | {p['unrealized_pnl']:,.2f} | {p['weight']:.2f}%")
        sections.append("\n".join(s4_lines))
        
    if not report_data.pending_orders:
        sections.append("no pending orders")
    else:
        s5_lines = []
        for o in report_data.pending_orders:
            name = f"{o['symbol']} {o['name']}".strip()
            s5_lines.append(f"{name} | {o['side']} | {o['qty']} shares")
        sections.append("\n".join(s5_lines))
        
    rs = report_data.risk_status
    s6 = (
        f"drawdown: {report_data.max_drawdown_pct:.2f}% | halt: {'yes' if rs['buying_halted'] else 'no'}\n"
        f"soft reduction: {'yes' if rs['is_soft_reduced'] else 'no'} | cooldowns: {rs['cooldown_count']}"
    )
    sections.append(s6)
    
    return "\n\n".join(sections)


def _get_secret(key: str) -> str:
    try:
        r = subprocess.run(["pass", "show", key], capture_output=True, text=True, check=True, timeout=5)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Failed to get secret for %s: %s", key, e)
        raise


def call_deepseek_summary(report_data: ReportData, config: dict) -> str | None:
    try:
        import requests
    except ImportError:
        return None

    try:
        api_key = _get_secret("ashare/deepseek-api-key")
    except Exception:
        return None

    model = config["paper"]["report"].get("deepseek_model", "deepseek-v4-flash")
    timeout = config["paper"]["report"].get("deepseek_timeout", 30)

    data = {
        "date": report_data.trade_date,
        "nav": report_data.total_nav,
        "daily_return_pct": report_data.daily_return_pct,
        "position_count": len(report_data.positions),
        "trade_count": report_data.trade_count,
        "industry_distribution": report_data.industry_distribution,
        "risk_status": {
            "buying_halted": report_data.risk_status["buying_halted"],
            "is_soft_reduced": report_data.risk_status["is_soft_reduced"],
        }
    }

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "用一句中文总结今日组合表现，包含主要驱动因素"},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False)}
                ],
                "thinking": {"type": "disabled"}
            },
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("deepseek summary failed: %s", e)
        return None


def split_report_text(text: str, max_length: int = 4000) -> list[str]:
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    paras = text.split("\n\n")
    current = []
    curr_len = 0
    
    for p in paras:
        plen = len(p)
        if curr_len + plen + 2 > max_length and current:
            chunks.append("\n\n".join(current))
            current = []
            curr_len = 0
            
        if plen > max_length:
            lines = p.split("\n")
            for line in lines:
                llen = len(line)
                if curr_len + llen + 1 > max_length and current:
                    chunks.append("\n\n".join(current) if '\n\n' in "\n\n".join(current) else "\n".join(current))
                    current = []
                    curr_len = 0
                if llen > max_length:
                    # chunk strictly
                    for i in range(0, llen, max_length):
                        chunks.append(line[i:i+max_length])
                else:
                    current.append(line)
                    curr_len += llen + 1
        else:
            current.append(p)
            curr_len += plen + 2
            
    if current:
        chunks.append("\n\n".join(current))
        
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
            raise RuntimeError(f"iLink error {resp.status}: {resp_body[:200]}")
        return await resp.json(content_type=None)


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

    async def _send_chunks(start_idx: int = 0) -> int:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for i in range(start_idx, len(chunks)):
                await send_text_ilink(session, wx_token, wx_chat_id, chunks[i], ilink_timeout)
                if i < len(chunks) - 1:
                    await asyncio.sleep(delay)
                start_idx = i + 1
        return start_idx

    try:
        asyncio.run(_send_chunks(0))
        insert_report(conn, trade_date, mode, report_text, "ilink", "sent")
        conn.commit()
        return "sent"
    except Exception as e:
        logger.warning("iLink first attempt failed: %s", e)
        time.sleep(5)
        try:
            # We don't track partial success reliably here for retry, just retry full
            asyncio.run(_send_chunks(0))
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

    insert_report(conn, trade_date, mode, report_text, None, "failed")
    conn.commit()
    return "failed"


def generate_and_send_report(trade_date: str, conn: sqlite3.Connection, config: dict, dry_run: bool = False, force_detailed: bool = False) -> int:
    nav_check = conn.execute("SELECT 1 FROM nav WHERE trade_date=?", (trade_date,)).fetchone()
    if not nav_check:
        print(f"No data for {trade_date}")
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

    mode = "detailed" if force_detailed else determine_report_mode(report_data, config["paper"]["report"]["detailed_triggers"])

    if mode == "simple":
        report_text = format_simple_report(report_data)
    else:
        llm_summary = call_deepseek_summary(report_data, config)
        if llm_summary is None:
            logger.warning("deepseek summary unavailable for %s, detailed report will omit D1 section", trade_date)
        report_text = format_detailed_report(report_data, llm_summary)

    if dry_run:
        insert_report(conn, trade_date, mode, report_text, None, "dry_run")
        conn.commit()
        print(report_text)
        return 0

    status = deliver_report(conn, trade_date, mode, report_text, config)
    return 0 if status == "sent" else 1
