"""Unit tests for report generation and delivery."""

import json
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from ashare_lab.paper.report import (
    ReportData, gather_report_data, format_chinese_report,
    split_report_text, send_text_ilink, send_pushplus, send_serverchan,
    deliver_report, generate_and_send_report, _get_secret
)

# -------------------------------------------------------------------
# Data aggregation and mode determination
# -------------------------------------------------------------------

def test_gather_report_data_nav_fields(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {}, {}, {"paper": paper_config})
    assert rd.total_nav == 300_000.0 * 0.97
    assert rd.cash == 50_000.0
    assert rd.daily_return_pct < 0
    assert rd.cumulative_return_pct < 0
    assert rd.max_drawdown_pct == 3.0
    assert rd.cash_ratio == pytest.approx(50000 / (300000*0.97) * 100)
    assert rd.benchmark_return_pct < 0
    assert rd.benchmark_csi1000 == 4800.0

def test_gather_report_data_trades(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {"SZ000001": "Test"}, {}, {"paper": paper_config})
    assert len(rd.trades) == 1
    t = rd.trades[0]
    assert t["symbol"] == "SZ000001"
    assert t["side"] == "buy"
    assert t["qty"] == 1000
    assert t["price"] == 10.0
    assert t["total_fee"] == 6.0

def test_gather_report_data_positions(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {"SZ000001": "Test"}, {}, {"paper": paper_config})
    assert len(rd.positions) == 1
    p = rd.positions[0]
    assert p["symbol"] == "SZ000001"
    assert p["qty"] == 1000
    assert p["market_value"] == 10000.0
    assert p["unrealized_pnl"] == 1000.0
    assert p["weight"] > 0

def test_gather_report_data_pending_orders(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {"SZ000002": "Test2"}, {}, {"paper": paper_config})
    assert len(rd.pending_orders) == 1
    o = rd.pending_orders[0]
    assert o["symbol"] == "SZ000002"
    assert o["side"] == "sell"
    assert o["qty"] == 500

def test_gather_report_data_risk_status(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {}, {}, {"paper": paper_config})
    rs = rd.risk_status
    assert rs["buying_halted"] is True  # 3% drawdown > 0.15 hard config? No, hard config is 15%. Wait.
    # Actually daily loss is 3%, which is equal to 3% config (<= -0.03).
    # Regime is halted because we inserted declining closes.
    assert rs["regime_halted"] is True
    assert rs["is_soft_reduced"] is False
    assert rs["sell_order_count"] == 1
    assert rs["cooldown_count"] == 1

def test_gather_report_data_no_trade_day(db_conn, paper_config):
    from ashare_lab.paper.ledger import record_nav
    record_nav(db_conn, "2025-01-06", 300_000.0, 0.0, 300_000.0, None, None, None, None)
    rd = gather_report_data(db_conn, "2025-01-06", {}, {}, {"paper": paper_config})
    assert rd.trade_count == 0
    assert len(rd.trades) == 0

def test_stock_names_resolve_cache(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {"SZ000001": "TestBank"}, {}, {"paper": paper_config})
    assert rd.trades[0]["name"] == "TestBank"


# -------------------------------------------------------------------
# Templates and LLM
# -------------------------------------------------------------------

@pytest.fixture
def dummy_report_data():
    return ReportData(
        trade_date="2025-01-06",
        total_nav=300000.0,
        cash=150000.0,
        daily_return_pct=1.5,
        daily_pnl=4500.0,
        cumulative_return_pct=10.0,
        max_drawdown_pct=5.0,
        cash_ratio=50.0,
        benchmark_csi1000=1000.0,
        benchmark_return_pct=0.5,
        trades=[{
            "symbol": "SZ000001",
            "name": "Test1",
            "side": "buy",
            "qty": 100,
            "price": 10.0,
            "commission": 1.0,
            "stamp": 1.0,
            "transfer_fee": 1.0,
            "total_fee": 3.0
        }],
        positions=[{
            "symbol": "SZ000001",
            "name": "Test1",
            "qty": 100,
            "market_value": 1000.0,
            "unrealized_pnl": 100.0,
            "weight": 0.33,
            "daily_change_pct": 0.01,
        }],
        pending_orders=[{
            "symbol": "SZ000002",
            "name": "Test2",
            "side": "sell",
            "qty": 200
        }],
        trade_count=1,
        risk_status={
            "buying_halted": False,
            "is_soft_reduced": False,
            "sell_order_count": 0,
            "cooldown_count": 0,
            "drawdown_halted": False,
            "regime_halted": False,
        },
        industry_distribution={"Bank": 1}
    )

def test_format_chinese_report_header(dummy_report_data):
    txt = format_chinese_report(dummy_report_data)
    assert "2025-01-06" in txt.split("\n")[0]
    assert "ashare-lab" in txt

def test_format_chinese_report_risk_normal(dummy_report_data):
    txt = format_chinese_report(dummy_report_data)
    assert "✅" in txt  # checkmark

def test_format_chinese_report_nav_pnl(dummy_report_data):
    """F14: verify daily_pnl appears in NAV line."""
    txt = format_chinese_report(dummy_report_data)
    # daily_pnl=4500.0, daily_return_pct=1.5
    assert "+1.50%" in txt
    assert "+4,500元" in txt

def test_format_chinese_report_trade_amount(dummy_report_data):
    """F15: verify trade amount in yuan appears."""
    txt = format_chinese_report(dummy_report_data)
    # qty=100, price=10.0 -> amount=1000
    assert "1,000元" in txt

@patch("subprocess.run")
def test_get_secret(mock_run):
    mock_run.return_value.stdout = "secret\n"
    assert _get_secret("key") == "secret"
    mock_run.assert_called_with(["pass", "show", "key"], capture_output=True, text=True, check=True, timeout=5)


# -------------------------------------------------------------------
# Delivery
# -------------------------------------------------------------------

def test_split_report_text_short():
    assert len(split_report_text("short", 100)) == 1

def test_split_report_text_paragraph():
    res = split_report_text("para1\n\npara2", 6)
    assert len(res) == 2
    assert res[0] == "para1"

def test_split_report_text_newline():
    res = split_report_text("line1\nline2", 6)
    assert len(res) == 2

def test_split_report_text_never_exceeds():
    text = "x" * 5000
    res = split_report_text(text, 4000)
    assert all(len(c) <= 4000 for c in res)

def test_split_oversized_paragraph_preserves_next_paragraph_boundary():
    text = "AA" + "\n\n" + "aaaa\nbbbb\ncccc\ndddd\neeee" + "\n\n" + "ffff"
    chunks = split_report_text(text, 20)
    assert all(len(c) <= 20 for c in chunks)
    # paragraph "ffff" must not be glued to the oversized para tail by "\n"
    assert not any("eeee\nffff" in c for c in chunks), \
        f"paragraph boundary collapsed: {chunks!r}"

@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post")
async def test_send_text_ilink_headers_payload(mock_post):
    class MockResp:
        status = 200
        async def json(self, **kwargs): return {}
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
    mock_post.return_value = MockResp()
    
    import aiohttp
    async with aiohttp.ClientSession() as session:
        await send_text_ilink(session, "tok", "chat", "txt")

    kwargs = mock_post.call_args[1]
    assert "AuthorizationType" in kwargs["headers"]
    assert "X-WECHAT-UIN" in kwargs["headers"]
    assert "Bearer tok" in kwargs["headers"]["Authorization"]

    body = json.loads(kwargs["data"].decode())
    assert body["msg"]["to_user_id"] == "chat"
    assert body["msg"]["from_user_id"] == ""
    assert body["msg"]["item_list"][0]["text_item"]["text"] == "txt"

@patch("requests.post")
def test_send_pushplus(mock_post):
    mock_post.return_value.json.return_value = {"code": 200}
    assert send_pushplus("tok", "title", "content") is True
    kwargs = mock_post.call_args[1]
    assert kwargs["json"]["token"] == "tok"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report._send_via_ilink", new_callable=AsyncMock)
def test_deliver_report_ilink_success(mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", {"paper":{}}) == "sent"
    r = db_conn.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1").fetchone()
    assert r["delivered_via"] == "ilink"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report._send_via_ilink", new_callable=AsyncMock)
@patch("time.sleep")
def test_deliver_report_ilink_retry(mock_sleep, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    mock_ilink.side_effect = [Exception("fail"), None]
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", {"paper":{}}) == "sent"
    r = db_conn.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1").fetchone()
    assert r["delivered_via"] == "ilink"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report._send_via_ilink", new_callable=AsyncMock)
@patch("time.sleep")
def test_deliver_report_retry_resumes_from_sent_chunk(mock_sleep, mock_ilink, mock_sec, db_conn):
    """Retry must resume from the last successfully sent chunk, not from 0."""
    mock_sec.return_value = "token"
    # Use enough text to produce multiple chunks.
    cfg = {"paper": {"report": {"ilink_max_message_length": 20}}}
    text = "A" * 60  # -> 3 chunks of 20

    call_count = {"n": 0}
    resume_log = []

    async def side_effect_fn(*args, **kwargs):
        sent_upto = kwargs.get("sent_upto")
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate: chunk 0 sent successfully, chunk 1 fails.
            sent_upto[0] = 1
            raise Exception("transient network error")
        else:
            # On retry, verify we resume from chunk 1, not 0.
            resume_log.append(sent_upto[0])

    mock_ilink.side_effect = side_effect_fn
    assert deliver_report(db_conn, "2025-01-06", "simple", text, cfg) == "sent"
    assert call_count["n"] == 2
    assert resume_log == [1], f"retry should resume from chunk 1, got {resume_log}"

@patch("requests.post")
def test_send_serverchan(mock_post):
    mock_post.return_value.json.return_value = {"code": 0}
    assert send_serverchan("SCTxxx", "title", "content") is True
    args = mock_post.call_args
    assert "sctapi.ftqq.com/SCTxxx.send" in args[0][0]
    assert args[1]["data"]["text"] == "title"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report._send_via_ilink", new_callable=AsyncMock)
@patch("ashare_lab.paper.report.send_serverchan")
@patch("time.sleep")
def test_deliver_report_fallback(mock_sleep, mock_sc, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    mock_ilink.side_effect = Exception("iLink down")
    mock_sc.return_value = True
    cfg = {"paper": {"report": {"fallback_service": "serverchan"}}}
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", cfg) == "sent"
    r = db_conn.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1").fetchone()
    assert r["delivered_via"] == "serverchan"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report._send_via_ilink", new_callable=AsyncMock)
@patch("ashare_lab.paper.report.send_serverchan")
@patch("time.sleep")
def test_deliver_report_all_fail(mock_sleep, mock_sc, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    mock_ilink.side_effect = Exception("iLink down")
    mock_sc.return_value = False
    cfg = {"paper": {"report": {"fallback_service": "serverchan"}}}
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", cfg) == "failed"
    r = db_conn.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1").fetchone()
    assert r["delivery_status"] == "failed"

def test_deliver_report_saves_always(db_conn):
    with patch("ashare_lab.paper.report._get_secret", side_effect=Exception("err")):
        deliver_report(db_conn, "2025-01-06", "simple", "text", {"paper":{}})
    r = db_conn.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1").fetchone()
    assert r is not None
    assert r["delivery_status"] == "failed"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report._send_via_ilink", new_callable=AsyncMock)
def test_deliver_report_splits(mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    deliver_report(db_conn, "2025-01-06", "simple", "x" * 5000, {"paper":{"report":{"ilink_max_message_length":4000}}})
    assert mock_ilink.call_count == 1
    # Verify chunks arg has 2 elements
    chunks_arg = mock_ilink.call_args[0][0]
    assert len(chunks_arg) == 2


# Integration
def test_generate_and_send_report_dry_run(populated_db, paper_config):
    paper_config["report"] = {}
    rc = generate_and_send_report("2025-01-06", populated_db, {"paper": paper_config}, dry_run=True)
    assert rc == 0
    r = populated_db.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 1").fetchone()
    assert r["delivery_status"] == "dry_run"

@patch("ashare_lab.paper.report.deliver_report")
def test_generate_and_send_report_chinese_mode(mock_deliver, db_conn, paper_config):
    mock_deliver.return_value = "sent"
    paper_config["report"] = {}
    from ashare_lab.paper.ledger import record_nav
    record_nav(db_conn, "2024-12-30", 300_000.0, 0.0, 300_000.0, None, None, None, None)
    record_nav(db_conn, "2025-01-06", 50000.0, 250000.0, 300000.0, None, None, 3000.0, 5000.0)
    rc = generate_and_send_report("2025-01-06", db_conn, {"paper": paper_config})
    assert rc == 0
    mock_deliver.assert_called()
    assert mock_deliver.call_args[0][2] == "chinese"


def test_cli_report_args():
    from ashare_lab.cli import main
    import sys
    # --detailed removed from cli.py (force_detailed param no longer exists in report.py)
    with patch.object(sys, "argv", ["ashare-lab", "paper", "report", "--date", "2025-01-06", "--dry-run"]):
        with patch("ashare_lab.cli.cmd_paper_report") as mock_cmd:
            mock_cmd.return_value = 0
            main()
            mock_cmd.assert_called_once()
            args = mock_cmd.call_args[0][0]
            assert args.date == "2025-01-06"
            assert args.dry_run is True

def test_generate_and_send_report_no_nav_row(db_conn, paper_config):
    rc = generate_and_send_report("2025-01-06", db_conn, {"paper": paper_config})
    assert rc == 1
