"""Unit tests for report generation and delivery."""

import json
from unittest.mock import patch, MagicMock
import pytest
from ashare_lab.paper.report import (
    ReportData, gather_report_data, determine_report_mode,
    format_simple_report, format_detailed_report, call_deepseek_summary,
    split_report_text, send_text_ilink, send_pushplus, send_serverchan,
    deliver_report,
    generate_and_send_report, _get_secret
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

def test_determine_report_mode_daily_loss(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {}, {}, {"paper": paper_config})
    assert determine_report_mode(rd, {"daily_loss_pct": 0.01, "min_trade_count": 5}) == "detailed"

def test_determine_report_mode_trade_count(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {}, {}, {"paper": paper_config})
    assert determine_report_mode(rd, {"daily_loss_pct": 0.1, "min_trade_count": 1}) == "detailed"

def test_determine_report_mode_risk_triggered(populated_db, paper_config):
    rd = gather_report_data(populated_db, "2025-01-06", {}, {}, {"paper": paper_config})
    assert determine_report_mode(rd, {"daily_loss_pct": 0.1, "min_trade_count": 5}) == "detailed"

def test_determine_report_mode_simple(db_conn, paper_config):
    from ashare_lab.paper.ledger import record_nav
    record_nav(db_conn, "2025-01-06", 300_000.0, 0.0, 300_000.0, None, None, None, None)
    rd = gather_report_data(db_conn, "2025-01-06", {}, {}, {"paper": paper_config})
    assert determine_report_mode(rd, {"daily_loss_pct": 0.1, "min_trade_count": 5}) == "simple"

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
            "weight": 0.33
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

def test_format_simple_report_s1(dummy_report_data):
    txt = format_simple_report(dummy_report_data)
    assert "2025-01-06 | 1 trades | NAV 300,000.00" in txt.split("\n\n")[0]

def test_format_simple_report_s2(dummy_report_data):
    txt = format_simple_report(dummy_report_data)
    assert "NAV: 300,000.00 | daily: +1.50%" in txt

def test_format_simple_report_s3(dummy_report_data):
    txt = format_simple_report(dummy_report_data)
    assert "SZ000001 Test1 | buy | 100 shares | 10.00 | fees 3.00" in txt

def test_format_simple_report_s4(dummy_report_data):
    txt = format_simple_report(dummy_report_data)
    assert "SZ000002 Test2 | sell | 200 shares" in txt

def test_format_simple_report_no_trade(dummy_report_data):
    from dataclasses import replace
    rd = replace(dummy_report_data, trade_count=0, trades=[])
    txt = format_simple_report(rd)
    assert "no trades today" in txt

def test_format_detailed_report_match_sections(dummy_report_data):
    txt = format_detailed_report(dummy_report_data, None)
    assert "NAV: 300,000.00" in txt
    assert "SZ000001 Test1 | 100 | 1,000.00 | 100.00 | 0.33%" in txt

def test_format_detailed_report_risk_status(dummy_report_data):
    txt = format_detailed_report(dummy_report_data, None)
    assert "drawdown: 5.00% | halt: no" in txt

def test_format_detailed_report_with_llm(dummy_report_data):
    txt = format_detailed_report(dummy_report_data, "This is LLM.")
    assert txt.startswith("This is LLM.\n\nNAV:")

def test_format_detailed_report_without_llm(dummy_report_data):
    txt = format_detailed_report(dummy_report_data, None)
    assert txt.startswith("NAV:")

def test_stock_name_display(dummy_report_data):
    txt = format_simple_report(dummy_report_data)
    assert "SZ000001 Test1" in txt

# Deepseek tests
@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_success(mock_sec, mock_post, dummy_report_data):
    mock_sec.return_value = "token"
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
    assert call_deepseek_summary(dummy_report_data, {"paper": {"report": {}}}) == "summary"

@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_error(mock_sec, mock_post, dummy_report_data):
    mock_sec.return_value = "token"
    mock_post.side_effect = Exception("error")
    assert call_deepseek_summary(dummy_report_data, {"paper": {"report": {}}}) is None

@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_timeout(mock_sec, mock_post, dummy_report_data):
    from requests.exceptions import Timeout
    mock_sec.return_value = "token"
    mock_post.side_effect = Timeout("timeout")
    assert call_deepseek_summary(dummy_report_data, {"paper": {"report": {}}}) is None

@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_malformed(mock_sec, mock_post, dummy_report_data):
    mock_sec.return_value = "token"
    mock_post.return_value.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
    assert call_deepseek_summary(dummy_report_data, {"paper": {"report": {}}}) is None

@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_model_id(mock_sec, mock_post, dummy_report_data):
    mock_sec.return_value = "token"
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
    call_deepseek_summary(dummy_report_data, {"paper": {"report": {"deepseek_model": "deepseek-v4-flash"}}})
    assert mock_post.call_args[1]["json"]["model"] == "deepseek-v4-flash"

@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_payload(mock_sec, mock_post, dummy_report_data):
    mock_sec.return_value = "token"
    call_deepseek_summary(dummy_report_data, {"paper": {"report": {}}})
    data = json.loads(mock_post.call_args[1]["json"]["messages"][1]["content"])
    assert "nav" in data
    assert "daily_return_pct" in data
    assert "industry_distribution" in data
    assert "risk_status" in data

@patch("requests.post")
@patch("ashare_lab.paper.report._get_secret")
def test_call_deepseek_thinking(mock_sec, mock_post, dummy_report_data):
    mock_sec.return_value = "token"
    call_deepseek_summary(dummy_report_data, {"paper": {"report": {}}})
    assert mock_post.call_args[1]["json"]["thinking"] == {"type": "disabled"}

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
@patch("ashare_lab.paper.report.send_text_ilink")
def test_deliver_report_ilink_success(mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", {"paper":{}}) == "sent"
    r = db_conn.execute("SELECT * FROM reports").fetchone()
    assert r["delivered_via"] == "ilink"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report.send_text_ilink")
@patch("time.sleep")
def test_deliver_report_ilink_retry(mock_sleep, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    mock_ilink.side_effect = [Exception("err"), {}]
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", {"paper":{}}) == "sent"
    r = db_conn.execute("SELECT * FROM reports").fetchone()
    assert r["delivered_via"] == "ilink"

@patch("requests.post")
def test_send_serverchan(mock_post):
    mock_post.return_value.json.return_value = {"code": 0}
    assert send_serverchan("SCTxxx", "title", "content") is True
    args = mock_post.call_args
    assert "sctapi.ftqq.com/SCTxxx.send" in args[0][0]
    assert args[1]["data"]["text"] == "title"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report.send_text_ilink")
@patch("ashare_lab.paper.report.send_serverchan")
@patch("time.sleep")
def test_deliver_report_fallback(mock_sleep, mock_sc, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    mock_ilink.side_effect = Exception("err")
    mock_sc.return_value = True
    cfg = {"paper": {"report": {"fallback_service": "serverchan"}}}
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", cfg) == "sent"
    r = db_conn.execute("SELECT * FROM reports").fetchone()
    assert r["delivered_via"] == "serverchan"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report.send_text_ilink")
@patch("ashare_lab.paper.report.send_serverchan")
@patch("time.sleep")
def test_deliver_report_all_fail(mock_sleep, mock_sc, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    mock_ilink.side_effect = Exception("err")
    mock_sc.return_value = False
    cfg = {"paper": {"report": {"fallback_service": "serverchan"}}}
    assert deliver_report(db_conn, "2025-01-06", "simple", "text", cfg) == "failed"
    r = db_conn.execute("SELECT * FROM reports").fetchone()
    assert r["delivery_status"] == "failed"

def test_deliver_report_saves_always(db_conn):
    with patch("ashare_lab.paper.report._get_secret", side_effect=Exception("err")):
        deliver_report(db_conn, "2025-01-06", "simple", "text", {"paper":{}})
    r = db_conn.execute("SELECT * FROM reports").fetchone()
    assert r is not None
    assert r["delivery_status"] == "failed"

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report.send_text_ilink")
@patch("asyncio.sleep")
def test_deliver_report_splits(mock_sleep, mock_ilink, mock_sec, db_conn):
    mock_sec.return_value = "token"
    deliver_report(db_conn, "2025-01-06", "simple", "x" * 5000, {"paper":{"report":{"ilink_max_message_length":4000}}})
    assert mock_ilink.call_count == 2
    mock_sleep.assert_called()

@patch("ashare_lab.paper.report._get_secret")
@patch("ashare_lab.paper.report.send_text_ilink")
@patch("asyncio.sleep")
@patch("time.sleep")
def test_deliver_report_retry_resumes_from_sent(mock_tsleep, mock_asleep, mock_ilink, mock_sec, db_conn):
    """F2: retry must resume from last successfully sent chunk, not re-send from 0."""
    mock_sec.return_value = "token"
    call_count = [0]
    sent_chunk_indices = []

    async def ilink_side_effect(session, token, chat_id, text, timeout=15):
        call_count[0] += 1
        # Fail on the 3rd call (chunk 2 of first attempt)
        if call_count[0] == 3:
            raise Exception("transient network error")
        sent_chunk_indices.append(text)

    mock_ilink.side_effect = ilink_side_effect

    # 3 chunks: force small max_length so "x"*5000 splits into 3
    cfg = {"paper": {"report": {"ilink_max_message_length": 2000}}}
    result = deliver_report(db_conn, "2025-01-06", "simple", "x" * 5000, cfg)
    assert result == "sent"
    # Total calls: 2 (first attempt chunks 0,1) + 1 fail (chunk 2) + 1 retry (chunk 2 only) = 4
    assert mock_ilink.call_count == 4, f"expected 4 calls (2 ok + 1 fail + 1 retry), got {mock_ilink.call_count}"
    # Verify no duplicate: each chunk text should appear exactly once in sent_chunk_indices
    # Chunks 0 and 1 were sent once in first attempt, chunk 2 failed then succeeded on retry
    assert len(sent_chunk_indices) == 3, f"expected 3 successful sends, got {len(sent_chunk_indices)}"


# Integration
def test_generate_and_send_report_dry_run(populated_db, paper_config):
    paper_config["report"] = {"detailed_triggers": {"daily_loss_pct": 0.02, "min_trade_count": 5}}
    rc = generate_and_send_report("2025-01-06", populated_db, {"paper": paper_config}, dry_run=True)
    assert rc == 0
    r = populated_db.execute("SELECT * FROM reports").fetchone()
    assert r["delivery_status"] == "dry_run"

@patch("ashare_lab.paper.report.deliver_report")
def test_generate_and_send_report_simple(mock_deliver, db_conn, paper_config):
    mock_deliver.return_value = "sent"
    paper_config["report"] = {"detailed_triggers": {"daily_loss_pct": 0.02, "min_trade_count": 5}}
    from ashare_lab.paper.ledger import record_nav
    record_nav(db_conn, "2024-12-30", 300_000.0, 0.0, 300_000.0, None, None, None, None)
    record_nav(db_conn, "2025-01-06", 50000.0, 250000.0, 300000.0, None, None, 3000.0, 5000.0) # total nav 300,000 = 0% return
    rc = generate_and_send_report("2025-01-06", db_conn, {"paper": paper_config})
    assert rc == 0
    mock_deliver.assert_called()
    assert mock_deliver.call_args[0][2] == "simple"


def test_cli_report_args():
    from ashare_lab.cli import main
    import sys
    with patch.object(sys, "argv", ["ashare-lab", "paper", "report", "--date", "2025-01-06", "--detailed", "--dry-run"]):
        with patch("ashare_lab.cli.cmd_paper_report") as mock_cmd:
            mock_cmd.return_value = 0
            main()
            mock_cmd.assert_called_once()
            args = mock_cmd.call_args[0][0]
            assert args.date == "2025-01-06"
            assert args.detailed is True
            assert args.dry_run is True

def test_generate_and_send_report_no_nav_row(db_conn, paper_config):
    rc = generate_and_send_report("2025-01-06", db_conn, {"paper": paper_config})
    assert rc == 1
