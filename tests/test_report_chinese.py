"""Tests for Chinese mobile-first report format (D-72-06)."""

from dataclasses import replace

import pytest

from ashare_lab.paper.report import ReportData, format_chinese_report


@pytest.fixture
def base_report():
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
        trades=[
            {
                "symbol": "SZ000001",
                "name": "PingAn",
                "side": "buy",
                "qty": 100,
                "price": 10.0,
                "total_fee": 3.0,
            },
            {
                "symbol": "SZ000002",
                "name": "WanKe",
                "side": "sell",
                "qty": 200,
                "price": 20.0,
                "total_fee": 5.0,
            },
        ],
        positions=[
            {
                "symbol": "SZ000001",
                "name": "PingAn",
                "qty": 100,
                "market_value": 1000.0,
                "unrealized_pnl": 100.0,
                "weight": 10.0,
                "daily_change_pct": 0.05,
            },
            {
                "symbol": "SZ000002",
                "name": "WanKe",
                "qty": 200,
                "market_value": 4000.0,
                "unrealized_pnl": -50.0,
                "weight": 40.0,
                "daily_change_pct": -0.03,
            },
            {
                "symbol": "SZ000003",
                "name": "ZhongXin",
                "qty": 300,
                "market_value": 5000.0,
                "unrealized_pnl": 0.0,
                "weight": 50.0,
                "daily_change_pct": 0.0,
            },
        ],
        pending_orders=[
            {"symbol": "SZ000004", "name": "GuoMao", "side": "buy", "qty": 500},
            {"symbol": "SZ000005", "name": "ZhaoShang", "side": "sell", "qty": 300},
        ],
        trade_count=2,
        risk_status={
            "buying_halted": False,
            "is_soft_reduced": False,
            "sell_order_count": 0,
            "cooldown_count": 0,
            "drawdown_halted": False,
            "regime_halted": False,
        },
        industry_distribution={"Bank": 1, "Real Estate": 1},
    )


def test_header_date_and_emoji(base_report):
    text = format_chinese_report(base_report)
    lines = text.split("\n")
    assert "2025-01-06" in lines[0]
    assert "\U0001f4ca" in lines[0]  # chart emoji
    assert "ashare-lab" in lines[0]


def test_nav_section(base_report):
    text = format_chinese_report(base_report)
    assert "\U0001f4b0" in text  # money bag
    assert "300,000.00" in text or "300000" in text
    assert "+1.50%" in text
    assert "10.00%" in text  # cumulative
    assert "5.00%" in text  # drawdown
    assert "50.0%" in text  # cash ratio


def test_excess_vs_csi1000(base_report):
    text = format_chinese_report(base_report)
    excess = base_report.daily_return_pct - base_report.benchmark_return_pct
    assert f"{excess:+.2f}%" in text


def test_trades_chinese_buy_verb(base_report):
    text = format_chinese_report(base_report)
    assert "买" in text  # buy verb
    assert "PingAn" in text
    assert "100" in text


def test_trades_chinese_sell_verb(base_report):
    text = format_chinese_report(base_report)
    assert "卖" in text  # sell verb
    assert "WanKe" in text


def test_positions_sorted_by_change_desc(base_report):
    text = format_chinese_report(base_report)
    pos_lines = []
    in_pos = False
    for line in text.split("\n"):
        if "持仓" in line:  # positions header
            in_pos = True
            continue
        if in_pos and line.strip() and not line.startswith("━"):
            if "明日" in line or "风控" in line or "✅" in line or "⚠" in line:
                break
            pos_lines.append(line)
    # PingAn (+5%) should come before ZhongXin (0%) before WanKe (-3%)
    names_in_order = []
    for pl in pos_lines:
        for name in ["PingAn", "ZhongXin", "WanKe"]:
            if name in pl:
                names_in_order.append(name)
    assert names_in_order == ["PingAn", "ZhongXin", "WanKe"]


def test_positive_change_up_triangle(base_report):
    text = format_chinese_report(base_report)
    # PingAn has +5% change, should have up-triangle U+1F53A
    for line in text.split("\n"):
        if "PingAn" in line:
            assert "\U0001f53a" in line
            assert "5.00%" in line
            break
    else:
        pytest.fail("PingAn line not found")


def test_negative_change_down_triangle(base_report):
    text = format_chinese_report(base_report)
    for line in text.split("\n"):
        if "WanKe" in line and "持仓" not in line:
            assert "\U0001f53b" in line
            assert "3.00%" in line
            break
    else:
        pytest.fail("WanKe position line not found")


def test_zero_change_no_indicator(base_report):
    text = format_chinese_report(base_report)
    for line in text.split("\n"):
        if "ZhongXin" in line:
            assert "\U0001f53a" not in line
            assert "\U0001f53b" not in line
            break
    else:
        pytest.fail("ZhongXin line not found")


def test_pending_orders_buy_verb(base_report):
    text = format_chinese_report(base_report)
    assert "拟买" in text  # planned buy
    assert "GuoMao" in text


def test_pending_orders_sell_verb(base_report):
    text = format_chinese_report(base_report)
    assert "拟卖" in text  # planned sell
    assert "ZhaoShang" in text


def test_risk_normal_checkmark(base_report):
    text = format_chinese_report(base_report)
    assert "✅" in text  # checkmark
    assert "风控正常" in text


def test_risk_warning_emoji(base_report):
    rd = replace(
        base_report,
        risk_status={
            **base_report.risk_status,
            "buying_halted": True,
            "drawdown_halted": True,
        },
    )
    text = format_chinese_report(rd)
    assert "⚠️" in text  # warning emoji


def test_empty_name_falls_back_to_code(base_report):
    rd = replace(
        base_report,
        trades=[
            {
                "symbol": "SZ000099",
                "name": "",
                "side": "buy",
                "qty": 100,
                "price": 5.0,
                "total_fee": 1.0,
            }
        ],
    )
    text = format_chinese_report(rd)
    assert "SZ000099" in text


def test_empty_trades_no_crash(base_report):
    rd = replace(base_report, trades=[], trade_count=0)
    text = format_chinese_report(rd)
    assert "ashare-lab" in text
    # no trades section header when empty
    assert "今日交易" not in text


def test_empty_positions_no_crash(base_report):
    rd = replace(base_report, positions=[])
    text = format_chinese_report(rd)
    assert "ashare-lab" in text
    assert "持仓分布" not in text


def test_sentiment_section_provided(base_report):
    sentiment = "sentiment vetoes:\n  SZ000001 | news | score -0.8"
    text = format_chinese_report(base_report, sentiment_section=sentiment)
    assert "sentiment" in text.lower()
    assert "SZ000001" in text


def test_sentiment_section_none_omits(base_report):
    text = format_chinese_report(base_report, sentiment_section=None)
    assert "sentiment" not in text.lower() or "veto" not in text.lower()
