"""T3 smoke tests: real eastmoney news fetch + real LLM scoring via OmniRoute.

Run on X500 only (needs pass store with api/omniroute).
All tests marked @pytest.mark.integration -- excluded from normal runs.
"""

from __future__ import annotations

import copy
import datetime
import sqlite3
import subprocess

import pytest

# Module-level skip: if OmniRoute API key is not available, skip everything.
try:
    subprocess.run(
        ["pass", "show", "api/omniroute"],
        capture_output=True, text=True, check=True, timeout=5,
    )
except Exception:
    pytest.skip("OmniRoute API key not available", allow_module_level=True)

from ashare_lab.config import load_config
from ashare_lab.paper.ledger import init_schema
from ashare_lab.paper.sentiment import (
    SentimentVetoResult,
    _build_stock_prompt,
    _call_llm_score,
    fetch_global_news,
    fetch_stock_news,
    run_sentiment_veto,
)


@pytest.fixture(scope="module")
def config():
    """Load baseline config with news_count=3 to minimize API calls."""
    cfg = copy.deepcopy(load_config())
    cfg["paper"]["sentiment"]["news_count"] = 3
    return cfg


@pytest.fixture(scope="module")
def sent_cfg(config):
    """Flat sentiment sub-config for internal function tests."""
    return config["paper"]["sentiment"]


@pytest.mark.integration
class TestSentimentSmoke:

    @pytest.mark.timeout(60)
    def test_fetch_stock_news_live(self, sent_cfg):
        articles = fetch_stock_news("000001", sent_cfg)
        assert isinstance(articles, list)
        assert len(articles) >= 1, "000001 (Ping An Bank) should always have news"
        for a in articles:
            assert "title" in a
            assert "content" in a
            assert "date" in a
            assert "media" in a
            assert isinstance(a["title"], str) and a["title"]

    @pytest.mark.timeout(60)
    def test_fetch_global_news_live(self, sent_cfg):
        articles = fetch_global_news(sent_cfg)
        assert isinstance(articles, list)
        assert len(articles) >= 1
        for a in articles:
            assert "title" in a
            assert "summary" in a
            assert "date" in a

    @pytest.mark.timeout(60)
    def test_llm_score_live(self, sent_cfg):
        prompt = _build_stock_prompt(
            "000001", "Ping An Bank",
            [{"title": "test", "content": "neutral news",
              "date": datetime.date.today().isoformat(), "media": "test"}],
        )
        score = _call_llm_score(prompt, sent_cfg)
        assert isinstance(score, int)
        assert -3 <= score <= 3

    @pytest.mark.timeout(120)
    def test_run_sentiment_veto_e2e(self, config):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_schema(conn)

        trade_date = datetime.date.today().isoformat()
        result = run_sentiment_veto(
            ["SZ000001"], trade_date,
            {"SZ000001": "banking"},
            conn, config,
            {"SZ000001": "Ping An Bank"},
        )

        assert isinstance(result, SentimentVetoResult)
        assert isinstance(result.vetoed_stocks, dict)
        assert isinstance(result.global_halted, bool)

        rows = conn.execute(
            "SELECT COUNT(*) FROM sentiment_scores WHERE trade_date = ?",
            (trade_date,),
        ).fetchone()
        assert rows[0] >= 1, "sentiment_scores should have cached rows"

        events = conn.execute(
            "SELECT COUNT(*) FROM sentiment_events WHERE trade_date = ?",
            (trade_date,),
        ).fetchone()
        assert events[0] >= 1, "sentiment_events should have logged events"

        conn.close()

    @pytest.mark.timeout(120)
    def test_run_sentiment_veto_no_stock_names(self, config):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_schema(conn)

        trade_date = datetime.date.today().isoformat()
        result = run_sentiment_veto(
            ["SZ000001"], trade_date,
            {"SZ000001": "banking"},
            conn, config,
            stock_names=None,
        )

        assert isinstance(result, SentimentVetoResult)
        assert isinstance(result.vetoed_stocks, dict)
        assert isinstance(result.global_halted, bool)

        conn.close()
