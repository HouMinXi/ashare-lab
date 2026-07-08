"""Unit tests for the three-layer sentiment veto module.

All network calls mocked.  Cache tests use in-memory SQLite.
Config convention: run_sentiment_veto tests use wrapped config
{"paper": {"sentiment": ...}}.  Internal function tests use
flat sentiment sub-config directly.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from ashare_lab.paper.ledger import init_schema
from ashare_lab.paper.sentiment import (
    SentimentVetoResult,
    _check_cache,
    _log_event,
    _parse_jsonp,
    _parse_score,
    _strip_html,
    _write_cache,
    apply_veto,
    run_sentiment_veto,
    score_news,
)


# -------------------------------------------------------------------
# _parse_jsonp
# -------------------------------------------------------------------


@pytest.mark.unit
class TestParseJsonp:
    def test_standard_callback(self):
        assert _parse_jsonp('cb({"a": 1})', "cb") == {"a": 1}

    def test_fallback_on_callback_mismatch(self):
        result = _parse_jsonp('xyz({"b": 2})', "cb")
        assert result == {"b": 2}

    def test_nested_json(self):
        result = _parse_jsonp('cb({"x": {"y": [1, 2]}})', "cb")
        assert result == {"x": {"y": [1, 2]}}

    def test_invalid_json_returns_empty(self):
        assert _parse_jsonp("cb({bad json})", "cb") == {}

    def test_no_parens_returns_empty(self):
        assert _parse_jsonp("just plain text", "cb") == {}


# -------------------------------------------------------------------
# _parse_score
# -------------------------------------------------------------------


@pytest.mark.unit
class TestParseScore:
    def test_clean_integer(self):
        assert _parse_score("2") == 2

    def test_negative(self):
        assert _parse_score("-3") == -3

    def test_embedded_in_prose(self):
        assert _parse_score("The score is -2 based on analysis") == -2

    def test_positive_with_sign(self):
        assert _parse_score("+3") == 3

    def test_garbage_returns_zero(self):
        assert _parse_score("no number here") == 0

    def test_empty_string_returns_zero(self):
        assert _parse_score("") == 0


# -------------------------------------------------------------------
# _strip_html
# -------------------------------------------------------------------


@pytest.mark.unit
class TestStripHtml:
    def test_strips_em_tags(self):
        assert _strip_html("<em>text</em>") == "text"

    def test_strips_nested(self):
        assert _strip_html("<b><em>x</em></b>") == "x"

    def test_no_tags_unchanged(self):
        assert _strip_html("plain text") == "plain text"


# -------------------------------------------------------------------
# apply_veto (pure logic, no network)
# -------------------------------------------------------------------


@pytest.mark.unit
class TestApplyVeto:
    def test_empty_scores_no_veto(self, sentiment_config):
        result = apply_veto(
            ["SZ000001"], {"stock_scores": {}, "industry_scores": {}},
            {}, sentiment_config,
        )
        assert result.vetoed_stocks == {}
        assert result.global_halted is False

    def test_stock_below_threshold_vetoed(self, sentiment_config):
        scores = {"stock_scores": {"SZ000001": -3}, "industry_scores": {}}
        result = apply_veto(["SZ000001"], scores, {}, sentiment_config)
        assert "SZ000001" in result.vetoed_stocks

    def test_stock_at_threshold_vetoed(self, sentiment_config):
        # score == threshold (-2 <= -2) -> vetoed
        scores = {"stock_scores": {"SZ000001": -2}, "industry_scores": {}}
        result = apply_veto(["SZ000001"], scores, {}, sentiment_config)
        assert "SZ000001" in result.vetoed_stocks

    def test_stock_above_threshold_not_vetoed(self, sentiment_config):
        scores = {"stock_scores": {"SZ000001": -1}, "industry_scores": {}}
        result = apply_veto(["SZ000001"], scores, {}, sentiment_config)
        assert "SZ000001" not in result.vetoed_stocks

    def test_industry_below_threshold_vetoes_stock(self, sentiment_config):
        scores = {
            "stock_scores": {"SZ000001": 0},
            "industry_scores": {"banking": -3},
        }
        industry_map = {"SZ000001": "banking"}
        result = apply_veto(["SZ000001"], scores, industry_map, sentiment_config)
        assert "SZ000001" in result.vetoed_stocks
        assert "banking" in result.vetoed_industries

    def test_industry_veto_blocks_all_stocks_in_industry(self, sentiment_config):
        scores = {
            "stock_scores": {"SZ000001": 0, "SZ000002": 1},
            "industry_scores": {"banking": -3},
        }
        industry_map = {"SZ000001": "banking", "SZ000002": "banking"}
        result = apply_veto(
            ["SZ000001", "SZ000002"], scores, industry_map, sentiment_config,
        )
        assert "SZ000001" in result.vetoed_stocks
        assert "SZ000002" in result.vetoed_stocks

    def test_global_halt_at_threshold(self, sentiment_config):
        # global_threshold = -3, score = -3 -> halted
        scores = {
            "stock_scores": {}, "industry_scores": {},
            "global_score": -3,
        }
        result = apply_veto([], scores, {}, sentiment_config)
        assert result.global_halted is True

    def test_global_not_halted_above_threshold(self, sentiment_config):
        # global_threshold = -3, score = -2 -> not halted
        scores = {
            "stock_scores": {}, "industry_scores": {},
            "global_score": -2,
        }
        result = apply_veto([], scores, {}, sentiment_config)
        assert result.global_halted is False

    def test_global_none_not_halted(self, sentiment_config):
        scores = {
            "stock_scores": {}, "industry_scores": {},
            "global_score": None,
        }
        result = apply_veto([], scores, {}, sentiment_config)
        assert result.global_halted is False

    def test_or_logic_industry_vetoes_even_if_stock_ok(self, sentiment_config):
        # stock score OK but industry vetoes -> stock still vetoed
        scores = {
            "stock_scores": {"SZ000001": 1},
            "industry_scores": {"tech": -3},
        }
        industry_map = {"SZ000001": "tech"}
        result = apply_veto(["SZ000001"], scores, industry_map, sentiment_config)
        assert "SZ000001" in result.vetoed_stocks


# -------------------------------------------------------------------
# Cache (in-memory SQLite)
# -------------------------------------------------------------------


@pytest.mark.unit
class TestCache:
    def test_cache_miss_returns_none(self, sentiment_db):
        assert _check_cache(sentiment_db, "2025-01-06", "stock", "SZ000001") is None

    def test_write_then_hit(self, sentiment_db):
        _write_cache(sentiment_db, "2025-01-06", "stock", "SZ000001", -2.0, 5)
        assert _check_cache(sentiment_db, "2025-01-06", "stock", "SZ000001") == -2.0

    def test_overwrite_second_wins(self, sentiment_db):
        _write_cache(sentiment_db, "2025-01-06", "stock", "SZ000001", -1.0, 3)
        _write_cache(sentiment_db, "2025-01-06", "stock", "SZ000001", -3.0, 7)
        assert _check_cache(sentiment_db, "2025-01-06", "stock", "SZ000001") == -3.0


# -------------------------------------------------------------------
# _log_event
# -------------------------------------------------------------------


@pytest.mark.unit
class TestLogEvent:
    def test_event_logged(self, sentiment_db):
        _log_event(sentiment_db, "2025-01-06", "veto", "stock", "SZ000001", -2, None)
        rows = sentiment_db.execute(
            "SELECT * FROM sentiment_events WHERE target = ?", ("SZ000001",),
        ).fetchall()
        assert len(rows) == 1

    def test_multiple_events_recorded(self, sentiment_db):
        _log_event(sentiment_db, "2025-01-06", "veto", "stock", "SZ000001", -2, None)
        _log_event(sentiment_db, "2025-01-06", "pass", "stock", "SZ000002", 1, None)
        rows = sentiment_db.execute(
            "SELECT * FROM sentiment_events WHERE trade_date = ?",
            ("2025-01-06",),
        ).fetchall()
        assert len(rows) == 2


# -------------------------------------------------------------------
# _call_llm_score request construction (mocked HTTP, real logic)
# -------------------------------------------------------------------


@pytest.mark.unit
class TestCallLlmScoreWireFormat:
    """Verify URL construction, headers, payload, and guard logic."""

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-test-key")
    @patch("requests.post")
    def test_url_and_headers(self, mock_post, mock_secret, sentiment_config):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "2"}}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        score = _call_llm_score("test prompt", sentiment_config)

        assert score == 2
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:20129/v1/chat/completions"
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test-key"
        body = call_args[1]["json"]
        assert body["model"] == "auto/smart"
        assert body["messages"] == [{"role": "user", "content": "test prompt"}]
        mock_secret.assert_called_once_with("api/omniroute")

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-mimo")
    @patch("requests.post")
    def test_custom_provider(self, mock_post, mock_secret):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "-1"}}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = {
            "llm_base_url": "https://api.xiaomimimo.com/v1",
            "llm_model": "mimo-v2.5-pro",
            "llm_api_key_pass": "ashare/mimo-pro-key",
            "llm_timeout": 60,
            "llm_extra_body": {"max_tokens": 1024},
        }
        score = _call_llm_score("test", cfg)

        assert score == -1
        url = mock_post.call_args[0][0]
        assert url == "https://api.xiaomimimo.com/v1/chat/completions"
        body = mock_post.call_args[1]["json"]
        assert body["max_tokens"] == 1024
        assert body["model"] == "mimo-v2.5-pro"
        mock_secret.assert_called_once_with("ashare/mimo-pro-key")

    def test_bad_scheme_returns_zero(self, sentiment_config):
        from ashare_lab.paper.sentiment import _call_llm_score

        cfg = {**sentiment_config, "llm_base_url": "ftp://bad.example.com"}
        assert _call_llm_score("test", cfg) == 0

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-test")
    @patch("requests.post")
    def test_extra_body_reserved_keys_dropped(self, mock_post, mock_secret):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "0"}}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = {
            "llm_base_url": "http://localhost:20129/v1",
            "llm_model": "auto/smart",
            "llm_api_key_pass": "api/omniroute",
            "llm_timeout": 5,
            "llm_extra_body": {"model": "WRONG", "max_tokens": 512},
        }
        _call_llm_score("test", cfg)

        body = mock_post.call_args[1]["json"]
        assert body["model"] == "auto/smart"
        assert body["max_tokens"] == 512

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-test")
    @patch("requests.post")
    def test_missing_choices_returns_zero(self, mock_post, mock_secret):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "bad request"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = {
            "llm_base_url": "http://localhost:20129/v1",
            "llm_model": "auto/smart",
            "llm_api_key_pass": "api/omniroute",
            "llm_timeout": 5,
            "llm_extra_body": {},
        }
        assert _call_llm_score("test", cfg) == 0

    def test_missing_required_keys_returns_zero(self):
        from ashare_lab.paper.sentiment import _call_llm_score

        # Old deepseek_* keys no longer recognized -- missing llm_* = return 0.
        cfg = {"deepseek_model": "some-model", "deepseek_timeout": 10}
        assert _call_llm_score("test", cfg) == 0

        # Partial config -- missing llm_model.
        cfg2 = {"llm_base_url": "http://localhost:20129/v1"}
        assert _call_llm_score("test", cfg2) == 0

    def test_non_dict_extra_body_ignored(self, sentiment_config):
        from ashare_lab.paper.sentiment import _call_llm_score

        cfg = {**sentiment_config, "llm_extra_body": "not-a-dict"}
        with patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-t"), \
             patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "0"}}],
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            score = _call_llm_score("test", cfg)
            assert score == 0
            body = mock_post.call_args[1]["json"]
            assert set(body.keys()) == {"model", "messages"}

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-t")
    @patch("requests.post")
    def test_messages_key_in_extra_body_dropped(self, mock_post, mock_secret):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "2"}}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = {
            "llm_base_url": "http://localhost:20129/v1",
            "llm_model": "auto/smart",
            "llm_api_key_pass": "api/omniroute",
            "llm_timeout": 5,
            "llm_extra_body": {"messages": [{"role": "system", "content": "BAD"}]},
        }
        _call_llm_score("test prompt", cfg)
        body = mock_post.call_args[1]["json"]
        assert body["messages"] == [{"role": "user", "content": "test prompt"}]

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-t")
    @patch("requests.post")
    def test_empty_content_returns_zero(self, mock_post, mock_secret):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": ""}}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = {
            "llm_base_url": "http://localhost:20129/v1",
            "llm_model": "auto/smart",
            "llm_api_key_pass": "api/omniroute",
            "llm_timeout": 5,
            "llm_extra_body": {},
        }
        assert _call_llm_score("test", cfg) == 0

    @patch("ashare_lab.paper.sentiment._get_secret", return_value="sk-t")
    @patch("requests.post")
    def test_trailing_slash_normalized(self, mock_post, mock_secret):
        from ashare_lab.paper.sentiment import _call_llm_score

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "1"}}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = {
            "llm_base_url": "http://localhost:20129/v1/",
            "llm_model": "auto/smart",
            "llm_api_key_pass": "api/omniroute",
            "llm_timeout": 5,
            "llm_extra_body": {},
        }
        _call_llm_score("test", cfg)
        url = mock_post.call_args[0][0]
        assert url == "http://localhost:20129/v1/chat/completions"


# -------------------------------------------------------------------
# score_news (mocked network)
# -------------------------------------------------------------------


@pytest.mark.unit
class TestScoreNews:
    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-2)
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_stock_news")
    def test_stock_scoring(self, mock_fetch, mock_global, mock_score,
                           sentiment_config, sentiment_db):
        mock_fetch.return_value = [{"title": "bad news", "content": "loss"}]
        result = score_news(
            ["SZ000001"], {}, "2025-01-06", sentiment_db,
            sentiment_config, {"SZ000001": "Test Bank"},
        )
        assert result["stock_scores"]["SZ000001"] == -2
        mock_score.assert_called_once()

    @patch("ashare_lab.paper.sentiment._call_llm_score")
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    def test_cache_hit_skips_api(self, mock_global, mock_score,
                                 sentiment_config, sentiment_db):
        _write_cache(sentiment_db, "2025-01-06", "stock", "SZ000001", -1.0, 5)
        result = score_news(
            ["SZ000001"], {}, "2025-01-06", sentiment_db,
            sentiment_config, {},
        )
        assert result["stock_scores"]["SZ000001"] == -1.0
        mock_score.assert_not_called()

    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-2)
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_stock_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_industry_news")
    def test_industry_scoring(self, mock_ind_fetch, mock_stock_fetch,
                              mock_global, mock_score,
                              sentiment_config, sentiment_db):
        mock_ind_fetch.return_value = [{"title": "sector trouble", "content": "bad"}]
        industry_map = {"SZ000001": "banking"}
        result = score_news(
            ["SZ000001"], industry_map, "2025-01-06", sentiment_db,
            sentiment_config, {},
        )
        assert result["industry_scores"]["banking"] == -2

    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-3)
    @patch("ashare_lab.paper.sentiment.fetch_global_news")
    @patch("ashare_lab.paper.sentiment.fetch_stock_news", return_value=[])
    def test_global_scoring(self, mock_stock, mock_global, mock_score,
                            sentiment_config, sentiment_db):
        mock_global.return_value = [{"title": "crash", "summary": "bad"}]
        result = score_news(
            ["SZ000001"], {}, "2025-01-06", sentiment_db,
            sentiment_config, {},
        )
        assert result["global_score"] == -3.0

    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_stock_news", return_value=[])
    def test_empty_articles_score_zero(self, mock_fetch, mock_global,
                                       sentiment_config, sentiment_db):
        result = score_news(
            ["SZ000001"], {}, "2025-01-06", sentiment_db,
            sentiment_config, {},
        )
        assert result["stock_scores"]["SZ000001"] == 0


# -------------------------------------------------------------------
# Fail-open behavior
# -------------------------------------------------------------------


@pytest.mark.unit
class TestFailOpen:
    @patch("ashare_lab.paper.sentiment.score_news", side_effect=ConnectionError("down"))
    def test_fetch_error_returns_empty(self, mock_score, sentiment_config, sentiment_db):
        cfg = {"paper": {"sentiment": sentiment_config}}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        assert result.vetoed_stocks == {}
        assert result.global_halted is False

    @patch("ashare_lab.paper.sentiment.score_news", side_effect=RuntimeError("boom"))
    def test_runtime_error_returns_empty(self, mock_score, sentiment_config, sentiment_db):
        cfg = {"paper": {"sentiment": sentiment_config}}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        assert isinstance(result, SentimentVetoResult)
        assert result.vetoed_stocks == {}

    def test_disabled_config_returns_empty(self, sentiment_db):
        cfg = {"paper": {"sentiment": {"enabled": False}}}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        assert result.vetoed_stocks == {}
        assert result.global_halted is False

    def test_missing_config_key_returns_empty(self, sentiment_db):
        # no "paper" key at all -> KeyError caught by fail-open
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, {},
        )
        assert result.vetoed_stocks == {}


# -------------------------------------------------------------------
# run_sentiment_veto (full pipeline, mocked network)
# -------------------------------------------------------------------


@pytest.mark.unit
class TestRunSentimentVeto:
    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-3)
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_stock_news")
    def test_happy_path_veto(self, mock_stock, mock_global, mock_score,
                             sentiment_config, sentiment_db):
        mock_stock.return_value = [{"title": "scandal", "content": "fraud"}]
        cfg = {"paper": {"sentiment": sentiment_config}}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        assert "SZ000001" in result.vetoed_stocks

    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=1)
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_stock_news")
    def test_happy_path_pass(self, mock_stock, mock_global, mock_score,
                             sentiment_config, sentiment_db):
        mock_stock.return_value = [{"title": "good", "content": "growth"}]
        cfg = {"paper": {"sentiment": sentiment_config}}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        assert "SZ000001" not in result.vetoed_stocks
        assert result.global_halted is False

    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-3)
    @patch("ashare_lab.paper.sentiment.fetch_global_news")
    @patch("ashare_lab.paper.sentiment.fetch_stock_news", return_value=[])
    def test_global_halt_propagates(self, mock_stock, mock_global, mock_score,
                                    sentiment_config, sentiment_db):
        mock_global.return_value = [{"title": "crash", "summary": "systemic"}]
        cfg = {"paper": {"sentiment": sentiment_config}}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        assert result.global_halted is True

    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-3)
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_industry_news")
    @patch("ashare_lab.paper.sentiment.fetch_stock_news")
    def test_industry_veto_via_or_logic(self, mock_stock, mock_industry,
                                        mock_global, mock_score,
                                        sentiment_config, sentiment_db):
        mock_stock.return_value = [{"title": "ok", "content": "fine"}]
        mock_industry.return_value = [{"title": "sector crash", "content": "bad"}]
        cfg = {"paper": {"sentiment": sentiment_config}}
        industry_map = {"SZ000001": "banking"}
        result = run_sentiment_veto(
            ["SZ000001"], "2025-01-06", industry_map, sentiment_db, cfg,
        )
        # industry score -3 <= threshold -2 -> vetoed via industry
        assert "SZ000001" in result.vetoed_stocks
        assert "banking" in result.vetoed_industries

    @patch("ashare_lab.paper.sentiment._call_llm_score", return_value=-2)
    @patch("ashare_lab.paper.sentiment.fetch_global_news", return_value=[])
    @patch("ashare_lab.paper.sentiment.fetch_stock_news")
    def test_events_logged(self, mock_stock, mock_global, mock_score,
                           sentiment_config, sentiment_db):
        mock_stock.return_value = [{"title": "bad", "content": "loss"}]
        cfg = {"paper": {"sentiment": sentiment_config}}
        run_sentiment_veto(
            ["SZ000001"], "2025-01-06", {}, sentiment_db, cfg,
        )
        rows = sentiment_db.execute(
            "SELECT * FROM sentiment_events WHERE trade_date = ?",
            ("2025-01-06",),
        ).fetchall()
        assert len(rows) >= 1
