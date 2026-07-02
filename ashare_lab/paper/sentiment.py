"""Three-layer news sentiment veto for the paper trading engine.

Layers: per-stock, per-industry, global macro.  Each layer fetches
news from eastmoney, scores via deepseek v4-flash, and compares
against configurable thresholds.  Infrastructure failure at any
point fails open (no veto applied, buys proceed).

Public entry point: run_sentiment_veto().
"""

from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "SentimentVetoResult",
    "run_sentiment_veto",
    "apply_veto",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentVetoResult:
    """Outcome of the three-layer sentiment veto.

    Attributes:
        vetoed_stocks: symbol -> score for stocks blocked by any layer.
        vetoed_industries: industry name -> score for industries that
            triggered the industry-layer veto.
        global_halted: True if global macro score tripped the threshold.
        global_score: the raw global score, or None if not scored.
        scores_detail: list of {target, layer, score} dicts for monitoring.
    """

    vetoed_stocks: dict[str, float]
    vetoed_industries: dict[str, float]
    global_halted: bool
    global_score: float | None
    scores_detail: list[dict]


_EMPTY_RESULT = SentimentVetoResult(
    vetoed_stocks={},
    vetoed_industries={},
    global_halted=False,
    global_score=None,
    scores_detail=[],
)


# ---------------------------------------------------------------------------
# JSONP / HTML helpers
# ---------------------------------------------------------------------------


def _parse_jsonp(text: str, callback: str = "cb") -> dict:
    """Strip JSONP callback wrapper and parse the inner JSON.

    Never raises -- returns {} on any parse failure (fail-open).
    """
    try:
        prefix = f"{callback}("
        if text.startswith(prefix):
            return json.loads(text[len(prefix):-1])
        idx = text.index("(")
        return json.loads(text[idx + 1 : text.rindex(")")])
    except Exception:
        return {}


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# Secret retrieval (intentional duplication from report.py --
# independent module, avoids circular imports; keep in sync if
# the pass key path changes)
# ---------------------------------------------------------------------------


def _get_secret(key: str) -> str:
    r = subprocess.run(
        ["pass", "show", key],
        capture_output=True, text=True, check=True, timeout=5,
    )
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Rate-limited HTTP
# ---------------------------------------------------------------------------


def _rate_limited_get(url, params, headers, config, timeout=15):
    """GET with sleep before request to respect eastmoney rate limits.

    config is the flat sentiment sub-config (has rate_limit_base etc.).
    """
    import requests  # noqa: PLC0415

    base = config.get("rate_limit_base", 1.0)
    jitter = config.get("rate_limit_jitter", 0.5)
    time.sleep(base + random.uniform(0, jitter))
    return requests.get(url, params=params, headers=headers, timeout=timeout)


# ---------------------------------------------------------------------------
# News fetching (three layers)
# ---------------------------------------------------------------------------

_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
_SEARCH_HEADERS = {
    "Referer": "https://so.eastmoney.com/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0",
}


def _build_search_params(keyword: str, page_size: int) -> dict:
    """Build params dict for search-api-web JSONP endpoint."""
    inner_param = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": page_size,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    return {
        "cb": "cb",
        "param": json.dumps(inner_param, ensure_ascii=False),
        "_": "1",
    }


def fetch_stock_news(symbol: str, config: dict) -> list[dict]:
    """Fetch per-stock news from eastmoney search-api-web.

    config is the flat sentiment sub-config.  Returns list of
    {title, content, date, media} dicts.  Empty list on any error.
    """
    try:
        code = symbol[-6:] if len(symbol) > 6 else symbol
        page_size = config.get("news_count", 10)
        params = _build_search_params(code, page_size)
        resp = _rate_limited_get(
            _SEARCH_URL, params, _SEARCH_HEADERS, config,
        )
        data = _parse_jsonp(resp.text, "cb")
        articles = data.get("result", {}).get("cmsArticleWebOld", [])
        return [
            {
                "title": item.get("title", ""),
                "content": _strip_html(item.get("content", "")),
                "date": item.get("date", ""),
                "media": item.get("mediaName", ""),
            }
            for item in articles
        ]
    except Exception:
        logger.warning("Failed to fetch stock news for %s", symbol, exc_info=True)
        return []


def fetch_industry_news(industry_name: str, config: dict) -> list[dict]:
    """Fetch per-industry news from eastmoney search-api-web.

    Same endpoint as stock news but with industry Chinese name as keyword.
    """
    try:
        page_size = config.get("news_count", 10)
        params = _build_search_params(industry_name, page_size)
        resp = _rate_limited_get(
            _SEARCH_URL, params, _SEARCH_HEADERS, config,
        )
        data = _parse_jsonp(resp.text, "cb")
        articles = data.get("result", {}).get("cmsArticleWebOld", [])
        return [
            {
                "title": item.get("title", ""),
                "content": _strip_html(item.get("content", "")),
                "date": item.get("date", ""),
                "media": item.get("mediaName", ""),
            }
            for item in articles
        ]
    except Exception:
        logger.warning(
            "Failed to fetch industry news for %s", industry_name, exc_info=True,
        )
        return []


def fetch_global_news(config: dict) -> list[dict]:
    """Fetch global macro news from eastmoney np-weblist.

    Returns list of {title, summary, date} dicts.
    """
    try:
        import requests  # noqa: PLC0415

        url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
        params = {
            "client": "web",
            "biz": "web_724",
            "fastColumn": "102",
            "sortEnd": "",
            "pageSize": "20",
            "req_trace": "1",
        }
        base = config.get("rate_limit_base", 1.0)
        jitter = config.get("rate_limit_jitter", 0.5)
        time.sleep(base + random.uniform(0, jitter))
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        news_list = data.get("data", {}).get("fastNewsList", [])
        return [
            {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "date": item.get("showTime", ""),
            }
            for item in news_list
        ]
    except Exception:
        logger.warning("Failed to fetch global news", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_stock_prompt(symbol: str, name: str, articles: list[dict]) -> str:
    name = name or symbol
    parts = [
        f"You are an A-share market news analyst. "
        f"Rate the following news articles' combined impact on "
        f"stock {symbol} ({name}). "
        f"Score from -3 (extremely negative) to +3 (extremely positive). "
        f"A-share context: "
        f"'li kong chu jin' (bearish exhaustion) = neutral not negative. "
        f"'zi chan chong zu' (asset restructuring rumor) = neutral. "
        f"'bai tuo fu mian' (escaping negative territory) = neutral. "
        f"Policy tightening on the specific sector = negative. "
        f"Routine regulatory filing = neutral.\n\n"
    ]
    for a in articles:
        content = _strip_html(a.get("content", ""))[:200]
        parts.append(f"- {a.get('title', '')}: {content}\n")
    parts.append("\nReply with ONLY a single integer from -3 to +3, nothing else.")
    return "".join(parts)


def _build_industry_prompt(industry_name: str, articles: list[dict]) -> str:
    parts = [
        f"You are an A-share market news analyst. "
        f"Rate the following news articles' combined impact on "
        f"the {industry_name} sector. "
        f"Score from -3 (extremely negative) to +3 (extremely positive). "
        f"A-share context: "
        f"'li kong chu jin' (bearish exhaustion) = neutral not negative. "
        f"'zi chan chong zu' (asset restructuring rumor) = neutral. "
        f"'bai tuo fu mian' (escaping negative territory) = neutral. "
        f"Policy tightening on the specific sector = negative. "
        f"Routine regulatory filing = neutral.\n\n"
    ]
    for a in articles:
        content = _strip_html(a.get("content", ""))[:200]
        parts.append(f"- {a.get('title', '')}: {content}\n")
    parts.append("\nReply with ONLY a single integer from -3 to +3, nothing else.")
    return "".join(parts)


def _build_global_prompt(articles: list[dict]) -> str:
    parts = [
        "You are an A-share market policy analyst. "
        "Rate the following policy and macro news' combined impact "
        "on the A-share market overall. "
        "Score from -3 (extremely negative) to +3 (extremely positive). "
        "A-share context: "
        "'li kong chu jin' (bearish exhaustion) = neutral not negative. "
        "'zi chan chong zu' (asset restructuring rumor) = neutral. "
        "'bai tuo fu mian' (escaping negative territory) = neutral. "
        "Policy tightening = negative. "
        "Routine regulatory filing = neutral.\n\n"
    ]
    for a in articles:
        # global news uses summary, not content
        parts.append(f"- {a.get('title', '')}: {a.get('summary', '')}\n")
    parts.append("\nReply with ONLY a single integer from -3 to +3, nothing else.")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def _parse_score(response_text: str) -> int:
    """Extract integer score from deepseek response.

    Returns 0 on parse failure (neutral = fail-open).
    """
    match = re.search(r"(?<!\d)[+-]?[0-3](?!\d)", response_text)
    if match:
        val = int(match.group())
        if -3 <= val <= 3:
            return val
    return 0


# ---------------------------------------------------------------------------
# LLM scoring (OpenAI chat/completions compatible)
# ---------------------------------------------------------------------------


_LLM_RESERVED_KEYS = frozenset({"model", "messages"})


def _call_llm_score(prompt: str, config: dict) -> int:
    """Call an OpenAI-compatible LLM API for sentiment scoring.

    Provider, model, and credentials are read from config (the flat
    sentiment sub-config).  Any provider that speaks the OpenAI
    chat/completions wire format works: DeepSeek, MiMo, Kimi,
    MiniMax, GLM, OpenAI, Gemini, Azure OpenAI, Bedrock.

    Returns integer score -3..+3, or 0 on any failure (fail-open).
    """
    try:
        import requests  # noqa: PLC0415

        # Read config with backward-compat fallback to old deepseek_* keys.
        base_url = config.get(
            "llm_base_url", config.get("deepseek_base_url", "https://api.deepseek.com"),
        )
        model = config.get(
            "llm_model", config.get("deepseek_model", "deepseek-v4-flash"),
        )
        api_key_pass = config.get("llm_api_key_pass", "ashare/deepseek-api-key")
        timeout = config.get(
            "llm_timeout", config.get("deepseek_timeout", 30),
        )
        extra_body = config.get("llm_extra_body", {})

        # Validate base_url scheme.
        if not base_url.startswith(("http://", "https://")):
            logger.warning("llm_base_url missing scheme: %s", base_url)
            return 0

        # Guard against non-dict or reserved-key collisions in extra_body.
        if not isinstance(extra_body, dict):
            logger.warning("llm_extra_body is not a dict, ignoring")
            extra_body = {}
        elif extra_body:
            collisions = _LLM_RESERVED_KEYS & set(extra_body)
            if collisions:
                    logger.warning(
                        "llm_extra_body contains reserved keys %s, dropping them",
                        collisions,
                    )
                    extra_body = {
                        k: v for k, v in extra_body.items()
                        if k not in _LLM_RESERVED_KEYS
                    }

        api_key = _get_secret(api_key_pass)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **extra_body,
        }
        url = f"{base_url.rstrip('/')}/chat/completions"
        logger.debug("llm scoring: %s model=%s", url, model)
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()

        # Defensive response parsing -- providers may return unexpected shapes.
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            logger.warning("llm response missing 'choices': %s", str(data)[:200])
            return 0
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            logger.warning("llm response empty content: %s", str(data)[:200])
            return 0
        return _parse_score(content)
    except Exception:
        logger.warning("llm scoring failed", exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------


def _check_cache(conn, trade_date: str, layer: str, target: str):
    """Return cached score or None if not cached."""
    row = conn.execute(
        "SELECT score FROM sentiment_scores "
        "WHERE trade_date = ? AND layer = ? AND target = ?",
        (trade_date, layer, target),
    ).fetchone()
    if row is not None:
        return float(row["score"])
    return None


def _write_cache(conn, trade_date: str, layer: str, target: str,
                 score: float, news_count: int) -> None:
    """Write score to cache.  Fail-open on write errors."""
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sentiment_scores "
            "(trade_date, layer, target, score, news_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (trade_date, layer, target, score, news_count),
        )
    except Exception:
        logger.warning(
            "Failed to cache sentiment score for %s/%s", layer, target,
            exc_info=True,
        )


def _log_event(conn, trade_date: str, event_type: str, layer: str,
               target: str, score, detail) -> None:
    """Log a sentiment event.  Fail-open on write errors."""
    try:
        conn.execute(
            "INSERT INTO sentiment_events "
            "(trade_date, event_type, layer, target, score, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (trade_date, event_type, layer, target, score, detail),
        )
    except Exception:
        logger.warning(
            "Failed to log sentiment event %s/%s", event_type, target,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Orchestrated scoring across all three layers
# ---------------------------------------------------------------------------


def score_news(
    buy_syms: list[str],
    industry_map: dict[str, str],
    trade_date: str,
    conn,
    config: dict,
    stock_names: dict[str, str],
) -> dict:
    """Score news for all three layers.  Returns dict with keys
    stock_scores, industry_scores, global_score.

    config is the flat sentiment sub-config.
    """
    stock_scores: dict[str, float] = {}
    industry_scores: dict[str, float] = {}

    # -- per-stock layer --
    # cache key uses sym as-is (with SZ/SH prefix from pipeline);
    # fetch_stock_news normalizes to 6-digit code for the API call
    for sym in buy_syms:
        cached = _check_cache(conn, trade_date, "stock", sym)
        if cached is not None:
            stock_scores[sym] = cached
            continue
        articles = fetch_stock_news(sym, config)
        if not articles:
            stock_scores[sym] = 0
            _write_cache(conn, trade_date, "stock", sym, 0, 0)
            continue
        prompt = _build_stock_prompt(
            sym, stock_names.get(sym, sym), articles,
        )
        score = _call_llm_score(prompt, config)
        stock_scores[sym] = score
        _write_cache(conn, trade_date, "stock", sym, score, len(articles))

    # -- per-industry layer --
    unique_industries = {
        i for i in (industry_map.get(s) for s in buy_syms) if i
    }
    for ind in unique_industries:
        cached = _check_cache(conn, trade_date, "industry", ind)
        if cached is not None:
            industry_scores[ind] = cached
            continue
        articles = fetch_industry_news(ind, config)
        if not articles:
            industry_scores[ind] = 0
            _write_cache(conn, trade_date, "industry", ind, 0, 0)
            continue
        prompt = _build_industry_prompt(ind, articles)
        score = _call_llm_score(prompt, config)
        industry_scores[ind] = score
        _write_cache(conn, trade_date, "industry", ind, score, len(articles))

    # -- global layer --
    global_score = None
    cached = _check_cache(conn, trade_date, "global", "global")
    if cached is not None:
        global_score = cached
    else:
        articles = fetch_global_news(config)
        if articles:
            prompt = _build_global_prompt(articles)
            global_score = float(_call_llm_score(prompt, config))
            _write_cache(
                conn, trade_date, "global", "global",
                global_score, len(articles),
            )

    return {
        "stock_scores": stock_scores,
        "industry_scores": industry_scores,
        "global_score": global_score,
    }


# ---------------------------------------------------------------------------
# Veto logic
# ---------------------------------------------------------------------------


def apply_veto(
    buy_syms: list[str],
    scores: dict,
    industry_map: dict[str, str],
    config: dict,
) -> SentimentVetoResult:
    """Apply three-layer veto thresholds to buy list.

    config is the flat sentiment sub-config.
    """
    stock_threshold = config.get("stock_threshold", -2)
    industry_threshold = config.get("industry_threshold", -2)
    global_threshold = config.get("global_threshold", -3)

    stock_scores = scores.get("stock_scores", {})
    industry_scores = scores.get("industry_scores", {})
    global_score = scores.get("global_score")

    vetoed_stocks: dict[str, float] = {}
    vetoed_industries: dict[str, float] = {}

    # global check (None = no news fetched, treat as neutral)
    global_halted = (
        global_score is not None and global_score <= global_threshold
    )

    # industry check -- veto all stocks in that industry
    for ind_name, ind_score in industry_scores.items():
        if ind_score <= industry_threshold:
            vetoed_industries[ind_name] = ind_score
            for sym in buy_syms:
                if industry_map.get(sym) == ind_name:
                    vetoed_stocks[sym] = ind_score

    # per-stock check
    for sym, sym_score in stock_scores.items():
        if sym_score <= stock_threshold and sym not in vetoed_stocks:
            vetoed_stocks[sym] = sym_score

    # scores_detail for monitoring
    scores_detail = []
    for sym, s in stock_scores.items():
        scores_detail.append({"target": sym, "layer": "stock", "score": s})
    for ind, s in industry_scores.items():
        scores_detail.append({"target": ind, "layer": "industry", "score": s})
    if global_score is not None:
        scores_detail.append(
            {"target": "global", "layer": "global", "score": global_score}
        )

    return SentimentVetoResult(
        vetoed_stocks=vetoed_stocks,
        vetoed_industries=vetoed_industries,
        global_halted=global_halted,
        global_score=global_score,
        scores_detail=scores_detail,
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_sentiment_veto(
    buy_syms: list[str],
    trade_date: str,
    industry_map: dict[str, str],
    conn,
    config: dict,
    stock_names: dict[str, str] | None = None,
) -> SentimentVetoResult:
    """Run the full sentiment veto pipeline: fetch, score, veto.

    Fails open on any error -- infrastructure failure must not block buys.
    config is the FULL nested config (has config["paper"]["sentiment"]).
    """
    try:
        stock_names = stock_names or {}
        sent_cfg = config.get("paper", {}).get("sentiment", {})

        if not sent_cfg.get("enabled", False):
            return _EMPTY_RESULT

        scores = score_news(
            buy_syms, industry_map, trade_date, conn, sent_cfg, stock_names,
        )
        result = apply_veto(buy_syms, scores, industry_map, sent_cfg)

        # log veto/pass events for the D7 report and Phase 6 monitoring
        for ind_name, ind_score in result.vetoed_industries.items():
            _log_event(
                conn, trade_date, "veto", "industry", ind_name, ind_score, None,
            )

        # log per-stock vetoes that are not solely from industry propagation
        for sym, sym_score in result.vetoed_stocks.items():
            sym_industry = industry_map.get(sym)
            if sym_industry not in result.vetoed_industries:
                _log_event(
                    conn, trade_date, "veto", "stock", sym, sym_score, None,
                )

        if result.global_halted:
            _log_event(
                conn, trade_date, "veto", "global", "global",
                result.global_score, None,
            )

        for sym in buy_syms:
            if sym not in result.vetoed_stocks:
                _log_event(
                    conn, trade_date, "pass", "stock", sym, None, None,
                )

        return result

    except Exception:
        logger.warning(
            "Sentiment veto failed, continuing without veto", exc_info=True,
        )
        return _EMPTY_RESULT
