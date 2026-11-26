"""newswire — the dated-news leg, borrowed from the news-pull skill.

`reddit-alpha` does not need its own news crawler: this workspace already has
one (`news-pull/scripts/news.py`, stdlib only, four tiers of sources, Google News
discovery). This module calls it, caches what comes back in the `news` table, and
keeps the one distinction that matters for citation — a Google News hit has a
usable headline and publisher but an unresolvable link, so it is stored with
`discovery=1` and must never be linked as a source.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import config as C
import store
import textproc as T

NEWS_PY = C.BASE / ".dsh" / "skills" / "news-pull" / "scripts" / "news.py"
CACHE_TTL = 1800          # seconds; headlines do not change faster than this


def available() -> bool:
    return NEWS_PY.exists()


def _run(args: list[str], timeout: int = 150) -> dict:
    if not NEWS_PY.exists():
        return {"items": [], "error": f"news.py not found at {NEWS_PY}"}
    cmd = [sys.executable, str(NEWS_PY), *args, "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(C.BASE))
    except subprocess.TimeoutExpired:
        return {"items": [], "error": "news.py timed out"}
    if proc.returncode != 0:
        return {"items": [], "error": (proc.stderr or proc.stdout or "")[-400:]}
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"items": [], "error": f"news.py returned non-JSON: {proc.stdout[:200]!r}"}
    return {"items": doc.get("items") or [], "error": None}


def _fresh(conn, key: str) -> bool:
    raw = store.meta_get(conn, f"news_fetch:{key}")
    if not raw:
        return False
    try:
        return (time.time() - float(raw)) < CACHE_TTL
    except ValueError:
        return False


def _stamp(conn, key: str) -> None:
    store.meta_set(conn, f"news_fetch:{key}", str(time.time()))
    conn.commit()


def stock(conn, tickers: list[str], *, hours: float | None = None,
          force: bool = False, limit: int = 25) -> dict:
    """Headlines for named tickers, cached. Returns per-ticker lists."""
    hours = hours or C.GEN_NEWS_HOURS
    tickers = [t.upper().lstrip("$") for t in tickers if t and t.strip()]
    if not tickers:
        return {"by_ticker": {}, "error": "no tickers"}
    key = "stock:" + ",".join(sorted(tickers)) + f":{int(hours)}"
    if force or not _fresh(conn, key):
        out = _run(["stock", *tickers, "--hours", str(hours), "--limit", str(limit)])
        if out["error"]:
            return {"by_ticker": {}, "error": out["error"]}
        by_ticker: dict[str, list[dict]] = {}
        for it in out["items"]:
            # news.py tags each item with the feed it came from: gnews:NVDA:q
            feed = it.get("feed") or ""
            for t in tickers:
                if f":{t}:" in feed or feed.endswith(f":{t}"):
                    store.upsert_news(conn, [it], ticker=t)
                    by_ticker.setdefault(t, []).append(_shape(it, t))
        _stamp(conn, key)
        conn.commit()
        return {"by_ticker": by_ticker, "error": None}
    # cached path: read the table rather than the network
    by_ticker = {}
    for t in tickers:
        rows = conn.execute(
            """SELECT title, url, publisher, published_at, discovery FROM news
               WHERE ticker=? ORDER BY COALESCE(published_at,'') DESC LIMIT ?""",
            (t, limit)).fetchall()
        by_ticker[t] = [dict(r) for r in rows]
    return {"by_ticker": by_ticker, "error": None, "cached": True}


def query(conn, q: str, *, hours: float | None = None, force: bool = False,
          limit: int = 25, topic: str | None = None) -> dict:
    """Free-text newsflow — used when an idea is a theme rather than a ticker."""
    hours = hours or C.GEN_NEWS_HOURS
    key = "query:" + q[:80] + f":{int(hours)}"
    if force or not _fresh(conn, key):
        out = _run(["query", q, "--hours", str(hours), "--limit", str(limit)])
        if out["error"]:
            return {"items": [], "error": out["error"]}
        store.upsert_news(conn, out["items"], topic=topic or q[:60])
        _stamp(conn, key)
        conn.commit()
        return {"items": [_shape(it, None) for it in out["items"]], "error": None}
    rows = conn.execute(
        """SELECT title, url, publisher, published_at, discovery FROM news
           WHERE topic=? ORDER BY COALESCE(published_at,'') DESC LIMIT ?""",
        (topic or q[:60], limit)).fetchall()
    return {"items": [dict(r) for r in rows], "error": None, "cached": True}


def _shape(it: dict, ticker: str | None) -> dict:
    return {
        "title": it.get("title"),
        "url": it.get("url"),
        "publisher": it.get("source") or it.get("publisher"),
        "published_at": it.get("published_utc") or it.get("published_at"),
        "discovery": bool(it.get("discovery")),
        "ticker": ticker,
    }


def recent(conn, *, hours: float = 72.0, tickers: list[str] | None = None,
           limit: int = 40) -> list[dict]:
    """What is already in the cache — no network."""
    since = T.iso(time.time() - hours * 3600)
    if tickers:
        names = [t.upper() for t in tickers]
        rows = conn.execute(
            f"""SELECT title, url, publisher, published_at, ticker, discovery FROM news
                WHERE ticker IN ({','.join('?' * len(names))})
                ORDER BY COALESCE(published_at,'') DESC LIMIT ?""",
            [*names, limit]).fetchall()
    else:
        rows = conn.execute(
            """SELECT title, url, publisher, published_at, ticker, discovery FROM news
               WHERE COALESCE(published_at,'') >= ? ORDER BY published_at DESC LIMIT ?""",
            (since, limit)).fetchall()
    return [dict(r) for r in rows]
