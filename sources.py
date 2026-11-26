"""sources — where Reddit records come from, behind one interface.

Three adapters, one contract:

  ArcticSource   the public archive at arctic-shift.photon-reddit.com. No key,
                 no login, live to within a few minutes on busy subreddits.
                 This is what ships enabled.
  OAuthSource    Reddit's own API. Needs a script app's client id and secret in
                 .env; once they are there it becomes the primary source with
                 no change to the store.
  FileSource     replay saved JSON (a dump, a fixture, a hand-exported thread).

Every adapter normalises to the same record shape, so `crawl` never learns
which one it is talking to. Records are deliberately narrow: fields Reddit's own
projection cannot return are derived (permalink, domain, is_self) or left NULL.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Iterable
from urllib.parse import urlparse

import config as C

try:                                        # preferred: TLS impersonation
    from curl_cffi import requests as _http
    _HAVE_CURL_CFFI = True
except Exception:                           # pragma: no cover
    import urllib.request as _urllib
    _HAVE_CURL_CFFI = False

# Projection lists verified against the archive's validator: anything absent
# here is rejected with a 400, so these are exactly the fields we may ask for.
POST_FIELDS = "id,subreddit,author,title,selftext,url,created_utc,link_flair_text,over_18,num_comments,score,author_flair_text"
COMMENT_FIELDS = "id,subreddit,author,created_utc,score,link_id,parent_id,body"


class SourceError(RuntimeError):
    """A source refused us or returned something unusable."""


class RateLimited(SourceError):
    """Back off; the caller should sleep and retry, not treat it as empty."""


class NotConfigured(SourceError):
    """The adapter needs credentials or files that are not present."""


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _domain_of(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return None
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def norm_post(raw: dict, subreddit: str | None = None) -> dict:
    pid = str(raw.get("id") or "").strip()
    sub = raw.get("subreddit") or subreddit or ""
    url = raw.get("url") or ""
    selftext = raw.get("selftext") or ""
    return {
        "id": pid,
        "subreddit": sub,
        "author": raw.get("author"),
        "author_flair": raw.get("author_flair_text"),
        "title": (raw.get("title") or "").strip(),
        "selftext": selftext,
        "url": url,
        "domain": raw.get("domain") or _domain_of(url),
        "permalink": f"/r/{sub}/comments/{pid}/",
        "created_utc": int(raw.get("created_utc") or 0),
        "flair": raw.get("link_flair_text"),
        "is_self": bool(raw.get("is_self")) or bool(selftext) or (
            "reddit.com" in (url or "") and bool(selftext)),
        "over_18": bool(raw.get("over_18")),
        "stickied": bool(raw.get("stickied") or raw.get("pinned")),
        "num_comments": int(raw.get("num_comments") or 0) or None,
        "score": int(raw.get("score") or 0),
        "upvote_ratio": raw.get("upvote_ratio"),
        "gilded": raw.get("gilded"),
    }


def norm_comment(raw: dict, subreddit: str | None = None) -> dict:
    cid = str(raw.get("id") or "").strip()
    link = str(raw.get("link_id") or "")
    post_id = link[3:] if link.startswith("t3_") else link
    sub = raw.get("subreddit") or subreddit or ""
    return {
        "id": cid,
        "post_id": post_id,
        "subreddit": sub,
        "author": raw.get("author"),
        "parent_id": raw.get("parent_id") or "",
        "body": raw.get("body") or "",
        "created_utc": int(raw.get("created_utc") or 0),
        "score": int(raw.get("score") or 0),
        "stickied": bool(raw.get("stickied")),
        "permalink": f"/r/{sub}/comments/{post_id}/comment/{cid}/",
    }


# --------------------------------------------------------------------------- #
# the source contract
# --------------------------------------------------------------------------- #
class Source:
    name = "base"
    label = "base"

    def fetch_posts(self, subreddit: str, *, after: int | None = None,
                    before: int | None = None, limit: int = C.ARCTIC_PAGE,
                    sort: str = "asc") -> list[dict]:
        raise NotImplementedError

    def fetch_comments(self, subreddit: str, *, after: int | None = None,
                       before: int | None = None, limit: int = C.ARCTIC_PAGE,
                       sort: str = "asc") -> list[dict]:
        raise NotImplementedError

    def fetch_thread(self, post_id: str, *, limit: int = 500) -> list[dict]:
        raise NotImplementedError

    def probe(self, subreddit: str, *, hours: float = 168.0) -> dict:
        """Recent activity, used to validate a subreddit name before trusting it."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# arctic-shift
# --------------------------------------------------------------------------- #
class ArcticSource(Source):
    name = "arctic"
    label = "arctic-shift.photon-reddit.com (public archive)"

    def __init__(self, base: str | None = None, pace: float | None = None,
                 timeout: int | None = None):
        self.base = (base or C.ARCTIC_BASE).rstrip("/")
        self.pace = C.ARCTIC_PACE if pace is None else pace
        self.timeout = timeout or C.ARCTIC_TIMEOUT
        self._lock = Lock()
        self._last = 0.0
        self._base_pace = float(self.pace)
        self.pace = float(self.pace)     # adapts upward when the archive asks us to slow down
        self.calls = 0
        self.bytes = 0
        self.retries = 0
        self.slowdowns = 0

    # -- plumbing ---------------------------------------------------------- #
    def _sleep(self) -> None:
        with self._lock:
            wait = self.pace - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _note_success(self) -> None:
        with self._lock:
            if self.pace > self._base_pace:
                self.pace = max(self._base_pace, self.pace * 0.90)

    def _note_pressure(self) -> None:
        """The archive said slow down: widen our own gap before it has to say it twice."""
        with self._lock:
            # Widening is a control loop, not a punishment. Capped at
            # ARCTIC_PACE_MAX, because an uncapped 6 s gap turned one loaded
            # minute into a 24-minute run on 2026-11-24 (280 requests, pace 5.8).
            self.pace = min(C.ARCTIC_PACE_MAX, max(self.pace, self._base_pace) * 1.4)
            self.slowdowns += 1

    def _get(self, path: str, params: dict[str, Any], *, attempts: int = 6) -> Any:
        url = f"{self.base}{path}"
        last: Exception | None = None
        for attempt in range(attempts):
            self._sleep()
            try:
                if _HAVE_CURL_CFFI:
                    r = _http.get(url, params=params, impersonate="chrome",
                                  timeout=self.timeout,
                                  headers={"Accept": "application/json",
                                           "User-Agent": C.USER_AGENT})
                    status, body = r.status_code, r.content
                else:  # pragma: no cover
                    import urllib.parse
                    full = url + "?" + urllib.parse.urlencode(params)
                    req = _urllib.Request(full, headers={
                        "Accept": "application/json", "User-Agent": C.USER_AGENT})
                    with _urllib.urlopen(req, timeout=self.timeout) as resp:
                        status, body = resp.status, resp.read()
                self.calls += 1
                self.bytes += len(body or b"")
                if status == 200:
                    self._note_success()
            except Exception as exc:                       # network-level
                last = exc
                time.sleep(1.2 * (attempt + 1))
                self.retries += 1
                continue

            if status == 200:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise SourceError(f"{path}: not JSON ({exc})") from exc
                if isinstance(payload, dict) and payload.get("error"):
                    msg = str(payload["error"])
                    low = msg.lower()
                    if ("rate" in low or "quota" in low or "timeout" in low
                            or "slow down" in low or "busy" in low):
                        last = RateLimited(msg)
                        self._note_pressure()
                        time.sleep(0.5 * (attempt + 1))     # pacing already spaces the retry
                        self.retries += 1
                        continue
                    raise SourceError(f"{path}: {msg}")
                return payload.get("data")
            if status in (422, 429, 500, 502, 503, 504):
                # 422 carries {"error": "Timeout. Maybe slow down a bit"} — the archive
                # telling us it is under load. Retryable, and a reason to widen our gap.
                last = RateLimited(f"HTTP {status} {body[:120]!r}")
                self._note_pressure()
                time.sleep(0.5 * (attempt + 1))         # the gap in _sleep is the real wait
                self.retries += 1
                continue
            if status == 404:
                return []
            raise SourceError(f"{path}: HTTP {status} {body[:200]!r}")
        raise last or SourceError(f"{path}: exhausted retries")

    # -- queries ----------------------------------------------------------- #
    def fetch_posts(self, subreddit, *, after=None, before=None,
                    limit=C.ARCTIC_PAGE, sort="asc") -> list[dict]:
        params: dict[str, Any] = {"subreddit": subreddit, "limit": int(limit),
                                  "sort": sort, "fields": POST_FIELDS}
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)
        data = self._get("/posts/search", params) or []
        return [norm_post(p, subreddit) for p in data if p.get("id")]

    def fetch_comments(self, subreddit, *, after=None, before=None,
                       limit=C.ARCTIC_PAGE, sort="asc") -> list[dict]:
        params: dict[str, Any] = {"subreddit": subreddit, "limit": int(limit),
                                  "sort": sort, "fields": COMMENT_FIELDS}
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)
        data = self._get("/comments/search", params) or []
        return [norm_comment(c, subreddit) for c in data if c.get("id")]

    def fetch_thread(self, post_id: str, *, limit: int = 500) -> list[dict]:
        """Every comment in one thread, walked forward by creation time."""
        out: list[dict] = []
        cursor: int | None = None
        while len(out) < limit:
            page = int(min(C.ARCTIC_PAGE, limit - len(out)))
            params: dict[str, Any] = {"link_id": post_id, "limit": page,
                                      "sort": "asc", "fields": COMMENT_FIELDS}
            if cursor is not None:
                params["after"] = cursor
            data = self._get("/comments/search", params) or []
            if not data:
                break
            batch = [norm_comment(c) for c in data if c.get("id")]
            out.extend(batch)
            if len(data) < page:
                break
            cursors = [int(c["created_utc"]) for c in data if c.get("created_utc")]
            if not cursors:
                break
            nxt = max(cursors)
            if cursor is not None and nxt <= cursor:
                break
            cursor = nxt
        return out

    def fetch_post(self, post_id: str) -> dict | None:
        data = self._get("/posts/ids", {"ids": post_id, "fields": POST_FIELDS}) or []
        return norm_post(data[0]) if data else None

    def probe(self, subreddit: str, *, hours: float = 168.0) -> dict:
        """Posts and comments seen in the window — the honest existence test."""
        since = int(time.time() - hours * 3600)
        posts, pages = 0, 0
        cursor = since
        while pages < 4:
            page = self._get("/posts/search", {
                "subreddit": subreddit, "limit": C.ARCTIC_PAGE, "sort": "asc",
                "after": cursor, "fields": "id,created_utc"}) or []
            posts += len(page)
            pages += 1
            if len(page) < C.ARCTIC_PAGE:
                break
            cursor = max(int(p["created_utc"]) for p in page) + 1

        comments, pages = 0, 0
        cursor = since
        while pages < 3:
            page = self._get("/comments/search", {
                "subreddit": subreddit, "limit": C.ARCTIC_PAGE, "sort": "asc",
                "after": cursor, "fields": "id,created_utc"}) or []
            comments += len(page)
            pages += 1
            if len(page) < C.ARCTIC_PAGE:
                break
            cursor = max(int(c["created_utc"]) for c in page) + 1

        return {"recent_posts": posts, "recent_comments": comments,
                "window_hours": hours, "exists": bool(posts or comments)}


# --------------------------------------------------------------------------- #
# Reddit's own API (a script app is enough)
# --------------------------------------------------------------------------- #
class OAuthSource(Source):
    """Reddit's API over OAuth. Ships disabled; fill .env to switch it on.

    Create a "script" app at https://www.reddit.com/prefs/apps and put the two
    values in .env as REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET. Rate limit is
    ~100 queries/minute, which is generous next to the archive.
    """
    name = "oauth"
    label = "reddit.com oauth (script app)"

    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    API = "https://oauth.reddit.com"

    def __init__(self, client_id: str | None = None, client_secret: str | None = None,
                 user_agent: str | None = None):
        self.client_id = client_id or C.REDDIT_CLIENT_ID
        self.client_secret = client_secret or C.REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or C.REDDIT_USER_AGENT
        self._token: str | None = None
        self._expires = 0.0
        if not (self.client_id and self.client_secret):
            raise NotConfigured(
                "OAuthSource needs REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env "
                "(create a script app at https://www.reddit.com/prefs/apps)")

    def _auth(self) -> str:
        if self._token and time.time() < self._expires - 30:
            return self._token
        import base64
        cred = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        if not _HAVE_CURL_CFFI:                        # pragma: no cover
            raise NotConfigured("curl_cffi is required for OAuthSource")
        r = _http.post(self.TOKEN_URL, data={"grant_type": "client_credentials"},
                       headers={"Authorization": f"Basic {cred}",
                                "User-Agent": self.user_agent}, timeout=30)
        if r.status_code != 200:
            raise SourceError(f"token: HTTP {r.status_code} {r.text[:200]!r}")
        j = r.json()
        self._token = j["access_token"]
        self._expires = time.time() + float(j.get("expires_in", 3600))
        return self._token

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        r = _http.get(self.API + path, params=params, timeout=30,
                      headers={"Authorization": f"bearer {self._auth()}",
                               "User-Agent": self.user_agent})
        if r.status_code == 429:
            raise RateLimited("oauth: HTTP 429")
        if r.status_code >= 400:
            raise SourceError(f"oauth {path}: HTTP {r.status_code} {r.text[:200]!r}")
        return r.json()

    def fetch_posts(self, subreddit, *, after=None, before=None,
                    limit=C.ARCTIC_PAGE, sort="asc") -> list[dict]:
        j = self._get(f"/r/{subreddit}/new", {"limit": int(limit)})
        children = [c["data"] for c in j.get("data", {}).get("children", [])]
        out = [norm_post(p, subreddit) for p in children]
        if after is not None:
            out = [p for p in out if p["created_utc"] > int(after)]
        return sorted(out, key=lambda p: p["created_utc"])

    def fetch_comments(self, subreddit, *, after=None, before=None,
                       limit=C.ARCTIC_PAGE, sort="asc") -> list[dict]:
        j = self._get(f"/r/{subreddit}/comments", {"limit": int(limit)})
        children = [c["data"] for c in j.get("data", {}).get("children", [])]
        out = [norm_comment(c, subreddit) for c in children]
        if after is not None:
            out = [c for c in out if c["created_utc"] > int(after)]
        return sorted(out, key=lambda c: c["created_utc"])

    def fetch_thread(self, post_id: str, *, limit: int = 500) -> list[dict]:
        j = self._get(f"/comments/{post_id}", {"limit": min(limit, 500), "depth": 10})
        if not isinstance(j, list) or len(j) < 2:
            return []
        out: list[dict] = []

        def walk(node: dict) -> None:
            d = node.get("data") or {}
            if node.get("kind") != "t1" or not d.get("id"):
                return
            out.append(norm_comment(d))
            rep = d.get("replies")
            if isinstance(rep, dict):
                for child in rep.get("data", {}).get("children", []):
                    if len(out) < limit:
                        walk(child)

        for child in j[1].get("data", {}).get("children", []):
            if len(out) < limit:
                walk(child)
        return out

    def fetch_post(self, post_id: str) -> dict | None:
        j = self._get(f"/comments/{post_id}", {"limit": 1})
        if isinstance(j, list) and j and j[0].get("data", {}).get("children"):
            return norm_post(j[0]["data"]["children"][0]["data"])
        return None

    def probe(self, subreddit: str, *, hours: float = 168.0) -> dict:
        j = self._get(f"/r/{subreddit}/about", {})
        about = j.get("data", {})
        if about.get("reason") == "private" or about.get("error"):
            return {"recent_posts": 0, "recent_comments": 0, "exists": False,
                    "error": about.get("reason") or about.get("error")}
        j2 = self._get(f"/r/{subreddit}/new", {"limit": 25})
        posts = [norm_post(c["data"], subreddit)
                 for c in j2.get("data", {}).get("children", [])]
        since = time.time() - hours * 3600
        return {"recent_posts": len([p for p in posts if p["created_utc"] >= since]),
                "recent_comments": None, "exists": True,
                "subscribers": about.get("subscribers")}


# --------------------------------------------------------------------------- #
# replayed files
# --------------------------------------------------------------------------- #
class FileSource(Source):
    """Read posts/comments from saved JSON.

    Accepts the shape `{"posts": [...], "comments": [...]}` or a bare list of
    Reddit objects (the archive's own `/api/.../search` response, with `data`).
    """
    name = "file"
    label = "local JSON dump"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise NotConfigured(f"no such file: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            payload = {"posts": payload["data"]}
        if isinstance(payload, list):
            payload = {"posts": payload}
        self.posts = [norm_post(p) for p in payload.get("posts", []) if p.get("id")]
        self.comments = [norm_comment(c) for c in payload.get("comments", []) if c.get("id")]

    def fetch_posts(self, subreddit, *, after=None, before=None, limit=C.ARCTIC_PAGE,
                    sort="asc") -> list[dict]:
        rows = [p for p in self.posts if p["subreddit"].lower() == subreddit.lower()]
        if after is not None:
            rows = [p for p in rows if p["created_utc"] > int(after)]
        if before is not None:
            rows = [p for p in rows if p["created_utc"] < int(before)]
        rows.sort(key=lambda p: p["created_utc"], reverse=(sort == "desc"))
        return rows[:limit]

    def fetch_comments(self, subreddit, *, after=None, before=None, limit=C.ARCTIC_PAGE,
                       sort="asc") -> list[dict]:
        rows = [c for c in self.comments if c["subreddit"].lower() == subreddit.lower()]
        if after is not None:
            rows = [c for c in rows if c["created_utc"] > int(after)]
        if before is not None:
            rows = [c for c in rows if c["created_utc"] < int(before)]
        rows.sort(key=lambda c: c["created_utc"], reverse=(sort == "desc"))
        return rows[:limit]

    def fetch_thread(self, post_id: str, *, limit: int = 500) -> list[dict]:
        return [c for c in self.comments if c["post_id"] == post_id][:limit]

    def probe(self, subreddit: str, *, hours: float = 168.0) -> dict:
        p = [x for x in self.posts if x["subreddit"].lower() == subreddit.lower()]
        c = [x for x in self.comments if x["subreddit"].lower() == subreddit.lower()]
        return {"recent_posts": len(p), "recent_comments": len(c),
                "exists": bool(p or c)}


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def get_source(kind: str | None = None, **kw) -> Source:
    """Pick a source by name, or the best one that is configured."""
    kind = (kind or os.environ.get("RDX_SOURCE") or "").lower()
    if kind in ("", "auto"):
        if C.REDDIT_CLIENT_ID and C.REDDIT_CLIENT_SECRET:
            return OAuthSource()
        return ArcticSource()
    if kind == "arctic":
        return ArcticSource()
    if kind == "oauth":
        return OAuthSource()
    if kind == "file":
        path = kw.get("path") or os.environ.get("RDX_FILE")
        if not path:
            raise NotConfigured("file source needs a path (--file PATH or RDX_FILE)")
        return FileSource(path)
    raise SourceError(f"unknown source {kind!r} (arctic | oauth | file)")
