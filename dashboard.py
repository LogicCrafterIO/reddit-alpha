"""dashboard — a local page for reading the flow and running the jobs.

Two jobs, the same division as the other apps in this workspace: start work as a
subprocess (so a slow crawl never blocks the request loop) and query the store
read-only. Nothing here writes to the database except through `store` helpers
called by the registry endpoints, and every long action is a job with a log.

    python3 rdx.py serve           # http://127.0.0.1:8792
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config as C
import ideas as ideas_mod
import newswire
import retrieve as R
import store
import textproc as T

HERE = Path(__file__).resolve().parent
app = FastAPI(title="reddit-alpha", docs_url=None, redoc_url=None)

# --------------------------------------------------------------------------- #
# store access — one connection per request, WAL lets the crawler write meanwhile
# --------------------------------------------------------------------------- #
def db():
    return store.connect()


class Job:
    """A single background subprocess with a bounded log."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.kind: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.lines: deque[str] = deque(maxlen=400)
        self.exit_code: int | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind: str, argv: list[str]) -> dict:
        with self.lock:
            if self.running():
                raise HTTPException(409, f"a {self.kind} job is already running")
            # Pin the store explicitly: a dashboard started with RDX_DB set must
            # not spawn jobs that write to the default database instead.
            cmd = [sys.executable, str(HERE / "rdx.py"), "--db", str(C.DB_PATH),
                   *argv]
            self.proc = subprocess.Popen(
                cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            self.kind = kind
            self.started_at = time.time()
            self.finished_at = None
            self.exit_code = None
            self.lines.clear()
            threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
            return self.status()

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.lines.append(line.rstrip("\n"))
        proc.wait()
        with self.lock:
            self.exit_code = proc.returncode
            self.finished_at = time.time()

    def status(self) -> dict:
        return {"kind": self.kind, "running": self.running(),
                "started_at": self.started_at, "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "seconds": round((self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at else None,
                "lines": list(self.lines)[-200:]}


JOB = Job()
ALLOWED_STATUSES = ("new", "keep", "acting", "discard")


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if (HERE / "static").exists():
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


# --------------------------------------------------------------------------- #
# read API
# --------------------------------------------------------------------------- #
@app.get("/api/overview")
def overview():
    conn = db()
    try:
        s = R.stats(conn)
        chk = R.check(conn)
        subs = store.list_subreddits(conn)
        return {
            "store": str(C.DB_PATH), "stats": s, "check": chk,
            "subs": {"total": len(subs), "enabled": sum(1 for x in subs if x["enabled"])},
            "job": JOB.status(), "now": T.now_iso(),
            "embed_model": C.EMBED_MODEL, "llm": {"configured": bool(C.OLLAMA_API_KEY),
                                                 "model": C.LLM_MODEL},
            "news_available": newswire.available(),
        }
    finally:
        conn.close()


@app.get("/api/flow")
def flow(hours: float = 48.0, sort: str = "velocity", limit: int = 40,
         subs: str | None = None, theme: str | None = None,
         q: str | None = None, ticker: str | None = None):
    conn = db()
    try:
        names = [s.strip() for s in subs.split(",")] if subs else None
        if q:
            return {"items": R.search(conn, q, subs=names, theme=theme, ticker=ticker,
                                      hours=hours, limit=limit, mode="hybrid")}
        items = R.trending(conn, subs=names, theme=theme, hours=hours, limit=limit, sort=sort)
        if ticker:
            sym = ticker.upper().lstrip("$")
            ids = {r["doc_id"] for r in conn.execute(
                "SELECT DISTINCT doc_id FROM mention WHERE symbol=?", (sym,))}
            items = [i for i in items if i["id"] in ids]
        return {"items": items}
    finally:
        conn.close()


@app.get("/api/thread/{post_id}")
def thread(post_id: str, limit: int = 120, sort: str = "score"):
    conn = db()
    try:
        return R.thread(conn, post_id, limit=limit, sort=sort)
    finally:
        conn.close()


@app.get("/api/show")
def show(ref: str, comments: int = 5):
    conn = db()
    try:
        return R.show(conn, ref, comments=comments)
    finally:
        conn.close()


@app.get("/api/search")
def search(q: str, hours: float | None = None, limit: int = 12,
           mode: str = "hybrid", subs: str | None = None, theme: str | None = None,
           ticker: str | None = None):
    conn = db()
    try:
        names = [s.strip() for s in subs.split(",")] if subs else None
        return {"items": R.search(conn, q, hours=hours, limit=limit, mode=mode,
                                  subs=names, theme=theme, ticker=ticker)}
    finally:
        conn.close()


@app.get("/api/tickers")
def tickers(hours: float = 24.0, limit: int = 25, theme: str | None = None):
    conn = db()
    try:
        return {"items": R.tickers(conn, hours=hours, limit=limit, theme=theme)}
    finally:
        conn.close()


@app.get("/api/ideas")
def ideas_list(limit: int = 40, direction: str | None = None, status: str | None = None,
               bookmarked: bool = False, sort: str = "newest"):
    conn = db()
    try:
        wanted = [s.strip() for s in (status or "").split(",") if s.strip()]
        try:
            rows = ideas_mod.list_ideas(conn, limit=limit, direction=direction,
                                        statuses=wanted or None, bookmarked=bookmarked,
                                        sort=sort)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        for r in rows:
            conn_id = r["id"]
            refs = conn.execute("SELECT refs FROM idea WHERE id=?", (conn_id,)).fetchone()
            try:
                r["refs"] = json.loads(refs["refs"]) if refs else []
            except (json.JSONDecodeError, TypeError):
                r["refs"] = []
        return {"items": rows}
    finally:
        conn.close()


@app.get("/api/idea/{idea_id}")
def idea_detail(idea_id: int):
    conn = db()
    try:
        doc = ideas_mod.get_idea(conn, idea_id)
        if not doc:
            raise HTTPException(404, f"no idea #{idea_id}")
        return doc
    finally:
        conn.close()


@app.get("/api/news")
def news(tickers: str | None = None, hours: float = 72.0, limit: int = 40):
    conn = db()
    try:
        syms = [t.strip().upper() for t in tickers.split(",")] if tickers else None
        return {"items": newswire.recent(conn, hours=hours, tickers=syms, limit=limit)}
    finally:
        conn.close()


@app.get("/api/subs")
def subs_list():
    conn = db()
    try:
        rows = store.list_subreddits(conn)
        for r in rows:
            r["posts"] = conn.execute("SELECT count(*) FROM post WHERE subreddit=? AND stub=0",
                                      (r["name"],)).fetchone()[0]
            r["comments"] = conn.execute("SELECT count(*) FROM comment WHERE subreddit=?",
                                         (r["name"],)).fetchone()[0]
        return {"items": rows, "themes": sorted({r["theme"] or "other" for r in rows})}
    finally:
        conn.close()


@app.get("/api/authors")
def authors(limit: int = 20):
    conn = db()
    try:
        return {"items": R.authors(conn, limit=limit)}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# write API — the registry, and only the registry
# --------------------------------------------------------------------------- #
@app.post("/api/subs")
def subs_upsert(payload: dict):
    name = (payload.get("name") or "").strip().lstrip("/").removeprefix("r/")
    if not name:
        raise HTTPException(400, "name is required")
    conn = db()
    try:
        store.upsert_subreddit(
            conn, name,
            theme=(payload.get("theme") or "other").strip(),
            weight=float(payload.get("weight") or 1.0),
            enabled=bool(payload.get("enabled", True)),
            notes=(payload.get("notes") or "").strip())
        row = conn.execute("SELECT * FROM subreddit WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        return {"ok": True, "sub": dict(row) if row else None}
    finally:
        conn.close()


@app.patch("/api/subs/{name}")
def subs_patch(name: str, payload: dict):
    conn = db()
    try:
        fields = {k: payload[k] for k in ("theme", "weight", "enabled", "notes") if k in payload}
        if "enabled" in fields:
            fields["enabled"] = 1 if fields["enabled"] else 0
        store.update_subreddit_fields(conn, name, **fields)
        row = conn.execute("SELECT * FROM subreddit WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        if not row:
            raise HTTPException(404, f"no subreddit {name}")
        return {"ok": True, "sub": dict(row)}
    finally:
        conn.close()


@app.delete("/api/subs/{name}")
def subs_delete(name: str):
    conn = db()
    try:
        store.delete_subreddit(conn, name)
        return {"ok": True, "removed": name}
    finally:
        conn.close()


@app.post("/api/subs/import")
def subs_import(payload: dict | None = None):
    import registry
    conn = db()
    try:
        return registry.load_seed(conn, (payload or {}).get("file"))
    finally:
        conn.close()


@app.post("/api/subs/export")
def subs_export(payload: dict | None = None):
    import registry
    conn = db()
    try:
        return registry.export_seed(conn, (payload or {}).get("file"))
    finally:
        conn.close()


@app.post("/api/promote")
def promote_ref(payload: dict):
    """One thread through the generator: an idea, a duplicate, or a reasoned refusal."""
    ref = (payload.get("ref") or "").strip()
    if not ref:
        raise HTTPException(400, "ref required (r/<sub>/<post_id>)")
    conn = db()
    try:
        res = ideas_mod.promote(conn, ref, model=payload.get("model"),
                                news=not payload.get("no_news"))
        return res
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        conn.close()


@app.post("/api/idea/{idea_id}/status")
def idea_status(idea_id: int, payload: dict):
    conn = db()
    try:
        row = conn.execute("SELECT status, marked_at FROM idea WHERE id=?",
                           (idea_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"no idea {idea_id}")
        status = (payload.get("status") or "new").strip().lower()
        if status not in ALLOWED_STATUSES:
            raise HTTPException(400, f"status must be one of {', '.join(ALLOWED_STATUSES)}")
        store.update_idea_status(conn, idea_id, status, payload.get("notes"))
        conn.commit()
        row = conn.execute("SELECT status, marked_at FROM idea WHERE id=?",
                           (idea_id,)).fetchone()
        return {"ok": True, "id": idea_id, "status": row["status"],
                "marked_at": row["marked_at"], "bookmarked": bool(row["marked_at"])}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
@app.post("/api/crawl")
def crawl_start(payload: dict | None = None):
    p = payload or {}
    argv = ["crawl", "--hours", str(p.get("hours") or 24),
            "--mode", p.get("mode") or "incremental",
            "--max-pages", str(p.get("max_pages") or C.MAX_PAGES_PER_SUB),
            "--budget", str(p.get("budget") or C.CRAWL_PAGE_BUDGET),
            "--sub-cap", str(p.get("sub_cap") or C.CRAWL_SUB_PAGE_CAP)]
    if p.get("subs"):
        argv += ["--subs", p["subs"]]
    if p.get("embed", True):
        argv.append("--embed")
    if p.get("no_hydrate"):
        argv.append("--no-hydrate")
    return JOB.start("crawl", argv)


@app.post("/api/embed")
def embed_start():
    return JOB.start("embed", ["embed"])


@app.post("/api/generate")
def generate_start(payload: dict | None = None):
    p = payload or {}
    n = int(p.get("n") or C.GEN_IDEAS)
    argv = ["generate", "--hours", str(p.get("hours") or 72),
            "--n", str(n),
            # asking for more ideas than the pool of candidates would cap below n
            "--candidates", str(p.get("candidates") or max(C.GEN_CANDIDATES, n))]
    if p.get("theme"):
        argv += ["--theme", p["theme"]]
    if p.get("model"):
        argv += ["--model", p["model"]]
    return JOB.start("generate", argv)


@app.post("/api/validate")
def validate_start():
    return JOB.start("validate", ["subs", "validate"])


@app.get("/api/job")
def job_status():
    return JOB.status()
