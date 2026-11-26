"""crawl — pull records from a source into the store, then index and embed.

Four things make this fast without being rude:

* **Fan-out, not a forward walk.** Pagination is latency-bound (~1.4 s per request
  here), so one fetcher spends nearly all of its time waiting. Four pages in flight
  returned in 2.0 s on 2026-11-25 where the same four serialised took 8.3 s.
* **One writer.** Only the calling thread touches the database. SQLite allows one
  writer at a time; three crawling processes are three writers colliding, which is
  exactly the `database is locked` that this module used to produce. Fetchers hand
  batches over a queue instead.
* **Adaptive pacing.** The archive answers `422 Timeout. Maybe slow down a bit`
  when it is loaded. `sources.ArcticSource` widens its own gap when that happens
  and narrows it again on success, so politeness is a control loop rather than a
  constant somebody guessed — and the widening is capped, because the uncapped
  loop once turned a loaded minute into a 24-minute run.
* **A page allowance.** A run may spend a fixed number of archive pages, and a
  subreddit may spend a fixed share of them. What is not fetched is not lost: the
  cursor stays behind and the next run resumes there.

Everything is idempotent: upserts key on Reddit's ids, and a re-run refreshes
rather than duplicates.
"""
from __future__ import annotations

import queue
import threading
import time

from typing import Callable, Iterable

import config as C
import store
import textproc as T
from sources import RateLimited, Source, SourceError

Progress = Callable[[dict], None]
_DONE = "__done__"


def _emit(progress: Progress | None, **evt) -> None:
    if progress:
        progress(evt)


def _slices(floor: int, ceiling: int, n: int) -> list[tuple[int, int]]:
    n = max(1, min(n, 12))
    if n == 1 or ceiling - floor < 3600:
        return [(floor, ceiling)]
    step = (ceiling - floor) / n
    return [(int(floor + step * i), int(floor + step * (i + 1))) for i in range(n)]


class _Pipeline:
    """Fetch in parallel across subreddits, write in a single thread.

    The cost of a daily crawl is not the archive's rate limit but the waiting:
    every page costs a network round trip, and a serialised run pays all of them
    one after another. Measured here on 2026-11-25, four 100-record pages to four
    subreddits returned HTTP 200 in **2.0 s wall** concurrently against **8.3 s**
    serialised — the archive no longer 422s four in flight, so the parallelism it
    refused in November is now the cheapest speedup available.

    Two things stay as they were. Writes happen in the calling thread only: SQLite
    has one writer, and three fetching processes are three writers colliding. And
    the source's own adaptive gap is still the politeness control — fanning out
    spreads the *waiting*, it does not raise the request rate, so a load signal
    from the archive still slows the whole run down together.
    """

    def __init__(self, conn, source: Source, *, parallel: int, max_pages: int,
                 budget: int | None, sub_cap: int | None, progress: Progress | None):
        self.conn = conn
        self.source = source
        self.max_pages = max_pages
        self.progress = progress
        self.parallel = max(1, parallel)
        self.pages_per_task = C.CRAWL_PAGES_PER_TASK
        self.stop = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.tasks: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.outstanding = 0
        self.budget = budget                     # total pages this run may spend
        self.sub_cap = sub_cap                   # pages per (subreddit, kind)
        self.spent = 0
        self.capped: set[str] = set()            # subreddits that ran out of allowance
        self.stats: dict[str, dict] = {}
        self.errors: dict[str, list[str]] = {}
        self.hydration: dict[str, list] = {}

    # -- accounting -------------------------------------------------------- #
    def stat(self, sub: str) -> dict:
        return self.stats.setdefault(sub, {
            "posts": 0, "posts_new": 0, "comments": 0, "comments_new": 0,
            "chunks": 0, "hydrated": 0, "errors": 0, "pages": 0, "cursor": 0})

    def _fail(self, sub: str, kind: str, exc: Exception) -> None:
        with self.lock:
            self.errors.setdefault(sub, []).append(f"{kind}: {exc}"[:300])
            self.stat(sub)["errors"] += 1

    # -- fetching (worker threads) ----------------------------------------- #
    def _allow(self, sub: str, kind: str) -> bool:
        """Take one page from the run's and this subreddit's allowance.

        A page costs an archive request, so the allowance is what keeps a run
        bounded: when it is gone the subreddit is marked capped and its cursor
        stays where it was, which makes the next run resume rather than refetch.
        """
        with self.lock:
            stat = self.stat(sub)
            if self.budget is not None and self.spent >= self.budget:
                self.capped.add(sub)
                return False
            if self.sub_cap is not None and stat["pages"] >= self.sub_cap:
                self.capped.add(sub)
                return False
            self.spent += 1
            stat["pages"] += 1
            every = self.spent if self.spent % 10 == 0 else 0
        if every:                                   # say so, rather than look stuck
            _emit(self.progress, event="page", pages=self.spent, budget=self.budget)
        return True

    def _fetch_pages(self, kind: str, sub: str, lo: int, hi: int,
                     cursor: int) -> tuple | None:
        """Fetch a short run of pages, and hand back the rest as a continuation.

        Yielding after a few pages is what gives the pool its round-robin: four
        workers are shared by seventy subreddits, so a chain that never let go
        would spend the whole allowance on one firehose (r/technology: 161 of
        625 requests on 2026-11-25).
        """
        pages = 0
        while pages < self.pages_per_task and not self.stop.is_set():
            if not self._allow(sub, kind):
                return None
            try:
                if kind == "posts":
                    batch = self.source.fetch_posts(sub, after=cursor, before=hi,
                                                    limit=C.ARCTIC_PAGE, sort="asc")
                else:
                    batch = self.source.fetch_comments(sub, after=cursor, before=hi,
                                                       limit=C.ARCTIC_PAGE, sort="asc")
            except (RateLimited, SourceError) as exc:
                self._fail(sub, kind, exc)
                return None
            if not batch:
                return None
            self.q.put(("batch", sub, kind, batch))
            pages += 1
            stamps = [int(b["created_utc"]) for b in batch if b.get("created_utc")]
            if not stamps:
                return None
            nxt = max(stamps)
            if nxt <= cursor:
                return None
            cursor = nxt + 1
            if len(batch) < C.ARCTIC_PAGE:
                return None
        return (kind, sub, lo, hi, cursor)

    def _thread(self, sub: str, post_id: str) -> None:
        try:
            rows = self.source.fetch_thread(post_id, limit=800)
        except (RateLimited, SourceError) as exc:
            self._fail(sub, "thread", exc)
            return
        if rows:
            self.q.put(("thread", sub, post_id, rows))

    def _worker(self) -> None:
        while not self.stop.is_set():
            try:
                task = self.tasks.get(timeout=0.2)
            except queue.Empty:
                with self.lock:
                    finished = self.outstanding <= 0
                if finished:
                    return
                continue
            try:
                if task[0] == "thread":
                    self._thread(task[1], task[2])
                    with self.lock:
                        self.outstanding -= 1
                    continue
                nxt = self._fetch_pages(*task)          # (kind, sub, lo, hi, cursor)
            except Exception as exc:                  # a worker must never strand a chain
                self._fail(task[1], task[0], exc)
                nxt = None
            if nxt is not None:
                self.tasks.put(nxt)                   # the chain lives on: count unchanged
            else:
                with self.lock:
                    self.outstanding -= 1

    # -- writer (calling thread) ------------------------------------------- #
    def add(self, tasks: list[tuple]) -> None:
        if not tasks:
            return
        for task in tasks:
            self.tasks.put(task)
        with self.lock:
            self.outstanding += len(tasks)
        workers = [threading.Thread(target=self._worker, daemon=True)
                   for _ in range(self.parallel)]
        for w in workers:
            w.start()
        self._drain()
        for w in workers:
            w.join()

    def _drain(self) -> None:
        """Write every batch the workers hand over, and return when none is left."""
        while True:
            with self.lock:
                done = self.outstanding == 0
            if done and self.q.empty():
                try:
                    item = self.q.get(timeout=0.2)
                except queue.Empty:
                    return
                self._write(item)
                continue
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._write(item)

    # -- writer (calling thread) ------------------------------------------- #
    def _write(self, item: tuple) -> None:
        conn, now = self.conn, int(time.time())
        _, sub, kind, payload = item
        stat = self.stat(sub)
        if kind == "posts":
            self._posts(conn, sub, payload, now, stat)
        elif kind == "comments":
            self._comments(conn, sub, payload, now, stat)
        elif kind == "thread":
            self._thread_rows(conn, sub, payload, now, stat)
            _emit(self.progress, event="hydrate", sub=sub, post=item[2],
                  comments=len(payload))

    def _posts(self, conn, sub: str, batch: list[dict], now: int, stat: dict) -> None:
        stat["posts"] += len(batch)
        for rec in batch:
            if store.upsert_post(conn, rec):
                stat["posts_new"] += 1
            store.record_metrics(conn, rec["id"], score=rec.get("score"),
                                 num_comments=rec.get("num_comments"),
                                 upvote_ratio=rec.get("upvote_ratio"))
            stat["cursor"] = max(stat["cursor"], int(rec["created_utc"]))
        conn.commit()
        ids = [r["id"] for r in batch]
        rows = conn.execute(
            f"SELECT * FROM post WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        for prow in rows:
            age = max(0.0, (now - prow["created_utc"]) / 3600.0)
            n, _ = store.index_post_row(conn, prow, age_hours=age)
            stat["chunks"] += n
            if (prow["num_comments"] or 0) >= C.HYDRATE_MIN_COMMENTS:
                self.hydration.setdefault(sub, []).append(prow)
        conn.commit()

    def _comments(self, conn, sub: str, batch: list[dict], now: int, stat: dict) -> None:
        stat["comments"] += len(batch)
        want = {c["post_id"] for c in batch if c.get("post_id")}
        authors: dict[str, str] = {}
        if want:
            for r in conn.execute(
                    f"SELECT id, author FROM post WHERE id IN ({','.join('?' * len(want))})",
                    list(want)):
                authors[r["id"]] = r["author"]
            for pid in want - set(authors):
                first = min(c["created_utc"] for c in batch if c["post_id"] == pid)
                store.ensure_post_stub(conn, pid, sub, first)
        stat["comments_new"] += store.upsert_comments(conn, batch, authors)
        for c in batch:
            stat["cursor"] = max(stat["cursor"], int(c["created_utc"]))
        conn.commit()
        ids = [c["id"] for c in batch]
        for crow in conn.execute(
                f"SELECT * FROM comment WHERE id IN ({','.join('?' * len(ids))})", ids):
            age = max(0.0, (now - crow["created_utc"]) / 3600.0)
            n, _ = store.index_comment_row(conn, crow, age_hours=age)
            stat["chunks"] += n
        conn.commit()

    def _thread_rows(self, conn, sub: str, rows: list[dict], now: int, stat: dict) -> None:
        pid = rows[0].get("post_id") if rows else None
        author = conn.execute("SELECT author FROM post WHERE id=?", (pid,)).fetchone()
        store.upsert_comments(conn, rows, {pid: author["author"] if author else None})
        for crow in conn.execute("SELECT * FROM comment WHERE post_id=?", (pid,)):
            age = max(0.0, (now - crow["created_utc"]) / 3600.0)
            store.index_comment_row(conn, crow, age_hours=age)
        conn.commit()
        stat["hydrated"] += 1


# --------------------------------------------------------------------------- #
# public: crawl
# --------------------------------------------------------------------------- #
def crawl(conn, *, subs: Iterable[str] | None = None, hours: float | None = None,
          source: Source, mode: str = "incremental", max_pages: int = C.MAX_PAGES_PER_SUB,
          hydrate: bool = True, hydrate_max: int | None = None,
          parallel: int | None = None, budget: int | None = None,
          sub_cap: int | None = None, progress: Progress | None = None) -> dict:
    """Crawl enabled subreddits (or the named ones) over a time window.

    mode="incremental" starts each subreddit at its own cursor; mode="backfill"
    always starts at `now - hours`, which is what a first run or a gap fill uses.

    Two passes over the same pool: every subreddit's posts first (so hydration can
    rank the whole window's threads), then comments and thread hydrations together.
    """
    hours = float(hours if hours is not None else C.DEFAULT_WINDOW_HOURS)
    parallel = int(parallel if parallel is not None else C.CRAWL_PARALLEL)
    budget = C.CRAWL_PAGE_BUDGET if budget is None else budget
    sub_cap = C.CRAWL_SUB_PAGE_CAP if sub_cap is None else sub_cap
    budget = None if budget <= 0 else int(budget)          # 0 = no ceiling
    sub_cap = None if sub_cap is None or sub_cap <= 0 else int(sub_cap)
    names = list(subs) if subs else store.subreddit_names(conn, enabled_only=True)
    if not names:
        return {"error": "no subreddits enabled — add one with `rdx.py subs add NAME`"}

    run_id = store.run_start(conn, "crawl", {"subs": names, "hours": hours, "mode": mode})
    started = time.time()
    now = int(started)
    floor = now - int(hours * 3600)
    slices = 1 if parallel <= 1 else max(1, min(8, int(hours / 12) or 1))
    pipe = _Pipeline(conn, source, parallel=parallel, max_pages=max_pages,
                     budget=budget, sub_cap=sub_cap, progress=progress)
    starts: dict[str, int] = {}

    # -- pass 1: posts ----------------------------------------------------- #
    post_tasks: list[tuple] = []
    for sub in names:
        row = conn.execute("SELECT * FROM subreddit WHERE name=? COLLATE NOCASE",
                           (sub,)).fetchone()
        if row is None:
            pipe._fail(sub, "registry", SourceError("not in registry"))
            continue
        start = floor
        if mode == "incremental" and row["last_after_utc"]:
            # Resume from the cursor, not from the window. A daily run should fetch
            # what arrived since the last run; treating `--hours` as a hard floor
            # silently skipped whatever the skipped runs had not reached, and made
            # every run re-walk the same window. The floor only bounds the backstop.
            backstop = now - int(C.MAX_CATCHUP_HOURS * 3600)
            start = max(int(row["last_after_utc"]) - 60, backstop)
        starts[sub] = start
        pipe.stat(sub)
        _emit(progress, event="sub_start", sub=sub, cursor=start, theme=row["theme"])
        post_tasks += [("posts", sub, lo, hi, lo)
                       for lo, hi in _slices(start, now, slices)]
    if post_tasks:
        _emit(progress, event="pass", name="posts", tasks=len(post_tasks))
        pipe.add(post_tasks)

    # -- pass 2: comments, and the threads worth reading in full ---------- #
    comment_tasks: list[tuple] = []
    for sub in list(pipe.stats):
        comment_tasks += [("comments", sub, lo, hi, lo)
                          for lo, hi in _slices(starts.get(sub, floor), now, slices)]
    thread_tasks = _hydration_tasks(conn, pipe.hydration, enabled=hydrate,
                                    max_threads=hydrate_max or C.HYDRATE_MAX_PER_SUB)
    if comment_tasks or thread_tasks:
        _emit(progress, event="pass", name="comments",
              tasks=len(comment_tasks) + len(thread_tasks))
        pipe.add(comment_tasks + thread_tasks)

    # -- close out --------------------------------------------------------- #
    totals = {"subs": 0, "posts": 0, "posts_new": 0, "comments": 0, "comments_new": 0,
              "chunks": 0, "hydrated": 0, "errors": 0}
    per_sub: dict[str, dict] = {}
    for sub, stat in pipe.stats.items():
        stats = dict(stat)
        stats.pop("cursor", None)
        errors = pipe.errors.get(sub, [])
        # A slice that failed leaves a hole behind the cursor, and so does a run
        # that ran out of allowance part-way through a subreddit. Advance only
        # when neither happened; otherwise the next incremental pass resumes from
        # the same cursor (upserts make the overlap harmless).
        clean = not errors and sub not in pipe.capped
        stats["capped"] = sub in pipe.capped
        try:
            store.touch_subreddit(conn, sub, cursor=(stat["cursor"] or None) if clean
                                  else None, error=errors[0] if errors else None)
            conn.commit()
        except Exception as exc:                       # never lose the rest of the run
            stats["error"] = f"{type(exc).__name__}: {exc}"[:400]
            totals["errors"] += 1
            _safe_touch(conn, sub, stats["error"])
        if errors:
            stats["error"] = errors[0]
        per_sub[sub] = stats
        totals["subs"] += 1
        for k in ("posts", "posts_new", "comments", "comments_new", "chunks",
                  "hydrated", "errors"):
            totals[k] += stats.get(k, 0)
        _emit(progress, event="sub_done", sub=sub, stats=stats)

    elapsed = round(time.time() - started, 1)
    try:
        store.refresh_authors(conn)
    except Exception:
        pass
    result = {"totals": totals, "per_sub": per_sub, "elapsed_s": elapsed,
              "pages": pipe.spent, "budget": budget,
              "capped_subs": len(pipe.capped),
              "window_hours": hours, "mode": mode, "source": source.name,
              "requests": getattr(source, "calls", 0),
              "slowdowns": getattr(source, "slowdowns", 0),
              "pace": round(getattr(source, "pace", 0.0), 2),
              "mb": round(getattr(source, "bytes", 0) / 1e6, 1),
              "db_mb": round(store.db_size() / 1e6, 1)}
    store.run_finish(conn, run_id, stats=result,
                     status="ok" if not totals["errors"] else "partial")
    _emit(progress, event="done", result=result)
    return result


def _hydration_tasks(conn, pool: dict[str, list], *, enabled: bool,
                     max_threads: int) -> list[tuple]:
    """Pick the threads that clearly matter and turn them into fetch tasks.

    A window misses the top comments of a thread that started before it but is
    still receiving replies; those are exactly the threads where the argument
    lives. Ranked, capped per subreddit, and skipped when the store already holds
    most of the thread.
    """
    if not enabled or not pool:
        return []
    tasks: list[tuple] = []
    for sub, posts in pool.items():
        seen: set[str] = set()
        ranked = []
        for p in sorted(posts, key=lambda p: -((p["score"] or 0) * (p["num_comments"] or 1))):
            if p["id"] in seen or p["stub"]:
                continue
            if (p["score"] or 0) < C.HYDRATE_MIN_SCORE:
                continue
            seen.add(p["id"])
            ranked.append(p)
            if len(ranked) >= max_threads:
                break
        for p in ranked:
            have = conn.execute("SELECT count(*) FROM comment WHERE post_id=?",
                                (p["id"],)).fetchone()[0]
            if p["num_comments"] and have >= (p["num_comments"] or 0) * 0.85:
                continue
            tasks.append(("thread", sub, p["id"], None))
    return tasks


def _safe_touch(conn, sub: str, error: str) -> None:
    for attempt in range(6):
        try:
            store.touch_subreddit(conn, sub, error=error)
            return
        except Exception:
            time.sleep(0.5 * (attempt + 1))


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import os
    os.environ.setdefault("HF_HOME", str(C.EMB_CACHE))
    from model2vec import StaticModel
    try:
        _MODEL = StaticModel.from_pretrained(C.EMBED_MODEL)
    except Exception:
        # the transcript store may already hold the same model; reuse it rather
        # than pull 30 MB again
        os.environ["HF_HOME"] = str(C.EMB_ALT_CACHE)
        _MODEL = StaticModel.from_pretrained(C.EMBED_MODEL)
    return _MODEL


def embed_pending(conn, *, min_chars: int | None = None, min_quality: float | None = None,
                  limit: int | None = None, batch: int = 512,
                  progress: Progress | None = None) -> dict:
    """Embed chunks that qualify for the semantic leg and do not have a vector."""
    import numpy as np

    rows = store.chunks_to_embed(
        conn,
        min_chars=min_chars if min_chars is not None else C.VEC_MIN_CHARS,
        min_quality=min_quality if min_quality is not None else C.VEC_MIN_QUALITY,
        limit=limit)
    if not rows:
        return {"embedded": 0, "pending": 0, "model": C.EMBED_MODEL}

    model = _load_model()
    dtype = C.VEC_DTYPE if C.VEC_DTYPE in ("float32", "float16") else "float32"
    np_dtype = np.float32 if dtype == "float32" else np.float16
    total = 0
    run_id = store.run_start(conn, "embed", {"pending": len(rows)})
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        vecs = model.encode([r["text"] for r in part])
        payload = []
        for r, v in zip(part, vecs):
            arr = np.asarray(v, dtype=np_dtype)
            payload.append((int(r["id"]), arr.tobytes()))
        total += store.store_vectors(conn, payload, model=C.EMBED_MODEL,
                                     dim=int(vecs.shape[1]), dtype=dtype)
        _emit(progress, event="embed", done=total, total=len(rows))
    store.run_finish(conn, run_id, stats={"embedded": total})
    return {"embedded": total, "pending": 0, "model": C.EMBED_MODEL}


def reindex(conn, *, subs: Iterable[str] | None = None, hours: float | None = None,
            progress: Progress | None = None) -> dict:
    """Rebuild chunks for documents already stored (after tuning the chunker)."""
    params: list = []
    where = ["1=1"]
    if subs:
        names = list(subs)
        where.append(f"subreddit IN ({','.join('?' * len(names))})")
        params.extend(names)
    if hours:
        where.append("created_utc >= ?")
        params.append(int(time.time() - hours * 3600))
    clause = " AND ".join(where)
    now = int(time.time())
    stats = {"posts": 0, "comments": 0, "chunks": 0}
    for prow in conn.execute(f"SELECT * FROM post WHERE {clause}", params).fetchall():
        age = max(0.0, (now - prow["created_utc"]) / 3600.0)
        n, _ = store.index_post_row(conn, prow, age_hours=age)
        stats["posts"] += 1
        stats["chunks"] += n
    conn.commit()
    for crow in conn.execute(f"SELECT * FROM comment WHERE {clause}", params):
        age = max(0.0, (now - crow["created_utc"]) / 3600.0)
        n, _ = store.index_comment_row(conn, crow, age_hours=age)
        stats["comments"] += 1
        stats["chunks"] += n
    conn.commit()
    stats["authors"] = store.refresh_authors(conn)
    store.run_finish(conn, store.run_start(conn, "reindex", stats), stats=stats)
    return stats
