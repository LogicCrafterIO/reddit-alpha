"""store — the SQLite layer. Schema, migrations, and every write/read primitive.

Design rules this file enforces, in order of importance:

1. Reddit's own ids are the keys. Re-crawling updates a row, never duplicates it.
2. Text is immutable, metrics append. `post_metric` is append-only so velocity is
   computable: a thread that doubled in score in forty minutes is a measurement.
3. One retrieval unit. `chunk` spans posts and comments; a searcher never has to
   care which kind of document carried the argument.
4. Evidence by reference. `idea_doc` points at documents; the words are re-read
   from the original row, so a stored idea can never drift from its evidence.
5. Additive migrations. `meta.schema_version` gates ALTERs; no rebuilds.

Every public function takes a connection so callers can batch inside one
transaction. `open_db()` is the convenience for scripts and the CLI.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import config as C
import textproc as T

SCHEMA_VERSION = 2

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS subreddit (
  name             TEXT PRIMARY KEY COLLATE NOCASE,
  display          TEXT,
  theme            TEXT NOT NULL DEFAULT 'other',
  weight           REAL NOT NULL DEFAULT 1.0,
  enabled          INTEGER NOT NULL DEFAULT 1,
  notes            TEXT,
  added_at         TEXT,
  last_after_utc   INTEGER,          -- incremental crawl cursor (epoch seconds)
  last_run_at      TEXT,
  last_error       TEXT,
  validated_at     TEXT,
  recent_posts     INTEGER,          -- from `subs validate`
  recent_comments  INTEGER
);

CREATE TABLE IF NOT EXISTS post (
  id            TEXT PRIMARY KEY,
  subreddit     TEXT NOT NULL,
  author        TEXT,
  author_flair  TEXT,
  title         TEXT NOT NULL DEFAULT '',
  selftext      TEXT,
  url           TEXT,
  domain        TEXT,
  permalink     TEXT,
  created_utc   INTEGER NOT NULL,
  flair         TEXT,
  is_self       INTEGER,
  over_18       INTEGER,
  stickied      INTEGER,
  num_comments  INTEGER,
  score         INTEGER,
  upvote_ratio  REAL,
  gilded        INTEGER,
  quality       REAL,
  chars         INTEGER,
  tickers       TEXT,                -- JSON list of primary tickers
  stub          INTEGER NOT NULL DEFAULT 0,   -- comment-only sighting
  first_seen    TEXT,
  last_seen     TEXT
);
CREATE INDEX IF NOT EXISTS idx_post_sub_time ON post(subreddit, created_utc);
CREATE INDEX IF NOT EXISTS idx_post_time     ON post(created_utc);

CREATE TABLE IF NOT EXISTS post_metric (
  post_id       TEXT NOT NULL REFERENCES post(id) ON DELETE CASCADE,
  seen_at       TEXT NOT NULL,
  score         INTEGER,
  num_comments  INTEGER,
  upvote_ratio  REAL,
  PRIMARY KEY (post_id, seen_at)
);

CREATE TABLE IF NOT EXISTS comment (
  id          TEXT PRIMARY KEY,
  post_id     TEXT NOT NULL,         -- deliberately not a FK: old threads get new replies
  subreddit   TEXT NOT NULL,
  author      TEXT,
  parent_id   TEXT,
  depth       INTEGER,
  body        TEXT,
  created_utc INTEGER NOT NULL,
  score       INTEGER,
  is_op       INTEGER,
  stickied    INTEGER,
  permalink   TEXT,
  quality     REAL,
  chars       INTEGER,
  tickers     TEXT,
  first_seen  TEXT,
  last_seen   TEXT
);
CREATE INDEX IF NOT EXISTS idx_comment_post     ON comment(post_id, created_utc);
CREATE INDEX IF NOT EXISTS idx_comment_sub_time ON comment(subreddit, created_utc);

CREATE TABLE IF NOT EXISTS chunk (
  id          INTEGER PRIMARY KEY,
  doc_kind    TEXT NOT NULL CHECK (doc_kind IN ('post', 'comment')),
  doc_id      TEXT NOT NULL,
  post_id     TEXT NOT NULL,
  subreddit   TEXT NOT NULL,
  seq         INTEGER NOT NULL DEFAULT 0,
  created_utc INTEGER,
  author      TEXT,
  score       INTEGER,
  heading     TEXT,
  text        TEXT NOT NULL,
  chars       INTEGER,
  quality     REAL,
  UNIQUE (doc_kind, doc_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc  ON chunk(doc_kind, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_post ON chunk(post_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  text, subreddit, author, post_id, doc_kind,
  tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS vector (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunk(id) ON DELETE CASCADE,
  model    TEXT NOT NULL,
  dim      INTEGER NOT NULL,
  dtype    TEXT NOT NULL DEFAULT 'float32',
  vec      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS mention (
  doc_kind    TEXT NOT NULL,
  doc_id      TEXT NOT NULL,
  symbol      TEXT NOT NULL,
  kind        TEXT NOT NULL,          -- cashtag | ticker | name
  confidence  REAL NOT NULL,
  subreddit   TEXT NOT NULL,
  created_utc INTEGER,
  PRIMARY KEY (doc_kind, doc_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_mention_symbol ON mention(symbol, created_utc);

CREATE TABLE IF NOT EXISTS author (
  name        TEXT PRIMARY KEY,
  posts       INTEGER NOT NULL DEFAULT 0,
  comments    INTEGER NOT NULL DEFAULT 0,
  score_sum   INTEGER NOT NULL DEFAULT 0,
  best        REAL NOT NULL DEFAULT 0,
  top_sub     TEXT,
  updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS news (
  id           INTEGER PRIMARY KEY,
  hash         TEXT NOT NULL UNIQUE,
  ticker       TEXT,
  topic        TEXT,
  title        TEXT NOT NULL,
  url          TEXT,
  publisher    TEXT,
  published_at TEXT,
  snippet      TEXT,
  discovery    INTEGER NOT NULL DEFAULT 0,   -- Google News hit: headline usable, link not
  source       TEXT,
  fetched_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker, published_at);
CREATE INDEX IF NOT EXISTS idx_news_time   ON news(published_at);

CREATE TABLE IF NOT EXISTS idea (
  id            INTEGER PRIMARY KEY,
  hash          TEXT NOT NULL UNIQUE,
  created_at    TEXT NOT NULL,
  window_hours  REAL,
  theme         TEXT,
  headline      TEXT NOT NULL,
  direction     TEXT CHECK (direction IN ('long','short','watch','avoid')),
  tickers       TEXT,                -- JSON list
  thesis        TEXT NOT NULL,
  mechanism     TEXT,
  catalyst      TEXT,
  catalyst_date TEXT,
  invalidation  TEXT,
  crowd_edge    TEXT,
  conviction    REAL,
  metrics       TEXT,                -- JSON: crowd metrics behind the idea
  refs          TEXT,                -- JSON list of reddit refs
  news_refs     TEXT,                -- JSON list of {title,url,published_at}
  model         TEXT,
  status        TEXT NOT NULL DEFAULT 'new',
  marked_at     TEXT,                -- when it was bookmarked or marked acting
  emb           BLOB,                -- headline+thesis vector, for novelty checks
  supersedes    INTEGER,
  notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_idea_time ON idea(created_at);

CREATE TABLE IF NOT EXISTS idea_doc (
  idea_id  INTEGER NOT NULL REFERENCES idea(id) ON DELETE CASCADE,
  doc_kind TEXT NOT NULL,
  doc_id   TEXT NOT NULL,
  weight   REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (idea_id, doc_kind, doc_id)
);

CREATE TABLE IF NOT EXISTS run (
  id          INTEGER PRIMARY KEY,
  kind        TEXT NOT NULL,        -- crawl | hydrate | embed | generate | news
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  status      TEXT,                 -- running | ok | error
  stats       TEXT,                 -- JSON
  error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_time ON run(started_at);
"""

# (target_version, [sql, ...]) — applied in order when meta.schema_version is lower.
# Additive only: each entry upgrades a store that predates it. An ALTER that a
# fresh store already has is tolerated, so one file serves both paths.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (2, ["ALTER TABLE idea ADD COLUMN marked_at TEXT"]),
]


# --------------------------------------------------------------------------- #
# connection & schema
# --------------------------------------------------------------------------- #
def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path or C.DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    conn.executescript(SCHEMA)
    cur = meta_get(conn, "schema_version")
    version = int(cur) if cur else 0
    if version == 0:
        meta_set(conn, "schema_version", str(SCHEMA_VERSION))
        conn.commit()
        return SCHEMA_VERSION
    for target, stmts in MIGRATIONS:
        if version < target:
            for sql in stmts:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            version = target
    if version < SCHEMA_VERSION:
        version = SCHEMA_VERSION
    meta_set(conn, "schema_version", str(version))
    conn.commit()
    return version


@contextmanager
def open_db(path: str | Path | None = None):
    conn = connect(path)
    try:
        migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def meta_get(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def meta_set(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def run_start(conn, kind: str, stats: dict | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO run(kind, started_at, status, stats) VALUES(?,?,?,?)",
        (kind, T.now_iso(), "running", json.dumps(stats or {})))
    conn.commit()
    return int(cur.lastrowid)


def run_finish(conn, run_id: int, *, stats: dict | None = None,
               status: str = "ok", error: str | None = None) -> None:
    conn.execute(
        "UPDATE run SET finished_at=?, status=?, stats=?, error=? WHERE id=?",
        (T.now_iso(), status, json.dumps(stats or {}), error, run_id))
    conn.commit()


# --------------------------------------------------------------------------- #
# subreddits — the editable registry
# --------------------------------------------------------------------------- #
def upsert_subreddit(conn, name: str, *, theme: str = "other", weight: float = 1.0,
                     enabled: bool = True, notes: str = "", display: str | None = None) -> None:
    name = name.strip().lstrip("/").removeprefix("r/")
    conn.execute(
        """INSERT INTO subreddit(name, display, theme, weight, enabled, notes, added_at)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(name) DO UPDATE SET
             theme=excluded.theme, weight=excluded.weight, enabled=excluded.enabled,
             notes=CASE WHEN excluded.notes<>'' THEN excluded.notes ELSE subreddit.notes END,
             display=COALESCE(excluded.display, subreddit.display)""",
        (name, display or name, theme, float(weight), 1 if enabled else 0, notes, T.now_iso()))
    conn.commit()


def list_subreddits(conn, enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM subreddit"
    if enabled_only:
        sql += " WHERE enabled=1"
    sql += " ORDER BY theme, name"
    return [dict(r) for r in conn.execute(sql)]


def subreddit_names(conn, enabled_only: bool = True) -> list[str]:
    return [r["name"] for r in list_subreddits(conn, enabled_only)]


def set_subreddit_enabled(conn, name: str, enabled: bool) -> None:
    conn.execute("UPDATE subreddit SET enabled=? WHERE name=? COLLATE NOCASE",
                 (1 if enabled else 0, name))
    conn.commit()


def update_subreddit_fields(conn, name: str, **fields) -> None:
    allowed = {"theme", "weight", "enabled", "notes", "display"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    clause = ", ".join(f"{k}=?" for k in sets)
    conn.execute(f"UPDATE subreddit SET {clause} WHERE name=? COLLATE NOCASE",
                 (*sets.values(), name))
    conn.commit()


def delete_subreddit(conn, name: str) -> None:
    conn.execute("DELETE FROM subreddit WHERE name=? COLLATE NOCASE", (name,))
    conn.commit()


def touch_subreddit(conn, name: str, *, error: str | None = None,
                    cursor: int | None = None) -> None:
    if error is None:
        conn.execute(
            "UPDATE subreddit SET last_run_at=?, last_error=NULL, "
            "last_after_utc=COALESCE(?, last_after_utc) WHERE name=? COLLATE NOCASE",
            (T.now_iso(), cursor, name))
    else:
        conn.execute("UPDATE subreddit SET last_run_at=?, last_error=? WHERE name=? COLLATE NOCASE",
                     (T.now_iso(), error[:500], name))


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def ensure_post_stub(conn, post_id: str, subreddit: str, created_utc: int) -> None:
    conn.execute(
        """INSERT INTO post(id, subreddit, created_utc, title, stub, first_seen, last_seen)
           VALUES(?,?,?,?,1,?,?)
           ON CONFLICT(id) DO NOTHING""",
        (post_id, subreddit, int(created_utc), "", T.now_iso(), T.now_iso()))


def upsert_post(conn, rec: dict) -> bool:
    """Insert or refresh one post. Returns True if it was new."""
    pid = rec["id"]
    now = T.now_iso()
    exists = conn.execute("SELECT 1 FROM post WHERE id=?", (pid,)).fetchone() is not None
    tickers = [t[0] for t in T.extract_tickers(rec.get("title", "") + "\n" + (rec.get("selftext") or ""))]
    conn.execute(
        """INSERT INTO post(id, subreddit, author, author_flair, title, selftext, url, domain,
                            permalink, created_utc, flair, is_self, over_18, stickied,
                            num_comments, score, upvote_ratio, gilded, tickers,
                            first_seen, last_seen, stub)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
           ON CONFLICT(id) DO UPDATE SET
             title=excluded.title, selftext=excluded.selftext, url=excluded.url,
             domain=excluded.domain, permalink=excluded.permalink, flair=excluded.flair,
             num_comments=excluded.num_comments, score=excluded.score,
             upvote_ratio=excluded.upvote_ratio, gilded=excluded.gilded,
             tickers=excluded.tickers, last_seen=excluded.last_seen, stub=0""",
        (pid, rec["subreddit"], rec.get("author"), rec.get("author_flair"), rec.get("title") or "",
         rec.get("selftext"), rec.get("url"), rec.get("domain"), rec.get("permalink"),
         int(rec["created_utc"]), rec.get("flair"), 1 if rec.get("is_self") else 0,
         1 if rec.get("over_18") else 0, 1 if rec.get("stickied") else 0,
         rec.get("num_comments"), rec.get("score"), rec.get("upvote_ratio"), rec.get("gilded"),
         json.dumps(tickers), now, now))
    return not exists


def record_metrics(conn, post_id: str, *, score=None, num_comments=None,
                   upvote_ratio=None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO post_metric(post_id, seen_at, score, num_comments, upvote_ratio) "
        "VALUES(?,?,?,?,?)",
        (post_id, T.now_precise(), score, num_comments, upvote_ratio))


def _comment_depths(conn, rows: Sequence[dict]) -> dict[str, int]:
    """Depth from the parent chain, using this batch first and the store second."""
    parent = {r["id"]: (r.get("parent_id") or "") for r in rows}
    depth: dict[str, int] = {}

    def resolve(cid: str, seen: set[str]) -> int:
        if cid in depth:
            return depth[cid]
        par = parent.get(cid, "")
        if not par.startswith("t1_"):
            depth[cid] = 0
            return 0
        pid = par[3:]
        if pid in seen:                      # malformed cycle guard
            depth[cid] = 1
            return 1
        if pid in parent:
            d = 1 + resolve(pid, seen | {cid})
        else:
            row = conn.execute("SELECT depth FROM comment WHERE id=?", (pid,)).fetchone()
            d = (row["depth"] + 1) if row and row["depth"] is not None else 1
        depth[cid] = d
        return d

    for r in rows:
        resolve(r["id"], set())
    return depth


def upsert_comments(conn, rows: Sequence[dict], post_authors: dict[str, str] | None = None) -> int:
    """Insert or refresh comments. Returns how many were new."""
    if not rows:
        return 0
    now = T.now_iso()
    depths = _comment_depths(conn, rows)
    post_authors = post_authors or {}
    # A comment can arrive for a thread that began before the window, or that the
    # source returns out of order. Keep the reference whole here rather than relying
    # on every caller to remember: a comment always has a parent row. Stubs never
    # count as posts in any statistic.
    known = {r["post_id"] for r in rows if r.get("post_id")}
    if known:
        have = {row[0] for row in conn.execute(
            f"SELECT id FROM post WHERE id IN ({','.join('?' * len(known))})", list(known))}
        for pid in known - have:
            first = min(int(r["created_utc"]) for r in rows if r.get("post_id") == pid)
            ensure_post_stub(conn, pid, rows[0]["subreddit"], first)
    new = 0
    for r in rows:
        body = r.get("body") or ""
        pid = r.get("post_id") or ""
        if not pid:
            continue
        exists = conn.execute("SELECT 1 FROM comment WHERE id=?", (r["id"],)).fetchone() is not None
        if not exists:
            new += 1
        tickers = [t[0] for t in T.extract_tickers(body)]
        is_op = 1 if (post_authors.get(pid) and r.get("author") == post_authors[pid]) else 0
        conn.execute(
            """INSERT INTO comment(id, post_id, subreddit, author, parent_id, depth, body,
                                   created_utc, score, is_op, stickied, permalink, tickers,
                                   first_seen, last_seen)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 body=excluded.body, score=excluded.score, depth=excluded.depth,
                 tickers=excluded.tickers, last_seen=excluded.last_seen""",
            (r["id"], pid, r["subreddit"], r.get("author"), r.get("parent_id"), depths.get(r["id"]),
             body, int(r["created_utc"]), r.get("score"), is_op,
             1 if r.get("stickied") else 0, r.get("permalink") or T.permalink(r["subreddit"], pid, r["id"]),
             json.dumps(tickers), now, now))
    return new


def prune_junk_chunks(conn) -> dict:
    """Drop chunks whose text the current junk filter now rejects.

    The filter improves; the index was built by the old one. Re-indexing every
    document to clear a few hundred rows is minutes of work for a rounding error,
    so this walks the chunks once and removes the ones `is_junk` would refuse today.
    """
    victims = [r["id"] for r in conn.execute("SELECT id, text FROM chunk")
               if T.is_junk(r["text"])]
    for cid in victims:
        conn.execute("DELETE FROM chunk_fts WHERE rowid=?", (cid,))
        conn.execute("DELETE FROM vector WHERE chunk_id=?", (cid,))
        conn.execute("DELETE FROM chunk WHERE id=?", (cid,))
    conn.commit()
    return {"removed": len(victims), "remaining": conn.execute(
        "SELECT count(*) FROM chunk").fetchone()[0]}


# --------------------------------------------------------------------------- #
# indexing — chunks, fts, mentions
# --------------------------------------------------------------------------- #
def _delete_chunks(conn, doc_kind: str, doc_id: str) -> None:
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM chunk WHERE doc_kind=? AND doc_id=?", (doc_kind, doc_id))]
    if not ids:
        return
    for cid in ids:
        conn.execute("DELETE FROM chunk_fts WHERE rowid=?", (cid,))
        # a re-chunk renumbers rows, so its vectors are stale by definition
        conn.execute("DELETE FROM vector WHERE chunk_id=?", (cid,))
    conn.execute("DELETE FROM chunk WHERE doc_kind=? AND doc_id=?", (doc_kind, doc_id))


def index_doc(conn, *, doc_kind: str, doc_id: str, post_id: str, subreddit: str,
              created_utc: int, author: str | None, score: int | None,
              text: str, quality: float) -> tuple[int, bool]:
    """Chunk one document into the retrieval index. Returns (n_chunks, changed).

    Unchanged text is left alone, so a re-crawl refreshes scores without
    discarding vectors it would then have to recompute.
    """
    body = T.clean(text)
    if not body:
        return 0, False
    pieces = T.chunk_text(body)
    if not pieces:
        return 0, False

    existing = [(r["seq"], r["text"]) for r in conn.execute(
        "SELECT seq, text FROM chunk WHERE doc_kind=? AND doc_id=? ORDER BY seq",
        (doc_kind, doc_id))]
    if existing == list(enumerate(pieces)):
        conn.execute(
            """UPDATE chunk SET score=?, quality=?, created_utc=?, author=?
               WHERE doc_kind=? AND doc_id=?""",
            (score, quality, int(created_utc), author, doc_kind, doc_id))
        return len(pieces), False

    _delete_chunks(conn, doc_kind, doc_id)
    conn.execute("DELETE FROM mention WHERE doc_kind=? AND doc_id=?", (doc_kind, doc_id))

    for seq, piece in enumerate(pieces):
        heading = T.heading_of(piece)
        cur = conn.execute(
            """INSERT INTO chunk(doc_kind, doc_id, post_id, subreddit, seq, created_utc,
                                 author, score, heading, text, chars, quality)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (doc_kind, doc_id, post_id, subreddit, seq, int(created_utc), author, score,
             heading, piece, len(piece), quality))
        cid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO chunk_fts(rowid, text, subreddit, author, post_id, doc_kind) "
            "VALUES(?,?,?,?,?,?)",
            (cid, piece, subreddit, author or "", post_id, doc_kind))

    for sym, kind, conf in T.extract_tickers(body):
        conn.execute(
            """INSERT OR REPLACE INTO mention(doc_kind, doc_id, symbol, kind, confidence,
                                              subreddit, created_utc)
               VALUES(?,?,?,?,?,?,?)""",
            (doc_kind, doc_id, sym, kind, conf, subreddit, int(created_utc)))
    return len(pieces), True


def index_post_row(conn, row: sqlite3.Row | dict, *, age_hours: float | None = None) -> tuple[int, bool]:
    d = dict(row)
    text = (d.get("title") or "") + ("\n\n" + d["selftext"] if d.get("selftext") else "")
    q = T.quality_score(text, score=d.get("score") or 0, num_comments=d.get("num_comments") or 0,
                        flair=d.get("flair"), kind="post", age_hours=age_hours)
    conn.execute("UPDATE post SET quality=?, chars=? WHERE id=?", (q, len(T.clean(text)), d["id"]))
    return index_doc(conn, doc_kind="post", doc_id=d["id"], post_id=d["id"],
                     subreddit=d["subreddit"], created_utc=int(d["created_utc"]),
                     author=d.get("author"), score=d.get("score"), text=text, quality=q)


def index_comment_row(conn, row: sqlite3.Row | dict, *, age_hours: float | None = None) -> tuple[int, bool]:
    d = dict(row)
    body = d.get("body") or ""
    if T.is_junk(body):
        return 0, False
    q = T.quality_score(body, score=d.get("score") or 0, kind="comment", age_hours=age_hours)
    conn.execute("UPDATE comment SET quality=?, chars=? WHERE id=?", (q, len(T.clean(body)), d["id"]))
    return index_doc(conn, doc_kind="comment", doc_id=d["id"], post_id=d["post_id"],
                     subreddit=d["subreddit"], created_utc=int(d["created_utc"]),
                     author=d.get("author"), score=d.get("score"), text=body, quality=q)


def refresh_authors(conn) -> int:
    """Rebuild the author roll from the store: who keeps writing things worth reading.

    Cheap enough to run at the end of every crawl (a few hundred thousand rows
    at most); it is a summary table, never a source of truth.
    """
    rows = conn.execute(
        """SELECT author,
                  SUM(kind='post')    AS posts,
                  SUM(kind='comment') AS comments,
                  SUM(score)          AS score_sum,
                  MAX(quality)        AS best,
                  subreddit
           FROM (
             SELECT author, 'post' AS kind, COALESCE(score, 0) AS score,
                    COALESCE(quality, 0) AS quality, subreddit
             FROM post WHERE stub=0 AND author IS NOT NULL AND author NOT IN ('', '[deleted]')
             UNION ALL
             SELECT author, 'comment', COALESCE(score, 0), COALESCE(quality, 0), subreddit
             FROM comment WHERE author IS NOT NULL AND author NOT IN ('', '[deleted]')
           )
           GROUP BY author, subreddit""").fetchall()

    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(r["author"], {"posts": 0, "comments": 0, "score_sum": 0,
                                         "best": 0.0, "subs": {}})
        a["posts"] += int(r["posts"] or 0)
        a["comments"] += int(r["comments"] or 0)
        a["score_sum"] += int(r["score_sum"] or 0)
        a["best"] = max(a["best"], float(r["best"] or 0))
        a["subs"][r["subreddit"]] = a["subs"].get(r["subreddit"], 0) + int(r["posts"] or 0)

    now = T.now_iso()
    conn.execute("DELETE FROM author")
    conn.executemany(
        "INSERT INTO author(name, posts, comments, score_sum, best, top_sub, updated_at) "
        "VALUES(?,?,?,?,?,?,?)",
        [(name, a["posts"], a["comments"], a["score_sum"], round(a["best"], 4),
          max(a["subs"], key=a["subs"].get) if a["subs"] else None, now)
         for name, a in agg.items()])
    conn.commit()
    return len(agg)


# --------------------------------------------------------------------------- #
# vectors
# --------------------------------------------------------------------------- #
def chunks_to_embed(conn, *, min_chars: int = C.VEC_MIN_CHARS,
                    min_quality: float = C.VEC_MIN_QUALITY,
                    limit: int | None = None) -> list[sqlite3.Row]:
    sql = """SELECT c.id, c.text FROM chunk c
             LEFT JOIN vector v ON v.chunk_id=c.id
             WHERE v.chunk_id IS NULL AND c.chars>=? AND c.quality>=?
             ORDER BY c.quality DESC"""
    params: list[Any] = [min_chars, min_quality]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(sql, params))


def store_vectors(conn, rows: Sequence[tuple[int, bytes]], *, model: str,
                  dim: int, dtype: str = "float32") -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO vector(chunk_id, model, dim, dtype, vec) VALUES(?,?,?,?,?)",
        [(cid, model, dim, dtype, blob) for cid, blob in rows])
    conn.commit()
    meta_set(conn, "embed_model", model)
    meta_set(conn, "embed_count", str(conn.execute("SELECT count(*) FROM vector").fetchone()[0]))
    conn.commit()
    return len(rows)


def vector_stats(conn) -> dict:
    n = conn.execute("SELECT count(*) FROM vector").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunk").fetchone()[0]
    return {"vectors": n, "chunks": chunks,
            "model": meta_get(conn, "embed_model"),
            "coverage": round(n / chunks, 3) if chunks else 0.0}


# --------------------------------------------------------------------------- #
# news
# --------------------------------------------------------------------------- #
def upsert_news(conn, items: Sequence[dict], *, ticker: str | None = None,
                topic: str | None = None) -> int:
    import hashlib

    n = 0
    for it in items:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        url = it.get("url") or ""
        h = hashlib.sha1(f"{title}|{url}".encode()).hexdigest()[:20]
        cur = conn.execute(
            """INSERT INTO news(hash, ticker, topic, title, url, publisher, published_at,
                                snippet, discovery, source, fetched_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(hash) DO NOTHING""",
            (h, ticker, topic, title, url, it.get("source") or it.get("publisher"),
             it.get("published_utc") or it.get("published_at"), it.get("snippet"),
             1 if it.get("discovery") else 0, it.get("feed") or it.get("source"),
             T.now_iso()))
        n += 0 if cur.rowcount == 0 else 1
    conn.commit()
    return n


# --------------------------------------------------------------------------- #
# ideas
# --------------------------------------------------------------------------- #
def insert_idea(conn, idea: dict, docs: Sequence[tuple[str, str, float]] = ()) -> int:
    import hashlib

    h = idea.get("hash") or hashlib.sha1(
        (idea["headline"] + "|" + idea["thesis"]).encode()).hexdigest()[:16]
    cur = conn.execute(
        """INSERT INTO idea(hash, created_at, window_hours, theme, headline, direction, tickers,
                            thesis, mechanism, catalyst, catalyst_date, invalidation, crowd_edge,
                            conviction, metrics, refs, news_refs, model, status, emb, notes)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(hash) DO UPDATE SET
             metrics=excluded.metrics, refs=excluded.refs, news_refs=excluded.news_refs,
             conviction=excluded.conviction""",
        (h, idea.get("created_at") or T.now_iso(), idea.get("window_hours"),
         idea.get("theme"), idea["headline"], idea.get("direction"), json.dumps(idea.get("tickers") or []),
         idea["thesis"], idea.get("mechanism"), idea.get("catalyst"), idea.get("catalyst_date"),
         idea.get("invalidation"), idea.get("crowd_edge"), idea.get("conviction"),
         json.dumps(idea.get("metrics") or {}), json.dumps(idea.get("refs") or []),
         json.dumps(idea.get("news_refs") or []), idea.get("model"), idea.get("status", "new"),
         idea.get("emb"), idea.get("notes")))
    idea_id = int(cur.lastrowid) if cur.rowcount else conn.execute(
        "SELECT id FROM idea WHERE hash=?", (h,)).fetchone()[0]
    for kind, doc_id, weight in docs:
        conn.execute("INSERT OR REPLACE INTO idea_doc(idea_id, doc_kind, doc_id, weight) "
                     "VALUES(?,?,?,?)", (idea_id, kind, doc_id, weight))
    conn.commit()
    return idea_id


BOOKMARK_STATUSES = ("keep", "acting")


def update_idea_status(conn, idea_id: int, status: str, notes: str | None = None) -> None:
    """Set an idea's status, keeping `marked_at` in step with it.

    Only the status the human sets moves `marked_at` — regeneration never touches
    it, so a bookmark survives every later run.
    """
    stamp = T.now_iso() if status in BOOKMARK_STATUSES else None
    if notes is None:
        conn.execute("UPDATE idea SET status=?, marked_at=? WHERE id=?",
                     (status, stamp, idea_id))
    else:
        conn.execute("UPDATE idea SET status=?, notes=?, marked_at=? WHERE id=?",
                     (status, notes, stamp, idea_id))
    conn.commit()


def idea_embeddings(conn) -> list[tuple[int, str, bytes | None]]:
    return [(r["id"], r["headline"], r["emb"]) for r in conn.execute(
        "SELECT id, headline, emb FROM idea ORDER BY id DESC LIMIT 500")]


# --------------------------------------------------------------------------- #
# counts used by the UI and `stats`
# --------------------------------------------------------------------------- #
def counts(conn) -> dict:
    def one(sql: str, *p) -> int:
        return int(conn.execute(sql, p).fetchone()[0])

    return {
        "subreddits": one("SELECT count(*) FROM subreddit"),
        "subreddits_enabled": one("SELECT count(*) FROM subreddit WHERE enabled=1"),
        "posts": one("SELECT count(*) FROM post WHERE stub=0"),
        "comments": one("SELECT count(*) FROM comment"),
        "chunks": one("SELECT count(*) FROM chunk"),
        "vectors": one("SELECT count(*) FROM vector"),
        "mentions": one("SELECT count(*) FROM mention"),
        "authors": one("SELECT count(*) FROM author"),
        "news": one("SELECT count(*) FROM news"),
        "ideas": one("SELECT count(*) FROM idea"),
        "ideas_marked": one("SELECT count(*) FROM idea WHERE marked_at IS NOT NULL"),
        "schema_version": int(meta_get(conn, "schema_version") or 0),
        "embed_model": meta_get(conn, "embed_model"),
    }


def db_size(path: str | Path | None = None) -> int:
    p = Path(path or C.DB_PATH)
    total = 0
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(p) + suffix)
        if f.exists():
            total += f.stat().st_size
    return total
