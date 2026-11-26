"""retrieve — the read side: search, show, thread, trending, tickers.

Two legs, fused: BM25 over every chunk (cheap, exact terms) and cosine over the
quality-gated vector subset (fuzzy, semantic), combined by reciprocal rank. The
contract matches the transcript store's, because that is the pattern this
workspace already reads fluently: **locate with `search`, then read with `show`**.
Never load a thread to answer a question a snippet can answer.
"""
from __future__ import annotations

import heapq
import math
import re
import time
from typing import Any, Iterable

import config as C
import store
import textproc as T

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have",
    "has", "had", "what", "which", "who", "whom", "how", "why", "when", "where",
    "any", "anyone", "someone", "something", "they", "them", "their", "it", "its",
    "this", "that", "these", "those", "about", "say", "said", "says", "tell", "me",
    "us", "we", "i", "you", "your", "can", "could", "would", "should", "will",
    "just", "also", "there", "here", "if", "then", "than", "so", "but", "not",
    "no", "yes", "much", "many", "more", "most", "some", "such", "very", "from",
    "into", "over", "up", "down", "out", "as", "than", "then",
}


def _fts_query(q: str, all_terms: bool = False) -> str:
    """FTS5 rejects NEAR/5 and its implicit AND returns nothing for a question.

    A plain multi-word query becomes an OR of its terms (BM25 still ranks the
    overlaps); anything that already looks like FTS syntax is passed through.
    """
    q = re.sub(r"(\w+)\s+NEAR/(\d+)\s+(\w+)", r"NEAR(\1 \3, \2)", q)
    q = re.sub(r"(\w+)\s+NEAR\s+(\w+)", r"NEAR(\1 \2, 10)", q)
    if all_terms or re.search(r'["*()]|\b(AND|OR|NOT|NEAR)\b', q):
        return q
    words = [w for w in re.findall(r"[\w'\u2019\-]+", q) if len(w) > 1]
    return " OR ".join(words) if len(words) > 1 else q


def _terms(q: str) -> list[str]:
    words = [w.lower() for w in re.findall(r"[\w'\u2019\-]+", q)]
    return [w for w in words if len(w) > 1 and w not in STOPWORDS] or words


def _snippet(text: str, terms: list[str], width: int = 260) -> str:
    """A readable window around the first term hit, never a whole chunk."""
    body = re.sub(r"\s+", " ", text or "").strip()
    if len(body) <= width:
        return body
    low = body.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
    if pos < 0:
        return body[:width].rstrip() + " …"
    start = max(0, pos - width // 3)
    end = min(len(body), start + width)
    out = body[start:end].strip()
    return ("… " if start > 0 else "") + out + (" …" if end < len(body) else "")


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def _where(*, subs: Iterable[str] | None = None, theme: str | None = None,
           ticker: str | None = None, since: float | None = None,
           kind: str | None = None, min_quality: float | None = None,
           min_score: int | None = None) -> tuple[str, list[Any]]:
    clause, params = ["1=1"], []
    if subs:
        names = list(subs)
        clause.append(f"c.subreddit IN ({','.join('?' * len(names))})")
        params.extend(names)
    if theme:
        clause.append("c.subreddit IN (SELECT name FROM subreddit WHERE theme=? COLLATE NOCASE)")
        params.append(theme)
    if kind:
        clause.append("c.doc_kind=?")
        params.append(kind)
    if min_quality is not None:
        clause.append("c.quality >= ?")
        params.append(float(min_quality))
    if min_score is not None:
        clause.append("c.score >= ?")
        params.append(int(min_score))
    if since:
        clause.append("c.created_utc >= ?")
        params.append(int(since))
    if ticker:
        sym = ticker.upper().lstrip("$")
        clause.append(
            "c.id IN (SELECT ch.id FROM chunk ch JOIN mention m ON m.doc_kind=ch.doc_kind "
            "AND m.doc_id=ch.doc_id WHERE m.symbol=?)")
        params.append(sym)
    return " AND ".join(clause), params


# --------------------------------------------------------------------------- #
# the two legs
# --------------------------------------------------------------------------- #
def _lexical(conn, q: str, *, clause: str, params: list[Any], limit: int,
             all_terms: bool) -> list[tuple[int, float]]:
    fq = _fts_query(q, all_terms)
    sql = (f"SELECT c.id AS cid, bm25(chunk_fts) AS rank FROM chunk_fts "
           f"JOIN chunk c ON c.id = chunk_fts.rowid "
           f"WHERE chunk_fts MATCH ? AND {clause} "
           f"ORDER BY rank LIMIT ?")
    try:
        rows = conn.execute(sql, [fq, *params, limit]).fetchall()
    except Exception:
        return []
    # bm25 is negative-better; normalise to a positive score for display
    return [(r["cid"], -float(r["rank"])) for r in rows]


def _vector(conn, q: str, *, clause: str, params: list[Any], limit: int) -> list[tuple[int, float]]:
    import numpy as np

    qvec = _embed_query(q)
    if qvec is None:
        return []
    sql = (f"SELECT v.vec AS vec, v.dim AS dim, v.dtype AS dtype, c.id AS cid "
           f"FROM vector v JOIN chunk c ON c.id = v.chunk_id WHERE {clause}")
    cur = conn.execute(sql, params)
    best: list[tuple[float, int]] = []
    while True:
        block = cur.fetchmany(4000)
        if not block:
            break
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float16 if r["dtype"] == "float16"
                                       else np.float32) for r in block])
        sims = mat @ qvec
        for i, s in enumerate(sims):
            cid = block[i]["cid"]
            if len(best) < limit:
                heapq.heappush(best, (float(s), cid))
            elif s > best[0][0]:
                heapq.heapreplace(best, (float(s), cid))
    return [(cid, s) for s, cid in sorted(best, reverse=True)]


_EMB = None


def _embed_query(q: str):
    global _EMB
    try:
        if _EMB is None:
            import os
            os.environ.setdefault("HF_HOME", str(C.EMB_CACHE))
            from model2vec import StaticModel
            try:
                _EMB = StaticModel.from_pretrained(C.EMBED_MODEL)
            except Exception:
                os.environ["HF_HOME"] = str(C.EMB_ALT_CACHE)
                _EMB = StaticModel.from_pretrained(C.EMBED_MODEL)
        import numpy as np
        v = np.asarray(_EMB.encode([q])[0], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v
    except Exception:
        return None


def _fuse(legs: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion — no score calibration needed between the legs."""
    acc: dict[int, float] = {}
    for hits in legs:
        for rank, (cid, _score) in enumerate(hits, start=1):
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(acc.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------- #
# public: search
# --------------------------------------------------------------------------- #
def search(conn, q: str, *, subs: Iterable[str] | None = None, theme: str | None = None,
           ticker: str | None = None, hours: float | None = None,
           since: float | None = None, kind: str | None = None,
           min_quality: float | None = None, min_score: int | None = None,
           limit: int = 8, mode: str = "hybrid", all_terms: bool = False,
           snippet_chars: int = 260) -> list[dict]:
    """Locate chunks, ranked. mode: lexical | vector | hybrid."""
    since_ts = since or (time.time() - hours * 3600 if hours else None)
    clause, params = _where(subs=subs, theme=theme, ticker=ticker, since=since_ts,
                            kind=kind, min_quality=min_quality, min_score=min_score)
    pool = max(limit * 6, 40)
    legs = []
    if mode in ("lexical", "hybrid"):
        legs.append(_lexical(conn, q, clause=clause, params=params, limit=pool,
                             all_terms=all_terms))
    if mode in ("vector", "hybrid"):
        legs.append(_vector(conn, q, clause=clause, params=params, limit=pool))
    if not legs or not any(legs):
        return []
    fused = _fuse([leg for leg in legs if leg])[:limit]
    if not fused:
        return []

    terms = _terms(q)
    ids = [cid for cid, _ in fused]
    rows = {r["id"]: r for r in conn.execute(
        f"""SELECT c.id, c.doc_kind, c.doc_id, c.post_id, c.subreddit, c.author,
                   c.created_utc, c.score, c.quality, c.heading, c.text, c.seq
            FROM chunk c WHERE c.id IN ({','.join('?' * len(ids))})""", ids)}
    out = []
    for cid, score in fused:
        r = rows.get(cid)
        if not r:
            continue
        ref = (T.post_ref(r["subreddit"], r["post_id"]) if r["doc_kind"] == "post"
               else T.comment_ref(r["subreddit"], r["post_id"], r["doc_id"]))
        out.append({
            "ref": ref,
            "score": round(score, 6),
            "doc_kind": r["doc_kind"],
            "subreddit": r["subreddit"],
            "author": r["author"],
            "document_score": r["score"],
            "quality": r["quality"],
            "created_utc": r["created_utc"],
            "created_at": T.iso(r["created_utc"]),
            "heading": r["heading"],
            "snippet": _snippet(r["text"], terms, snippet_chars),
            "url": T.permalink(r["subreddit"], r["post_id"],
                               r["doc_id"] if r["doc_kind"] == "comment" else None),
        })
    return out


# --------------------------------------------------------------------------- #
# public: read one document
# --------------------------------------------------------------------------- #
def show(conn, ref: str, *, comments: int = 8, comment_sort: str = "score",
         max_chars: int = 8000) -> dict:
    """Re-read the document a reference points at, with just enough context."""
    kind, sub, post_id, cid = T.parse_ref(ref)
    post = conn.execute("SELECT * FROM post WHERE id=?", (post_id,)).fetchone()
    out: dict[str, Any] = {"ref": ref, "post": dict(post) if post else None}

    if kind == "comment":
        row = conn.execute("SELECT * FROM comment WHERE id=?", (cid,)).fetchone()
        out["comment"] = dict(row) if row else None
        if row and row["parent_id"].startswith("t1_"):
            par = conn.execute("SELECT * FROM comment WHERE id=?", (row["parent_id"][3:],)).fetchone()
            out["parent_comment"] = dict(par) if par else None
    if post:
        order = "score DESC" if comment_sort == "score" else "created_utc ASC"
        rows = conn.execute(
            f"SELECT id, author, parent_id, depth, body, score, created_utc, is_op "
            f"FROM comment WHERE post_id=? ORDER BY {order} LIMIT ?",
            (post_id, max(0, comments))).fetchall()
        out["top_comments"] = [dict(r) for r in rows]
        out["metrics"] = [dict(r) for r in conn.execute(
            "SELECT seen_at, score, num_comments, upvote_ratio FROM post_metric "
            "WHERE post_id=? ORDER BY seen_at", (post_id,))]
        out["velocity"] = velocity_of(conn, post_id)
    # trim anything enormous so a caller never accidentally reads 80 KB
    if out.get("post") and out["post"].get("selftext"):
        full = out["post"]["selftext"]
        out["post_chars"] = len(full)
        out["post"]["selftext"] = full[:max_chars]
        out["post"]["selftext_truncated"] = len(full) > max_chars
        out["post_truncated"] = len(full) > max_chars
    return out


def thread(conn, post_id: str, *, limit: int = 60, sort: str = "score") -> dict:
    post = conn.execute("SELECT * FROM post WHERE id=?", (post_id,)).fetchone()
    if not post:
        return {"post": None, "comments": []}
    order = {"score": "score DESC", "new": "created_utc ASC",
             "old": "created_utc ASC", "controversial": "score ASC"}.get(sort, "score DESC")
    rows = conn.execute(
        f"""SELECT id, parent_id, depth, author, body, score, created_utc, is_op, quality
            FROM comment WHERE post_id=? ORDER BY {order} LIMIT ?""",
        (post_id, limit)).fetchall()
    return {"post": dict(post), "comments": [dict(r) for r in rows],
            "velocity": velocity_of(conn, post_id)}


def velocity_of(conn, post_id: str) -> float | None:
    """Upvotes per hour between the first and latest metric sample.

    Accepts either a bare post id or the store's own reference form.
    """
    if post_id.startswith("r/"):
        post_id = T.parse_ref(post_id)[2]
    rows = conn.execute(
        "SELECT seen_at, score FROM post_metric WHERE post_id=? ORDER BY seen_at",
        (post_id,)).fetchall()
    if len(rows) < 2:
        return None
    from datetime import datetime

    def ts(s: str) -> float:
        return datetime.fromisoformat(s).timestamp()

    t0, t1 = ts(rows[0]["seen_at"]), ts(rows[-1]["seen_at"])
    dt = (t1 - t0) / 3600.0
    if dt <= 0:
        return None
    return round(((rows[-1]["score"] or 0) - (rows[0]["score"] or 0)) / dt, 1)


# --------------------------------------------------------------------------- #
# public: the cross-sections
# --------------------------------------------------------------------------- #
def trending(conn, *, subs: Iterable[str] | None = None, theme: str | None = None,
             hours: float = 24.0, limit: int = 20, min_comments: int = 0,
             sort: str = "velocity") -> list[dict]:
    """Posts ranked by how fast they are being read, not by absolute score.

    Absolute score favours old threads; velocity favours the one that just took
    off, which is what an idea generator needs.
    """
    since = time.time() - hours * 3600
    clause, params = _where(subs=subs, theme=theme, since=since)
    rows = conn.execute(
        f"""SELECT c.id, c.subreddit, c.author, c.title, c.permalink, c.created_utc,
                   c.score, c.num_comments, c.quality, c.flair, c.url, c.domain
            FROM post c WHERE {clause} AND c.stub=0
            {"AND COALESCE(c.num_comments,0) >= ?" if min_comments else ""}
            ORDER BY c.created_utc DESC LIMIT 400""",
        params + ([min_comments] if min_comments else [])).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        v = velocity_of(conn, d["id"])
        age_h = max(0.25, (time.time() - d["created_utc"]) / 3600.0)
        d["velocity"] = v if v is not None else round((d["score"] or 0) / age_h, 1)
        d["age_hours"] = round(age_h, 1)
        d["ref"] = T.post_ref(d["subreddit"], d["id"])
        d["url"] = T.permalink(d["subreddit"], d["id"])
        out.append(d)
    if sort == "quality":
        out.sort(key=lambda d: -(d["quality"] or 0))
    elif sort == "score":
        out.sort(key=lambda d: -(d["score"] or 0))
    elif sort == "comments":
        out.sort(key=lambda d: -(d["num_comments"] or 0))
    else:
        out.sort(key=lambda d: -(d["velocity"] or 0))
    return out[:limit]


def tickers(conn, *, hours: float = 24.0, theme: str | None = None,
            subs: Iterable[str] | None = None, limit: int = 25,
            symbol: str | None = None) -> list[dict]:
    """Mention counts, and the change against the equal-length window before.

    The delta is the whole point: a ticker going from 40 to 300 mentions in a day
    is a rotation, and it is visible here before it is visible in price.
    """
    now = time.time()
    since, prior = now - hours * 3600, now - 2 * hours * 3600
    sub_clause, sub_params = "", []
    if subs:
        names = list(subs)
        sub_clause = f" AND subreddit IN ({','.join('?' * len(names))})"
        sub_params = names
    if theme:
        sub_clause += " AND subreddit IN (SELECT name FROM subreddit WHERE theme=? COLLATE NOCASE)"
        sub_params = sub_params + [theme]

    def counts(lo: float, hi: float) -> dict[str, int]:
        rows = conn.execute(
            f"""SELECT symbol, count(*) AS n FROM mention
                WHERE created_utc >= ? AND created_utc < ?{sub_clause}
                GROUP BY symbol""", [int(lo), int(hi), *sub_params])
        return {r["symbol"]: r["n"] for r in rows}

    cur = counts(since, now)
    prev = counts(prior, since)
    if symbol:
        sym = symbol.upper().lstrip("$")
        cur = {sym: cur.get(sym, 0)}
        prev = {sym: prev.get(sym, 0)}
    out = []
    for sym, n in sorted(cur.items(), key=lambda kv: -kv[1])[:limit]:
        before = prev.get(sym, 0)
        delta = n - before
        out.append({
            "symbol": sym,
            "mentions": n,
            "prior": before,
            "delta": delta,
            "change": round(delta / before, 2) if before else None,
            "documents": n,
            "top": _top_docs_for(conn, sym, since, sub_clause, sub_params),
        })
    return out


def _top_docs_for(conn, symbol: str, since: float, sub_clause: str,
                  sub_params: list[Any], limit: int = 3) -> list[dict]:
    rows = conn.execute(
        f"""SELECT m.doc_kind, m.doc_id, m.subreddit, m.created_utc,
                   COALESCE(p.score, 0) AS score
            FROM mention m
            LEFT JOIN post p ON p.id = CASE WHEN m.doc_kind='post' THEN m.doc_id ELSE
                                            (SELECT post_id FROM comment WHERE id=m.doc_id) END
            WHERE m.symbol=? AND m.created_utc >= ?{sub_clause.replace('subreddit', 'm.subreddit')}
            ORDER BY score DESC LIMIT ?""",
        [symbol, int(since), *sub_params, limit]).fetchall()
    return [{"ref": (T.post_ref(r["subreddit"], r["doc_id"]) if r["doc_kind"] == "post"
                     else T.comment_ref(r["subreddit"], _post_of_comment(conn, r["doc_id"]), r["doc_id"])),
             "subreddit": r["subreddit"], "score": r["score"],
             "created_at": T.iso(r["created_utc"])} for r in rows]


def _post_of_comment(conn, comment_id: str) -> str:
    row = conn.execute("SELECT post_id FROM comment WHERE id=?", (comment_id,)).fetchone()
    return row["post_id"] if row else ""


def authors(conn, *, limit: int = 20, min_comments: int = 2) -> list[dict]:
    return [dict(r) for r in conn.execute(
        """SELECT name, posts, comments, score_sum, best, top_sub, updated_at
           FROM author WHERE comments >= ? ORDER BY best DESC LIMIT ?""",
        (min_comments, limit))]


# --------------------------------------------------------------------------- #
# public: stats & integrity
# --------------------------------------------------------------------------- #
def stats(conn) -> dict:
    c = store.counts(conn)
    now = time.time()
    day = conn.execute("SELECT count(*) FROM post WHERE created_utc >= ? AND stub=0",
                       (int(now - 86400),)).fetchone()[0]
    day_c = conn.execute("SELECT count(*) FROM comment WHERE created_utc >= ?",
                         (int(now - 86400),)).fetchone()[0]
    kinds = {r["doc_kind"]: r["n"] for r in conn.execute(
        "SELECT doc_kind, count(*) AS n FROM chunk GROUP BY doc_kind")}
    top_subs = [dict(r) for r in conn.execute(
        """SELECT subreddit, count(*) AS posts, max(created_utc) AS newest
           FROM post WHERE stub=0 GROUP BY subreddit ORDER BY posts DESC LIMIT 25""")]
    last_runs = [dict(r) for r in conn.execute(
        "SELECT kind, started_at, finished_at, status FROM run ORDER BY id DESC LIMIT 8")]
    return {**c, "posts_24h": day, "comments_24h": day_c, "chunks_by_kind": kinds,
            "top_subs": top_subs, "last_runs": last_runs,
            "db_mb": round(store.db_size() / 1e6, 1), "vectors_detail": store.vector_stats(conn)}


def check(conn) -> dict:
    """Structural health: does the index cover the documents it claims to?"""
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    posts = one("SELECT count(*) FROM post WHERE stub=0")
    comments = one("SELECT count(*) FROM comment")
    chunk_posts = one("SELECT count(DISTINCT doc_id) FROM chunk WHERE doc_kind='post'")
    chunk_comments = one("SELECT count(DISTINCT doc_id) FROM chunk WHERE doc_kind='comment'")
    fts_rows = one("SELECT count(*) FROM chunk_fts")
    chunk_rows = one("SELECT count(*) FROM chunk")
    orphans = one("SELECT count(*) FROM chunk c WHERE c.doc_kind='post' "
                  "AND NOT EXISTS (SELECT 1 FROM post p WHERE p.id=c.doc_id)")
    bad_refs = one("SELECT count(*) FROM chunk WHERE doc_kind NOT IN ('post','comment')")
    vectors_missing_text = one(
        "SELECT count(*) FROM vector v WHERE NOT EXISTS (SELECT 1 FROM chunk c WHERE c.id=v.chunk_id)")
    issues = []
    if posts and chunk_posts < posts * 0.6:
        issues.append(f"only {chunk_posts}/{posts} posts indexed (junk-filtered or failed)")
    if comments and chunk_comments < comments * 0.2:
        issues.append(f"only {chunk_comments}/{comments} comments indexed — check MIN_CHUNK_CHARS")
    if fts_rows != chunk_rows:
        issues.append(f"fts/chunk mismatch: {fts_rows} vs {chunk_rows}")
    if orphans:
        issues.append(f"{orphans} chunks point at missing posts")
    if bad_refs:
        issues.append(f"{bad_refs} chunks with an invalid doc_kind")
    if vectors_missing_text:
        issues.append(f"{vectors_missing_text} vectors point at missing chunks")
    return {"posts": posts, "comments": comments, "chunk_docs": {"post": chunk_posts,
            "comment": chunk_comments}, "chunks": chunk_rows, "fts_rows": fts_rows,
            "issues": issues, "ok": not issues}
