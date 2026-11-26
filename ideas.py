"""ideas — turn the store into investment ideas, with the evidence attached.

Two stages, deliberately:

  1. **select** — one cheap call over compact summaries of every candidate cluster
     in the window. Its only job is to rank; it never writes an idea.
  2. **synthesize** — one call per selected cluster, with the full evidence pack
     (post bodies, the comments that carry the argument, mention deltas, dated
     news). This is where an idea is written.

The split keeps the expensive call pointed at the two or three threads that
matter instead of the whole window, and it means a bad selection cannot silently
become a bad idea: the second call sees the words, not the first call's summary.

Every idea stores references, never paraphrases, so `rdx.py show <ref>` always
returns the original sentence an idea rests on.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Iterable, Sequence

import config as C
import llm
import newswire
import retrieve
import store
import textproc as T

DIRECTIONS = {"long", "short", "watch", "avoid"}


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]{3,}", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]{3,}", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def candidates(conn, *, hours: float = 72.0, theme: str | None = None,
               subs: Iterable[str] | None = None, limit: int = 400,
               min_quality: float | None = None, per_cluster: int = 4) -> list[dict]:
    """Cluster the window's strongest posts into candidate themes.

    A cluster is keyed on the strongest ticker the post mentions, or on its
    subreddit's theme when it names no ticker. Near-duplicate titles inside a
    cluster are folded together, because five people posting the same headline
    is one story, not five.
    """
    since = int(time.time() - hours * 3600)
    min_quality = C.GEN_MIN_QUALITY if min_quality is None else min_quality
    clause, params = ["created_utc >= ?", "stub=0", "COALESCE(quality,0) >= ?"], [since, min_quality]
    if subs:
        names = list(subs)
        clause.append(f"subreddit IN ({','.join('?' * len(names))})")
        params.extend(names)
    if theme:
        clause.append("subreddit IN (SELECT name FROM subreddit WHERE theme=? COLLATE NOCASE)")
        params.append(theme)
    rows = conn.execute(
        f"""SELECT id, subreddit, author, title, selftext, score, num_comments, quality,
                   created_utc, tickers, flair, permalink
            FROM post WHERE {' AND '.join(clause)}
            ORDER BY COALESCE(quality,0) * LOG(2 + COALESCE(score,0)) DESC
            LIMIT ?""", [*params, limit]).fetchall()

    groups: dict[str, list] = {}
    for r in rows:
        if T.is_junk(r["title"]):
            continue
        tick = None
        try:
            tickers = json.loads(r["tickers"] or "[]")
            tick = tickers[0] if tickers else None
        except json.JSONDecodeError:
            pass
        theme_of = conn.execute("SELECT theme FROM subreddit WHERE name=? COLLATE NOCASE",
                                (r["subreddit"],)).fetchone()
        key = tick or (theme_of["theme"] if theme_of else r["subreddit"].lower())
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for key, posts in groups.items():
        posts = sorted(posts, key=lambda p: -((p["quality"] or 0) * math.log1p(p["score"] or 0)))
        head = posts[0]
        kept = [head]
        for p in posts[1:]:
            if len(kept) >= per_cluster:
                break
            if all(_jaccard(p["title"], k["title"]) < 0.45 for k in kept):
                kept.append(p)
        engagement = sum((p["score"] or 0) for p in posts)
        comments = sum((p["num_comments"] or 0) for p in posts)
        authors = len({p["author"] for p in posts if p["author"]})
        subs_in = {p["subreddit"] for p in posts}
        quality = max((p["quality"] or 0) for p in posts)
        out.append({
            "key": key,
            "ticker": head_key_ticker(conn, head),
            "theme": theme or "mixed",
            "posts": [dict(p) for p in kept],
            "n_posts": len(posts),
            "n_subs": len(subs_in),
            "subs": sorted(subs_in)[:6],
            "authors": authors,
            "engagement": engagement,
            "comments": comments,
            "quality": round(quality, 3),
            "score": round(quality * math.log1p(engagement) * (1 + 0.25 * min(4, len(subs_in))), 3),
        })
    out.sort(key=lambda c: -c["score"])
    return out


def head_key_ticker(conn, post) -> str | None:
    try:
        tickers = json.loads(post["tickers"] or "[]")
    except (json.JSONDecodeError, TypeError):
        return None
    return tickers[0] if tickers else None


def cluster_metrics(conn, cluster: dict, *, hours: float) -> dict:
    """The crowd numbers behind a candidate: how loud it is, and how fast it got loud."""
    metrics = {"posts": cluster["n_posts"], "posts_kept": len(cluster["posts"]),
               "subs": cluster["subs"], "authors": cluster["authors"],
               "engagement": cluster["engagement"], "comments": cluster["comments"],
               "window_hours": hours}
    tick = cluster.get("ticker")
    if tick:
        rows = retrieve.tickers(conn, hours=hours, symbol=tick, limit=1)
        if rows:
            metrics["mentions"] = {k: rows[0][k] for k in ("symbol", "mentions", "prior",
                                                            "delta", "change")}
    subs = cluster["subs"][:6]
    if subs:
        rows = conn.execute(
            f"""SELECT subreddit, count(*) AS n, sum(score) AS score
                FROM post WHERE subreddit IN ({','.join('?' * len(subs))})
                  AND created_utc >= ? GROUP BY subreddit ORDER BY n DESC""",
            [*subs, int(time.time() - hours * 3600)]).fetchall()
        metrics["by_sub"] = [dict(r) for r in rows]
    return metrics


# --------------------------------------------------------------------------- #
# the evidence pack
# --------------------------------------------------------------------------- #
def evidence_pack(conn, cluster: dict, *, budget: int | None = None,
                  news: list[dict] | None = None) -> tuple[str, list[str]]:
    """A bounded block of primary text, plus the refs that are legal to cite."""
    budget = budget or C.GEN_SNIPPET_BUDGET
    parts: list[str] = []
    refs: list[str] = []
    used = 0

    for p in cluster["posts"][:3]:
        ref = T.post_ref(p["subreddit"], p["id"])
        body = T.clean(p.get("selftext") or "")[:2200]
        block = (f"### POST {ref}  r/{p['subreddit']} · score {p['score']} · "
                 f"{p['num_comments']} comments · flair {p.get('flair') or '-'}\n"
                 f"{p['title']}\n{body}\n")
        if used + len(block) > budget:
            break
        parts.append(block)
        refs.append(ref)
        used += len(block)

    # comments carry most of the argument: take the highest score×quality ones
    for p in cluster["posts"][:3]:
        rows = conn.execute(
            """SELECT id, author, body, score, quality, created_utc
               FROM comment WHERE post_id=? AND COALESCE(chars,0) >= 120
               ORDER BY COALESCE(score,0) * (0.4 + COALESCE(quality,0)) DESC LIMIT 8""",
            (p["id"],)).fetchall()
        for c in rows:
            if used >= budget:
                break
            body = T.clean(c["body"])[:1100]
            ref = T.comment_ref(p["subreddit"], p["id"], c["id"])
            block = (f"--- COMMENT {ref} · u/{c['author']} · score {c['score']}\n{body}\n")
            if used + len(block) > budget:
                continue
            parts.append(block)
            refs.append(ref)
            used += len(block)

    if news:
        lines = []
        for n in news[:8]:
            when = (n.get("published_at") or "")[:16]
            link = "link not resolvable (Google News hit)" if n.get("discovery") else n.get("url")
            lines.append(f"- {when} · {n.get('publisher') or '?'} · {n.get('title')} · {link}")
        if lines:
            parts.append("### RECENT NEWS (dated, for cross-checking the crowd's claim)\n"
                         + "\n".join(lines))
    return "\n".join(parts), refs


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
SYSTEM = """You mine Reddit for investment and trading ideas. The value of this work is that \
certain subreddits contain arguments that are earlier and more concrete than sell-side \
coverage: a customer's capex line, a procurement award nobody wrote up, an operator noticing \
a backlog, a margin structure described by someone who works in it.

Rules, and they are not negotiable:
- Write only what the supplied evidence supports. If the evidence is thin, say so in the \
thesis rather than filling the gap with plausible narrative.
- Cite the evidence you used by its reference string, exactly as given (r/<sub>/<post id>). \
Never invent a reference.
- No price targets, no "should be worth X". This is a read on a mechanism, not a valuation.
- The thesis is at most three sentences. The mechanism names the causal chain and, if you can, \
the link in it that breaks first.
- `crowd_edge` says what the Reddit crowd saw, or saw earlier, than the mainstream — this is \
the whole point of reading Reddit, so be specific about it.
- If nothing in the evidence is tradeable, return an empty ideas list. An honest empty answer \
is better than a manufactured idea.

Return JSON only, matching the requested schema."""

SYNTH_PROMPT = """A candidate theme was selected from {window:.0f} hours of Reddit flow.

CLUSTER: {key}
SUBJECT: {subject}
CROWD METRICS: {metrics}

EVIDENCE (primary text, with references):
{evidence}

Recent news below the evidence is dated and from a news wire, not from Reddit; treat it as a 
cross-check on the crowd's claim, not as the idea itself.

Write at most {n} ideas. Each idea must be one of:
- a directional view a trader could act on (long / short)
- a watch item with a specific trigger dated ahead
- an avoid, where the crowd is wrong in a way that is about to be tested

Return exactly this JSON shape:
{{"ideas": [{{"headline": "<= 110 chars, specific, no hype",
  "direction": "long|short|watch|avoid",
  "tickers": ["NVDA"],
  "thesis": "<= 3 sentences, the claim and why it is true now",
  "mechanism": "the causal chain, and which link breaks first",
  "catalyst": "the event or observation that will test this",
  "catalyst_date": "YYYY-MM-DD or null",
  "invalidation": "the observable that proves this wrong",
  "crowd_edge": "what this crowd saw before the mainstream did",
  "conviction": 0.0,
  "refs": ["r/sub/postid"],
  "new_evidence": "the one fact in the pack that is not already in the news"}}]}}

If the evidence does not support an idea, return {{"ideas": []}}."""

SELECT_PROMPT = """Here are {n} candidate themes from {window:.0f} hours of Reddit flow. Each has \
the number of posts, the engagement, and its strongest titles.

{candidates}

Pick the {k} that most plausibly contain a tradeable insight that is NOT already consensus. \
Prefer themes where the evidence is specific (a number, a named customer, a dated event) and \
where the crowd is arguing rather than cheering.

Return JSON: {{"picks": [{{"key": "...", "reason": "one sentence", "direction_hint": \
"long|short|watch|avoid"}}]}}"""


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _norm_idea(raw: dict, cluster: dict, allowed_refs: set[str], news: list[dict],
               *, hours: float, model: str) -> dict | None:
    if not isinstance(raw, dict):
        return None
    headline = str(raw.get("headline") or "").strip()
    thesis = str(raw.get("thesis") or "").strip()
    if len(headline) < 12 or len(thesis) < 40:
        return None
    direction = str(raw.get("direction") or "watch").strip().lower()
    if direction not in DIRECTIONS:
        direction = "watch"
    refs = [r for r in (raw.get("refs") or []) if isinstance(r, str)]
    refs = [r for r in refs if r in allowed_refs] or sorted(allowed_refs)[:2]
    tickers = [str(t).upper().lstrip("$") for t in (raw.get("tickers") or []) if str(t).strip()]
    if not tickers and cluster.get("ticker"):
        tickers = [cluster["ticker"]]
    try:
        conviction = float(raw.get("conviction") or 0.5)
    except (TypeError, ValueError):
        conviction = 0.5
    conviction = max(0.0, min(1.0, conviction))
    news_refs = [{"title": n.get("title"), "url": n.get("url"),
                  "published_at": n.get("published_at"), "discovery": n.get("discovery")}
                 for n in news[:3]]
    return {
        "headline": headline[:180],
        "direction": direction,
        "tickers": tickers[:5],
        "thesis": thesis[:1200],
        "mechanism": str(raw.get("mechanism") or "").strip()[:900] or None,
        "catalyst": str(raw.get("catalyst") or "").strip()[:400] or None,
        "catalyst_date": _date(raw.get("catalyst_date")),
        "invalidation": str(raw.get("invalidation") or "").strip()[:500] or None,
        "crowd_edge": str(raw.get("crowd_edge") or "").strip()[:600] or None,
        "conviction": round(conviction, 2),
        "refs": refs,
        "news_refs": news_refs,
        "metrics": {"cluster": cluster["key"], "crowd": {
            "posts": cluster["n_posts"], "comments": cluster["comments"],
            "authors": cluster["authors"], "subs": cluster["subs"],
            "engagement": cluster["engagement"]}, "window_hours": hours},
        "model": model,
        "new_evidence": str(raw.get("new_evidence") or "").strip()[:500] or None,
    }


def _date(v) -> str | None:
    if not v or not isinstance(v, str):
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", v.strip())
    return m.group(0) if m else None


def _cosine(a, b) -> float:
    import numpy as np
    if a is None or b is None:
        return 0.0
    va, vb = np.frombuffer(a, dtype=np.float32), np.frombuffer(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def _embed_idea(text: str):
    import numpy as np
    vec = retrieve._embed_query(text)
    return None if vec is None else np.asarray(vec, dtype=np.float32)


def generate(conn, *, hours: float = 72.0, theme: str | None = None,
             subs: Iterable[str] | None = None, n_ideas: int | None = None,
             candidates_n: int | None = None, select: bool = True,
             news: bool = True, model: str | None = None,
             progress=None) -> dict:
    """Select candidates, synthesize ideas, attach news, store what survives."""
    n_ideas = n_ideas or C.GEN_IDEAS
    candidates_n = candidates_n or C.GEN_CANDIDATES
    # You cannot write more ideas than clusters you looked at, so asking for more
    # ideas silently widens the pool instead of quietly under-delivering.
    candidates_n = max(candidates_n, n_ideas)
    model = model or C.LLM_MODEL
    if not llm.configured():
        return {"error": "no OLLAMA_API_KEY in the workspace .env — the generator needs one "
                         "model call per idea; `rdx.py topics` lists the candidate clusters "
                         "it would have used"}

    run_id = store.run_start(conn, "generate", {"hours": hours, "theme": theme})
    pool = candidates(conn, hours=hours, theme=theme, subs=subs, limit=400)
    if not pool:
        store.run_finish(conn, run_id, status="ok", stats={"ideas": 0, "pool": 0})
        return {"ideas": [], "candidates": 0, "note": "nothing in the window passed the quality gate"}

    if progress:
        progress({"event": "candidates", "n": len(pool)})

    # ---- stage 1: rank ---------------------------------------------------- #
    chosen = pool[:candidates_n]
    if select and len(pool) > 1:
        compact = [{
            "key": c["key"],
            "posts": c["n_posts"],
            "engagement": c["engagement"],
            "comments": c["comments"],
            "subs": c["subs"],
            "titles": [p["title"][:110] for p in c["posts"][:3]],
        } for c in pool[:24]]
        try:
            picked = llm.chat_json([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": SELECT_PROMPT.format(
                    n=len(compact), window=hours, k=min(n_ideas, candidates_n),
                    candidates=json.dumps(compact, indent=1))}],
                model=model, tag="select", num_predict=1200)
            picks = picked.get("picks") if isinstance(picked, dict) else None
            if isinstance(picks, list) and picks:
                want = [str(p.get("key")) for p in picks if isinstance(p, dict)]
                ordered = [c for k in want for c in pool if c["key"] == k]
                chosen = (ordered or pool[:candidates_n])[:candidates_n]
                if progress:
                    progress({"event": "selected", "keys": [c["key"] for c in chosen],
                              "reasons": [p.get("reason") for p in picks if isinstance(p, dict)]})
        except llm.LLMError as exc:
            if progress:
                progress({"event": "select_failed", "error": str(exc)[:200]})

    # ---- stage 2: synthesize --------------------------------------------- #
    news_map: dict[str, list[dict]] = {}
    if news:
        tickers = sorted({c["ticker"] for c in chosen if c.get("ticker")})
        if tickers:
            got = newswire.stock(conn, tickers[:6], hours=C.GEN_NEWS_HOURS)
            news_map = got.get("by_ticker") or {}
            if progress:
                progress({"event": "news", "tickers": tickers[:6],
                          "error": got.get("error"), "found": sum(len(v) for v in news_map.values())})

    created: list[dict] = []
    merged = 0
    for cluster in chosen:
        if len(created) >= n_ideas:
            break
        tick = cluster.get("ticker")
        pack_news = news_map.get(tick, []) if tick else []
        if not tick and news and (theme or cluster["subs"]):
            topic = theme or cluster["key"].replace("_", " ")
            got = newswire.query(conn, topic, hours=C.GEN_NEWS_HOURS, limit=8)
            pack_news = got.get("items") or []
        evidence, allowed = evidence_pack(conn, cluster, news=pack_news)
        if len(evidence) < 400:
            if progress:
                progress({"event": "skip", "key": cluster["key"],
                          "why": f"evidence too thin ({len(evidence)} chars)"})
            continue
        metrics = cluster_metrics(conn, cluster, hours=hours)
        subject = f"{tick or cluster['key']} · subs {', '.join(cluster['subs'])}"
        try:
            out = llm.chat_json([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": SYNTH_PROMPT.format(
                    window=hours, key=cluster["key"], subject=subject,
                    metrics=json.dumps(metrics, default=str),
                    evidence=evidence, n=min(2, n_ideas - len(created)))}],
                model=model, tag="synthesize")
        except llm.LLMError as exc:
            if progress:
                progress({"event": "synth_failed", "key": cluster["key"], "error": str(exc)[:200]})
            continue

        ideas_raw = out.get("ideas") if isinstance(out, dict) else out
        if not isinstance(ideas_raw, list):
            continue
        for raw in ideas_raw:
            idea = _norm_idea(raw, cluster, set(allowed), pack_news, hours=hours, model=model)
            if not idea:
                continue
            emb = _embed_idea(idea["headline"] + ". " + idea["thesis"])
            blob = None if emb is None else emb.astype("float32").tobytes()
            # novelty: an idea that repeats one we already hold is evidence, not a new idea
            dup_id = _nearest(conn, idea["headline"] + ". " + idea["thesis"], emb)
            if dup_id:
                _merge_into(conn, dup_id, idea)
                merged += 1
                continue
            idea["emb"] = blob
            idea["created_at"] = T.now_iso()
            idea["notes"] = idea.pop("new_evidence", None)
            docs = []
            for ref in idea["refs"]:
                try:
                    kind, _sub, _pid, cid = T.parse_ref(ref)
                except ValueError:
                    continue
                docs.append((kind, cid if kind == "comment" else _pid, 1.0))
            idea_id = store.insert_idea(conn, idea, docs)
            idea["id"] = idea_id
            created.append(idea)
            if progress:
                progress({"event": "idea", "id": idea_id, "headline": idea["headline"],
                          "direction": idea["direction"], "conviction": idea["conviction"]})

    result = {"ideas": created, "merged": merged, "candidates": len(pool),
              "examined": [c["key"] for c in chosen],
              "stored": conn.execute("SELECT count(*) FROM idea").fetchone()[0],
              "usage": llm.usage_snapshot(reset=True), "model": model}
    store.run_finish(conn, run_id, stats={"ideas": len(created), "merged": merged,
                                          "candidates": len(pool), "examined": result["examined"]})
    return result


def _nearest(conn, text: str, emb) -> int | None:
    """The id of an existing idea this one is near-duplicating, if any."""
    if emb is None:
        return None
    best_id, best = None, 0.0
    for idea_id, _headline, blob in store.idea_embeddings(conn):
        if not blob:
            continue
        cos = _cosine(emb.astype("float32").tobytes(), blob)
        if cos > best:
            best_id, best = idea_id, cos
    return best_id if best >= C.GEN_DEDUPE_COSINE else None


def _merge_into(conn, idea_id: int, idea: dict) -> None:
    """Add the new evidence to the idea it duplicates rather than minting a twin."""
    row = conn.execute("SELECT refs, metrics FROM idea WHERE id=?", (idea_id,)).fetchone()
    if not row:
        return
    refs = set(json.loads(row["refs"] or "[]")) | set(idea.get("refs") or [])
    conn.execute("UPDATE idea SET refs=? WHERE id=?", (json.dumps(sorted(refs)), idea_id))
    for ref in idea.get("refs") or []:
        try:
            kind, _s, pid, cid = T.parse_ref(ref)
        except ValueError:
            continue
        conn.execute("INSERT OR REPLACE INTO idea_doc(idea_id, doc_kind, doc_id, weight) "
                     "VALUES(?,?,?,1.0)", (idea_id, kind, cid if kind == "comment" else pid))
    conn.commit()


# --------------------------------------------------------------------------- #
# reading ideas back
# --------------------------------------------------------------------------- #
# Sort keys are a closed set: the value reaches SQL, so it is mapped, never interpolated.
IDEA_SORTS = {
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
    "conviction": "(conviction IS NULL), conviction DESC, created_at DESC",
    "conviction-asc": "(conviction IS NULL), conviction ASC, created_at DESC",
    # actionable first, not alphabetical: long, short, watch, avoid
    "direction": ("CASE direction WHEN 'long' THEN 0 WHEN 'short' THEN 1 "
                  "WHEN 'watch' THEN 2 WHEN 'avoid' THEN 3 ELSE 4 END, "
                  "(conviction IS NULL), conviction DESC"),
    "marked": "(marked_at IS NULL), marked_at DESC, created_at DESC",
    "catalyst": "(catalyst_date IS NULL), catalyst_date ASC, created_at DESC",
}


PROMOTE_PROMPT = """A reader pointed at one thread and asked whether it becomes an idea. The batch generator normally clusters first and asks which cluster is worth reading; that question is already answered here, so decide only whether *this* thread carries a tradeable claim.

REFERENCE: {ref}
WHERE: r/{sub} · post score {score} · {comments} comments · age {age_hours:.0f}h
TICKERS THE CROWD NAMED: {tickers}
CROWD METRICS: {metrics}

EVIDENCE (primary text, references are citable exactly as written):
{evidence}

Recent news below is dated and from a wire, not from Reddit; it is a cross-check, not the idea.

Return exactly this JSON shape:
{{"verdict": "idea" | "no_claim",
  "why": "one sentence — if no_claim, name the specific thing that is missing",
  "ideas": [{{"headline": "<= 110 chars", "direction": "long|short|watch|avoid",
    "tickers": ["NVDA"], "thesis": "<= 3 sentences",
    "mechanism": "the causal chain and which link breaks first",
    "catalyst": "the event or observation that tests it",
    "catalyst_date": "YYYY-MM-DD or null",
    "invalidation": "the observable that proves it wrong",
    "crowd_edge": "what this thread saw before the mainstream",
    "conviction": 0.0, "refs": ["r/sub/postid"],
    "new_evidence": "the fact here that is not already in the news"}}]}}

"no_claim" is a legitimate and common answer: a rant, a loss screenshot, a question with no answer, a thread that only restates the news. Do not manufacture an idea to be agreeable. When the verdict is no_claim, return an empty ideas list."""


def promote(conn, ref: str, *, model: str | None = None, news: bool = True,
            progress=None) -> dict:
    """Run one thread through the generator: does this become an idea?

    One synthesis call, no ranking call, no clustering — the human already chose the
    thread. Returns either the idea (stored, or folded into the one it duplicates) or
    an explicit refusal with the reason, which is itself a useful answer.
    """
    model = model or C.LLM_MODEL
    kind, sub, post_id, comment_id = T.parse_ref(ref)
    post = conn.execute(
        """SELECT id, subreddit, author, title, selftext, score, num_comments,
                  created_utc, tickers, flair, quality
           FROM post WHERE lower(subreddit)=lower(?) AND id=?""", (sub, post_id)).fetchone()
    if post is None:
        return {"ref": ref, "verdict": "no_claim", "idea": None,
                "why": "that post is not in the store — crawl it first (rdx.py crawl --hours N)"}
    sub = post["subreddit"]
    age_hours = max(0.0, (time.time() - (post["created_utc"] or time.time())) / 3600.0)

    try:
        tickers = json.loads(post["tickers"] or "[]")
    except json.JSONDecodeError:
        tickers = []
    tick = tickers[0] if tickers else None

    parts, allowed, used = [], [], 0
    body = T.clean(post["selftext"] or "")[:2400]
    block = (f"### POST {T.post_ref(sub, post_id)} · r/{sub} · score {post['score']} · "
             f"{post['num_comments']} comments · flair {post['flair'] or '-'}\n"
             f"{post['title']}\n{body}\n")
    parts.append(block)
    allowed.append(T.post_ref(sub, post_id))
    used += len(block)

    if comment_id:
        c = conn.execute("SELECT id, author, body, score FROM comment WHERE post_id=? AND id=?",
                         (post_id, comment_id)).fetchone()
        if c is not None:
            text = T.clean(c["body"] or "")[:1800]
            block = f"--- THE COMMENT THE READER POINTED AT {T.comment_ref(sub, post_id, comment_id)}\n{text}\n"
            parts.append(block)
            allowed.append(T.comment_ref(sub, post_id, comment_id))
            used += len(block)

    rows = conn.execute(
        """SELECT id, author, body, score, quality FROM comment
           WHERE post_id=? AND COALESCE(chars,0) >= 120 AND id IS NOT ?
           ORDER BY COALESCE(score,0) * (0.4 + COALESCE(quality,0)) DESC LIMIT 10""",
        (post_id, comment_id or "")).fetchall()
    for c in rows:
        if used >= C.GEN_SNIPPET_BUDGET:
            break
        text = T.clean(c["body"] or "")[:1100]
        ref_c = T.comment_ref(sub, post_id, c["id"])
        block = f"--- COMMENT {ref_c} · u/{c['author']} · score {c['score']}\n{text}\n"
        if used + len(block) > C.GEN_SNIPPET_BUDGET:
            continue
        parts.append(block)
        allowed.append(ref_c)
        used += len(block)

    pack_news: list[dict] = []
    if news and tick:
        got = newswire.stock(conn, [tick], hours=C.GEN_NEWS_HOURS)
        pack_news = (got.get("by_ticker") or {}).get(tick, [])
        if pack_news:
            lines = [f"- {(n.get('published_at') or '')[:16]} · {n.get('publisher') or '?'} · "
                     f"{n.get('title')} · {n.get('url') or 'link not resolvable'}"
                     for n in pack_news[:6]]
            parts.append("### RECENT NEWS (dated, cross-check only)\n" + "\n".join(lines))

    evidence = "\n".join(parts)
    if len(evidence) < 300:
        return {"ref": ref, "verdict": "no_claim", "idea": None,
                "why": "there is almost nothing in this thread — no body and no comments to read"}

    cluster = {"key": tick or sub.lower(), "ticker": tick, "theme": None, "subs": [sub],
               "posts": [dict(post)], "n_posts": 1,
               "engagement": post["score"] or 0, "comments": post["num_comments"] or 0,
               "authors": 1}
    metrics = {"key": cluster["key"], "posts": 1, "comments": cluster["comments"],
               "engagement": cluster["engagement"], "age_hours": round(age_hours, 1)}

    run_id = store.run_start(conn, "promote", {"ref": ref})
    try:
        out = llm.chat_json([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMOTE_PROMPT.format(
                ref=ref, sub=sub, score=post["score"], comments=post["num_comments"],
                age_hours=age_hours, tickers=", ".join(tickers) or "none named",
                metrics=json.dumps(metrics), evidence=evidence)}],
            model=model, tag="promote")
    except llm.LLMError as exc:
        store.run_finish(conn, run_id, status="error", error=str(exc)[:300])
        return {"ref": ref, "verdict": "error", "idea": None,
                "why": f"the model call failed: {exc}"}

    verdict = str(out.get("verdict") or "").strip().lower() if isinstance(out, dict) else ""
    why = str(out.get("why") or "").strip() if isinstance(out, dict) else ""
    raws = out.get("ideas") if isinstance(out, dict) else None
    raws = [r for r in (raws or []) if isinstance(r, dict)]
    if verdict == "no_claim" or not raws:
        store.run_finish(conn, run_id, status="ok", stats={"verdict": "no_claim"})
        return {"ref": ref, "verdict": "no_claim", "idea": None,
                "why": why or "the model read the thread and found no claim worth acting on",
                "model": model, "evidence_chars": len(evidence)}

    idea = _norm_idea(raws[0], cluster, set(allowed), pack_news, hours=age_hours, model=model)
    if not idea:
        store.run_finish(conn, run_id, status="ok", stats={"verdict": "rejected"})
        return {"ref": ref, "verdict": "no_claim", "idea": None,
                "why": why or "the model's answer did not survive the idea checks "
                              "(missing thesis, or a reference it could not cite)"}

    # A second read of the same thread in the same direction is not a new idea, even
    # when the wording has drifted past the embedding threshold.
    post_ref = T.post_ref(sub, post_id)
    for row in conn.execute("SELECT id, direction FROM idea WHERE refs LIKE ?",
                            (f'%"{post_ref}"%',)).fetchall():
        if (row["direction"] or "") == idea["direction"]:
            _merge_into(conn, row["id"], idea)
            store.run_finish(conn, run_id, status="ok",
                             stats={"verdict": "duplicate", "id": row["id"]})
            return {"ref": ref, "verdict": "duplicate", "idea": None, "merged_into": row["id"],
                    "why": f"#{row['id']} already rests on this thread in the same direction — "
                           "the extra evidence was added to it",
                    "model": model, "evidence_chars": len(evidence)}

    emb = _embed_idea(idea["headline"] + ". " + idea["thesis"])
    dup_id = _nearest(conn, idea["headline"] + ". " + idea["thesis"], emb)
    if dup_id:
        _merge_into(conn, dup_id, idea)
        store.run_finish(conn, run_id, status="ok", stats={"verdict": "duplicate", "id": dup_id})
        return {"ref": ref, "verdict": "duplicate", "idea": None, "merged_into": dup_id,
                "why": "this is the same idea as one already stored — its evidence was added "
                       "to that idea instead of minting a twin",
                "model": model, "evidence_chars": len(evidence)}

    if emb is not None:
        idea["emb"] = emb.astype("float32").tobytes()
    docs = [("post", post_id, 1.0)]
    if comment_id:
        docs.append(("comment", comment_id, 1.0))
    idea_id = store.insert_idea(conn, idea, docs)
    conn.commit()
    store.run_finish(conn, run_id, status="ok", stats={"verdict": "idea", "id": idea_id})
    if progress:
        progress({"event": "promoted", "id": idea_id, "ref": ref})
    return {"ref": ref, "verdict": "idea", "idea": {**idea, "id": idea_id}, "idea_id": idea_id,
            "why": why, "model": model, "evidence_chars": len(evidence),
            "usage": llm.usage_snapshot()}


def list_ideas(conn, *, limit: int = 40, direction: str | None = None,
               theme: str | None = None, status: str | None = None,
               statuses: Sequence[str] | None = None,
               bookmarked: bool = False, sort: str = "newest") -> list[dict]:
    """Ideas in the order asked for; bookmarked ones default to when they were marked."""
    wanted = list(statuses) if statuses else ([status] if status else [])
    if bookmarked:
        wanted = list(store.BOOKMARK_STATUSES)
    if sort not in IDEA_SORTS:
        raise ValueError(f"sort must be one of {', '.join(IDEA_SORTS)}")
    clause, params = ["1=1"], []
    if direction:
        clause.append("direction=?")
        params.append(direction)
    if theme:
        clause.append("theme=?")
        params.append(theme)
    if wanted:
        clause.append(f"status IN ({','.join('?' * len(wanted))})")
        params.extend(wanted)
    rows = conn.execute(
        f"""SELECT id, created_at, marked_at, headline, direction, tickers, thesis,
                   conviction, theme, catalyst_date, status
            FROM idea WHERE {' AND '.join(clause)}
            ORDER BY {IDEA_SORTS[sort]} LIMIT ?""",
        [*params, limit]).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tickers"] = json.loads(d["tickers"] or "[]")
        d["bookmarked"] = bool(d.get("marked_at"))
        out.append(d)
    return out


def get_idea(conn, idea_id: int, *, evidence_chars: int = 700) -> dict | None:
    row = conn.execute("SELECT * FROM idea WHERE id=?", (idea_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d.pop("emb", None)
    for k in ("tickers", "metrics", "refs", "news_refs"):
        try:
            d[k] = json.loads(d[k] or "null")
        except json.JSONDecodeError:
            d[k] = None
    d["evidence"] = []
    for ref_row in conn.execute(
            "SELECT doc_kind, doc_id FROM idea_doc WHERE idea_id=? ORDER BY weight DESC",
            (idea_id,)):
        try:
            shown = retrieve.show(conn, _ref_from_row(conn, ref_row), comments=1)
        except Exception:
            continue
        for key in ("post", "comment"):
            doc = shown.get(key)
            if doc and (doc.get("selftext") or doc.get("body")):
                full = doc.get("selftext") or doc.get("body") or ""
                d["evidence"].append({
                    "ref": _ref_from_row(conn, ref_row),
                    "kind": key,
                    "author": doc.get("author"),
                    "score": doc.get("score"),
                    "created_at": T.iso(doc.get("created_utc")),
                    "text": full[:evidence_chars],
                    # say so plainly: the excerpt is a peek, the document is in the store
                    "text_chars": len(full),
                    "text_truncated": len(full) > evidence_chars,
                })
                break
    return d


def _ref_from_row(conn, row) -> str:
    if row["doc_kind"] == "post":
        p = conn.execute("SELECT subreddit FROM post WHERE id=?", (row["doc_id"],)).fetchone()
        return T.post_ref(p["subreddit"], row["doc_id"]) if p else f"r/unknown/{row['doc_id']}"
    c = conn.execute("SELECT subreddit, post_id FROM comment WHERE id=?", (row["doc_id"],)).fetchone()
    if not c:
        return f"r/unknown/{row['doc_id']}"
    return T.comment_ref(c["subreddit"], c["post_id"], row["doc_id"])
