#!/usr/bin/env python3
"""rdx — the command line for reddit-alpha.

One file, one store, every entry point the rest of the workspace needs:

    crawl      pull new posts and comments into the store (incremental by default)
    reindex    rebuild chunks for documents already stored
    embed      compute vectors for the chunks that qualify
    search     locate evidence, ranked (lexical, vector, or fused)
    show       re-read one post or one comment by reference
    thread     a whole thread, ranked
    trending   what is taking off right now
    tickers    mention counts and their change against the prior window
    subs       the editable subreddit registry (list/add/rm/enable/validate/import/export)
    news       dated headlines for tickers or a theme (via news-pull)
    generate   turn the window into ideas, with evidence attached
    ideas      list stored ideas; `idea <id>` opens one
    stats      what is in the store
    check      structural health of the index
    export     documents out as jsonl/json/markdown for another tool
    serve      the dashboard

Run it with the workspace interpreter, which has numpy and model2vec:

    ../.venv/bin/python rdx.py stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as C          # noqa: E402
import store                # noqa: E402
import textproc as T        # noqa: E402
import retrieve as R        # noqa: E402


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _progress(evt: dict) -> None:
    kind = evt.get("event")
    if kind == "sub_start":
        print(f"· r/{evt['sub']}", file=sys.stderr, flush=True)
    elif kind == "sub_done":
        s = evt["stats"]
        err = f"  ⚠ {s['error']}" if s.get("error") else ""
        print(f"  posts {s['posts']} (+{s['posts_new']}) · comments {s['comments']} "
              f"(+{s['comments_new']}) · chunks {s['chunks']} · hydrated {s['hydrated']}{err}",
              file=sys.stderr, flush=True)
    elif kind == "hydrate":
        print(f"  ↳ hydrated r/{evt['sub']}/{evt['post']} ({evt['comments']} comments)",
              file=sys.stderr, flush=True)
    elif kind == "page":
        total = evt.get("budget")
        print(f"  … {evt['pages']} pages" + (f" / {total}" if total else " spent"),
              file=sys.stderr, flush=True)
    elif kind == "embed":
        print(f"  embedded {evt['done']}/{evt['total']}", file=sys.stderr, flush=True)
    elif kind == "candidates":
        print(f"· {evt['n']} candidate clusters", file=sys.stderr, flush=True)
    elif kind == "selected":
        print(f"· selected: {', '.join(evt['keys'])}", file=sys.stderr, flush=True)
    elif kind == "idea":
        print(f"  ✦ {evt['direction'].upper():5s} {evt['headline']}  "
              f"(conviction {evt['conviction']})", file=sys.stderr, flush=True)
    elif kind in ("synth_failed", "select_failed"):
        print(f"  ⚠ {kind}: {evt.get('error')}", file=sys.stderr, flush=True)


def _fmt_score(n) -> str:
    if n is None:
        return "-"
    return f"{n:,.0f}"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_crawl(args, conn) -> int:
    import crawl as crawl_mod
    import sources

    src = sources.get_source(args.source, path=args.file)
    subs = [s.strip().lstrip("/").removeprefix("r/") for s in (args.subs or "").split(",") if s.strip()]
    # a sub named explicitly on the command line is a statement of intent: register it
    for name in subs:
        if not conn.execute("SELECT 1 FROM subreddit WHERE name=? COLLATE NOCASE", (name,)).fetchone():
            store.upsert_subreddit(conn, name, theme="other",
                                   notes=f"added by explicit --subs on {T.now_iso()[:10]}")
            print(f"+ r/{name} registered (theme other — set one with `subs add`)", file=sys.stderr)
    res = crawl_mod.crawl(conn, subs=subs or None, hours=args.hours, source=src,
                          mode=args.mode, max_pages=args.max_pages,
                          hydrate=not args.no_hydrate, hydrate_max=args.hydrate_max,
                          parallel=args.parallel, budget=args.budget,
                          sub_cap=args.sub_cap,
                          progress=None if args.quiet else _progress)
    if args.embed:
        res["embed"] = crawl_mod.embed_pending(conn, progress=None if args.quiet else _progress)
    if args.json:
        _dump(res)
        return 0
    t = res.get("totals", {})
    print(f"\ncrawled {t.get('subs', 0)} subs · {t.get('posts', 0)} posts "
          f"(+{t.get('posts_new', 0)} new) · {t.get('comments', 0)} comments "
          f"(+{t.get('comments_new', 0)} new) · {t.get('chunks', 0)} chunks · "
          f"{t.get('errors', 0)} errors")
    rps = (res.get("requests", 0) / res.get("elapsed_s", 1)) if res.get("elapsed_s") else 0
    print(f"{res.get('requests', 0)} requests · {rps:.2f} req/s · {res.get('mb', 0)} MB · "
          f"{res.get('elapsed_s', 0)}s · {res.get('slowdowns', 0)} slowdowns · "
          f"pace {res.get('pace', 0)}s · db {res.get('db_mb', 0)} MB")
    budget = res.get("budget")
    spent = res.get("pages", 0)
    print(f"allowance {spent}/{budget if budget else '∞'} pages · "
          f"{res.get('capped_subs', 0)} subreddits capped (they resume next run)")
    if res.get("embed"):
        print(f"embedded {res['embed'].get('embedded', 0)} new chunks")
    for sub, s in (res.get("per_sub") or {}).items():
        if s.get("error"):
            print(f"  ⚠ r/{sub}: {s['error']}")
    return 0


def cmd_embed(args, conn) -> int:
    import crawl as crawl_mod
    res = crawl_mod.embed_pending(conn, min_chars=args.min_chars, min_quality=args.min_quality,
                                  limit=args.limit, progress=None if args.quiet else _progress)
    _dump(res) if args.json else print(f"embedded {res['embedded']} chunks · model {res['model']}")
    return 0


def cmd_reindex(args, conn) -> int:
    import crawl as crawl_mod
    subs = [s.strip() for s in (args.subs or "").split(",") if s.strip()]
    res = crawl_mod.reindex(conn, subs=subs or None, hours=args.hours,
                            progress=None if args.quiet else _progress)
    _dump(res) if args.json else print(f"reindexed {res['posts']} posts and "
                                       f"{res['comments']} comments into {res['chunks']} chunks")
    return 0


def cmd_search(args, conn) -> int:
    hits = R.search(conn, args.query, subs=_subs(args), theme=args.theme, ticker=args.ticker,
                    hours=args.hours, kind=args.kind, min_quality=args.min_quality,
                    min_score=args.min_score, limit=args.limit, mode=args.mode,
                    all_terms=args.all)
    if args.json:
        _dump(hits)
        return 0
    if not hits:
        print("no hits — try --mode lexical, a shorter query, or a wider --hours")
        return 1
    for h in hits:
        meta = f"r/{h['subreddit']} · {h['doc_kind']} · score {_fmt_score(h['document_score'])}"
        if h.get("quality") is not None:
            meta += f" · q {h['quality']:.2f}"
        if h.get("created_at"):
            meta += f" · {h['created_at'][:10]}"
        print(f"\n{h['ref']}\n  {meta}\n  {h['snippet']}")
    return 0


def cmd_show(args, conn) -> int:
    doc = R.show(conn, args.ref, comments=args.comments)
    if args.json:
        _dump(doc)
        return 0
    post = doc.get("post")
    if post:
        print(f"r/{post['subreddit']} · {post['score']} pts · {post['num_comments']} comments"
              f" · {T.iso(post['created_utc'], ) or ''}")
        print(f"u/{post['author']} · flair {post.get('flair') or '-'} · {T.permalink(post['subreddit'], post['id'])}")
        print(f"\n# {post['title']}\n")
        if post.get("selftext"):
            print(post["selftext"][:args.max_chars])
    if doc.get("comment"):
        c = doc["comment"]
        print(f"\n--- comment {T.comment_ref(c['subreddit'], c['post_id'], c['id'])} "
              f"· u/{c['author']} · {c['score']} pts · {T.iso(c['created_utc'])}")
        print(c["body"][:args.max_chars])
    if doc.get("parent_comment"):
        p = doc["parent_comment"]
        print(f"\n--- parent comment (u/{p['author']}, {p['score']} pts)\n{p['body'][:1200]}")
    if doc.get("top_comments"):
        print(f"\n--- top comments ({len(doc['top_comments'])})")
        for c in doc["top_comments"]:
            print(f"  [{c['score']:>5}] u/{c['author']}{' (OP)' if c['is_op'] else ''}: "
                  f"{T.clean(c['body'])[:220]}")
    if doc.get("velocity") is not None:
        print(f"\nvelocity: {doc['velocity']} points/hour across {len(doc.get('metrics') or [])} samples")
    return 0


def cmd_thread(args, conn) -> int:
    doc = R.thread(conn, args.post_id, limit=args.limit, sort=args.sort)
    if args.json:
        _dump(doc)
        return 0 if doc.get("post") else 1
    post = doc.get("post")
    if not post:
        print(f"no post {args.post_id} in the store")
        return 1
    print(f"# {post['title']}\nr/{post['subreddit']} · {post['score']} pts · "
          f"{post['num_comments']} comments · {T.iso(post['created_utc'])}")
    if post.get("selftext"):
        print(f"\n{post['selftext'][:3000]}\n")
    for c in doc["comments"]:
        pad = "  " * min(6, c.get("depth") or 0)
        print(f"{pad}[{c['score']:>5}] u/{c['author']}: {T.clean(c['body'])[:260]}")
    return 0


def cmd_trending(args, conn) -> int:
    rows = R.trending(conn, subs=_subs(args), theme=args.theme, hours=args.hours,
                      limit=args.limit, sort=args.sort)
    if args.json:
        _dump(rows)
        return 0
    if not rows:
        print("nothing in the window — crawl first")
        return 1
    print(f"{'velocity':>9}  {'score':>6}  {'comments':>8}  post")
    for r in rows:
        print(f"{r['velocity']:>9}  {r['score'] or 0:>6}  {r['num_comments'] or 0:>8}  "
              f"{r['ref']}  {r['title'][:80]}")
    return 0


def cmd_tickers(args, conn) -> int:
    rows = R.tickers(conn, hours=args.hours, theme=args.theme, subs=_subs(args),
                     limit=args.limit, symbol=args.symbol)
    if args.json:
        _dump(rows)
        return 0
    if not rows:
        print("no mentions in the window")
        return 1
    # The delta column is a comparison only if the window before it was crawled too.
    import time as _t
    oldest = conn.execute("SELECT min(created_utc) FROM chunk").fetchone()[0] or 0
    if oldest > _t.time() - 2 * args.hours * 3600:
        print("note: the prior window predates what is stored, so `prior` counts are "
              "incomplete — a delta here is coverage, not a comparison\n")
    print(f"{'symbol':8s} {'mentions':>9} {'prior':>7} {'delta':>7}  top document")
    for r in rows:
        top = (r.get("top") or [{}])[0].get("ref", "")
        # A multiplier off a three-mention base is noise dressed as a signal, and
        # before the store holds two full windows every multiplier is that. Say which.
        thin_base = (r["prior"] or 0) < 8
        if r.get("change") is None:
            chg = "new" if r["delta"] else "-"
        else:
            chg = f"{r['change']:+.2f}x" if not thin_base else f"{r['delta']:+d}~"
        print(f"{r['symbol']:8s} {r['mentions']:>9} {r['prior']:>7} {chg:>7}  {top}")
    return 0


def cmd_topics(args, conn) -> int:
    """The clusters the generator would consider — cheap, and it costs no tokens."""
    import ideas as ideas_mod
    rows = ideas_mod.candidates(conn, hours=args.hours, theme=args.theme,
                                subs=_subs(args), limit=args.limit)
    if args.json:
        _dump([{k: v for k, v in r.items() if k != "posts"} |
               {"titles": [p["title"] for p in r["posts"][:3]]} for r in rows])
        return 0
    if not rows:
        print("no clusters in the window — crawl first, or widen it")
        return 1
    print(f"{'cluster':26s} {'posts':>5} {'cmts':>6} {'engage':>7}  titles")
    for r in rows:
        titles = " | ".join(p["title"][:56] for p in r["posts"][:2])
        print(f"{r['key'][:26]:26s} {r['n_posts']:>5} {r['comments']:>6} "
              f"{r['engagement']:>7}  {titles}")
    return 0


def cmd_authors(args, conn) -> int:
    rows = R.authors(conn, limit=args.limit)
    if args.json:
        _dump(rows)
        return 0
    for r in rows:
        print(f"{r['best']:.2f}  u/{r['name']:<24} posts {r['posts']:>4} comments {r['comments']:>5} "
              f"score {r['score_sum']:>7}  {r['top_sub'] or ''}")
    return 0


def cmd_subs(args, conn) -> int:
    import registry
    action = args.action
    if action == "list":
        rows = store.list_subreddits(conn)
        if args.json:
            _dump(rows)
            return 0
        print(f"{'theme':20s} {'sub':24s} {'w':>4} {'on':>3} {'last run':16s} {'24h posts':>9}  note")
        for r in rows:
            print(f"{r['theme'] or '-':20s} {r['name'][:24]:24s} {r['weight']:>4.1f} "
                  f"{'yes' if r['enabled'] else 'no':>3} {(r['last_run_at'] or '-')[:16]:16s} "
                  f"{(r['recent_posts'] if r['recent_posts'] is not None else '-'):>9}  "
                  f"{(r['notes'] or '')[:60]}")
        return 0
    if action == "add":
        for name in args.names:
            store.upsert_subreddit(conn, name, theme=args.theme, weight=args.weight,
                                   enabled=True, notes=args.notes or "")
            print(f"+ r/{name.lstrip('/').removeprefix('r/')} ({args.theme}, weight {args.weight})")
        return 0
    if action in ("rm", "remove"):
        for name in args.names:
            store.delete_subreddit(conn, name)
            print(f"- r/{name} removed from the registry (its stored posts stay)")
        return 0
    if action in ("enable", "disable"):
        for name in args.names:
            store.set_subreddit_enabled(conn, name, action == "enable")
            print(f"{'+' if action == 'enable' else '-'} r/{name}")
        return 0
    if action == "validate":
        import sources
        src = sources.get_source(args.source, path=args.file)
        rows = registry.validate(conn, src, names=args.names or None,
                                 disable_empty=not args.keep_empty,
                                 progress=None if args.json or args.quiet else _progress)
        if args.json:
            _dump(rows)
            return 0
        for r in rows:
            if r.get("error"):
                print(f"  ⚠ {r['name']}: {r['error']}")
            else:
                flag = "ok " if r["alive"] else "dead"
                print(f"  {flag} {r['name']:24s} posts/7d {r['recent_posts']:>4} "
                      f"comments/7d {str(r['recent_comments']):>5}  {r.get('note') or ''}")
        return 0
    if action == "import":
        _dump(registry.load_seed(conn, args.file))
        return 0
    if action == "export":
        _dump(registry.export_seed(conn, args.file))
        return 0
    print(f"unknown subs action: {action}")
    return 2


def cmd_news(args, conn) -> int:
    import newswire
    if args.query:
        res = newswire.query(conn, args.query, hours=args.hours, force=args.force,
                             limit=args.limit)
        items = res.get("items") or []
    elif args.tickers:
        res = newswire.stock(conn, args.tickers, hours=args.hours, force=args.force,
                             limit=args.limit)
        items = [i for v in (res.get("by_ticker") or {}).values() for i in v]
    else:
        items = newswire.recent(conn, hours=args.hours or 72.0, limit=args.limit)
        res = {"error": None}
    if args.json:
        _dump({"items": items, "error": res.get("error")})
        return 0 if not res.get("error") else 1
    if res.get("error"):
        print(f"news failed: {res['error']}")
        return 1
    for it in items:
        mark = "⁺" if it.get("discovery") else " "
        print(f"{mark} {(it.get('published_at') or '')[:16]}  {it.get('publisher') or '?':22s} "
              f"{it['title'][:100]}")
        if it.get("url") and not it.get("discovery"):
            print(f"    {it['url']}")
    if any(it.get("discovery") for it in items):
        print("\n⁺ = Google News discovery hit: headline and publisher are usable, "
              "the link is not. Resolve before citing.")
    return 0


def cmd_generate(args, conn) -> int:
    import crawl as crawl_mod
    import ideas as ideas_mod
    if getattr(args, "think", False):
        ideas_mod.C.LLM_THINK = True
    subs = _subs(args)
    if not args.no_embed:
        # vectors make the novelty check and hybrid search work; keep them current
        try:
            crawl_mod.embed_pending(conn)
        except Exception as exc:                 # never block generation on the embedder
            print(f"  ⚠ embed skipped: {exc}", file=sys.stderr)
    res = ideas_mod.generate(conn, hours=args.hours, theme=args.theme, subs=subs,
                             n_ideas=args.n, candidates_n=args.candidates,
                             select=not args.no_select, news=not args.no_news,
                             model=args.model, progress=None if args.quiet else _progress)
    if args.json:
        _dump(res)
        return 0 if not res.get("error") else 1
    if res.get("error"):
        print(f"error: {res['error']}")
        return 1
    created = res.get("ideas") or []
    if not created:
        total = res.get("stored") or store.counts(conn)["ideas"]
        print(f"no new ideas written ({res.get('candidates', 0)} candidates examined: "
              f"{', '.join(res.get('examined') or []) or 'none'}) · "
              f"{total} already stored, none disturbed")
        return 0
    for idea in created:
        print(f"\n#{idea['id']}  [{idea['direction'].upper()}]  conviction {idea['conviction']}"
              f"  {', '.join(idea['tickers']) or '-'}")
        print(f"  {idea['headline']}")
        print(f"  {idea['thesis']}")
        if idea.get("catalyst"):
            print(f"  catalyst: {idea['catalyst']}"
                  + (f" ({idea['catalyst_date']})" if idea.get("catalyst_date") else ""))
        if idea.get("invalidation"):
            print(f"  wrong if: {idea['invalidation']}")
        print(f"  evidence: {', '.join(idea['refs'])}")
    usage = res.get("usage") or {}
    total = res.get("stored") or store.counts(conn)["ideas"]
    print(f"\n{len(created)} new · {res.get('merged', 0)} folded into an existing idea · "
          f"{total} stored in total (nothing was deleted) · "
          f"{usage.get('calls', 0)} llm calls · "
          f"{usage.get('tokens_in', 0)} in / {usage.get('tokens_out', 0)} out tokens · "
          f"model {res.get('model')}")
    return 0


def cmd_ideas(args, conn) -> int:
    import ideas as ideas_mod
    rows = ideas_mod.list_ideas(conn, limit=args.limit, direction=args.direction,
                                status=args.status, bookmarked=args.bookmarked,
                                sort=args.sort)
    if args.json:
        _dump(rows)
        return 0
    if not rows:
        total = store.counts(conn)["ideas"]
        if args.status or args.bookmarked or args.direction:
            print(f"no ideas match that filter ({total} stored — drop the filter to see them)")
        elif total:
            print(f"{total} ideas stored but none selected")
        else:
            print("no ideas yet — run `rdx.py generate`")
        return 0
    marked = sum(1 for r in rows if r.get("bookmarked"))
    for r in rows:
        star = "★" if r.get("bookmarked") else " "
        stamp = f"{r['created_at'][:10]}"
        print(f"{star} #{r['id']:<3} {stamp} [{r['direction'] or '?':5s}] "
              f"conv {r['conviction'] if r['conviction'] is not None else '-'}  "
              f"{','.join(r['tickers']) or '-':12s} {r['headline']}")
    if marked:
        print(f"\n{marked} of {len(rows)} bookmarked · `rdx.py ideas --bookmarked` lists only those")
    return 0


def ideas_sort_keys() -> list[str]:
    import ideas as ideas_mod
    return list(ideas_mod.IDEA_SORTS)


def cmd_promote(args, conn) -> int:
    """Put one thread in front of the model — the batch generator's question, asked once."""
    import ideas as ideas_mod
    import llm
    if not llm.configured():
        print("no OLLAMA_API_KEY in .env — `rdx.py show <ref>` reads it without the model")
        return 1
    res = ideas_mod.promote(conn, args.ref, model=args.model, news=not args.no_news)
    if args.json:
        _dump(res)
        return 0 if res.get("verdict") in ("idea", "duplicate") else 1
    verdict = res.get("verdict")
    if verdict == "idea":
        idea = res["idea"]
        print(f"IDEAS → #{res['idea_id']}  [{idea['direction'].upper()}]  "
              f"conviction {idea['conviction']}")
        print(f"\n{idea['headline']}\n")
        print(f"THESIS      {idea['thesis']}")
        for field, label in (("mechanism", "MECHANISM"), ("catalyst", "CATALYST"),
                             ("invalidation", "WRONG IF"), ("crowd_edge", "CROWD EDGE"),
                             ("new_evidence", "NEW EVIDENCE")):
            if idea.get(field):
                print(f"{label:12s}{idea[field]}")
        print(f"{'EVIDENCE':12s}{', '.join(idea['refs'])}")
    elif verdict == "duplicate":
        print(f"ALREADY AN IDEA → #{res['merged_into']}: {res['why']}")
    elif verdict == "no_claim":
        print(f"NO IDEA — {res['why']}")
    else:
        print(f"{verdict}: {res.get('why')}")
    return 0 if verdict in ("idea", "duplicate") else 1


def cmd_mark(args, conn) -> int:
    """Bookmark ideas, or move them along — the human's own layer over the machine's."""
    changed = []
    for idea_id in args.ids:
        row = conn.execute("SELECT id FROM idea WHERE id=?", (idea_id,)).fetchone()
        if row is None:
            print(f"  no idea #{idea_id}")
            continue
        store.update_idea_status(conn, idea_id, args.status, args.notes)
        conn.commit()
        row = conn.execute("SELECT status, marked_at FROM idea WHERE id=?",
                           (idea_id,)).fetchone()
        changed.append((idea_id, row["status"], row["marked_at"]))
    if args.json:
        _dump([{"id": i, "status": s_, "marked_at": m} for i, s_, m in changed])
        return 0
    for idea_id, status, marked_at in changed:
        when = f" · marked {marked_at[:16].replace('T', ' ')}" if marked_at else ""
        print(f"  #{idea_id} → {status}{when}")
    return 0


def cmd_idea(args, conn) -> int:
    import ideas as ideas_mod
    doc = ideas_mod.get_idea(conn, args.id)
    if not doc:
        print(f"no idea #{args.id}")
        return 1
    if args.json:
        _dump(doc)
        return 0
    star = "★ " if doc.get("marked_at") else ""
    print(f"{star}#{doc['id']} · {doc['created_at']} · {doc['direction']} · "
          f"conviction {doc['conviction']} · status {doc.get('status') or 'new'}"
          + (f" (marked {doc['marked_at'][:16].replace('T', ' ')})" if doc.get("marked_at") else ""))
    print(f"\n{doc['headline']}\n")
    print(f"THESIS      {doc['thesis']}")
    for field, label in (("mechanism", "MECHANISM"), ("catalyst", "CATALYST"),
                         ("invalidation", "WRONG IF"), ("crowd_edge", "CROWD EDGE"),
                         ("notes", "NEW EVIDENCE")):
        if doc.get(field):
            print(f"{label:<12s} {doc[field]}")
    print(f"TICKERS     {', '.join(doc.get('tickers') or []) or '-'}")
    print(f"METRICS     {json.dumps(doc.get('metrics') or {}, default=str)[:400]}")
    if doc.get("news_refs"):
        print("\nNEWS")
        for n in doc["news_refs"]:
            mark = "⁺ " if n.get("discovery") else "  "
            print(f"{mark}{(n.get('published_at') or '')[:16]}  {n.get('title')}"
                  + ("" if n.get("discovery") else f"\n     {n.get('url')}"))
    if doc.get("evidence"):
        print("\nEVIDENCE")
        for e in doc["evidence"]:
            print(f"  {e['ref']}  u/{e.get('author')} · {e.get('score')} pts · {e.get('created_at','')[:10]}")
            print(f"    {e['text'][:400]}")
    return 0


def cmd_stats(args, conn) -> int:
    s = R.stats(conn)
    if args.json:
        _dump(s)
        return 0
    print(f"store       {C.DB_PATH}  ({s['db_mb']} MB)")
    print(f"subreddits  {s['subreddits_enabled']}/{s['subreddits']} enabled")
    print(f"documents   {s['posts']} posts · {s['comments']} comments")
    print(f"last 24h    {s['posts_24h']} posts · {s['comments_24h']} comments")
    print(f"chunks      {s['chunks']}  "
          f"({s['chunks_by_kind'].get('post', 0)} post · {s['chunks_by_kind'].get('comment', 0)} comment)")
    print(f"vectors     {s['vectors']}  (model {s['vectors_detail'].get('model') or '-'}, "
          f"coverage {s['vectors_detail'].get('coverage', 0):.0%})")
    print(f"mentions    {s['mentions']} · authors {s['authors']} · news {s['news']} · "
          f"ideas {s['ideas']} ({s.get('ideas_marked', 0)} bookmarked)")
    print(f"schema      v{s['schema_version']}")
    if s["last_runs"]:
        print("\nrecent runs")
        for r in s["last_runs"]:
            print(f"  {r['started_at'][:19]}  {r['kind']:9s} {r['status'] or '-':8s} "
                  f"{(r['finished_at'] or '')[:19]}")
    return 0


def cmd_prune(args, conn) -> int:
    res = store.prune_junk_chunks(conn)
    if args.json:
        _dump(res)
        return 0
    print(f"removed {res['removed']} chunks the current junk filter rejects · "
          f"{res['remaining']} remain")
    return 0


def cmd_check(args, conn) -> int:
    res = R.check(conn)
    if args.json:
        _dump(res)
        return 0 if res["ok"] else 1
    print(f"posts {res['posts']} · comments {res['comments']} · chunks {res['chunks']} · "
          f"fts {res['fts_rows']}")
    print(f"indexed documents: {res['chunk_docs']['post']} posts, "
          f"{res['chunk_docs']['comment']} comments")
    if res["issues"]:
        print("\nissues")
        for i in res["issues"]:
            print(f"  ⚠ {i}")
    else:
        print("no structural issues")
    return 0 if res["ok"] else 1


def cmd_export(args, conn) -> int:
    since = args.hours and int(time.time() - args.hours * 3600)
    clause, params = ["stub=0"], []
    if since:
        clause.append("created_utc >= ?")
        params.append(since)
    rows = conn.execute(
        f"SELECT * FROM post WHERE {' AND '.join(clause)} ORDER BY created_utc DESC LIMIT ?",
        [*params, args.limit]).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        comments = [dict(c) for c in conn.execute(
            "SELECT id, parent_id, depth, author, body, score, created_utc FROM comment "
            "WHERE post_id=? ORDER BY COALESCE(score,0) DESC LIMIT ?", (r["id"], args.comments))]
        d["comments"] = comments
        out.append(d)

    if args.out:
        path = Path(args.out)
    else:
        path = HERE / "data" / f"export-{T.now_iso()[:10]}.{args.format}"
    if args.format == "jsonl":
        text = "\n".join(json.dumps(d, default=str, ensure_ascii=False) for d in out)
    elif args.format == "md":
        chunks = []
        for d in out:
            chunks.append(f"## {d['title']}\n\nr/{d['subreddit']} · {d['score']} pts · "
                          f"{T.iso(d['created_utc'])} · {T.permalink(d['subreddit'], d['id'])}\n\n"
                          f"{d.get('selftext') or ''}\n")
            for c in d["comments"][:args.comments]:
                chunks.append(f"> **u/{c['author']}** ({c['score']} pts): "
                              f"{T.clean(c['body'])[:600]}\n")
        text = "\n".join(chunks)
    else:
        text = json.dumps(out, indent=2, default=str, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {len(out)} posts to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_serve(args, conn) -> int:
    import uvicorn
    port = args.port or C.PORT
    print(f"reddit-alpha dashboard -> http://{C.HOST}:{port}")
    print(f"store -> {C.DB_PATH}")
    uvicorn.run("dashboard:app", host=C.HOST, port=port, log_level="info",
                app_dir=str(HERE))
    return 0


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def _subs(args) -> list[str] | None:
    raw = getattr(args, "subs", "") or ""
    names = [s.strip().lstrip("/").removeprefix("r/") for s in raw.split(",") if s.strip()]
    return names or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rdx", description="Reddit investment-idea miner: scrape, store, retrieve, generate.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--db", help=f"store path (default {C.DB_PATH})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("crawl", help="pull posts and comments into the store")
    s.add_argument("--hours", type=float, default=C.DEFAULT_WINDOW_HOURS)
    s.add_argument("--subs", help="comma-separated subset (default: every enabled sub)")
    s.add_argument("--mode", choices=("incremental", "backfill"), default="incremental")
    s.add_argument("--source", choices=("arctic", "oauth", "file", "auto"), default="auto")
    s.add_argument("--file", help="path for --source file")
    s.add_argument("--max-pages", type=int, default=C.MAX_PAGES_PER_SUB)
    s.add_argument("--hydrate-max", type=int, default=None,
                   help="full-thread fetches per sub (default %d)" % C.HYDRATE_MAX_PER_SUB)
    s.add_argument("--no-hydrate", action="store_true", help="skip full-thread fetches")
    s.add_argument("--parallel", type=int, default=C.CRAWL_PARALLEL,
                   help="concurrent fetchers across subreddits (default %d; "
                        "1 = the old serialised walk)" % C.CRAWL_PARALLEL)
    s.add_argument("--budget", type=int, default=C.CRAWL_PAGE_BUDGET,
                   help="max archive pages this run may spend (default %d, 0 = no "
                        "ceiling); the run resumes where it stopped" % C.CRAWL_PAGE_BUDGET)
    s.add_argument("--sub-cap", type=int, default=C.CRAWL_SUB_PAGE_CAP,
                   help="max pages one subreddit may spend per kind (default %d, "
                        "0 = no cap)" % C.CRAWL_SUB_PAGE_CAP)
    s.add_argument("--embed", action="store_true", help="embed after crawling")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_crawl)

    s = sub.add_parser("embed", help="compute vectors for qualifying chunks")
    s.add_argument("--min-chars", type=int, default=None)
    s.add_argument("--min-quality", type=float, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser("reindex", help="rebuild chunks for stored documents")
    s.add_argument("--subs")
    s.add_argument("--hours", type=float)
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_reindex)

    s = sub.add_parser("search", help="locate evidence")
    s.add_argument("query")
    s.add_argument("--subs")
    s.add_argument("--theme")
    s.add_argument("--ticker")
    s.add_argument("--hours", type=float)
    s.add_argument("--kind", choices=("post", "comment"))
    s.add_argument("--min-quality", type=float)
    s.add_argument("--min-score", type=int)
    s.add_argument("--limit", type=int, default=8)
    s.add_argument("--mode", choices=("lexical", "vector", "hybrid"), default="hybrid")
    s.add_argument("--all", action="store_true", help="AND the terms instead of OR")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("show", help="re-read one document by reference")
    s.add_argument("ref", help="r/<sub>/<post_id>[#c<comment_id>]")
    s.add_argument("--comments", type=int, default=6)
    s.add_argument("--max-chars", type=int, default=6000)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("thread", help="read a whole thread")
    s.add_argument("post_id")
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--sort", choices=("score", "new", "old", "controversial"), default="score")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_thread)

    s = sub.add_parser("trending", help="what is taking off now")
    s.add_argument("--hours", type=float, default=24.0)
    s.add_argument("--subs")
    s.add_argument("--theme")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--sort", choices=("velocity", "quality", "score", "comments"), default="velocity")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_trending)

    s = sub.add_parser("promote", help="one thread through the model: does it become an idea?")
    s.add_argument("ref", help="r/<sub>/<post_id> or r/<sub>/<post_id>#c<comment_id>")
    s.add_argument("--model")
    s.add_argument("--no-news", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("mark", help="star an idea, or move it along (keep/acting/discard/new)")
    s.add_argument("ids", type=int, nargs="+")
    s.add_argument("--status", choices=("keep", "acting", "discard", "new"), default="keep")
    s.add_argument("--notes")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_mark)

    s = sub.add_parser("prune", help="drop chunks the current junk filter now rejects")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_prune)

    s = sub.add_parser("topics", help="candidate clusters the generator would rank (no LLM)")
    s.add_argument("--hours", type=float, default=72)
    s.add_argument("--theme")
    s.add_argument("--subs", help="comma-separated; auto-registers unknown names")
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_topics)

    s = sub.add_parser("tickers", help="mention counts and their change")
    s.add_argument("--hours", type=float, default=24.0)
    s.add_argument("--subs")
    s.add_argument("--theme")
    s.add_argument("--symbol", help="just this one")
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_tickers)

    s = sub.add_parser("authors", help="who keeps writing things worth reading")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_authors)

    s = sub.add_parser("subs", help="the editable subreddit registry")
    s.add_argument("action", choices=("list", "add", "rm", "remove", "enable", "disable",
                                      "validate", "import", "export"))
    s.add_argument("names", nargs="*")
    s.add_argument("--theme", default="other")
    s.add_argument("--weight", type=float, default=1.0)
    s.add_argument("--notes", default="")
    s.add_argument("--file", help="YAML path for import/export")
    s.add_argument("--source", default="auto")
    s.add_argument("--keep-empty", action="store_true", help="do not switch off empty subs")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_subs)

    s = sub.add_parser("news", help="dated headlines for tickers or a theme")
    s.add_argument("tickers", nargs="*")
    s.add_argument("--query")
    s.add_argument("--hours", type=float)
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--force", action="store_true", help="ignore the 30-minute cache")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_news)

    s = sub.add_parser("generate", help="turn the window into ideas")
    s.add_argument("--hours", type=float, default=72.0)
    s.add_argument("--theme")
    s.add_argument("--subs")
    s.add_argument("--n", type=int, default=C.GEN_IDEAS, help="ideas to keep")
    s.add_argument("--candidates", type=int, default=C.GEN_CANDIDATES)
    s.add_argument("--think", action="store_true",
                   help="let the model reason before answering (slower, more tokens)")
    s.add_argument("--no-select", action="store_true", help="skip the ranking call")
    s.add_argument("--no-news", action="store_true", help="skip the news leg")
    s.add_argument("--no-embed", action="store_true")
    s.add_argument("--model")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("ideas", help="list stored ideas")
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--direction", choices=tuple(sorted(ideas_dirs())))
    s.add_argument("--status")
    s.add_argument("--bookmarked", action="store_true",
                   help="only the ones you starred, newest mark first")
    s.add_argument("--sort", default="newest", choices=tuple(ideas_sort_keys()),
                   help="order: newest, oldest, conviction, conviction-asc, direction, "
                        "marked, catalyst")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ideas)

    s = sub.add_parser("idea", help="open one idea with its evidence")
    s.add_argument("id", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_idea)

    s = sub.add_parser("stats", help="what is in the store")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("check", help="structural health of the index")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("export", help="documents out for another tool")
    s.add_argument("--format", choices=("jsonl", "json", "md"), default="jsonl")
    s.add_argument("--hours", type=float, default=168.0)
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--comments", type=int, default=10)
    s.add_argument("--out")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("serve", help="the dashboard")
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)
    return p


def ideas_dirs():
    from ideas import DIRECTIONS
    return DIRECTIONS


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db = args.db or C.DB_PATH
    conn = store.connect(db)
    try:
        return args.func(args, conn)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
