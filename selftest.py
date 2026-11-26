"""selftest — offline checks over an isolated store.

No network, no model download, no LLM: everything here is the store, the
chunker, the retrieval contract and the registry. Run with

    ../.venv/bin/python selftest.py          # quiet pass/fail per check
    ../.venv/bin/python selftest.py -v       # show what each check did

It exists because the expensive failures in this app are not crashes — they are
silent ones: a re-crawl that duplicates rows, a stub that counts as a post, a
reference that resolves to the wrong document, a cursor that advances past a hole.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C                                    # noqa: E402
import registry                                        # noqa: E402
import retrieve                                        # noqa: E402
import sources                                         # noqa: E402
import store                                           # noqa: E402
import textproc as T                                   # noqa: E402

DB = HERE / "data" / "_selftest.db"
VERBOSE = "-v" in sys.argv
NOW = int(time.time())

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  ok    {name}")
        except Exception as exc:
            _results.append((name, False, f"{exc}"))
            print(f"  FAIL  {name}: {exc}")
            if VERBOSE:
                traceback.print_exc()
        return fn
    return wrap


def fresh():
    for suffix in ("", "-wal", "-shm"):
        p = pathlib.Path(str(DB) + suffix)
        if p.exists():
            p.unlink()
    conn = store.connect(DB)
    conn.execute("INSERT OR IGNORE INTO subreddit(name, theme, weight, enabled, added_at) "
                 "VALUES ('stocks','value_investing',1.0,1,?)", (T.now_iso(),))
    conn.commit()
    return conn


def a_post(pid="p1", title="Samsung locks 2027-2029 memory capacity",
           body="Samsung signed long-term agreements with 10 major customers and is "
                "negotiating with 10 more. Fulfilment rates are 60% to 70% and HBM is "
                "sold out, because hyperscaler capex converts into prepayments and "
                "price floors through 2031. $MU is the cleanest exposure.",
           **kw):
    raw = {"id": pid, "subreddit": "stocks", "author": "rdh2dmd", "title": title,
           "selftext": body, "created_utc": NOW - 7200, "score": 182,
           "num_comments": 3, "link_flair_text": "Industry News", "over_18": False}
    raw.update(kw)
    return sources.norm_post(raw)


def a_comment(cid="c1", pid="p1", body=None, **kw):
    raw = {"id": cid, "subreddit": "stocks", "author": "someguy", "link_id": f"t3_{pid}",
           "parent_id": f"t3_{pid}", "created_utc": NOW - 5400, "score": 41,
           "body": body or ("The 60-70% fulfilment rate is the detail that matters, because "
                            "it means the locked capacity is not being delivered either. "
                            "That is a price floor for anyone buying spot DRAM, not a glut.")}
    raw.update(kw)
    return sources.norm_comment(raw)


# --------------------------------------------------------------------------- #
def main() -> int:
    print(f"selftest · db {DB}")
    conn = fresh()

    @check("normalize: permalink, domain and self-post flag derived from the archive record")
    def _():
        p = a_post()
        assert p["id"] == "p1" and p["subreddit"] == "stocks"
        assert p["permalink"] == "/r/stocks/comments/p1/", p["permalink"]
        assert p["is_self"] is True or p["is_self"] == 1, "a post with a selftext is a self-post"
        assert not p["domain"], p["domain"]   # a self-post has no external domain

    @check("normalize: a link post points at its own domain and is not a self-post")
    def _():
        p = a_post(pid="p2", title="link", body="", url="https://www.sec.gov/filing/1")
        assert p["is_self"] == 0 and p["domain"] == "sec.gov", p

    @check("ingest: post and comment land once, with chunks and a working FTS index")
    def _():
        p, c = a_post(), a_comment()
        assert store.upsert_post(conn, p) is True, "first insert reports new"
        store.upsert_comments(conn, [c], {"p1": "rdh2dmd"})
        conn.commit()
        prow = conn.execute("SELECT * FROM post WHERE id='p1'").fetchone()
        store.index_post_row(conn, prow, age_hours=2.0)
        for row in conn.execute("SELECT * FROM comment WHERE post_id='p1'").fetchall():
            store.index_comment_row(conn, row, age_hours=1.5)
        conn.commit()
        hits = retrieve.search(conn, "fulfilment rate", hours=168, limit=5)
        assert hits, "lexical search found nothing"
        refs = {h["ref"] for h in hits}
        assert "r/stocks/p1" in refs or "r/stocks/p1#cc1" in refs, refs

    @check("idempotence: re-ingesting the same records changes no count")
    def _():
        before = store.counts(conn)
        store.upsert_post(conn, a_post())
        store.upsert_comments(conn, [a_comment()], {"p1": "rdh2dmd"})
        conn.commit()
        after = store.counts(conn)
        assert before["posts"] == after["posts"], (before, after)
        assert before["comments"] == after["comments"], (before, after)

    @check("idempotence: re-indexing unchanged text does not duplicate chunks")
    def _():
        before = store.counts(conn)["chunks"]
        for _i in range(2):
            prow = conn.execute("SELECT * FROM post WHERE id='p1'").fetchone()
            store.index_post_row(conn, prow, age_hours=2.0)
            conn.commit()
        assert store.counts(conn)["chunks"] == before, (before, store.counts(conn)["chunks"])

    @check("references: post and comment refs round-trip, and a bad ref is refused")
    def _():
        ref = T.comment_ref("stocks", "p1", "c1")
        assert ref == "r/stocks/p1#cc1", ref
        kind, sub, pid, cid = T.parse_ref(ref)
        assert (kind, sub, pid, cid) == ("comment", "stocks", "p1", "c1"), (kind, sub, pid, cid)
        assert T.parse_ref("r/stocks/p1") == ("post", "stocks", "p1", None)
        for bad in ("stocks/p1", "r/stocks", "r//p1", "x/stocks/p1#zzz"):
            try:
                T.parse_ref(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} should not parse")

    @check("show: a comment ref resolves to that comment's own text")
    def _():
        doc = retrieve.show(conn, "r/stocks/p1#cc1")
        assert doc["comment"] and doc["comment"]["id"] == "c1", doc.get("comment")
        assert "fulfilment rate is the detail" in doc["comment"]["body"], doc["comment"]["body"][:80]
        assert doc["post"]["id"] == "p1", "a comment reference still carries its thread"

    @check("stubs: an orphan comment creates a stub thread that no statistic counts")
    def _():
        c = a_comment(cid="c9", pid="orphan", body="Context from a thread older than the window.")
        store.upsert_comments(conn, [c], {})
        conn.commit()
        row = conn.execute("SELECT stub FROM post WHERE id='orphan'").fetchone()
        assert row and row["stub"] == 1, row
        st = retrieve.stats(conn)
        assert st["posts"] == 1, st          # p1 only, never the orphan stub

    @check("thread: returns the post and its comments in one shape")
    def _():
        th = retrieve.thread(conn, "p1")
        assert th["post"]["id"] == "p1", th["post"]["id"]
        assert any(c["id"] == "c1" for c in th["comments"]), th["comments"]

    @check("quality: a one-liner scores below a substantive comment, junk scores near zero")
    def _():
        good = T.quality_score(a_comment()["body"], score=41, num_comments=3, age_hours=1.0)
        thin = T.quality_score("lol same", score=900, num_comments=3, age_hours=1.0)
        junk = T.quality_score("This comment was removed by a moderator.", score=0,
                               num_comments=0, age_hours=1.0)
        assert good > thin, (good, thin)
        assert good > 0.3, good
        assert junk < 0.2, junk
        assert T.is_junk("Your comment has been removed because it violates Rule 7"), True

    @check("tickers: cashtags outrank bare uppercase, aliases map, stopwords do not")
    def _():
        got = T.extract_tickers("Loaded $MU and $STX, also NVDA IMO, the FED and HBM matter")
        kinds = {sym: conf for sym, _kind, conf in got}
        assert "MU" in kinds and "STX" in kinds, kinds
        assert kinds["MU"] >= 0.9, kinds          # cashtag
        assert "IMO" not in kinds and "FED" not in kinds, kinds
        assert "NVDA" in kinds, kinds

    @check("mentions: extracted symbols are queryable per document")
    def _():
        n = conn.execute("SELECT count(*) FROM mention").fetchone()[0]
        assert n > 0, "no mentions were recorded during indexing"

    @check("metrics: two observations make velocity measurable, not guessed")
    def _():
        store.record_metrics(conn, "p1", score=182, num_comments=3)
        time.sleep(0.01)
        store.record_metrics(conn, "p1", score=400, num_comments=9)
        conn.commit()
        rows = conn.execute("SELECT score FROM post_metric WHERE post_id='p1' "
                            "ORDER BY seen_at").fetchall()
        assert len(rows) >= 2, len(rows)
        v = retrieve.velocity_of(conn, "r/stocks/p1")
        assert v is not None, "velocity unavailable with two samples"

    @check("trending: ranks by the requested key and survives a single sample")
    def _():
        rows = retrieve.trending(conn, hours=168, sort="velocity", limit=5)
        assert isinstance(rows, list)
        rows = retrieve.trending(conn, hours=168, sort="score", limit=5)
        assert rows and rows[0]["ref"].startswith("r/stocks/"), rows[:1]

    @check("registry: add, disable, enable and export/import round-trip")
    def _():
        store.upsert_subreddit(conn, "TestSub", theme="quant", weight=1.4, notes="selftest")
        assert "TestSub" in store.subreddit_names(conn, enabled_only=False)
        store.set_subreddit_enabled(conn, "TestSub", False)
        assert "TestSub" not in store.subreddit_names(conn, enabled_only=True)
        store.set_subreddit_enabled(conn, "TestSub", True)
        assert "TestSub" in store.subreddit_names(conn, enabled_only=True)
        out = HERE / "data" / "_selftest_subs.yaml"
        registry.export_seed(conn, out)
        text = out.read_text()
        assert "TestSub" in text, text[:120]
        out.unlink()

    @check("stats and check: a fresh store reports itself and passes the structural checks")
    def _():
        st = retrieve.stats(conn)
        assert st["posts"] >= 1 and st["chunks"] > 0, st
        res = retrieve.check(conn)
        assert res["ok"], res["issues"]

    @check("source contract: a JSON dump crawls end to end without the network")
    def _():
        import json
        dump = HERE / "data" / "_selftest_dump.json"
        dump.write_text(json.dumps({"posts": [
            {"id": "fp1", "subreddit": "quant", "author": "q1",
             "title": "Cross-sectional momentum decayed after the 2024 crowding",
             "selftext": "The signal still works, because the crowding is in the same "
                         "twenty names and the ranking has stopped rotating.",
             "created_utc": NOW - 3600, "score": 55, "num_comments": 2},
        ], "comments": [
            {"id": "fc1", "subreddit": "quant", "author": "q2", "link_id": "t3_fp1",
             "parent_id": "t3_fp1", "created_utc": NOW - 3000, "score": 12,
             "body": "Crowding is only half of it, because the borrow cost moved too."},
        ]}), encoding="utf-8")
        import crawl as crawl_mod
        store.upsert_subreddit(conn, "quant", theme="quant")   # the registry is the gate
        src = sources.FileSource(dump)
        res = crawl_mod.crawl(conn, subs=["quant"], hours=48, source=src, mode="backfill",
                              hydrate=False)
        assert res["totals"]["posts"] == 1, res["totals"]
        assert res["totals"]["comments"] == 1, res["totals"]
        hit = retrieve.search(conn, "crowding borrow cost", hours=48, limit=3)
        assert hit and hit[0]["ref"].startswith("r/quant/"), hit
        dump.unlink()

    @check("vectors: a vector round-trips through the store at the configured width")
    def _():
        import numpy as np
        vec = np.zeros(256, dtype=np.float32)
        vec[0] = 1.0
        cid = conn.execute("SELECT id FROM chunk LIMIT 1").fetchone()[0]
        store.store_vectors(conn, [(cid, vec.tobytes())], model="selftest",
                            dim=int(vec.shape[0]), dtype="float32")
        conn.commit()
        got = store.vector_stats(conn)
        assert got["vectors"] >= 1, got
        row = conn.execute("SELECT dtype, dim, vec FROM vector WHERE chunk_id=?", (cid,)).fetchone()
        assert row["dim"] == int(vec.shape[0]), row["dim"]
        back = np.frombuffer(row["vec"], dtype=np.float32)
        assert abs(back[0] - 1.0) < 1e-6, back[:3]

    @check("re-chunking: replaced chunks take their stale vectors with them")
    def _():
        import numpy as np
        prow = conn.execute("SELECT * FROM post WHERE id='p1'").fetchone()
        n, _changed = store.index_post_row(conn, prow, age_hours=2.0)
        conn.commit()
        cid = conn.execute("SELECT id FROM chunk WHERE doc_kind='post' AND doc_id='p1'").fetchone()[0]
        store.store_vectors(conn, [(cid, np.ones(256, dtype=np.float32).tobytes())],
                            model="selftest", dim=256, dtype="float32")
        conn.commit()
        extra = (" Additional paragraph because the thread kept going, with numbers: "
                 "4.2% and 18 months.")
        conn.execute("UPDATE post SET selftext=selftext || ? WHERE id='p1'", (extra,))
        conn.commit()
        prow = conn.execute("SELECT * FROM post WHERE id='p1'").fetchone()
        store.index_post_row(conn, prow, age_hours=2.0)
        conn.commit()
        orphans = conn.execute("SELECT count(*) FROM vector v WHERE NOT EXISTS "
                               "(SELECT 1 FROM chunk c WHERE c.id=v.chunk_id)").fetchone()[0]
        assert orphans == 0, f"{orphans} vectors now point at deleted chunks"

    @check("bookmarks: a starred idea keeps its mark through a regeneration pass")
    def _():
        import ideas as ideas_mod
        idea = {"headline": "Watch the bottleneck claim in bonding tools",
                "thesis": "The scarcity premium rests on a constraint that may sit in the "
                          "front-end fabs rather than in the bonder, which would make the "
                          "margin a capex artefact rather than durable pricing power.",
                "direction": "watch", "tickers": ["MU"], "conviction": 0.4,
                "refs": ["r/stocks/p1"], "created_at": T.now_iso(), "model": "selftest"}
        iid = store.insert_idea(conn, idea, [("post", "p1", 1.0)])
        conn.commit()
        store.update_idea_status(conn, iid, "keep")
        conn.commit()
        row = conn.execute("SELECT status, marked_at FROM idea WHERE id=?", (iid,)).fetchone()
        assert row["status"] == "keep" and row["marked_at"], dict(row)
        first_mark = row["marked_at"]
        # the generator re-writing the same idea (same hash) must not clear the mark
        store.insert_idea(conn, {**idea, "conviction": 0.55}, [("post", "p1", 1.0)])
        conn.commit()
        row = conn.execute("SELECT status, marked_at, conviction FROM idea WHERE id=?", (iid,)).fetchone()
        assert row["status"] == "keep" and row["marked_at"] == first_mark, dict(row)
        assert abs(row["conviction"] - 0.55) < 1e-9, "the refresh should still update the numbers"
        starred = ideas_mod.list_ideas(conn, bookmarked=True)
        assert [r["id"] for r in starred] == [iid], [r["id"] for r in starred]
        store.update_idea_status(conn, iid, "new")
        conn.commit()
        assert not ideas_mod.list_ideas(conn, bookmarked=True), "un-starring must clear the mark"

    @check("sorting: conviction, direction and the unknown-key refusal")
    def _():
        import ideas as ideas_mod
        base = {"thesis": "A thesis long enough to be a real sentence about a real mechanism.",
                "tickers": ["ZZ"], "refs": ["r/stocks/p1"], "created_at": T.now_iso(),
                "model": "selftest"}
        made = []
        for head, direction, conv in (("Zed low avoid", "avoid", 0.2),
                                      ("Alpha high long", "long", 0.9),
                                      ("Mid watch", "watch", 0.5)):
            made.append(store.insert_idea(conn, {**base, "headline": head,
                                                 "direction": direction,
                                                 "conviction": conv},
                                          [("post", "p1", 1.0)]))
        conn.commit()
        mine = [r for r in ideas_mod.list_ideas(conn, limit=50) if r["tickers"] == ["ZZ"]]
        by_conv = [r["headline"] for r in ideas_mod.list_ideas(conn, limit=50, sort="conviction")
                   if r["tickers"] == ["ZZ"]]
        assert by_conv[:3] == ["Alpha high long", "Mid watch", "Zed low avoid"], by_conv
        by_dir = [r["headline"] for r in ideas_mod.list_ideas(conn, limit=50, sort="direction")
                  if r["tickers"] == ["ZZ"]]
        assert by_dir[:3] == ["Alpha high long", "Mid watch", "Zed low avoid"], by_dir
        try:
            ideas_mod.list_ideas(conn, sort="; DROP TABLE idea")
            raise AssertionError("an unknown sort key must be refused, not interpolated")
        except ValueError:
            pass
        for i in made:
            conn.execute("DELETE FROM idea WHERE id=?", (i,))
        conn.commit()

    @check("promote: one thread becomes an idea, and a second read of it does not twin")
    def _():
        import ideas as ideas_mod
        import llm
        post = a_post(pid="prom1")
        store.upsert_post(conn, post)
        store.upsert_comments(conn, [a_comment(cid="pc1", pid="prom1",
                                               body="Hyperscaler prepayments are the tell; "
                                                    "if the LTAs carry floors, the cycle "
                                                    "argument is wrong for two years.")])
        conn.commit()

        fixed = {"verdict": "idea", "why": "contract terms are specific",
                 "ideas": [{"headline": "Memory contracts with floors mute the cycle",
                            "direction": "long", "tickers": ["MU"],
                            "thesis": "The long-term agreements carry prepayments and price "
                                      "floors through 2031, so the spot cycle turning does not "
                                      "reach reported margins for two years.",
                            "mechanism": "floors hold price, prepayments hold the buyer",
                            "catalyst": "next contract disclosure", "catalyst_date": None,
                            "invalidation": "a signed deal without a floor",
                            "crowd_edge": "terms from the negotiators, not the wire",
                            "conviction": 0.5, "refs": ["r/stocks/prom1"],
                            "new_evidence": "60-70% fulfilment"}],
                 }
        real = llm.chat_json
        llm.chat_json = lambda *a, **k: fixed          # offline: no model call
        try:
            first = ideas_mod.promote(conn, "r/stocks/prom1", news=False)
            assert first["verdict"] == "idea", first
            assert first["idea"]["direction"] == "long", first["idea"]["direction"]
            second = ideas_mod.promote(conn, "r/stocks/prom1", news=False)
            assert second["verdict"] == "duplicate", second
            assert second["merged_into"] == first["idea_id"], second
            llm.chat_json = lambda *a, **k: {"verdict": "no_claim",
                                             "why": "a loss screenshot", "ideas": []}
            third = ideas_mod.promote(conn, "r/stocks/prom1", news=False)
            assert third["verdict"] == "no_claim" and third["why"], third
            llm.chat_json = lambda *a, **k: {"verdict": "idea", "ideas": [
                {"headline": "x", "thesis": "too short"}]}
            fourth = ideas_mod.promote(conn, "r/stocks/prom1", news=False)
            assert fourth["verdict"] == "no_claim", fourth
        finally:
            llm.chat_json = real
        assert ideas_mod.promote(conn, "r/stocks/nosuchpost", news=False)["verdict"] == "no_claim"
        try:
            ideas_mod.promote(conn, "not-a-ref", news=False)
            raise AssertionError("a malformed ref must be refused")
        except ValueError:
            pass
        conn.execute("DELETE FROM idea WHERE refs LIKE '%prom1%'")
        conn.commit()

    conn.close()
    failed = [r for r in _results if not r[1]]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    for name, _ok, msg in failed:
        print(f"  · {name}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
