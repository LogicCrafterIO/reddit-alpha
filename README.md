# reddit-alpha

A Reddit scanner that treats certain subreddits as what they sometimes are: an
earlier, more concrete source of investment argument than the sell-side. It
pulls posts and comments into a local store, indexes them two ways, ranks them
for whether they carry an argument, and turns the strongest threads in a window
into investment ideas **with the evidence attached** — the reference points back
at the original words, so nothing is a paraphrase you cannot check.

It also keeps a dated news cache, so an idea can be read against what the wires
said afterwards, and so "the latest updates" does not depend on Reddit being
reachable.

```
 reddit (archive or API)  →  store  →  index  →  retrieve  →  generate  →  ideas
                              │         │          │
                              │         │          └─ search / show / thread
                              │         └─ FTS5 (all chunks) + vectors (quality-gated)
                              └─ posts · comments · metrics · mentions · news · ideas
```

## The one thing to know before anything else

**Reddit blocks this machine.** Anonymous `www.reddit.com/r/<sub>/hot.json`
returns 403 to every user agent and every TLS-impersonation profile `curl_cffi`
offers (`chrome`, `chrome124`, `safari17`, `firefox133`); `old.reddit.com` serves
a consent wall; `api.reddit.com` and `oauth.reddit.com` inherit the 403; every
`.rss` feed returns 429 and public redlib instances sit behind a proof-of-work
wall. That is IP reputation on Reddit's side, not a fault in this code.

So the default source is **`arctic-shift.photon-reddit.com`**, a public archive of
public Reddit content that answers without a key or a login. It throttles rather
than refusing outright: four pages in flight returned `200` in **2.0 s wall** on
2026-11-25 against **8.3 s** for the same four serialised, so the crawler now fans
out four fetchers and keeps the writing in one thread. What it will not do is serve
a high *rate* — the same afternoon, eleven of fourteen requests spaced 0.6 s apart
came back `422 Timeout. Maybe slow down a bit`, and a run settles around **0.8
pages/s** while the archive is loaded. That ceiling, not our concurrency, is why a
24-hour sweep of seventy subreddits cost 625 requests and 13.6 minutes before it
was bounded: see [What a run is allowed to cost](#what-a-run-is-allowed-to-cost).

Measured on 2026-11-22: newest r/wallstreetbets post **6 minutes** old, comments
**5 minutes** old; r/investing lags about an hour. A seven-day crawl of
r/wallstreetbets returns about 2 300 posts and 20 000 comments, and r/AIBubble — a
much smaller board — about 320 posts and 15 000 comments, so the volume is in the
comments and comment volume is where the argument is.

The archive is not Reddit. It is a mirror with a lag, and it is the honest way to
get this data from here. If you want Reddit's own API — for liveness or for a
subreddit the archive does not carry — register a **script** app at
<https://www.reddit.com/prefs/apps> and put two values in the workspace `.env`:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

`get_source()` in `sources.py` then prefers OAuth automatically. The store does
not change: both adapters emit the same normalised records.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # fastapi, uvicorn, numpy, model2vec, pyyaml
```

```bash
cd reddit-alpha
../.venv/bin/python rdx.py subs import          # load subreddits.yaml into the registry
../.venv/bin/python rdx.py subs validate        # probe every sub; switch off the dead ones
../.venv/bin/python rdx.py crawl --hours 24 --embed
../.venv/bin/python rdx.py stats
./run.sh                                        # dashboard on http://127.0.0.1:8792
```

Every command takes `--json`. The store, the crawler, the lexical search and the
dashboard need only the four core packages; `numpy` and `model2vec` add the vector
leg and the embedder, and without them retrieval degrades to lexical instead of
failing. The `../.venv/bin/python` above is this checkout's layout — with the venv
activated, plain `python` works.

No API key is required to crawl or search. `crawl` reads Reddit through the public
archive; the optional idea generator (`rdx.py generate`) wants an `OLLAMA_API_KEY`
in a `.env` beside this directory, and Reddit's own OAuth API wants
`REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`.

## What a run is allowed to cost

A page is an archive request, and requests are the only thing a crawl really spends.
So the run is budgeted in pages rather than left to finish whenever it finishes:

```bash
../.venv/bin/python rdx.py crawl --hours 24 --budget 240 --sub-cap 30 --embed
```

* `--budget` — pages this run may spend (default 240, `0` = no ceiling). Seventy
  subreddits need ~140 pages before each has even been touched, so a smaller
  budget leaves most of them capped and re-walking the same page next run. At the
  archive's loaded-hour rate 240 pages took **6.5 min** on 2026-11-25 evening
  (224 pages, 0.58 pages/s, 81 refusals); on a quiet hour the same sweep is ~2.5
  min. A dozen core boards with `--subs` and `--budget 60` took **41 s** for the
  same 24 hours, which is the cheap daily run.
* `--sub-cap` — pages one subreddit may spend per kind (default 30). Without it a
  single firehose takes the run: r/technology alone spent **161 of 625** requests
  on a 24-hour window on 2026-11-25.
* `--parallel` — fetchers in flight (default 4). Fanning out overlaps the waiting;
  it does not raise the request rate, which is the archive's to set.

Incremental runs resume from each subreddit's cursor rather than from `--hours`,
so a skipped day is fetched by the next run instead of being silently dropped by
the window; the backstop is 168 h. `--hours` therefore sets the backfill window
(`--mode backfill`) and the window used when a subreddit has no cursor yet.

What the budget does not do is lose data. A subreddit whose chain ran out of
allowance is marked `capped`, and its cursor is left where it was, so the next run
resumes from that point instead of skipping the window — the run report carries
`pages`, `capped_subs`, `pace` and `slowdowns` so it is visible when a sweep is
running behind.

## The retrieval contract

Locate first, read second. Never load a thread to answer something a snippet can.

```bash
# 1. locate — snippets, ranked, ~260 chars each
../.venv/bin/python rdx.py search "backlog conversion" --hours 168 --limit 8
../.venv/bin/python rdx.py search "hyperscaler capex" --ticker NVDA --mode vector
../.venv/bin/python rdx.py search "gamma squeeze" --mode lexical --all

# 2. read exactly the document a claim rests on
../.venv/bin/python rdx.py show "r/wallstreetbets/1y25458#c19szw98e3i1k4"

# 3. or the whole thread, ranked
../.venv/bin/python rdx.py thread 1y25458 --sort score --limit 40
```

References are the contract, and `show` is what makes them honest:

| form | means |
|---|---|
| `r/wallstreetbets/1y2e1xc` | a post |
| `r/wallstreetbets/1y2e1xc#c19szw98e3i1k4` | one comment inside that post |

Cross-sections, for when the question is "what is moving" rather than "what was
said":

```bash
rdx.py trending --hours 24 --sort velocity     # what is being read fastest, not what is oldest
rdx.py tickers --hours 24 --theme ai_semis     # mention counts + change vs the prior equal window
rdx.py authors --limit 20                      # who keeps writing things worth reading
```

`tickers` is the one that earns its keep: a symbol going from 40 mentions to 300
in a day is a rotation, and it shows up here before it shows up in price.

## The idea generator

```bash
../.venv/bin/python rdx.py topics --hours 72        # what the generator would rank (no LLM)
../.venv/bin/python rdx.py generate --hours 72 --n 5
../.venv/bin/python rdx.py ideas
../.venv/bin/python rdx.py idea 3
```

`topics` is the cheap half on its own: it prints the candidate clusters with their crowd
metrics and strongest titles, so you can see what the window actually contains — and whether
a run is worth a model call — without spending tokens.

### Promoting one thread

The batch generator decides what to read. When you have already decided — you are looking at a
thread in Flow and want to know whether it is an idea — promote it:

```bash
../.venv/bin/python rdx.py promote r/stocks/1y22xxa
../.venv/bin/python rdx.py promote r/wallstreetbets/1y1krky#c19svdp2nlcog0   # one comment
```

In the dashboard it is **does this become an idea?** in the Flow thread drawer, the Search
document drawer, and as a **→ idea** button on every search hit — so the three places you meet a
document (flow, search, an idea's evidence) all let you put it in front of the model. One model
call, no ranking call, no clustering; the evidence pack is that thread — its body, the comment
you pointed at, and its ten strongest comments — plus dated news for any ticker it names. The
answer is one of three, and all three are useful:

- **an idea**, stored and opened (`IDEAS → #31`), with the comment-level references it used;
- **already an idea**, when the thread is the same evidence for something you already hold, in
  which case the new evidence is folded in rather than minting a twin;
- **no idea here**, with the specific reason — a rant, a loss screenshot, a question with no
  answer, a thread that only restates the news. A confident premise that the thread's own
  informed commenters refute counts as an *avoid*, not a refusal.

A refusal costs the same as an idea (about 0.05 cents on the default model) and is worth having:
it is the difference between "I read the thread" and "I checked whether the thread says anything".

### Bookmarking the ones that matter

The generator writes; you mark. Those are separate layers, and the mark is the durable one.

```bash
../.venv/bin/python rdx.py ideas --bookmarked      # only the starred, newest mark first
../.venv/bin/python rdx.py mark 1 4               # star them (default status: keep)
../.venv/bin/python rdx.py mark 4 --status acting  # or: keep / acting / discard / new
../.venv/bin/python rdx.py mark 4 --status new     # un-star
```

In the dashboard every card carries a star in its top-right corner — one click, no drawer, and
a `★ bookmarked` filter in the bar shows just those. The bar also sorts: conviction high or low,
direction (actionable first, then watch, then avoid), recently starred, soonest catalyst, or
date. The same order is available on the CLI with `rdx.py ideas --sort conviction`, and the
controls survive the page reloads this app does after every job. `ideas.status` holds the state and
`idea.marked_at` holds when you set it, so a bookmarked list orders by *your* marks rather
than by when the machine wrote the idea.

Two guarantees, both tested: **a bookmark survives regeneration** (the generator upserts on a
headline+thesis hash and only refreshes metrics, refs, news and conviction; it never touches
`status` or `marked_at`), and **nothing is deleted** by a run. `ideas_marked` is in `stats`
and in the dashboard header, so the count of what you kept is visible without opening the tab.

Two calls, on purpose:

1. **select** — one cheap call over compact summaries of every candidate cluster
   in the window. It ranks; it never writes an idea.
2. **synthesize** — one call per selected cluster, given the full evidence pack
   (post bodies, the comments that carry the argument, mention deltas, dated
   news). This is where an idea is written.

That split points the expensive call at the two or three threads that matter and
means a bad ranking cannot quietly become a bad idea: the second call sees the
words, not the first call's summary.

An idea is stored with `headline`, `direction` (long/short/watch/avoid),
`thesis`, `mechanism`, a dated `catalyst`, `invalidation`, `crowd_edge` (what
this crowd saw before the mainstream), `conviction`, the crowd metrics behind it,
the references it cited, and the news it was read against. Thesis is capped at
three sentences and no price target is permitted — the register this workspace
uses for everything else.

Duplicate ideas are merged rather than multiplied: a new idea whose headline and
thesis embed within cosine 0.92 of an existing one has its evidence folded into
that idea (`gen_dedupe` in `config.py`).

Cost: bounded by `GEN_CANDIDATES` and `GEN_SNIPPET_BUDGET`. Measured usage is
printed after every run. The endpoint's models (`glm-5.3`, `deepseek-v4.1-flash`)
are reasoning models: they write `thinking` before `content`, and at the default
budget glm-5.3 spent 20 000+ characters reasoning and returned *nothing* —
`done_reason: length` with an empty answer, which reads like a broken API and is
not one. Structured calls therefore send `think: false` by default (one call,
under a second, content only), and `--think` turns reasoning back on for a run that
wants it. On an empty answer `llm.py` switches reasoning off before it considers
growing the budget, because a bigger budget only feeds the same field.

## What it costs

Only one part of this app is metered: the generator. Crawling hits a public archive
that needs no key, embeddings run locally (`model2vec`, 22 616 chunks in 2.2 s),
and the news leg scrapes news sites — all of that is free of charge and free of
tokens.

A generation run is one cheap **select** call plus one **synthesize** call per
candidate cluster. The default model is `glm-5.3-flash` (`LLM_MODEL` in `config.py`, or
`RDX_LLM_MODEL` in the environment); pass `--model glm-5.3` when an idea matters enough to
want the stronger reader. Measured on this store: five ideas cost four calls and
12 191 input / 2 878 output tokens; three ideas cost three calls and 9 127 /
1 581. That is roughly 3 000 tokens in and 500–700 out per call, because the
evidence pack is capped by `GEN_SNIPPET_BUDGET`.

At [Ollama's published rates](https://ollama.com/pricing) that is about
**0.6 cents per idea** on `glm-5.3` and **0.07 cents per idea** on `glm-5.3-flash`,
which is the default for exactly that reason — and on a 168-hour window twenty ideas
came back in 13 calls for 46 647 input and 11 103 output tokens, about 1.3 cents.

Three levers remain after the model. `--candidates` sets how many clusters get read,
one call each. `--hours` sets how much material rides on every call, so a 24-hour
window is cheaper than a week. `GEN_SNIPPET_BUDGET` in `config.py` caps the evidence
pack itself.

Raising `--n` past `--candidates` now widens the candidate pool automatically, so
asking for more ideas cannot silently under-deliver. `--no-select` saves the one
ranking call; it is rarely worth it, because that call is the cheapest one and it
is what keeps the expensive calls pointed at the right threads.

## Getting more ideas

Ideas accumulate — a run adds to the set and never replaces it. This store holds 29:
nine written by `glm-5.3` while the pipeline was being built, twenty by a single
`glm-5.3-flash` pass over the full week of crawled data. Each stays until you remove
it, and a run that finds nothing new changes nothing.

The count comes from three things, not one:

- **The candidate pool** — the clusters in the window that clear `GEN_MIN_QUALITY`.
  `rdx.py topics --hours 72` prints exactly this list for free, so you can see
  whether a window has anything before spending a token.
- **`--candidates`** — how many of those clusters get read (one call each).
- **`--n`** — how many ideas to keep from the run.

So more ideas means a wider window or more crawled subreddits *plus* a bigger
`--candidates`. Adding subreddits is the compounding move: seventeen of the seventy
enabled boards are loaded, and each new board can introduce a cluster the others
did not have. The dashboard's Ideas tab exposes the window, the count and the model
as three controls, so a cheap breadth pass is a two-click thing.

## Where the data lives, and why it is shaped this way

One SQLite file, `data/reddit.db`, WAL mode, so the dashboard reads while a crawl
writes.

Five rules do the design work:

1. **Reddit's own ids are the keys.** Re-crawling updates a row; it never
   duplicates one. `post.id`, `comment.id`, and the pair `(doc_kind, doc_id)` for
   chunks.
2. **Text is immutable, metrics append.** `post.score` is refreshed in place, but
   every observation also lands in `post_metric` with its timestamp, so velocity —
   points per hour between the first and last sample — is a measurement rather
   than an impression.
3. **One retrieval unit.** `chunk` spans posts and comments (`doc_kind`, `doc_id`,
   `post_id`, `seq`, `heading`). A searcher never has to know whether the argument
   was the submission or the top reply.
4. **Evidence by reference.** `idea_doc` links an idea to documents. The words are
   re-read from the original row on every `show`, so a stored idea cannot drift
   from what it was built on.
5. **Additive migrations.** `meta.schema_version` gates `ALTER`s in
   `store.MIGRATIONS`. Nothing needs a rebuild to gain a column.

Two index tiers, because volume and value are not the same distribution:

- **FTS5** (`chunk_fts`, porter unicode61) over *every* chunk — cheap, exact terms.
- **Vectors** (`vector`, `model2vec` `potion-base-8M`, 256-dim, stored as BLOBs)
  over only the chunks that clear `VEC_MIN_CHARS` and `VEC_MIN_QUALITY`. Every WSB
  one-liner at 1 KB of float32 per vector is 200 MB of noise inside a fortnight;
  the gate is a knob, not a hardcode.

Retrieval fuses the two legs by reciprocal rank fusion, so neither leg's score
scale has to be calibrated against the other.

| table | what it holds |
|---|---|
| `subreddit` | the editable registry: theme, weight, enabled, cursor, health |
| `post` | one row per submission, with `quality` and extracted `tickers` |
| `post_metric` | append-only score/comment observations, one per crawl pass |
| `comment` | one row per reply, with `depth`, `is_op`, `quality`, `tickers` |
| `chunk` + `chunk_fts` | the retrieval unit and its lexical index |
| `vector` | quality-gated embeddings |
| `mention` | `(doc_kind, doc_id, symbol, kind, confidence)` — the join table other apps want |
| `author` | per-author roll: posts, comments, score, best quality, home sub |
| `news` | dated headlines, `discovery=1` for Google News hits (headline usable, link not) |
| `idea` + `idea_doc` | the generator's output and its evidence backlinks |
| `run` | crawl / hydrate / embed / generate / validate runs with stats |

`post.stub` is the one subtlety: a comment can arrive for a thread older than the
crawl window, so a stub post is created to keep the thread coherent and
`thread <id>` can hydrate it later. Stubs never count as posts in any statistic.

## The subreddit list is a first-class object

Three ways to edit it, one source of truth:

```bash
# 1. the CLI
../.venv/bin/python rdx.py subs add "r/SomeSub" --theme defence --weight 1.2 --notes "why"
../.venv/bin/python rdx.py subs enable SomeSub
../.venv/bin/python rdx.py subs rm SomeSub        # removes the entry; stored posts stay
../.venv/bin/python rdx.py subs list
../.venv/bin/python rdx.py subs validate          # probe, record, and switch off empties

# 2. the YAML seed — edit and merge
$EDITOR subreddits.yaml
../.venv/bin/python rdx.py subs import

# 3. the dashboard's Sources tab — toggle, retheme, reweight, add, remove, export
```

`subreddits.yaml` is the seed *and* the export format, so the list is reviewable
in a diff and editable in a text editor. The database is what a crawl reads.

`validate` is deliberately blunt about what it can prove. It probes each name and
records posts/comments seen in the last 7 days; a name that returns nothing is
switched off **with a note saying that is a statement about the source, not proof
the subreddit is dead**. Right now that is the difference between the two subs you
named and the archive: `sidestreetbets` has not been crawled in about 135 days, and
`blackstreetbets` does not appear under any case variant at all. Both are in the
registry, disabled, with the date and the reason — and both become crawlable the
moment OAuth credentials are added, if they exist on Reddit.

The shipped seed covers 74 subreddits across twelve themes: `retail_trading`,
`options`, `value_investing`, `ai_semis`, `tech_software`, `defence`,
`macro_rates`, `energy_commodities`, `healthcare`, `crypto`, `china_em`, `quant`.
Every enabled name was probed against the archive on 2026-11-22 before shipping.

## What "quality" means here

`textproc.quality_score` answers one narrow question: should this cost the
generator a snippet slot? It is not a sentiment score and it is not a forecast.
It combines text substance (log length), argument cues (`because`, `guidance`,
`backlog`, `margin`, `capex`, `dilution`…), numeric density (`%`, `b`, `$`), flair
(`DD`/`Thesis`/`Discussion` up, `Meme`/`Loss` down), and compressed engagement. A
one-liner with 3 000 upvotes is a signal, not an insight, so short bodies are
capped hard.

That score gates three things: the vector tier, the generator's candidate pool,
and the ranking inside `flow`. It is one number with a visible formula in one
place — if a thread you rate is being missed, the fix is a term in that function,
then `rdx.py reindex` to re-score what is already stored.

## Extending it

**Another source.** Subclass `sources.Source`, implement `fetch_posts`,
`fetch_comments`, `fetch_thread`, `probe`, emit records through `norm_post` /
`norm_comment`, and add it to `get_source()`. Nothing downstream changes. There is
already an `OAuthSource` and a `FileSource` (replay a saved JSON dump) to copy
from.

**Another consumer in this workspace.** Import, do not shell out:

```python
import store, retrieve
conn = store.connect()                       # migrates on connect
hits = retrieve.search(conn, "backlog", ticker="NVDA", hours=168, limit=5)
doc = retrieve.show(conn, hits[0]["ref"])    # the original words
rows = conn.execute("SELECT symbol, count(*) FROM mention GROUP BY symbol")
```

`mention` and `chunk` are the two tables worth joining against. The options book,
the driver ledger and the news scanner all key on tickers; a mention table with
timestamps is what lets them ask "was this name getting louder on Reddit before
it moved".

**A new field.** Add the column to `SCHEMA` for fresh stores *and* append an
`ALTER TABLE` entry to `store.MIGRATIONS` for existing ones. Bump
`SCHEMA_VERSION`. That is the whole migration story.

**A different idea register.** `ideas.SYSTEM`, `SYNTH_PROMPT` and `SELECT_PROMPT`
are plain strings at the top of the module; the JSON shape an idea must return is
one `_norm_idea` function.

## Export for another tool

```bash
rdx.py export --format jsonl --hours 168 --limit 500 --comments 10 --out out.jsonl
rdx.py export --format md    --hours 24  --limit 50            # paste-ready digest
```

## Tests

```bash
../.venv/bin/python selftest.py       # 22 offline checks, ~1 s
```

No network, no model download, no LLM. It covers exactly the failures that do not
crash: a re-crawl that duplicates rows, a stub counted as a post, a reference
resolving to the wrong document, junk boilerplate entering the index, two metric
samples merging inside one second, a bookmark lost to a regeneration pass. Run it after
touching `store.py`, `textproc.py` or `retrieve.py`.

## Honest limitations

- **The archive is a mirror with a lag.** Between minutes (WSB) and hours
  (r/investing). It is not live, and a deleted comment stays deleted in the
  mirror only after it is re-crawled.
- **Coverage is uneven.** Some subs are absent (`China_Stocks`, `biotech_stocks`)
  and some are dormant in the archive (`sidestreetbets`, `defence`,
  `Stock_Picks`). `subs validate` is the authority on which, and when.
- **Upvote ratio and gilding are not in the archive's field projection**, so they
  are NULL when crawling from it. `score`, `num_comments` and `created_utc` — the
  three the ranking actually uses — are present.
- **`quality_score` is a heuristic, not a judgement.** It can promote a confident
  rant and miss a dry but correct comment. The generator reads the text; the score
  only decides what gets read.
- **The generator can still write a bad idea.** What it cannot do is write one
  without a reference you can open, or silently drop the evidence that contradicts
  it — `invalidation` is a required field of the schema, and an empty `ideas` list
  is a valid, expected answer.
