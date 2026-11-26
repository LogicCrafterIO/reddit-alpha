"""Configuration: paths, env, tunables.

Everything the rest of the app needs to know about *where* things live and how
big a bite it may take. No imports from the app itself, so this module is safe
to import first from anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                      # the workspace root
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

DB_PATH = Path(os.environ.get("RDX_DB", str(DATA / "reddit.db")))
SUBS_YAML = HERE / "subreddits.yaml"
STATIC = HERE / "static"

# --------------------------------------------------------------------------- #
# .env (the workspace keeps one key file at the root; no external dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path = BASE / ".env") -> None:
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
ARCTIC_BASE = os.environ.get("RDX_ARCTIC_BASE", "https://arctic-shift.photon-reddit.com/api")
ARCTIC_PACE = float(os.environ.get("RDX_ARCTIC_PACE", "0.6"))      # seconds between call starts
ARCTIC_TIMEOUT = int(os.environ.get("RDX_ARCTIC_TIMEOUT", "45"))
ARCTIC_PAGE = 100                                                  # hard API cap
USER_AGENT = os.environ.get(
    "RDX_UA", "reddit-alpha/0.1 (local research workspace; read-only archive reader)")

# Reddit's own OAuth API. Empty by default; fill these and the OAuth adapter
# becomes the primary source without any schema change.
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "reddit-alpha/0.1 by local-user")

# --------------------------------------------------------------------------- #
# Index / retrieval
# --------------------------------------------------------------------------- #
EMBED_MODEL = os.environ.get("RDX_EMBED_MODEL", "minishlab/potion-base-8M")
EMB_CACHE = DATA / "emb-cache"
# Reuse the transcript store's cache when it has the model, so a first run does
# not re-download 30 MB that is already on disk.
_ALT_EMB_CACHE = BASE / "transcripts" / "data" / "emb-cache"
if _ALT_EMB_CACHE.exists() and any(_ALT_EMB_CACHE.rglob("model.safetensors")):
    EMB_ALT_CACHE = _ALT_EMB_CACHE
else:
    EMB_ALT_CACHE = EMB_CACHE

CHUNK_CHARS = 1100          # target size of one retrieval unit
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 90        # shorter than this is not worth a row of its own

# Which chunks get a vector. Lexical search covers everything; the vector leg
# is reserved for text that plausibly carries an argument.
VEC_MIN_CHARS = int(os.environ.get("RDX_VEC_MIN_CHARS", "320"))
VEC_MIN_QUALITY = float(os.environ.get("RDX_VEC_MIN_QUALITY", "0.18"))
VEC_DTYPE = os.environ.get("RDX_VEC_DTYPE", "float32")   # float16 halves the footprint

# Text that carries no idea: removed/empty comments, one-word reactions.
# Moderator and automod boilerplate is long enough to pass a length test and
# carries words like "because", so it has to be excluded by pattern, not length.
JUNK_PATTERNS = (
    r"your comment has been removed",
    r"comment has been removed by (a )?moderator",
    r"(this|your) (post|submission|thread) (was|has been) removed",
    r"removed by reddit",
    r"removed by (a )?moderator",
    r"^\[removed\]$",
    r"^\[deleted\]$",
    r"this content (is|was) no longer available",
    r"i am a bot, and this action was performed automatically",
    # community safety nets and loss threads. They dominate a bad week's engagement
    # and carry no tradeable claim, so they must not become candidate themes.
    r"suicide hotline",
    r"your life is irreplaceable",
    r"if you or someone you know is struggling",
    r"reach out if you(’|')?re struggling",
)
JUNK_BODIES = {
    "", "[deleted]", "[removed]", "[removed by reddit]", "deleted", "removed",
    "this", "lol", "same", "what", "why", "yes", "no", "no.", "ok", "okay",
    "thanks", "thank you", "this.", "based", "cope", "nice", "+1", "^",
}

# --------------------------------------------------------------------------- #
# Crawl policy
# --------------------------------------------------------------------------- #
DEFAULT_WINDOW_HOURS = 24.0
# Four pages in flight at once. The archive answered 422 to concurrent calls when
# this was measured on 2026-11-22 and answered 200 to four on 2026-11-25, in 2.0 s
# wall against 8.3 s serialised; RDX_CRAWL_PARALLEL=1 restores the old behaviour.
CRAWL_PARALLEL = int(os.environ.get("RDX_CRAWL_PARALLEL", "4"))
ARCTIC_PACE_MAX = float(os.environ.get("RDX_ARCTIC_PACE_MAX", "2.5"))  # cap on adaptive widening
# A page is an archive request, so pages are the run's real currency. The budget
# bounds a run in minutes instead of leaving it to the archive's mood: 2026-11-25
# spent 625 requests and 13.6 min on one 24 h sweep, 139 of them refused as load.
# Measured 2026-11-25 under load: the archive grants ~0.6-1.1 pages/s and refuses
# 11 of 14 requests even spaced 0.6 s apart, so a run's length in minutes is
# roughly pages / 0.8. Seventy subreddits need ~140 pages before each of them has
# even been touched (a posts page plus a comments page each), so a budget below
# that leaves most subs capped, and a capped sub redraws the same pages next run —
# the sweep chases its own tail. 240 pages (~4 min loaded, ~2.5 min quiet) touches
# everyone with room to spare and costs 6.5 min at the loaded-hour rate measured
# that evening, 2.5 min when quiet. 0 means no ceiling, for a deliberate deep
# sweep; --subs with a dozen core boards costs 40 s for the same 24 hours.
CRAWL_PAGE_BUDGET = int(os.environ.get("RDX_PAGE_BUDGET", "240"))
CRAWL_SUB_PAGE_CAP = int(os.environ.get("RDX_SUB_PAGE_CAP", "20"))   # per subreddit per kind
CRAWL_PAGES_PER_TASK = int(os.environ.get("RDX_PAGES_PER_TASK", "4"))

HYDRATE_MIN_COMMENTS = 25      # thread-level hydration floor (num_comments)
HYDRATE_MIN_SCORE = 40
HYDRATE_MAX_PER_SUB = 12       # cap on full-thread fetches per sub per run
MAX_CATCHUP_HOURS = 168        # how far back an incremental run may resume from a stale cursor
MAX_PAGES_PER_SUB = 600        # hard stop per kind; 600 * 100 = 60 000 records

# --------------------------------------------------------------------------- #
# LLM (Ollama Cloud, same key the news scanner uses)
# --------------------------------------------------------------------------- #
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
# glm-5.3-flash is the default: ten times cheaper per token than the full model and
# good enough for this job, where the evidence pack does most of the work and the
# model mostly has to read it faithfully. Pass --model glm-5.3 when an idea matters
# enough to want the stronger reader.
LLM_MODEL = os.environ.get("RDX_LLM_MODEL", "glm-5.3-flash")
LLM_MODEL_FALLBACK = os.environ.get("RDX_LLM_FALLBACK", "deepseek-v4.1-flash")
# Both models above are reasoning models: they spend tokens on a `thinking`
# field before writing `content`, so a stingy budget yields empty content.
LLM_NUM_PREDICT = int(os.environ.get("RDX_LLM_NUM_PREDICT", "2600"))
LLM_TEMPERATURE = float(os.environ.get("RDX_LLM_TEMPERATURE", "0.35"))
# The reasoning field is a budget sink: glm-5.3 spent 20 000+ characters of
# thinking and produced nothing at num_predict=5200. For schema-bound JSON the
# reasoning buys little, so it is off by default and available with --think.
LLM_THINK = os.environ.get("RDX_LLM_THINK", "0").strip().lower() in ("1", "true", "yes", "on")

# --------------------------------------------------------------------------- #
# The idea generator
# --------------------------------------------------------------------------- #
GEN_CANDIDATES = int(os.environ.get("RDX_GEN_CANDIDATES", "10"))   # clusters per run
GEN_IDEAS = int(os.environ.get("RDX_GEN_IDEAS", "5"))              # ideas kept per run
GEN_SNIPPET_BUDGET = int(os.environ.get("RDX_GEN_BUDGET", "11000"))  # chars of evidence
GEN_MIN_QUALITY = float(os.environ.get("RDX_GEN_MIN_QUALITY", "0.28"))
GEN_DEDUPE_COSINE = float(os.environ.get("RDX_GEN_DEDUPE", "0.92"))
GEN_NEWS_HOURS = float(os.environ.get("RDX_GEN_NEWS_HOURS", "72"))

# --------------------------------------------------------------------------- #
# Web
# --------------------------------------------------------------------------- #
PORT = int(os.environ.get("PORT", "8792"))
HOST = os.environ.get("HOST", "127.0.0.1")
