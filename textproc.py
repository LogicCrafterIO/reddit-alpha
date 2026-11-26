"""Text processing: cleaning, chunking, ticker extraction, quality scoring.

No database and no network here, which is what makes every rule testable in
isolation. `store` calls into this module when it indexes a document; `retrieve`
calls into it when it parses a reference.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import config as C

# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0]+")


def clean(text: str | None) -> str:
    """Normalise a Reddit body to plain text without destroying its structure."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _ZERO_WIDTH.sub("", t)
    t = _SPACES.sub(" ", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


_JUNK_RE = __import__("re").compile("|".join(C.JUNK_PATTERNS), __import__("re").I)


def is_junk(text: str | None) -> bool:
    t = clean(text).lower().strip(" .!?")
    if len(t) < 3:
        return True
    if t in C.JUNK_BODIES:
        return True
    if _JUNK_RE.search(t):
        return True
    # a body that is only emoji/punctuation carries nothing
    if not re.search(r"[a-z0-9]", t):
        return True
    return False


# --------------------------------------------------------------------------- #
# chunking — the retrieval unit
# --------------------------------------------------------------------------- #
_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_long(par: str, limit: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries, hard-split as last resort."""
    out, cur = [], ""
    for s in _SENT.split(par):
        if not s:
            continue
        if len(cur) + len(s) + 1 <= limit:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                out.append(cur)
            while len(s) > limit:
                out.append(s[:limit])
                s = s[limit:]
            cur = s
    if cur:
        out.append(cur)
    return [o for o in out if o]


def chunk_text(text: str, limit: int = C.CHUNK_CHARS,
               min_chars: int = C.MIN_CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into ~limit-char units that never split mid-sentence.

    A short reply is one chunk. A 9 000-char DD post becomes ~8 chunks with the
    paragraph structure preserved, which is what makes a snippet readable.
    """
    t = clean(text)
    if not t:
        return []
    if len(t) <= limit:
        return [t] if len(t) >= 1 else []

    pieces: list[str] = []
    for par in re.split(r"\n\s*\n", t):
        par = par.strip()
        if not par:
            continue
        pieces.extend(_split_long(par, limit) if len(par) > limit else [par])

    chunks, cur = [], ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= limit:
            cur = f"{cur}\n\n{p}"
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    # fold a runt tail into its predecessor rather than emitting a stub
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def heading_of(text: str) -> str:
    """First non-empty line, trimmed — gives a chunk a human label."""
    for line in clean(text).splitlines():
        line = line.strip().lstrip("#*-> ").strip()
        if line:
            return line[:120]
    return ""


# --------------------------------------------------------------------------- #
# ticker extraction
# --------------------------------------------------------------------------- #
# Words that look like tickers and are not. Anything all-caps in a trading
# subreddit matches \b[A-Z]{2,5}\b, so this list does the heavy lifting.
STOP = set("""
DD YOLO OP EDIT TLDR TL;DR TIL IMO IMHO AFAIK FOMO FUD ATH ATL EPS PE PEG IPO
ETF ETP SPAC REIT BDC CEF HY IG FX DCF NPV IRR ROIC ROE ROA EV EBITDA FCF GAAP
CEO CFO CTO COO CMO CRO CIO CISO CTO VP SVP MD GM SVP PM AM
AI ML LLM LLMS AGI GPU GPUS CPU CPUS TPU TPUS FPGA HBM DRAM NAND SSD ASIC ASICS
IV ITM OTM ATM DTE PDT HFT RSI MACD EMA SMA VWAP ATR AUM NAV LTV CAC ARR MRR TCV
RPO OPM TAM SAM CAGR YOY QOQ MOM YTD MTD QTD EOD EOW EOM EOY
FED FOMC CPI PPI PCE GDP GNP PMI ISM NFP JOLTS CPI MO
SEC IRS IRA FDIC FTC DOJ FCC FDA EPA FAA DOE DOD NATO OPEC IEA EIA CDC NIH
USD EUR GBP JPY CNY CAD AUD CHF INR US USA UK EU UAE EUA EZ
Q1 Q2 Q3 Q4 H1 H2 FY
WSB GME APES APE BOT BOTs EDIT2 TLDR
LOL LMAO ROFL WTF OMG BTFD HODL FOMO DIY ELI5 PSA AMA AMA
NSFW NSFL NFA DYOR FWIW TBH IIRC SMH IMHO
CALL CALLS PUT PUTS STOCK STOCKS BUY SELL HOLD LONG SHORT BULL BEAR
SPX NDX VIX
""".split()) - {"SPX", "NDX", "VIX"}

# Words that are real tickers but so common as English that a bare match is
# noise; they need a cashtag (${'X'}) or an explicit "the stock" cue.
AMBIGUOUS = set("""
A I AM ALL ANY ARE ARM AS AT BE BY CAN CAR CAT COST DO FOR GO HAD HAS HAVE HE
HIM HIS HOW IF IN IS IT ITS KEY LAST LIKE LOW MAY ME MOST MUCH MY NEW NO NOR
NOT NOW OF OFF ON ONE ONLY OR OUR OUT OVER OW OWE OWN PAY PER PUT RUN SEE SHE
SO SOME SUCH THE THEIR THEM THEN THERE THESE THEY THIS TOO TWO UP US USE WAS
WAY WE WELL WERE WHAT WHEN WHO WHY WILL WITH YOU YOUR
C CORE DECK FAST FLEX GAME GOAL HOPE HUNT LION LOVE MAIN MARK MIND MOVE OPEN
PAY PLAN PLAY REAL RISE ROCK ROOM SAVE SHOP SITE SIZE STEP TIME TURN UNIT VALUE
VERY WANT WIDE WISE WORK
""".split())

_CASHTAG = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z]{1,5})\b")
_BARE = re.compile(r"(?<![A-Za-z0-9_$#/])([A-Z]{2,5})(?![A-Za-z0-9_])")

# Company / product / person -> ticker. Aliases shorter than four characters
# are excluded on purpose: prose collides with them far more often than it
# means them.
ALIASES: dict[str, str] = {
    # mega-cap tech & semis
    "nvidia": "NVDA", "geforce": "NVDA", "blackwell": "NVDA", "rubin": "NVDA",
    "apple": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT", "azure": "MSFT", "windows": "MSFT",
    "amazon": "AMZN", "aws": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "youtube": "GOOGL",
    "facebook": "META", "instagram": "META", "metaverse": "META",
    "tesla": "TSLA", "spacex": "TSLA",
    "broadcom": "AVGO", "micron": "MU", "tsmc": "TSM", "intel": "INTC",
    "qualcomm": "QCOM", "marvell": "MRVL", "supermicro": "SMCI",
    "oracle": "ORCL", "salesforce": "CRM", "servicenow": "NOW",
    "palantir": "PLTR", "snowflake": "SNOW", "databricks": "SNOW",
    "crowdstrike": "CRWD", "zscaler": "ZS", "palo alto": "PANW",
    "fortinet": "FTNT", "tenable": "TENB", "cloudflare": "NET",
    "datadog": "DDOG", "mongodb": "MDB", "atlassian": "TEAM", "workday": "WDAY",
    "adobe": "ADBE", "netflix": "NFLX", "disney": "DIS", "spotify": "SPOT",
    "roblox": "RBLX", "take-two": "TTWO", "take two": "TTWO",
    "airbnb": "ABNB", "uber": "UBER", "lyft": "LYFT", "doordash": "DASH",
    # crypto & fintech
    "coinbase": "COIN", "robinhood": "HOOD", "sofi": "SOFI",
    "microstrategy": "MSTR", "strategy inc": "MSTR", "mara holdings": "MARA",
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "dogecoin": "DOGE",
    # health
    "moderna": "MRNA", "pfizer": "PFE", "eli lilly": "LLY", "lilly": "LLY",
    "unitedhealth": "UNH", "united health": "UNH", "johnson": "JNJ",
    "merck": "MRK", "abbvie": "ABBV", "amgen": "AMGN", "gilead": "GILD",
    "biogen": "BIIB", "novo nordisk": "NVO", "astrazeneca": "AZN",
    "sanofi": "SNY", "hims": "HIMS", "teladoc": "TDOC",
    # consumer / retail
    "altria": "MO", "philip morris": "PM", "mondelez": "MDLZ",
    "coca cola": "KO", "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "walmart": "WMT", "costco": "COST", "target": "TGT",
    "home depot": "HD", "nike": "NKE", "starbucks": "SBUX",
    "mcdonald": "MCD", "chipotle": "CMG", "carvana": "CVNA",
    "caesars": "CZR", "macy": "M", "lululemon": "LULU", "elf beauty": "ELF",
    # energy & materials
    "chevron": "CVX", "exxon": "XOM", "conocophillips": "COP",
    "occidental": "OXY", "halliburton": "HAL", "schlumberger": "SLB",
    "newmont": "NEM", "freeport": "FCX", "albemarle": "ALB",
    # banks & payments
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman": "GS",
    "morgan stanley": "MS", "schwab": "SCHW", "blackrock": "BLK",
    "visa": "V", "mastercard": "MA", "paypal": "PYPL", "block inc": "SQ",
    "berkshire": "BRK.B",
    # defence / aerospace
    "lockheed": "LMT", "raytheon": "RTX", "northrop": "NOC",
    "general dynamics": "GD", "boeing": "BA", "l3harris": "LHX",
    "huntington ingalls": "HII", "kratos": "KTOS", "aerovironment": "AVAV",
    "rocket lab": "RKLB", "rocketlab": "RKLB", "textron": "TXT",
    "transdigm": "TDG", "howmet": "HWM", "curtis wright": "CW",
    "heico": "HEI", "axon": "AXON", "elbit": "ESLT", "leonardo": "DRS",
    "palantir technologies": "PLTR", "anduril": "PLTR",
    # china / EM
    "alibaba": "BABA", "tencent": "TCEHY", "baidu": "BIDU",
    "nio": "NIO", "xpeng": "XPEV", "pdd": "PDD", "temu": "PDD",
    "jd.com": "JD", "byd": "BYDDY", "xiaomi": "XIACY",
    # indices / ETFs people talk about by name
    "s and p": "SPY", "sp500": "SPY", "spx": "SPY", "s&p": "SPY",
    "nasdaq": "QQQ", "russell": "IWM", "semiconductor etf": "SMH",
    "soxx": "SOXX", "tlt": "TLT", "arkk": "ARKK", "xle": "XLE",
}

_ALIAS_KEYS = sorted((k for k in ALIASES if len(k) >= 4),
                     key=len, reverse=True)
_ALIAS_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(k) for k in _ALIAS_KEYS) + r")(?![a-z])",
                       re.IGNORECASE)

def extract_tickers(text: str, limit: int = 40) -> list[tuple[str, str, float]]:
    """Return [(symbol, kind, confidence)] from a body of Reddit text.

    Three signals, cheapest first: a cashtag is near-certain, a bare uppercase
    token is probable if it is not a stopword, and a company name is probable
    if the alias is long enough not to collide with prose.
    """
    if not text:
        return []
    t = clean(text)
    found: dict[str, tuple[str, float]] = {}

    for m in _CASHTAG.finditer(t):
        sym = m.group(1).upper()
        if sym in STOP or len(sym) < 2:
            continue
        found[sym] = ("cashtag", 0.95)

    # strip cashtags so the bare-token pass does not double-count them
    bare_src = _CASHTAG.sub(" ", t)
    for m in _BARE.finditer(bare_src):
        sym = m.group(1)
        if sym in STOP or sym in AMBIGUOUS:
            continue
        found.setdefault(sym, ("ticker", 0.55))

    for m in _ALIAS_RE.finditer(t):
        sym = ALIASES.get(m.group(1).lower())
        if sym and sym not in found:
            found[sym] = ("name", 0.6)

    # A dollar sign next to a symbol is a price, not a ticker: "$10" etc. is
    # already excluded by requiring a leading \$ and 1-5 letters right after.
    out = [(s, k, c) for s, (k, c) in found.items()]
    out.sort(key=lambda x: -x[2])
    return out[:limit]


def primary_ticker(text: str) -> str | None:
    got = extract_tickers(text, limit=1)
    return got[0][0] if got else None


# --------------------------------------------------------------------------- #
# quality — "is this worth storing as an insight?"
# --------------------------------------------------------------------------- #
_ARGUMENT_CUES = re.compile(
    r"\b(because|therefore|however|whereas|implies|suggests|guidance|backlog|"
    r"margin|margins|multiple|capex|dilution|free cash flow|balance sheet|"
    r"valuation|bottleneck|substitut|regulat|contract|award|utilization|"
    r"inventory|pricing power|depreciation|writedown|impairment|hedg|"
    r"unwind|refinanc|short interest|open interest|gamma|skew|volatility|"
    r"liquidity|earnings|revenue|estimate|consensus|revision)\b", re.I)
_NUM_CUE = re.compile(r"[$€£¥]\s?\d|\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?(?:b|bn|m|k)\b", re.I)
_MEME_FLAIR = re.compile(r"meme|loss|gain|cope|shitpost|yolo|guh|ape|tendies|flair", re.I)
_GOOD_FLAIR = re.compile(r"\bdd\b|thesis|analysis|research|discussion|due diligence|news|technicals?|idea", re.I)


def quality_score(text: str, *, score: int = 0, num_comments: int = 0,
                  flair: str | None = None, kind: str = "comment",
                  age_hours: float | None = None) -> float:
    """0..1 — how much of an argument this text carries, plus how much it landed.

    Deliberately not a sentiment score and not a prediction: it answers one
    narrow question, "should this cost the generator a snippet slot", and
    everything in it is visible in the text or the record.
    """
    t = clean(text)
    chars = len(t)
    if chars == 0:
        return 0.0

    substance = min(1.0, math.log1p(chars) / math.log1p(2600))
    nums = _NUM_CUE.findall(t)
    num_score = min(1.0, len(nums) / 6)
    cues = min(1.0, len(_ARGUMENT_CUES.findall(t)) / 7)
    # engagement, compressed: 10 upvotes and 5 000 upvotes are not 500x apart
    eng = min(1.0, math.log1p(max(0, score)) / math.log1p(3000))
    talk = min(1.0, math.log1p(max(0, num_comments)) / math.log1p(400))

    q = (0.34 * substance + 0.24 * cues + 0.16 * num_score
         + 0.14 * eng + 0.12 * talk)

    if flair:
        if _GOOD_FLAIR.search(flair):
            q += 0.10
        if _MEME_FLAIR.search(flair):
            q -= 0.16

    # a one-liner with 3 000 upvotes is a signal, not an insight: cap it
    if chars < 160:
        q *= 0.55
    if chars < 60:
        q *= 0.5
    # walls of text with no numbers and no argument cues are usually rants
    if chars > 3000 and num_score == 0 and cues == 0:
        q *= 0.7
    # age decay, mild: a strong six-day-old thread still counts
    if age_hours is not None and age_hours > 0:
        q *= 1.0 / (1.0 + max(0.0, age_hours - 72) / 240.0)

    return round(max(0.0, min(1.0, q)), 4)


# --------------------------------------------------------------------------- #
# references — the contract with the reader
# --------------------------------------------------------------------------- #
# r/wallstreetbets/1y2e1xc            a post
# r/wallstreetbets/1y2e1xc#c19szw98   one comment inside that post
_REF = re.compile(r"^r/([A-Za-z0-9_]+)/([A-Za-z0-9]+)(?:#c([A-Za-z0-9]+))?$")


def post_ref(sub: str, post_id: str) -> str:
    return f"r/{sub}/{post_id}"


def comment_ref(sub: str, post_id: str, comment_id: str) -> str:
    return f"r/{sub}/{post_id}#c{comment_id}"


def parse_ref(ref: str) -> tuple[str, str, str, str | None]:
    """`ref` -> (kind, subreddit, post_id, comment_id|None). Raises ValueError."""
    m = _REF.match(ref.strip())
    if not m:
        raise ValueError(
            f"not a reddit reference: {ref!r} (expected r/<sub>/<post_id>[#c<comment_id>])")
    sub, post, cid = m.group(1), m.group(2), m.group(3)
    return ("comment" if cid else "post"), sub, post, cid


def permalink(sub: str, post_id: str, comment_id: str | None = None) -> str:
    base = f"https://www.reddit.com/r/{sub}/comments/{post_id}/"
    return f"{base}comment/{comment_id}/" if comment_id else base


def iso(epoch: int | float | None) -> str | None:
    from datetime import datetime, timezone
    if not epoch:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="seconds")


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_precise() -> str:
    """Microsecond ISO stamp — two metric samples in the same second must not merge."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def iter_batches(rows: Iterable, n: int) -> Iterable[list]:
    batch: list = []
    for r in rows:
        batch.append(r)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
