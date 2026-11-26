"""registry — the subreddit list as a file, in and out.

The database is the runtime source of truth; this module is how the list gets
into it and back out again. `subreddits.yaml` is the seed and the export format,
so the list stays editable in a text editor, reviewable in a diff, and honest
about subs the source cannot actually serve.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

import config as C
import store
import textproc as T

HEADER = """\
# reddit-alpha — the subreddit registry.
# Written by `rdx.py subs export` on {when}. The table `subreddit` is the runtime
# source of truth; edit this file and run `rdx.py subs import` to merge changes.
#
#   name    the subreddit, without r/     theme   grouping used for filters and the UI
#   weight  0.2–1.5, how much it counts   enabled false keeps history, stops crawling
"""


def load_seed(conn, path: str | Path | None = None) -> dict:
    """Merge a YAML registry into the subreddit table. Never deletes."""
    p = Path(path or C.SUBS_YAML)
    if not p.exists():
        return {"error": f"no such file: {p}"}
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = doc.get("subreddits") or []
    added = updated = 0
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        exists = conn.execute("SELECT 1 FROM subreddit WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        store.upsert_subreddit(
            conn, name,
            theme=str(row.get("theme") or "other"),
            weight=float(row.get("weight") or 1.0),
            enabled=bool(row.get("enabled", True)),
            notes=str(row.get("note") or ""))
        added += 0 if exists else 1
        updated += 1 if exists else 0
    return {"file": str(p), "added": added, "updated": updated,
            "total": len(store.list_subreddits(conn))}


def export_seed(conn, path: str | Path | None = None) -> dict:
    """Write the current registry back out in the seed's own format."""
    p = Path(path or C.SUBS_YAML)
    rows = store.list_subreddits(conn)
    by_theme: dict[str, list[dict]] = {}
    for r in rows:
        by_theme.setdefault(r["theme"] or "other", []).append(r)
    out: list[dict] = []
    for theme in sorted(by_theme):
        for r in sorted(by_theme[theme], key=lambda x: (-x["weight"], x["name"].lower())):
            item = {"name": r["name"], "theme": r["theme"], "weight": round(r["weight"], 2)}
            if not r["enabled"]:
                item["enabled"] = False
            if r["notes"]:
                item["note"] = r["notes"]
            out.append(item)
    text = HEADER.format(when=T.now_iso()) + "version: 1\nsubreddits:\n" + yaml.safe_dump(
        out, sort_keys=False, allow_unicode=True, width=100, default_flow_style=False)
    p.write_text(text, encoding="utf-8")
    return {"file": str(p), "written": len(out)}


def validate(conn, source, *, names: Iterable[str] | None = None, hours: float = 168.0,
             disable_empty: bool = True, progress=None) -> list[dict]:
    """Probe every subreddit and record what the source actually serves.

    A name that returns nothing is not proof it does not exist on Reddit — it is
    proof that *this source* has nothing. The note says exactly that.
    """
    todo = list(names) if names else [r["name"] for r in store.list_subreddits(conn)]
    out: list[dict] = []
    for name in todo:
        row = conn.execute("SELECT * FROM subreddit WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        if row is None:
            out.append({"name": name, "error": "not in registry"})
            continue
        try:
            info = source.probe(name, hours=hours)
        except Exception as exc:
            out.append({"name": name, "error": str(exc)[:200]})
            continue
        posts = info.get("recent_posts") or 0
        comments = info.get("recent_comments") or 0
        alive = bool(posts or comments)
        conn.execute(
            "UPDATE subreddit SET validated_at=?, recent_posts=?, recent_comments=? WHERE name=?",
            (T.now_iso(), posts, comments, name))
        note = None
        if not alive and disable_empty and row["enabled"]:
            note = (f"no data at {source.name} in the last {hours / 24:.0f}d "
                    f"(validated {T.now_iso()[:10]}) — switched off; not proof the sub is dead")
            conn.execute("UPDATE subreddit SET enabled=0, notes=? WHERE name=?", (note, name))
        elif alive and not row["enabled"] and (row["notes"] or "").startswith("no data at"):
            conn.execute("UPDATE subreddit SET enabled=1, notes='' WHERE name=?", (name,))
        conn.commit()
        out.append({"name": name, "recent_posts": posts, "recent_comments": comments,
                    "alive": alive, "theme": row["theme"], "enabled": bool(row["enabled"]),
                    "note": note, "error": None})
        if progress:
            progress({"event": "validated", **out[-1]})
    return out
