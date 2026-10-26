"""Household memory: where things are kept.

The hard part is not storage, it is naming. The same object is called
"nail cutter", "नखं कापायचं", "नख कातरणी". String distance cannot bridge
Marathi and English, so the design is:

  * every item has one canonical name plus any number of aliases
  * aliases accumulate over time (from the LLM, or from Aai correcting Banku)
  * lookup is exact-on-alias first, then fuzzy within the same script

That keeps behaviour predictable and debuggable, which matters more than
looking clever. A wrong "it's in the kitchen" is worse than "मला समजलं नाही".
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from rapidfuzz import fuzz

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS aliases (
    id         INTEGER PRIMARY KEY,
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    norm       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS placements (
    id         INTEGER PRIMARY KEY,
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    place      TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_placements_item ON placements(item_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_aliases_item ON aliases(item_id);
"""

_PUNCT = re.compile(r"[।.,!?;:()\[\]\"'“”‘’/\\-]+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold away everything that should not affect matching."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200d", "").replace("\u200c", "")
    text = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", text).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Found:
    item: str
    place: str
    note: str | None
    since: str
    score: int

    def as_sentence(self) -> str:
        """What Banku actually says. Kept here so the wording is one place."""
        when = self.since[:10]
        base = f"{self.item} {self.place} मध्ये आहे."
        if self.note:
            base += f" ({self.note})"
        return f"{base} शेवटचं {when} ला सांगितलं होतं."


@dataclass(frozen=True)
class Remembered:
    item: str
    place: str
    created: bool
    aliases: tuple[str, ...]


class Memory:
    def __init__(self, db_path: Path | None = None) -> None:
        self.path = db_path or config.paths.db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ write

    def remember(
        self,
        item: str,
        place: str,
        note: str | None = None,
        *,
        aliases: list[str] | None = None,
    ) -> Remembered:
        """Record that `item` lives in `place`.

        Repeating a statement is normal ("the keys are on the hook" every day),
        so this updates the placement rather than refusing as a duplicate.
        """
        item, place = item.strip(), place.strip()
        if not item or not place:
            raise ValueError("both item and place are required")

        norm = normalize(item)
        with self._conn() as c:
            row = c.execute(
                "SELECT i.id, i.name FROM aliases a JOIN items i ON i.id = a.item_id "
                "WHERE a.norm = ? AND i.deleted_at IS NULL",
                (norm,),
            ).fetchone()

            created = row is None
            if created:
                cur = c.execute(
                    "INSERT INTO items (name, created_at) VALUES (?, ?)", (item, _now())
                )
                item_id = cur.lastrowid
                c.execute(
                    "INSERT INTO aliases (item_id, text, norm, created_at) VALUES (?,?,?,?)",
                    (item_id, item, norm, _now()),
                )
                canonical = item
            else:
                item_id = row["id"]
                canonical = row["name"]

            for extra in aliases or []:
                extra = extra.strip()
                if not extra:
                    continue
                en = normalize(extra)
                if en and not c.execute(
                    "SELECT 1 FROM aliases WHERE norm = ?", (en,)
                ).fetchone():
                    c.execute(
                        "INSERT INTO aliases (item_id, text, norm, created_at) VALUES (?,?,?,?)",
                        (item_id, extra, en, _now()),
                    )

            c.execute(
                "INSERT INTO placements (item_id, place, note, created_at) VALUES (?,?,?,?)",
                (item_id, place, note, _now()),
            )
            stored = tuple(
                r["text"]
                for r in c.execute(
                    "SELECT text FROM aliases WHERE item_id = ? ORDER BY id", (item_id,)
                )
            )

        return Remembered(item=canonical, place=place, created=created, aliases=stored)

    def add_alias(self, item: str, alias: str) -> bool:
        """Attach another name to an existing item. Returns False if it was taken."""
        target = self.find(item, limit=1)
        if not target:
            return False
        with self._conn() as c:
            try:
                c.execute(
                    "INSERT INTO aliases (item_id, text, norm, created_at) "
                    "SELECT id, ?, ?, ? FROM items WHERE name = ? AND deleted_at IS NULL",
                    (alias, normalize(alias), _now(), target[0].item),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def forget(self, item: str) -> bool:
        """Soft delete: keeps the history if you ever need to explain a mistake."""
        hits = self.find(item, limit=1)
        if not hits:
            return False
        with self._conn() as c:
            c.execute("UPDATE items SET deleted_at = ? WHERE name = ?", (_now(), hits[0].item))
        return True

    # ------------------------------------------------------------------- read

    def _ranked(self, query: str, low: int) -> list[Found]:
        """Score every known alias against `query`, best first.

        Measured on real ASR output (26 Oct 2026): a Marathi speaker's "nail
        cutter" comes back as "neel cutter" or "nail katter", which score 81.8 —
        two substitutions on an eleven-character word. A cutoff of 82 therefore
        missed it by 0.2. The cutoff is 78 now, and anything between 60 and 78 is
        reported separately so Banku can ask "did you mean ...?" rather than
        either guessing or giving up.
        """
        nq = normalize(query)
        if not nq:
            return []

        with self._conn() as c:
            rows = c.execute(
                """
                SELECT i.name AS item, p.place, p.note, p.created_at, a.norm
                FROM aliases a
                JOIN items i ON i.id = a.item_id
                JOIN placements p ON p.id = (
                    SELECT id FROM placements WHERE item_id = i.id ORDER BY id DESC LIMIT 1
                )
                WHERE i.deleted_at IS NULL
                """
            ).fetchall()

        scored: dict[str, tuple[int, sqlite3.Row]] = {}
        for row in rows:
            if row["norm"] == nq:
                score = 100
            else:
                score = round(
                    max(
                        fuzz.WRatio(nq, row["norm"], score_cutoff=low),
                        fuzz.token_set_ratio(nq, row["norm"], score_cutoff=low),
                        fuzz.partial_ratio(nq, row["norm"], score_cutoff=90) * 0.95,
                    )
                )
            if score < low:
                continue
            best = scored.get(row["item"])
            if best is None or score > best[0]:
                scored[row["item"]] = (score, row)

        ranked = sorted(scored.values(), key=lambda t: -t[0])
        return [
            Found(
                item=row["item"],
                place=row["place"],
                note=row["note"],
                since=row["created_at"],
                score=int(score),
            )
            for score, row in ranked
        ]

    def find(self, query: str, *, limit: int = 5, threshold: int | None = None) -> list[Found]:
        """Return best placements for `query`, best first.

        More than one result means "I am not sure"; the caller should ask
        rather than guess.
        """
        cutoff = config.match.threshold if threshold is None else threshold
        return [h for h in self._ranked(query, cutoff) if h.score >= cutoff][
            : clamp(limit, 1, 20)
        ]

    def near(self, query: str, *, limit: int = 3, low: int = 60) -> list[Found]:
        """Almost-matches, for asking a clarifying question.

        These are deliberately NOT answers. They exist so that Banku can say
        "do you mean the nail cutter?" instead of "I don't know", without ever
        asserting a location it is not sure about.
        """
        cutoff = config.match.threshold
        hits = [h for h in self._ranked(query, low) if low <= h.score < cutoff]
        return hits[: max(1, min(limit, 5))]


    def is_ambiguous(self, hits: list[Found]) -> bool:
        """Two close answers means ask, do not pick."""
        return len(hits) > 1 and (hits[0].score - hits[1].score) < config.match.ambiguous_gap

    def keyterms(self, limit: int = 50) -> list[str]:
        """Aliases to bias the STT engine toward.

        This is the nice feedback loop: every item Banku learns makes it better
        at hearing that item's name next time.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT a.text FROM aliases a JOIN items i ON i.id = a.item_id "
                "WHERE i.deleted_at IS NULL ORDER BY LENGTH(a.text) DESC LIMIT ?",
                (clamp(limit, 1, 50),),
            ).fetchall()
        return [r["text"] for r in rows]

    def all_items(self) -> list[tuple[str, str, str | None]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT i.name, p.place, p.note
                FROM items i JOIN placements p ON p.id = (
                    SELECT id FROM placements WHERE item_id = i.id ORDER BY id DESC LIMIT 1
                )
                WHERE i.deleted_at IS NULL ORDER BY i.name
                """
            ).fetchall()
        return [(r["name"], r["place"], r["note"]) for r in rows]


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


if __name__ == "__main__":  # tiny self-check, run with: python -m banku.memory
    import tempfile

    db = Path(tempfile.mkdtemp()) / "t.db"
    m = Memory(db)
    m.remember("nail cutter", "TV unit chya dusrya drawer", aliases=["नखं कापायचं", "nail cutter"])
    m.remember("passport", "cupboard var", note="blue cover")
    assert m.find("nail cutter")[0].place.startswith("TV unit")
    assert m.find("नखं कापायचं")[0].item == "nail cutter"
    assert m.find("napkin cutter")[0].item == "nail cutter", m.find("napkin cutter")
    assert not m.find("unicorn")
    assert m.forget("passport")
    assert not m.find("passport")
    print("memory self-check OK")
