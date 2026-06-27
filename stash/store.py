"""SQLite-backed store of saved items.

Each saved item records the source URL, a title, the raw transcript, a list of
tags, the generated summary, and a saved-at timestamp. Tags are stored as a
JSON array in a single column to keep the schema simple.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    url        TEXT,
    title      TEXT,
    transcript TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',
    summary    TEXT,
    saved_at   TEXT NOT NULL
);
"""


@dataclass
class Item:
    """A saved item. ``id`` is assigned by the store on save."""

    transcript: str
    url: str | None = None
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    summary: str | None = None
    saved_at: str | None = None
    id: int | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """A thin SQLite wrapper. Use ``":memory:"`` for an ephemeral store."""

    def __init__(self, path: str | Path = "stash.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def save(self, item: Item) -> Item:
        """Insert ``item`` and return it with its assigned ``id``/``saved_at``."""
        saved_at = item.saved_at or _utcnow()
        cur = self._conn.execute(
            "INSERT INTO items (url, title, transcript, tags, summary, saved_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                item.url,
                item.title,
                item.transcript,
                json.dumps(item.tags),
                item.summary,
                saved_at,
            ),
        )
        self._conn.commit()
        item.id = int(cur.lastrowid)
        item.saved_at = saved_at
        return item

    # -- reads -------------------------------------------------------------
    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> Item:
        return Item(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            transcript=row["transcript"],
            tags=json.loads(row["tags"]),
            summary=row["summary"],
            saved_at=row["saved_at"],
        )

    def get(self, item_id: int) -> Item | None:
        row = self._conn.execute(
            "SELECT * FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def all(self) -> list[Item]:
        rows = self._conn.execute(
            "SELECT * FROM items ORDER BY id"
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def find_by_tag(self, tag: str) -> list[Item]:
        """Return items carrying ``tag`` (exact slug match)."""
        return [it for it in self.all() if tag in it.tags]

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()
        return int(row["n"])
