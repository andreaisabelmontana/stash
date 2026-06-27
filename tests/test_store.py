"""Tests for the SQLite store."""

from __future__ import annotations

from stash.store import Item, Store


def test_save_and_load_round_trip(tmp_path):
    db = tmp_path / "stash.db"
    with Store(db) as store:
        item = store.save(
            Item(
                url="https://youtu.be/abc12345678",
                title="A talk",
                transcript="This is the transcript.",
                tags=["ai", "ml"],
                summary="A short summary.",
            )
        )
        assert item.id is not None
        assert item.saved_at is not None

    # Reopen from disk: data must persist exactly.
    with Store(db) as store2:
        loaded = store2.get(item.id)
        assert loaded is not None
        assert loaded.url == "https://youtu.be/abc12345678"
        assert loaded.title == "A talk"
        assert loaded.transcript == "This is the transcript."
        assert loaded.tags == ["ai", "ml"]
        assert loaded.summary == "A short summary."
        assert loaded.saved_at == item.saved_at


def test_all_and_len():
    with Store(":memory:") as store:
        assert len(store) == 0
        store.save(Item(transcript="one"))
        store.save(Item(transcript="two"))
        assert len(store) == 2
        items = store.all()
        assert [i.transcript for i in items] == ["one", "two"]


def test_find_by_tag():
    with Store(":memory:") as store:
        store.save(Item(transcript="a", tags=["finance"]))
        store.save(Item(transcript="b", tags=["ai", "ml"]))
        store.save(Item(transcript="c", tags=["finance", "ml"]))
        finance = store.find_by_tag("finance")
        assert {i.transcript for i in finance} == {"a", "c"}
        ml = store.find_by_tag("ml")
        assert {i.transcript for i in ml} == {"b", "c"}


def test_get_missing_returns_none():
    with Store(":memory:") as store:
        assert store.get(999) is None
