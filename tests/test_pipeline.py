"""Tests for ingestion, the LLM hook, and the end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from stash.ingest import TranscriptUnavailable, from_youtube, ingest, video_id
from stash.llm import llm_available, llm_summarize
from stash.pipeline import stash_item
from stash.store import Store

DATA = Path(__file__).resolve().parent.parent / "data"


# -- ingestion ------------------------------------------------------------
def test_video_id_parsing():
    assert video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id("not a url") is None


def test_ingest_from_text_normalizes_whitespace():
    assert ingest(text="  hello   world\n\nthere  ") == "hello world there"


def test_ingest_from_file():
    text = ingest(file=str(DATA / "transformers_talk.txt"))
    assert "attention" in text.lower()
    assert "\n" not in text  # normalized to single spaces


def test_ingest_requires_a_source():
    with pytest.raises(ValueError):
        ingest()


def test_from_youtube_degrades_gracefully_on_bad_id():
    # No network needed: an unparseable id fails fast and cleanly.
    with pytest.raises(TranscriptUnavailable):
        from_youtube("definitely not a youtube url")


# -- optional LLM hook ----------------------------------------------------
def test_llm_not_available_without_key():
    assert llm_available(env={}) is False
    assert llm_summarize("some transcript", env={}) is None


# -- full offline pipeline ------------------------------------------------
def test_pipeline_runs_fully_offline_on_committed_transcript():
    transcript = (DATA / "transformers_talk.txt").read_text(encoding="utf-8")
    with Store(":memory:") as store:
        # use_llm left default; with no API key in env it must stay extractive.
        result = stash_item(
            store,
            text=transcript,
            title="How transformers work",
            url="https://youtu.be/transformers",
            num_sentences=3,
        )
        assert result.used_llm is False
        assert result.llm_summary is None
        # Extractive summary is real and non-empty, bounded by request.
        assert 1 <= len(result.extractive.sentences) <= 3
        assert result.item.summary == result.extractive.summary
        assert result.item.summary
        # Keywords + tags were derived and persisted.
        assert any("attention" in k for k in result.keywords)
        assert result.item.tags
        assert result.item.id is not None

        # And it actually persisted.
        reloaded = store.get(result.item.id)
        assert reloaded.summary == result.item.summary
        assert reloaded.tags == result.item.tags


def test_pipeline_summary_length_bounded():
    transcript = (DATA / "index_funds_talk.txt").read_text(encoding="utf-8")
    with Store(":memory:") as store:
        result = stash_item(store, text=transcript, num_sentences=2)
        assert len(result.extractive.sentences) <= 2
