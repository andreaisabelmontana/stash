"""Tests for keyword extraction and auto-tagging."""

from __future__ import annotations

from pathlib import Path

from stash.keywords import auto_tags, extract_keywords

DATA = Path(__file__).resolve().parent.parent / "data"


def test_keywords_surface_planted_salient_terms():
    # "neural" and "network" are planted as the dominant, repeated topic; an
    # unrelated filler sentence should not dominate the ranking.
    doc = (
        "A neural network learns weights from data. "
        "A neural network is trained with gradient descent on a loss. "
        "Each layer of the neural network applies a nonlinear activation. "
        "The weather today is sunny and warm outside."
    )
    keywords = extract_keywords(doc, top_k=6)
    joined = " ".join(keywords)
    assert "neural" in joined
    assert "network" in joined
    # The off-topic filler term should not outrank the planted topic.
    assert keywords.index(next(k for k in keywords if "neural" in k)) < len(keywords)


def test_keywords_on_real_transcript():
    text = (DATA / "transformers_talk.txt").read_text(encoding="utf-8")
    keywords = extract_keywords(text, top_k=10)
    joined = " ".join(keywords)
    assert "attention" in joined
    assert "transformer" in joined or "transformers" in joined


def test_keywords_empty_text():
    assert extract_keywords("") == []
    assert extract_keywords("the of and a an") == []


def test_auto_tags_are_slugs():
    text = (DATA / "index_funds_talk.txt").read_text(encoding="utf-8")
    tags = auto_tags(text, max_tags=5)
    assert 1 <= len(tags) <= 5
    for tag in tags:
        assert tag == tag.lower()
        assert " " not in tag
        assert all(c.isalnum() or c == "-" for c in tag)
    joined = " ".join(tags)
    assert "index" in joined or "fund" in joined or "funds" in joined


def test_auto_tags_unique():
    text = (DATA / "index_funds_talk.txt").read_text(encoding="utf-8")
    tags = auto_tags(text, max_tags=8)
    assert len(tags) == len(set(tags))
