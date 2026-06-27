"""Tests for the extractive summarization engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from stash.summarize import (
    SummaryResult,
    split_sentences,
    summarize,
    textrank_scores,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third one? Yes."
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
        "Yes.",
    ]


def test_split_sentences_keeps_abbreviation_together():
    sents = split_sentences("Dr. Smith arrived. He was late.")
    assert sents == ["Dr. Smith arrived.", "He was late."]


# A constructed document with a known centre. The first sentence is the hub:
# it shares one distinctive term ("light", "water", "sugar") with each of the
# next three sentences, which in turn share nothing with each other. The last
# sentence is off-topic noise. A centrality ranker should rank the hub first.
HUB_DOC = (
    "Photosynthesis lets plants turn light, water, and air into sugar. "
    "Bright light from the sun powers the first stage of the reaction. "
    "Water is absorbed through the roots and split inside the leaf. "
    "The resulting sugar is transported to feed the rest of the plant. "
    "A bicycle has two wheels and a metal frame."
)


def test_textrank_picks_the_central_sentence():
    sentences = split_sentences(HUB_DOC)
    scores = textrank_scores(sentences)
    top_idx = int(scores.argmax())
    assert sentences[top_idx].startswith("Photosynthesis")
    # The off-topic sentence must score lowest.
    assert int(scores.argmin()) == len(sentences) - 1


def test_textrank_central_sentence_in_summary():
    result = summarize(HUB_DOC, num_sentences=1, method="textrank")
    assert result.summary.startswith("Photosynthesis")


def test_summary_length_is_bounded():
    text = (DATA / "transformers_talk.txt").read_text(encoding="utf-8")
    for n in (1, 2, 3, 5):
        result = summarize(text, num_sentences=n, method="textrank")
        assert len(result.sentences) <= n
        assert len(result.sentences) >= 1


def test_summary_more_sentences_than_document():
    doc = "Only one sentence here."
    result = summarize(doc, num_sentences=10)
    assert len(result.sentences) == 1


def test_summary_preserves_reading_order():
    text = (DATA / "index_funds_talk.txt").read_text(encoding="utf-8")
    full = split_sentences(text)
    result = summarize(text, num_sentences=3, method="textrank")
    positions = [full.index(s) for s in result.sentences]
    assert positions == sorted(positions)


def test_tfidf_baseline_runs():
    text = (DATA / "index_funds_talk.txt").read_text(encoding="utf-8")
    result = summarize(text, num_sentences=3, method="tfidf_baseline")
    assert isinstance(result, SummaryResult)
    assert result.method == "tfidf_baseline"
    assert 1 <= len(result.sentences) <= 3


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        summarize("Some text. More text.", method="nope")


def test_empty_text():
    result = summarize("", num_sentences=3)
    assert result.summary == ""
    assert result.sentences == []
