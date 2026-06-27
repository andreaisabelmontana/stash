"""Stash - save items, ingest transcripts, summarize them extractively.

The core engine is real, runnable extractive summarization (TextRank over
TF-IDF sentence vectors, plus a TF-IDF baseline) with keyword-based
auto-tagging. An LLM hook is available but strictly optional.
"""

from __future__ import annotations

from .ingest import TranscriptUnavailable, from_youtube, ingest, video_id
from .keywords import auto_tags, extract_keywords
from .llm import llm_available, llm_summarize
from .pipeline import StashResult, stash_item
from .store import Item, Store
from .summarize import (
    SummaryResult,
    summarize,
    textrank_scores,
    split_sentences,
)

__version__ = "0.1.0"

__all__ = [
    "Store",
    "Item",
    "summarize",
    "SummaryResult",
    "textrank_scores",
    "split_sentences",
    "extract_keywords",
    "auto_tags",
    "ingest",
    "from_youtube",
    "video_id",
    "TranscriptUnavailable",
    "llm_summarize",
    "llm_available",
    "stash_item",
    "StashResult",
]
