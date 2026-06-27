"""The save-and-summarize pipeline.

``stash_item`` is the one call that ties everything together: ingest a
transcript, run the extractive summarizer, extract keywords, derive tags, and
persist the result. The LLM is consulted only if a key is configured; the
extractive summary is always computed and is what gets stored by default.
"""

from __future__ import annotations

from .ingest import TranscriptUnavailable, ingest
from .keywords import auto_tags, extract_keywords
from .llm import llm_summarize
from .store import Item, Store
from .summarize import SummaryResult, summarize


class StashResult:
    """Bundle returned by :func:`stash_item`."""

    def __init__(
        self,
        item: Item,
        extractive: SummaryResult,
        keywords: list[str],
        llm_summary: str | None,
        used_llm: bool,
    ) -> None:
        self.item = item
        self.extractive = extractive
        self.keywords = keywords
        self.llm_summary = llm_summary
        self.used_llm = used_llm

    @property
    def summary(self) -> str:
        return self.item.summary or ""


def stash_item(
    store: Store,
    *,
    text: str | None = None,
    file: str | None = None,
    url: str | None = None,
    title: str | None = None,
    num_sentences: int = 3,
    method: str = "textrank",
    use_llm: bool = True,
    max_tags: int = 5,
) -> StashResult:
    """Run the full pipeline and persist the resulting item.

    The transcript is resolved from ``text`` / ``file`` / ``url``. The
    extractive summary is always computed. If ``use_llm`` is set and an API key
    is configured, the LLM hook is also tried; its output, when present,
    replaces the stored summary. Otherwise the extractive summary is stored.
    """
    transcript = ingest(text=text, file=file, url=url)

    extractive = summarize(transcript, num_sentences=num_sentences, method=method)
    keywords = extract_keywords(transcript, top_k=max_tags * 2)
    tags = auto_tags(transcript, max_tags=max_tags)

    llm_summary = llm_summarize(transcript, num_sentences) if use_llm else None
    used_llm = llm_summary is not None
    final_summary = llm_summary if used_llm else extractive.summary

    item = store.save(
        Item(
            url=url,
            title=title,
            transcript=transcript,
            tags=tags,
            summary=final_summary,
        )
    )
    return StashResult(item, extractive, keywords, llm_summary, used_llm)


__all__ = ["stash_item", "StashResult", "TranscriptUnavailable"]
