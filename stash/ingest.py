"""Transcript ingestion.

Transcripts can come from three sources, tried in this order of preference for
``ingest``:

1. text passed directly,
2. a local file,
3. a real YouTube fetch via the optional ``youtube-transcript-api`` package.

The YouTube path is strictly best-effort: if the package is not installed or
there is no network, it raises ``TranscriptUnavailable`` and the caller can fall
back to a supplied transcript. Nothing here requires the network to import.
"""

from __future__ import annotations

import re
from pathlib import Path

# youtu.be/<id>, youtube.com/watch?v=<id>, /embed/<id>, /shorts/<id>, or a bare id
_ID_PATTERNS = [
    re.compile(r"(?:v=|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$"),
]


class TranscriptUnavailable(RuntimeError):
    """Raised when a transcript could not be fetched from YouTube."""


def video_id(url_or_id: str) -> str | None:
    """Extract an 11-character YouTube video id from a URL or bare id."""
    if not url_or_id:
        return None
    text = url_or_id.strip()
    for pattern in _ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def from_text(text: str) -> str:
    """Normalize a transcript supplied directly as text."""
    return re.sub(r"\s+", " ", text or "").strip()


def from_file(path: str | Path) -> str:
    """Read and normalize a transcript from a local file."""
    return from_text(Path(path).read_text(encoding="utf-8"))


def from_youtube(url_or_id: str, languages: tuple[str, ...] = ("en",)) -> str:
    """Fetch a real transcript from YouTube. Best-effort, may raise.

    Raises ``TranscriptUnavailable`` if ``youtube-transcript-api`` is not
    installed, the id is unparseable, or the fetch fails (e.g. offline).
    """
    vid = video_id(url_or_id)
    if not vid:
        raise TranscriptUnavailable(f"could not parse a video id from {url_or_id!r}")

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # package not installed
        raise TranscriptUnavailable(
            "youtube-transcript-api is not installed; pass a transcript instead"
        ) from exc

    try:
        # Support both the classic classmethod API and the newer instance API.
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            chunks = YouTubeTranscriptApi.get_transcript(vid, languages=list(languages))
        else:  # pragma: no cover - depends on installed version
            fetched = YouTubeTranscriptApi().fetch(vid, languages=list(languages))
            chunks = [{"text": s.text} for s in fetched]
    except Exception as exc:  # network errors, disabled transcripts, etc.
        raise TranscriptUnavailable(str(exc)) from exc

    text = " ".join(chunk["text"] for chunk in chunks)
    return from_text(text)


def ingest(
    *,
    text: str | None = None,
    file: str | Path | None = None,
    url: str | None = None,
) -> str:
    """Resolve a transcript from whichever source is provided.

    Priority: ``text`` > ``file`` > ``url`` (YouTube fetch). The YouTube fetch
    is best-effort and degrades gracefully: if it fails and no other source was
    given, a ``TranscriptUnavailable`` error propagates so the caller can react.
    """
    if text:
        return from_text(text)
    if file:
        return from_file(file)
    if url:
        return from_youtube(url)
    raise ValueError("provide one of: text, file, url")
