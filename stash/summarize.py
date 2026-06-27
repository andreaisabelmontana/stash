"""Extractive summarization engine.

Two real algorithms, both implemented here:

* ``textrank`` - graph-based sentence centrality. We build a similarity graph
  whose nodes are sentences and whose edge weights are the cosine similarity
  between TF-IDF sentence vectors, then run power iteration (the PageRank
  recurrence) to score each sentence by how central it is to the document.
* ``tfidf_baseline`` - a simpler baseline that scores each sentence by the sum
  of the TF-IDF weights of the terms it contains.

Both return the highest-scoring sentences in their *original reading order*, so
the summary still reads top to bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A sentence splitter that is deliberately conservative: it breaks on ., ! or ?
# followed by whitespace, but does not break common abbreviations apart. Good
# enough for transcripts, which are the input we care about.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "no", "approx", "dept", "inc", "ltd", "co",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into a list of trimmed, non-empty sentences."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    raw = _SENT_SPLIT.split(text)
    sentences: list[str] = []
    buf = ""
    for piece in raw:
        candidate = (buf + " " + piece).strip() if buf else piece
        # Re-attach if the previous chunk ended in a known abbreviation.
        last_word = _WORD.findall(buf.lower())[-1] if buf else ""
        if buf and last_word in _ABBREV:
            buf = candidate
            continue
        if buf:
            sentences.append(buf)
        buf = piece
    if buf:
        sentences.append(buf)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class SummaryResult:
    """Output of a summarization run."""

    summary: str               # the selected sentences joined into one string
    sentences: list[str]       # the selected sentences, in reading order
    scores: list[float]        # centrality / importance score per selected sentence
    method: str                # "textrank" or "tfidf_baseline"


def _tfidf_matrix(sentences: list[str]):
    """Build a TF-IDF matrix (one row per sentence). Returns (matrix, vectorizer)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"[A-Za-z][A-Za-z'-]+",
    )
    matrix = vectorizer.fit_transform(sentences)
    return matrix, vectorizer


def _power_iteration(
    transition: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 200,
    tol: float = 1.0e-8,
) -> np.ndarray:
    """Run the PageRank power iteration on a column-stochastic ``transition``.

    ``r_{k+1} = (1 - d) / N + d * M @ r_k`` until the L1 change drops below
    ``tol``. Returns a probability vector that sums to 1.
    """
    n = transition.shape[0]
    rank = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - damping) / n)
    for _ in range(max_iter):
        nxt = teleport + damping * transition.dot(rank)
        if np.abs(nxt - rank).sum() < tol:
            rank = nxt
            break
        rank = nxt
    total = rank.sum()
    return rank / total if total else rank


def textrank_scores(sentences: list[str]) -> np.ndarray:
    """Return a TextRank centrality score for every sentence."""
    if not sentences:
        return np.array([])
    if len(sentences) == 1:
        return np.array([1.0])

    matrix, _ = _tfidf_matrix(sentences)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)          # no self-loops

    col_sums = sim.sum(axis=0)
    # Dangling sentences (no similarity to anything) get a uniform column so the
    # walk does not lose probability mass.
    transition = np.zeros_like(sim)
    n = sim.shape[0]
    for j in range(n):
        if col_sums[j] > 0:
            transition[:, j] = sim[:, j] / col_sums[j]
        else:
            transition[:, j] = 1.0 / n

    return _power_iteration(transition)


def _tfidf_sentence_scores(sentences: list[str]) -> np.ndarray:
    """Score each sentence by the sum of its TF-IDF term weights."""
    if not sentences:
        return np.array([])
    matrix, _ = _tfidf_matrix(sentences)
    return np.asarray(matrix.sum(axis=1)).ravel()


def _select(sentences: list[str], scores: np.ndarray, num_sentences: int):
    """Pick the top ``num_sentences`` by score, returned in reading order."""
    n = len(sentences)
    k = max(1, min(num_sentences, n))
    # argsort descending; take top k indices, then sort those by position.
    top = np.argsort(scores)[::-1][:k]
    ordered = sorted(top.tolist())
    chosen = [sentences[i] for i in ordered]
    chosen_scores = [float(scores[i]) for i in ordered]
    return chosen, chosen_scores


def summarize(
    text: str,
    num_sentences: int = 3,
    method: str = "textrank",
) -> SummaryResult:
    """Summarize ``text`` extractively.

    Parameters
    ----------
    text:
        The document (e.g. a video transcript).
    num_sentences:
        Upper bound on how many sentences the summary may contain. The result
        is guaranteed to hold at most this many sentences.
    method:
        ``"textrank"`` (default) or ``"tfidf_baseline"``.
    """
    sentences = split_sentences(text)
    if not sentences:
        return SummaryResult(summary="", sentences=[], scores=[], method=method)

    if method == "textrank":
        scores = textrank_scores(sentences)
    elif method == "tfidf_baseline":
        scores = _tfidf_sentence_scores(sentences)
    else:
        raise ValueError(f"unknown method: {method!r}")

    chosen, chosen_scores = _select(sentences, scores, num_sentences)
    return SummaryResult(
        summary=" ".join(chosen),
        sentences=chosen,
        scores=chosen_scores,
        method=method,
    )
