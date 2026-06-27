"""Keyword extraction and auto-tagging.

Keyword extraction ranks terms by their aggregate TF-IDF weight across the
sentences of a document. Both single words and bigrams are considered, so
salient multi-word phrases ("index funds", "multi-head attention") surface
alongside single tokens. Auto-tagging is a thin slug-ifying wrapper on top.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .summarize import split_sentences

# Spoken-video filler that is rarely the topic of a transcript. Added on top of
# scikit-learn's English stop word list so keywords reflect content, not intros.
_TRANSCRIPT_FILLER = {
    "welcome", "channel", "video", "today", "going", "talk", "thanks",
    "watching", "subscribe", "like", "hey", "everyone", "guys", "hi",
    "let", "want", "really", "actually", "gonna", "okay", "yeah", "right",
    "know", "think", "see", "time", "way", "thing", "things", "lot",
}
_STOP_WORDS = list(ENGLISH_STOP_WORDS | _TRANSCRIPT_FILLER)


def extract_keywords(text: str, top_k: int = 8) -> list[str]:
    """Return the ``top_k`` most salient terms/phrases in ``text``.

    Terms are ranked by their summed TF-IDF weight over the document's
    sentences. Unigrams and bigrams compete in the same ranking; a bigram only
    survives if it is genuinely more salient than its parts.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=_STOP_WORDS,
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"[A-Za-z][A-Za-z'-]+",
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Happens when the text is entirely stop words / punctuation.
        return []

    weights = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(weights)[::-1]

    keywords: list[str] = []
    seen_words: set[str] = set()
    for idx in order:
        term = terms[idx]
        if weights[idx] <= 0:
            break
        # Skip a bigram if both of its words already appear as accepted unigrams,
        # and skip a unigram already covered by an accepted bigram, to keep the
        # list diverse rather than redundant.
        parts = term.split()
        if len(parts) == 2 and all(p in seen_words for p in parts):
            continue
        keywords.append(term)
        seen_words.update(parts)
        if len(keywords) >= top_k:
            break
    return keywords


def _slugify(term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
    return slug


def auto_tags(text: str, max_tags: int = 5) -> list[str]:
    """Derive up to ``max_tags`` tag slugs from a document's keywords."""
    keywords = extract_keywords(text, top_k=max_tags * 2)
    tags: list[str] = []
    for kw in keywords:
        slug = _slugify(kw)
        if slug and slug not in tags:
            tags.append(slug)
        if len(tags) >= max_tags:
            break
    return tags
