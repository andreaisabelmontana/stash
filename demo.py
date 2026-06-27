"""Run the full Stash pipeline offline on a committed sample transcript.

    python demo.py

Saves a sample transcript, computes the extractive summary, extracts keywords,
derives tags, and prints them. No network and no API key required - the LLM hook
is optional and is skipped unless a key is configured.
"""

from __future__ import annotations

from pathlib import Path

from stash import Store, extract_keywords, stash_item, summarize
from stash.llm import llm_available

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    transcript = (DATA / "transformers_talk.txt").read_text(encoding="utf-8")

    with Store(":memory:") as store:
        result = stash_item(
            store,
            text=transcript,
            title="How transformers actually work",
            url="https://youtu.be/transformers-talk",
            num_sentences=3,
            method="textrank",
        )

    print("=" * 70)
    print("STASH DEMO - save & summarize (fully offline, extractive engine)")
    print("=" * 70)
    print(f"\nSaved item #{result.item.id}: {result.item.title}")
    print(f"URL:      {result.item.url}")
    print(f"Saved at: {result.item.saved_at}")
    print(f"LLM used: {result.used_llm}  (LLM key configured: {llm_available()})")

    print("\n--- TextRank summary (top 3 central sentences) ---")
    for i, (sent, score) in enumerate(
        zip(result.extractive.sentences, result.extractive.scores), 1
    ):
        print(f"  {i}. ({score:.4f}) {sent}")

    print("\n--- Keywords (TF-IDF over sentences) ---")
    print("  " + ", ".join(result.keywords))

    print("\n--- Auto tags ---")
    print("  " + ", ".join(result.item.tags))

    # Show the TF-IDF baseline for contrast.
    baseline = summarize(transcript, num_sentences=3, method="tfidf_baseline")
    print("\n--- TF-IDF baseline summary (for comparison) ---")
    for i, sent in enumerate(baseline.sentences, 1):
        print(f"  {i}. {sent}")


if __name__ == "__main__":
    main()
