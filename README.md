# Stash

Save items, ingest their transcripts, and summarize them **extractively** in
Python. The core engine is real and runs fully offline; the LLM step is
optional.

🔗 **Showcase:** https://andreaisabelmontana.github.io/stash/

## What it does

- **Stores saved items** in SQLite (`url`, `title`, `transcript`, `tags`,
  `summary`, `saved_at`).
- **Ingests transcripts** from text you pass in, from a file, or — best-effort —
  by fetching a real YouTube transcript with `youtube-transcript-api` from a
  video id. The YouTube path degrades gracefully when the package is missing or
  you are offline.
- **Summarizes extractively** with two real algorithms:
  - **TextRank** — builds a graph whose nodes are sentences and whose edge
    weights are the cosine similarity between TF-IDF sentence vectors, then runs
    the PageRank power iteration to score each sentence by centrality. The most
    central sentences (in reading order) become the summary.
  - **TF-IDF baseline** — scores each sentence by the summed TF-IDF weight of
    its terms.
- **Extracts keywords** by ranking unigrams/bigrams by aggregate TF-IDF weight
  (with a transcript-filler stop list), and **auto-tags** items from them.
- **Optional LLM hook** (`llm_summarize`) — only runs if an API key is set in
  the environment; otherwise it returns `None` and the stored summary is the
  extractive one. Nothing here calls the network at import time.

## Install

```bash
pip install -r requirements.txt          # numpy + scikit-learn
# optional extras:
pip install youtube-transcript-api openai
```

## Run the demo

```bash
python demo.py
```

It saves a committed sample transcript, computes the summary/keywords/tags, and
prints them — no network or API key needed.

## Real example

Running the pipeline on the committed `data/transformers_talk.txt`
(`method="textrank"`, `num_sentences=3`) produces:

**TextRank summary**

1. The single most important idea in a transformer is the attention mechanism,
   which lets every token look at every other token and decide what matters.
2. Attention works like a soft dictionary lookup: each token produces a query
   vector, every token produces a key vector, and the dot product between a
   query and a key tells you how relevant that token is.
3. If you remember nothing else from this video, remember that attention is the
   core idea that makes transformers work.

**Keywords:** attention, token, transformer, work, remember, weights,
transformers, different

**Tags:** attention, token, transformer, work, remember

And on `data/index_funds_talk.txt`:

**TextRank summary**

1. An index fund is a single investment that buys a whole basket of stocks
   tracking a market index, instead of trying to pick individual winners.
2. The big advantage of an index fund is that it gives you broad diversification
   and very low fees at the same time.
3. So the takeaway is simple: buy a broad low-cost index fund, contribute
   regularly, and hold for the long term.

**Keywords:** index, fund, broad, market, long, funds, returns, fees

## Library usage

```python
from stash import Store, stash_item, summarize, extract_keywords

# One-shot pipeline: ingest -> summarize -> keyword -> tag -> persist.
with Store("stash.db") as store:
    result = stash_item(
        store,
        text=open("data/index_funds_talk.txt").read(),
        title="Index funds explained simply",
        url="https://youtu.be/index-funds",
        num_sentences=3,
        method="textrank",      # or "tfidf_baseline"
    )
    print(result.item.summary)
    print(result.item.tags)

# Or call the engine directly:
res = summarize(some_text, num_sentences=3, method="textrank")
print(res.summary)
print(extract_keywords(some_text, top_k=8))
```

Fetching a real transcript (best-effort, needs the optional package + network):

```python
from stash import from_youtube, TranscriptUnavailable
try:
    transcript = from_youtube("https://youtu.be/<id>")
except TranscriptUnavailable:
    transcript = "...fall back to a transcript you already have..."
```

### Optional LLM summary

Set an API key and `llm_summarize` will use it; with no key it returns `None`
and the extractive summary is used. The default, tested behavior is fully
extractive.

```bash
export OPENAI_API_KEY=...      # or GROQ_API_KEY for an OpenAI-compatible endpoint
```

## Tests

```bash
pip install pytest scikit-learn numpy
python -m pytest -q
```

Covers TextRank centrality on a constructed hub document, summary-length bounds,
keyword salience, store save/load round-trips, and a fully offline end-to-end
pipeline run on a committed transcript.

## Layout

```
stash/
  summarize.py   TextRank + TF-IDF extractive summarizer, sentence splitting
  keywords.py    TF-IDF keyword extraction + auto-tagging
  ingest.py      transcript ingestion (text / file / best-effort YouTube)
  store.py       SQLite store of saved items
  llm.py         optional LLM hook (no-op without an API key)
  pipeline.py    save-and-summarize pipeline
data/            committed sample transcripts (run offline)
tests/           pytest suite
demo.py          end-to-end demo
```

## License

MIT — see [LICENSE](LICENSE).
