# Stash — Interactive Showcase

An interactive static showcase for **Stash**, an app that solves the "save for later but never
revisit" problem: save a YouTube URL and it extracts the transcript and summarizes it with an LLM —
with auth, metrics, and production-ready DevOps.

🔗 **Live site:** https://andreaisabelmontana.github.io/stash/

## What it does
- **Save a URL** — persist any YouTube link to your library.
- **Extract transcript** — the backend fetches and normalizes the transcript.
- **AI summary** — condensed into skimmable points via Groq's Llama model.
- **Clean architecture** — SOLID + dependency injection (ServiceContainer, Protocols for inversion).
- **Production-ready** — JWT auth, health + Prometheus metrics, tests with coverage gates, Docker, CI/CD on Azure.

**Stack:** FastAPI (Python 3.12) · Groq / Llama · SQLite · JWT · Docker · GitHub Actions · Prometheus · Azure.

## About this repo
An original, hand-built static site (single `index.html`, no framework) presenting the project, with a
scripted interactive demo (save a video → transcript excerpt → AI summary). Built from scratch;
videos, transcripts, and summaries are sample data.
