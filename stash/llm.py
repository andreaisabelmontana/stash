"""Optional LLM summarization hook.

This is entirely optional. The real, default summarizer in Stash is the
extractive engine in :mod:`stash.summarize`. ``llm_summarize`` only does
anything if an API key is present in the environment; otherwise it returns
``None`` and the caller falls back to the extractive summary.

No network call is ever made at import time, and none is made unless a key is
explicitly configured. This keeps the whole project fully runnable offline.
"""

from __future__ import annotations

import os


def llm_available(env: dict | None = None) -> bool:
    """True if an LLM API key is configured in the environment."""
    env = os.environ if env is None else env
    return bool(
        env.get("OPENAI_API_KEY")
        or env.get("GROQ_API_KEY")
        or env.get("ANTHROPIC_API_KEY")
    )


def llm_summarize(
    transcript: str,
    num_sentences: int = 3,
    *,
    model: str | None = None,
    env: dict | None = None,
) -> str | None:
    """Summarize via an LLM, or return ``None`` if no key is configured.

    This is the optional hook. With ``OPENAI_API_KEY`` (or ``GROQ_API_KEY`` via
    its OpenAI-compatible endpoint) set and the ``openai`` package installed, it
    asks the model for a short summary. In every other case it returns ``None``
    so the pipeline transparently uses the extractive engine instead.
    """
    env = os.environ if env is None else env
    if not llm_available(env):
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = env.get("OPENAI_API_KEY") or env.get("GROQ_API_KEY")
    base_url = env.get("OPENAI_BASE_URL")
    if env.get("GROQ_API_KEY") and not env.get("OPENAI_API_KEY"):
        base_url = base_url or "https://api.groq.com/openai/v1"
        model = model or "llama-3.1-8b-instant"

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize transcripts into concise bullet points.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following transcript in at most "
                        f"{num_sentences} short sentences:\n\n{transcript}"
                    ),
                },
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Any failure (network, auth, quota) degrades to the extractive path.
        return None
