"""Answer a tax question from retrieved chunks, with citations.

The prompt is deliberately strict about grounding: this corpus is tax year 2025
and includes provisions (the Schedule 1-A deductions, the $2,200 child tax
credit, the $40,000 SALT cap) that postdate most model training, so an answer
drawn from parametric memory is likely to be confidently wrong. Everything must
come from the supplied context.
"""
import time

import anthropic

from . import config
from .store import cite

# List pricing, USD per million tokens (input, output).
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
}

# The server-side refusal fallback is documented for the Opus and Fable tier.
# Other models go through the plain endpoint.
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}

SYSTEM = f"""You answer questions about U.S. federal income tax for tax year {config.TAX_YEAR}, using only the excerpts supplied to you.

Rules:
- Use only the numbered excerpts. Do not use prior knowledge about tax law, and do not correct the excerpts against what you remember -- the excerpts are the {config.TAX_YEAR} publications and your memory is likely older.
- Cite the excerpt number(s) you used inline, like [2].
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Lead with the direct answer -- the dollar amount, the rate, the yes or no -- then explain briefly.
- Keep it under 120 words."""


def build_prompt(question, chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {cite(c)}\n{c['text']}")
    excerpts = "\n\n".join(blocks)
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def cost_usd(usage, model=None):
    price_in, price_out = PRICES[model or config.GEN_MODEL]
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000


def create_message(client, **kwargs):
    """messages.create, with the refusal fallback on models that support it."""
    if kwargs["model"] in FALLBACK_MODELS:
        return client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
        )
    return client.messages.create(**kwargs)


def answer(question, chunks, client=None):
    """Generate a cited answer. Returns a dict of the answer plus usage metrics."""
    client = client or anthropic.Anthropic()

    t0 = time.perf_counter()
    response = create_message(
        client,
        model=config.GEN_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        output_config={"effort": config.GEN_EFFORT},
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        return {
            "text": f"[model declined to answer: {category}]",
            "refused": True,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": cost_usd(response.usage),
            "generate_ms": elapsed_ms,
        }

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return {
        "text": text,
        "refused": False,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": cost_usd(response.usage),
        "generate_ms": elapsed_ms,
    }
