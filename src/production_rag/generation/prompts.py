"""Prompt assembly, and the one place that owns the context format.

    RetrievalHit[] -> numbered context blocks -> [system, user] messages

The numbering is load-bearing. A citation in this project is an ordinal into the
list of chunks that were actually put in the prompt — ``[2]`` means "the second
block above", not "the second retrieved chunk" and not a document id the model
invented. That only holds if exactly one module decides what the blocks look
like and how they are numbered, so this module renders them *and* parses them
back (:func:`parse_context_blocks`). :class:`~production_rag.generation.llm.FakeLLM`
reads its answer out of the same text the real model is shown, rather than being
handed the hits through a side channel that could drift from the prompt.

Two ceilings apply, and they are not the same thing:

* ``prompt.max_chunks_in_prompt`` — how many blocks to include at all.
* ``generation.max_context_tokens`` — a budget on their combined size.

Chunks are dropped from the *tail* when either bites, so retrieval order is also
truncation order: the passage the retriever ranked last is the one that goes.

The token count here is an estimate (``~4 characters per token``), not a
tokeniser call. It is a budget guard, not an accounting figure — being a few
percent off costs a little unused headroom, while importing a tokeniser to be
exact would put a model-specific dependency in the prompt path. The estimate is
deliberately reported next to the prompt so nothing downstream mistakes it for a
billed number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from production_rag.config_loader import GenerationConfig
from production_rag.retrieval.hybrid import RetrievalHit

_log = structlog.get_logger(__name__)

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"  # noqa: S105 - a prompt sentinel, not a credential
"""What the model must say when the context does not support an answer.

A sentinel rather than prose, because "I don't know" has a thousand spellings and
a refusal that has to be detected by fuzzy matching is a refusal that will
sometimes be served as an answer.
"""

CHARS_PER_TOKEN = 4
"""Rough characters-per-token ratio used to size the context budget."""

DEFAULT_SYSTEM_PROMPT = f"""\
You answer questions using ONLY the numbered context blocks provided by the user.

Rules:
1. Use only what the context blocks say. Do not use prior knowledge, and do not
   fill gaps with what is usually true.
2. Cite every claim with the bracketed number of the block it came from, e.g.
   [1] or [2]. Put the marker at the end of the sentence it supports. A sentence
   may carry more than one marker.
3. Never cite a number that is not in the context.
4. If the context does not contain enough to answer, reply with exactly
   {ABSTAIN_TOKEN} and nothing else. A wrong answer is worse than no answer.
5. Be concise and factual. Do not restate the question, and do not add a
   preamble, a summary of the sources, or advice the context does not support.
"""
"""The built-in system prompt, used when no prompt file is on disk.

Kept in code as well as in ``configs/prompts/system.md`` so the library answers
correctly from a fresh clone, a wheel, or a container that did not ship the
configs directory — a missing file degrades the prompt, it must not break it.
"""

_BLOCK_HEADER = re.compile(r"^\[(\d+)\]\s*(.*)$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

CONTEXT_HEADER = "Context blocks:"
QUESTION_HEADER = "Question:"


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One numbered chunk as it appears in the prompt.

    ``marker`` is the number the model is asked to cite, 1-based over the blocks
    that survived truncation — which is why a citation can be resolved without
    trusting anything the model wrote except an integer.
    """

    marker: int
    hit: RetrievalHit

    def render(self, *, include_heading_path: bool = True) -> str:
        """Render the block: one metadata header line, then the chunk text."""
        parts = [f"source={self.hit.source_path or 'unknown'}"]
        if self.hit.title:
            parts.append(f"title={self.hit.title}")
        if include_heading_path and self.hit.heading_path:
            parts.append(f"heading={self.hit.heading_path}")
        return f"[{self.marker}] {' | '.join(parts)}\n{self.hit.text}"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat completion request."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        """Provider-shaped mapping, as the OpenAI chat API expects it."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """The messages to send, plus what went into them.

    ``blocks`` is the authority for citation mapping downstream: it records
    which hits were actually shown and under which number, so a marker resolves
    against the prompt rather than against the retrieval result. Those two lists
    differ exactly when truncation happened, and that is precisely the case
    where trusting the wrong one produces a citation pointing at a passage the
    model never saw.
    """

    messages: tuple[ChatMessage, ...]
    blocks: tuple[ContextBlock, ...]
    dropped_hits: int = 0
    estimated_context_tokens: int = 0

    @property
    def hits_used(self) -> int:
        """How many retrieved chunks reached the model."""
        return len(self.blocks)

    def as_dicts(self) -> list[dict[str, str]]:
        """The messages in provider shape."""
        return [message.as_dict() for message in self.messages]


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text*.

    Deliberately arithmetic, not a tokeniser call: this sizes a budget, and the
    cost of being a few percent conservative is a little unused headroom.
    """
    return len(text) // CHARS_PER_TOKEN + 1


def load_system_prompt(path: str | Path | None = None) -> str:
    """Read the system prompt from *path*, falling back to the built-in one.

    Args:
        path: Prompt file, usually ``generation.prompt.system_path``. ``None``
            skips the lookup entirely.

    Returns:
        The file's contents, or :data:`DEFAULT_SYSTEM_PROMPT` when the file is
        absent or empty. A missing prompt file is a degraded prompt, not a
        broken service: the library has to work from a wheel that never shipped
        ``configs/``.

    Note:
        HTML comments are stripped. The prompt file is Markdown a human edits,
        and a note to the next editor is not an instruction to the model —
        shipping one would spend tokens telling the model about its own
        configuration.
    """
    if path is None:
        return DEFAULT_SYSTEM_PROMPT
    candidate = Path(path)
    if not candidate.is_file():
        _log.info("system_prompt_absent", path=str(candidate), using="builtin")
        return DEFAULT_SYSTEM_PROMPT
    text = _HTML_COMMENT.sub("", candidate.read_text(encoding="utf-8")).strip()
    if not text:
        _log.warning("system_prompt_empty", path=str(candidate), using="builtin")
        return DEFAULT_SYSTEM_PROMPT
    return text


def select_blocks(
    hits: Sequence[RetrievalHit],
    *,
    max_chunks: int = 8,
    max_context_tokens: int = 6000,
) -> tuple[tuple[ContextBlock, ...], int]:
    """Number the hits that fit, and report how many did not.

    Args:
        hits: Retrieved chunks, best first.
        max_chunks: Hard ceiling on blocks, whatever the budget allows.
        max_context_tokens: Estimated token budget for the rendered blocks.

    Returns:
        ``(blocks, dropped)``. The first block is always included even when it
        alone exceeds the budget — returning nothing would turn an oversized
        chunk into a refusal, which reads to a user as "the system has no
        information about this" when the truth is "the system found it and threw
        it away".
    """
    blocks: list[ContextBlock] = []
    used = 0
    dropped = 0
    for hit in hits:
        if len(blocks) >= max_chunks:
            dropped += 1
            continue
        block = ContextBlock(marker=len(blocks) + 1, hit=hit)
        cost = estimate_tokens(block.render())
        if blocks and used + cost > max_context_tokens:
            dropped += 1
            continue
        blocks.append(block)
        used += cost
    if dropped:
        _log.info("context_truncated", kept=len(blocks), dropped=dropped, tokens=used)
    return tuple(blocks), dropped


def build_prompt(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    config: GenerationConfig | None = None,
    system_prompt: str | None = None,
) -> RenderedPrompt:
    """Assemble the chat messages for *query* over *hits*.

    Args:
        query: The user's question, verbatim.
        hits: Retrieved chunks, best first. May be empty — the caller decides
            whether that is a refusal; this function does not silently invent a
            different prompt for it.
        config: The ``generation`` block; documented defaults when omitted.
        system_prompt: Overrides both the configured prompt file and the
            built-in default. The seam tests use it, and so would an experiment
            comparing two system prompts.

    Returns:
        The rendered prompt, carrying the numbered blocks it used.
    """
    settings = config or GenerationConfig()
    blocks, dropped = select_blocks(
        hits,
        max_chunks=settings.prompt.max_chunks_in_prompt,
        max_context_tokens=settings.max_context_tokens,
    )
    rendered = [
        block.render(include_heading_path=settings.prompt.include_heading_path) for block in blocks
    ]
    context = "\n\n".join(rendered) if rendered else "(no context blocks were retrieved)"
    user = f"{CONTEXT_HEADER}\n\n{context}\n\n{QUESTION_HEADER} {query.strip()}"
    system = (
        system_prompt
        if system_prompt is not None
        else load_system_prompt(settings.prompt.system_path)
    )
    return RenderedPrompt(
        messages=(
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ),
        blocks=blocks,
        dropped_hits=dropped,
        estimated_context_tokens=sum(estimate_tokens(text) for text in rendered),
    )


def parse_context_blocks(text: str) -> list[tuple[int, str]]:
    """Recover ``(marker, body)`` pairs from a rendered user message.

    The inverse of :meth:`ContextBlock.render`, and the reason the two live in
    one module: :class:`~production_rag.generation.llm.FakeLLM` answers from the
    prompt it was actually given, so the offline path exercises the real prompt
    format instead of a parallel one that can drift from it.

    A body is everything after the header line up to the next header, with the
    metadata line excluded — the fake should quote the passage, not the file
    path. Parsing stops at the question line, so the user's own words can never
    end up inside the last block and be quoted back as if they were evidence.
    """
    blocks: list[tuple[int, str]] = []
    marker: int | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith(QUESTION_HEADER):
            break
        header = _BLOCK_HEADER.match(line)
        if header:
            if marker is not None:
                blocks.append((marker, "\n".join(body).strip()))
            marker = int(header.group(1))
            body = []
            continue
        if marker is not None:
            body.append(line)
    if marker is not None:
        blocks.append((marker, "\n".join(body).strip()))
    return [(number, content) for number, content in blocks if content]
