r"""Server-sent events: the wire format, and nothing else.

One dataclass and one encoder, kept away from the route that uses them, because
the streaming tests assert **exact bytes**. A framing rule that lives inside a
handler is a framing rule that gets adjusted to make a test pass; here it is a
function with its own tests, and the route is a caller like any other.

The format is the EventSource one, deliberately unextended:

    event: delta\\n
    data: {"text":"Hybrid "}\\n
    \\n

Every payload is a JSON object on a single ``data:`` line. That is not a
limitation of SSE — the protocol allows several ``data:`` lines, which a client
joins with newlines — but a single line is the shape ``json.dumps`` already
produces, since it escapes every newline inside strings. Sending one line means
a client's parse step is ``JSON.parse(event.data)`` with no reassembly, and
there is no second code path to get wrong.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SSE_MEDIA_TYPE = "text/event-stream"
"""The content type. Charset is not appended: the spec fixes SSE at UTF-8."""

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # Nginx buffers proxied responses by default, which turns a stream into one
    # delivery at the end — the failure looks exactly like a slow backend and is
    # diagnosed by nobody. The header is inert without a proxy in front.
    "X-Accel-Buffering": "no",
}
"""Headers a streamed response needs to survive the hop to a browser."""

EVENT_META = "meta"
"""Sent first, before any work: carries the request id.

First on purpose. It is the only event guaranteed to be sent, so a client that
gives up, times out or crashes mid-stream still has the id that makes the server
side of its own failure searchable.
"""

EVENT_DELTA = "delta"
"""Provisional model output. Not an answer, and never to be rendered as one.

Whether the pipeline serves this text is decided after generation finishes, by
the guardrails, against citations that do not exist yet. A client renders deltas
as a draft and replaces them wholesale when :data:`EVENT_RESULT` arrives.
"""

EVENT_RESULT = "result"
"""The authoritative outcome: the same body ``POST /v1/query`` returns.

Terminal. A grounded answer and a refusal both arrive here — a refusal is an
outcome, not an error — and this is the first point at which any text on the
wire is safe to present as an answer.
"""

EVENT_ERROR = "error"
"""The run failed. Terminal, and never a refusal.

Carries ``error_type`` so a client branches on a value rather than on wording.
A client that has already rendered deltas must discard them: text that was never
finished was never grounded.
"""


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One named event with a JSON object payload."""

    event: str
    data: Mapping[str, Any]

    def encode(self) -> bytes:
        r"""Render the event as the bytes that go on the wire.

        UTF-8, with non-ASCII characters left as themselves rather than escaped
        into ASCII: the transport already encodes them, so escaping would be a
        second encoding of the same character, and it makes the byte length of
        a delta depend on which alphabet the corpus happens to be written in.
        """
        payload = json.dumps(dict(self.data), ensure_ascii=False, separators=(",", ":"))
        return f"event: {self.event}\ndata: {payload}\n\n".encode()


__all__ = [
    "EVENT_DELTA",
    "EVENT_ERROR",
    "EVENT_META",
    "EVENT_RESULT",
    "SSE_HEADERS",
    "SSE_MEDIA_TYPE",
    "SSEEvent",
]
