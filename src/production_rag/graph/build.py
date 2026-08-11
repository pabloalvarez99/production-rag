"""Graph wiring, and the only file a LangGraph version bump should have to touch.

    START -> retrieve -> rerank -> guard -+-> generate -> cite -> finalise -> END
                                          |
                                          +-> END        (refused: no evidence)

That branch is the point of using a graph here. The refusal path is a declared
edge, visible in the topology, rather than an early ``return`` buried inside a
function — which is what ADR-0002 argues for, and what makes "does this system
refuse when it has no evidence?" a question answerable by reading the wiring.

Everything else in this file is binding: each node is closed over its
:class:`~production_rag.graph.nodes.QueryDeps` because LangGraph calls
single-argument callables, and putting live clients into the state instead would
make the state unserialisable and untestable on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from production_rag.graph.nodes import (
    QueryDeps,
    cite_node,
    finalise_node,
    generate_node,
    guard_node,
    rerank_node,
    retrieve_node,
    should_generate,
)
from production_rag.graph.state import (
    CITE_NODE,
    FINALISE_NODE,
    GENERATE_NODE,
    GUARD_NODE,
    RERANK_NODE,
    RETRIEVE_NODE,
    QueryState,
)
from production_rag.observability.context import current_request_id
from production_rag.observability.tracer import guarded_span, span_attributes

_NODES: tuple[tuple[str, Callable[[QueryState, QueryDeps], dict[str, object]]], ...] = (
    (RETRIEVE_NODE, retrieve_node),
    (RERANK_NODE, rerank_node),
    (GUARD_NODE, guard_node),
    (GENERATE_NODE, generate_node),
    (CITE_NODE, cite_node),
    (FINALISE_NODE, finalise_node),
)


class _BoundNode(Protocol):
    """A node with its dependencies already supplied.

    Spelled as a Protocol with a *named* ``state`` parameter rather than a bare
    ``Callable``, because that is the shape LangGraph's ``add_node`` overloads
    accept — a ``Callable`` alias declares positional-only parameters and does
    not match.
    """

    def __call__(self, state: QueryState) -> dict[str, object]: ...


def _bind(
    name: str,
    node: Callable[[QueryState, QueryDeps], dict[str, object]],
    deps: QueryDeps,
) -> _BoundNode:
    """Close *node* over its dependencies, and wrap it in a span.

    Tracing attaches here rather than inside the node functions, for the reason
    ADR-0002 keeps logic out of them: a node is an adapter, and an adapter that
    also opens spans is one more thing to copy correctly the next time a node is
    added. One wrapper, applied to every node, cannot be forgotten — and the
    span name is the node name the graph registered, so a trace and a
    ``timings_ms`` key are guaranteed to be the same string.
    """

    def run(state: QueryState) -> dict[str, object]:
        with guarded_span(deps.tracer, name, **span_attributes(request_id=current_request_id())):
            return node(state, deps)

    run.__name__ = node.__name__
    return run


def build_query_graph(deps: QueryDeps) -> Any:
    """Compile the query graph against *deps*.

    Args:
        deps: The live retriever, model and configuration. One graph per set of
            dependencies: compilation is cheap, and a graph that reaches for a
            client at call time is a graph that cannot be reasoned about.

    Returns:
        The compiled graph. Typed as ``Any`` deliberately — the concrete return
        type is LangGraph's and naming it here would spread the dependency into
        every caller's annotations, which is the coupling ADR-0002 set out to
        avoid.
    """
    graph: StateGraph[QueryState, None, QueryState, QueryState] = StateGraph(QueryState)
    for name, node in _NODES:
        graph.add_node(name, _bind(name, node, deps))

    graph.add_edge(START, RETRIEVE_NODE)
    graph.add_edge(RETRIEVE_NODE, RERANK_NODE)
    graph.add_edge(RERANK_NODE, GUARD_NODE)
    graph.add_conditional_edges(
        GUARD_NODE,
        should_generate,
        {"generate": GENERATE_NODE, "refused": END},
    )
    graph.add_edge(GENERATE_NODE, CITE_NODE)
    graph.add_edge(CITE_NODE, FINALISE_NODE)
    graph.add_edge(FINALISE_NODE, END)
    return graph.compile()


def run_graph(graph: Any, state: QueryState) -> QueryState:
    """Invoke a compiled *graph* and return the final state as a typed object.

    LangGraph hands back a mapping. Re-validating it into
    :class:`~production_rag.graph.state.QueryState` here means exactly one place
    in the codebase touches the framework's return shape.
    """
    final = graph.invoke(state)
    if isinstance(final, QueryState):  # pragma: no cover - depends on the version
        return final
    return QueryState.model_validate(final)
