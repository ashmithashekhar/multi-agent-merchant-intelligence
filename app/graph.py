"""A minimal LangGraph-style state graph framework in pure Python.

This module implements the core ideas of LangGraph — typed shared state,
nodes as functions, directed edges, and conditional routing — without any
external dependency, so agent pipelines can be built, unit-tested, and run
entirely offline.

Concepts
--------
* **State** — a plain ``dict`` that flows through the graph. Each node
  receives the current state and returns a *partial update* dict, which is
  merged into the state (shallow merge; list values under a key ending in
  ``"__append"`` are appended instead of replaced — see ``_merge``).
* **Node** — any callable ``state -> dict``.
* **Edge** — a fixed transition from one node to the next.
* **Conditional edge** — a router callable ``state -> str`` whose return
  value selects the next node via a mapping.
* **END** — the sentinel that terminates execution.

The compiled graph records an execution trace (node names, in order) under
``state["_trace"]`` which makes pipelines easy to debug and test.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Mapping, Optional

END = "__end__"

State = Dict[str, object]
NodeFn = Callable[[State], Dict[str, object]]
RouterFn = Callable[[State], Hashable]


def _merge(state: State, update: Dict[str, object]) -> State:
    """Merge a node's partial update into the shared state.

    Keys ending in ``__append`` extend the list stored under the base key
    instead of replacing it, which is handy for accumulating findings:

        return {"findings__append": ["chargeback rate is elevated"]}
    """
    for key, value in update.items():
        if key.endswith("__append"):
            base = key[: -len("__append")]
            existing = state.get(base)
            if isinstance(existing, list) and isinstance(value, list):
                state[base] = existing + value
            elif isinstance(value, list):
                state[base] = list(value)
            else:
                state[base] = value
        else:
            state[key] = value
    return state


class StateGraph:
    """Builder for a state graph. Call :meth:`compile` to run it."""

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeFn] = {}
        self._edges: Dict[str, str] = {}
        self._conditional: Dict[str, tuple[RouterFn, Mapping[Hashable, str]]] = {}
        self._entry: Optional[str] = None

    # -- construction -----------------------------------------------------

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name == END:
            raise ValueError(f"{END!r} is reserved")
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        mapping: Mapping[Hashable, str],
    ) -> "StateGraph":
        self._conditional[source] = (router, dict(mapping))
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    # -- compilation ------------------------------------------------------

    def compile(self, max_steps: int = 50) -> "CompiledGraph":
        if self._entry is None:
            raise ValueError("no entry point set")
        missing = {self._entry} | set(self._edges.values())
        for _, mapping in self._conditional.values():
            missing |= set(mapping.values())
        missing -= set(self._nodes) | {END}
        if missing:
            raise ValueError(f"edges reference unknown nodes: {sorted(missing)}")
        return CompiledGraph(self, max_steps=max_steps)


class CompiledGraph:
    """An executable graph produced by :meth:`StateGraph.compile`."""

    def __init__(self, graph: StateGraph, max_steps: int = 50) -> None:
        self._graph = graph
        self._max_steps = max_steps

    def invoke(self, initial_state: Optional[State] = None) -> State:
        """Run the graph to END and return the final state."""
        state: State = dict(initial_state or {})
        state["_trace"] = []
        current = self._graph._entry
        steps = 0
        while current != END:
            steps += 1
            if steps > self._max_steps:
                raise RuntimeError(
                    f"graph exceeded max_steps={self._max_steps} "
                    f"(possible cycle at node {current!r})"
                )
            node = self._graph._nodes[current]
            update = node(state) or {}
            _merge(state, update)
            state["_trace"].append(current)  # type: ignore[union-attr]
            current = self._next(current, state)
        return state

    def _next(self, current: str, state: State) -> str:
        graph = self._graph
        if current in graph._conditional:
            router, mapping = graph._conditional[current]
            key = router(state)
            if key not in mapping:
                raise KeyError(
                    f"router for node {current!r} returned {key!r}, "
                    f"which is not in the mapping {sorted(mapping)}"
                )
            return mapping[key]
        if current in graph._edges:
            return graph._edges[current]
        return END
