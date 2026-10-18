"""Tests for the LangGraph-style state graph framework."""

import pytest

from app.graph import END, StateGraph


def test_linear_graph_merges_state_and_traces():
    graph = StateGraph()
    graph.add_node("a", lambda s: {"x": 1})
    graph.add_node("b", lambda s: {"y": s["x"] + 1})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = graph.compile().invoke({})
    assert result["x"] == 1
    assert result["y"] == 2
    assert result["_trace"] == ["a", "b"]


def test_conditional_routing():
    graph = StateGraph()
    graph.add_node("check", lambda s: {"flag": s.get("flag", False)})
    graph.add_node("yes", lambda s: {"route": "yes"})
    graph.add_node("no", lambda s: {"route": "no"})
    graph.set_entry_point("check")
    graph.add_conditional_edges(
        "check",
        router=lambda s: "y" if s["flag"] else "n",
        mapping={"y": "yes", "n": "no"},
    )
    graph.add_edge("yes", END)
    graph.add_edge("no", END)
    compiled = graph.compile()

    assert compiled.invoke({"flag": True})["route"] == "yes"
    assert compiled.invoke({"flag": False})["route"] == "no"


def test_append_merge_semantics():
    graph = StateGraph()
    graph.add_node("one", lambda s: {"items__append": ["first"]})
    graph.add_node("two", lambda s: {"items__append": ["second"]})
    graph.set_entry_point("one")
    graph.add_edge("one", "two")
    graph.add_edge("two", END)

    result = graph.compile().invoke({})
    assert result["items"] == ["first", "second"]


def test_unknown_edge_target_rejected():
    graph = StateGraph()
    graph.add_node("a", lambda s: {})
    graph.set_entry_point("a")
    graph.add_edge("a", "missing")
    with pytest.raises(ValueError):
        graph.compile()


def test_cycle_guard():
    graph = StateGraph()
    graph.add_node("loop", lambda s: {})
    graph.set_entry_point("loop")
    graph.add_edge("loop", "loop")
    with pytest.raises(RuntimeError):
        graph.compile(max_steps=5).invoke({})
