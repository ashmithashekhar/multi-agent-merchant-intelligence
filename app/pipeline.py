"""Pipeline assembly: wires the agents into a state graph.

Graph topology::

    collect ──(has data)──▶ risk ──▶ insights ──▶ write ──▶ END
       │
       └──(no data)──▶ insufficient_data ──▶ END

Running an analysis is a single ``run_analysis`` call.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from . import agents
from .graph import END, CompiledGraph, StateGraph
from .llm import LLMClient, get_llm


def build_graph(session: Session, llm: Optional[LLMClient] = None) -> CompiledGraph:
    llm = llm or get_llm()
    graph = StateGraph()
    graph.add_node("collect", agents.DataCollectorAgent(session))
    graph.add_node("risk", agents.RiskAnalystAgent())
    graph.add_node("insights", agents.MarketInsightsAgent(session))
    graph.add_node("write", agents.ReportWriterAgent(session, llm))
    graph.add_node("insufficient_data", agents.InsufficientDataAgent(session))

    graph.set_entry_point("collect")
    graph.add_conditional_edges(
        "collect",
        router=lambda state: "ok" if state.get("has_data") else "empty",
        mapping={"ok": "risk", "empty": "insufficient_data"},
    )
    graph.add_edge("risk", "insights")
    graph.add_edge("insights", "write")
    graph.add_edge("write", END)
    graph.add_edge("insufficient_data", END)
    return graph.compile()


def run_analysis(
    merchant_id: int,
    session: Session,
    llm: Optional[LLMClient] = None,
) -> dict:
    """Run the full multi-agent pipeline for one merchant.

    Returns the final graph state, including ``report_id``, ``risk_score``,
    ``risk_level``, ``risk_flags``, ``insights``, ``summary`` and the
    execution ``_trace``.
    """
    graph = build_graph(session, llm=llm)
    return graph.invoke({"merchant_id": merchant_id})
