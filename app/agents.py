"""The agent team.

Each agent follows the CrewAI pattern of an explicit *role* and *goal*,
and is implemented as a node in the state graph: it reads what it needs
from the shared state, does one job, and returns a partial state update.

Pipeline order:

1. ``DataCollectorAgent``  — loads the merchant and its transactions and
   computes descriptive statistics into ``state["profile"]``.
2. ``RiskAnalystAgent``    — rule-based risk scoring on the profile;
   writes ``risk_score``, ``risk_level`` and ``risk_flags``.
3. ``MarketInsightsAgent`` — benchmarks the merchant against same-category
   peers; appends human-readable ``insights``.
4. ``ReportWriterAgent``   — turns the structured findings into a
   narrative via the pluggable LLM client, persists a ``Report`` row,
   and writes ``state["report"]``.

A fifth node, ``InsufficientDataAgent``, handles merchants with no
transactions (reached through a conditional edge from the collector).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List

from sqlalchemy.orm import Session

from .llm import LLMClient
from .models import Merchant, Report, Transaction


class BaseAgent:
    role: str = "Agent"
    goal: str = ""

    def __call__(self, state: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError


def _daily_volumes(txns: List[Transaction]) -> dict:
    daily: dict = defaultdict(float)
    for t in txns:
        daily[t.occurred_at.date()] += t.amount
    return dict(sorted(daily.items()))


def _trend_pct(daily: dict) -> float:
    """Percent change of daily volume, first half vs second half."""
    values = list(daily.values())
    if len(values) < 4:
        return 0.0
    mid = len(values) // 2
    first = statistics.fmean(values[:mid])
    second = statistics.fmean(values[mid:])
    if first == 0:
        return 0.0
    return (second - first) / first * 100.0


class DataCollectorAgent(BaseAgent):
    role = "Data Collector"
    goal = "Gather a merchant's transactions and compute a factual activity profile."

    def __init__(self, session: Session) -> None:
        self.session = session

    def __call__(self, state: dict) -> dict:
        merchant_id = state["merchant_id"]
        merchant = self.session.get(Merchant, merchant_id)
        if merchant is None:
            raise LookupError(f"merchant {merchant_id} not found")
        txns = (
            self.session.query(Transaction)
            .filter(Transaction.merchant_id == merchant_id)
            .order_by(Transaction.occurred_at)
            .all()
        )
        total_volume = sum(t.amount for t in txns)
        chargebacks = sum(1 for t in txns if t.is_chargeback)
        refunds = sum(1 for t in txns if t.is_refund)
        daily = _daily_volumes(txns)
        amounts = [t.amount for t in txns]
        channel_mix = {
            ch: round(n / len(txns), 3) if txns else 0.0
            for ch, n in Counter(t.channel for t in txns).items()
        }
        profile = {
            "merchant_name": merchant.name,
            "category": merchant.category,
            "region": merchant.region,
            "transaction_count": len(txns),
            "total_volume": round(total_volume, 2),
            "avg_ticket": round(statistics.fmean(amounts), 2) if amounts else 0.0,
            "amount_volatility": (
                round(statistics.pstdev(amounts), 2) if len(amounts) > 1 else 0.0
            ),
            "chargeback_count": chargebacks,
            "chargeback_rate": (chargebacks / len(txns)) if txns else 0.0,
            "refund_rate": (refunds / len(txns)) if txns else 0.0,
            "volume_trend_pct": round(_trend_pct(daily), 2),
            "active_days": len(daily),
            "channel_mix": channel_mix,
        }
        return {"merchant": merchant, "profile": profile, "has_data": len(txns) > 0}


class RiskAnalystAgent(BaseAgent):
    role = "Risk Analyst"
    goal = "Score merchant risk from observed behaviour and explain every flag."

    # Transparent rule weights (points out of 100). Deliberately simple and
    # explainable — a stand-in for a learned model in this reference build.
    CB_WATCH, CB_HIGH = 0.01, 0.025
    REFUND_WATCH = 0.03
    DECLINE_WATCH = -15.0

    def __call__(self, state: dict) -> dict:
        p = state["profile"]
        score = 5.0  # baseline
        flags: List[str] = []

        cb = p["chargeback_rate"]
        if cb >= self.CB_HIGH:
            score += 45
            flags.append(
                f"chargeback rate {cb:.2%} is above the {self.CB_HIGH:.1%} high-risk threshold"
            )
        elif cb >= self.CB_WATCH:
            score += 25
            flags.append(
                f"chargeback rate {cb:.2%} exceeds the {self.CB_WATCH:.1%} monitoring threshold"
            )
        elif cb > 0:
            score += 8

        if p["refund_rate"] >= self.REFUND_WATCH:
            score += 12
            flags.append(f"refund rate {p['refund_rate']:.2%} is elevated")

        if p["volume_trend_pct"] <= self.DECLINE_WATCH:
            score += 18
            flags.append(
                f"transaction volume declined {abs(p['volume_trend_pct']):.1f}% across the window"
            )

        if p["avg_ticket"] > 0 and p["amount_volatility"] / p["avg_ticket"] > 0.9:
            score += 7
            flags.append("transaction amounts are highly volatile relative to the average ticket")

        score = round(min(score, 100.0), 1)
        if score >= 70:
            level = "critical"
        elif score >= 50:
            level = "high"
        elif score >= 30:
            level = "moderate"
        else:
            level = "low"
        if not flags:
            flags.append("no material risk signals detected in the observation window")
        return {"risk_score": score, "risk_level": level, "risk_flags": flags}


class MarketInsightsAgent(BaseAgent):
    role = "Market Insights Analyst"
    goal = "Benchmark the merchant against same-category peers and surface context."

    def __init__(self, session: Session) -> None:
        self.session = session

    def __call__(self, state: dict) -> dict:
        merchant: Merchant = state["merchant"]
        p = state["profile"]
        peers = (
            self.session.query(Merchant)
            .filter(
                Merchant.category == merchant.category,
                Merchant.id != merchant.id,
            )
            .all()
        )
        insights: List[str] = []
        if not peers:
            insights.append("no same-category peers available for benchmarking")
            return {"insights": insights}

        peer_stats = []
        for peer in peers:
            txns = (
                self.session.query(Transaction)
                .filter(Transaction.merchant_id == peer.id)
                .all()
            )
            if not txns:
                continue
            volume = sum(t.amount for t in txns)
            cb_rate = sum(1 for t in txns if t.is_chargeback) / len(txns)
            peer_stats.append((volume, cb_rate))

        if peer_stats:
            volumes = sorted(v for v, _ in peer_stats)
            rank = sum(1 for v in volumes if v <= p["total_volume"])
            percentile = rank / len(volumes) * 100
            insights.append(
                f"volume ranks at the {percentile:.0f}th percentile among "
                f"{len(peer_stats)} same-category peers"
            )
            peer_cb = statistics.fmean(cb for _, cb in peer_stats)
            if peer_cb > 0:
                ratio = p["chargeback_rate"] / peer_cb
                if ratio >= 1.5:
                    insights.append(
                        f"chargeback rate is {ratio:.1f}x the category peer average ({peer_cb:.2%})"
                    )
                elif ratio <= 0.67:
                    insights.append(
                        f"chargeback rate is below the category peer average ({peer_cb:.2%})"
                    )
        if p["volume_trend_pct"] >= 15:
            insights.append("merchant is in a clear growth phase within the window")
        concentrated = max(p["channel_mix"].values(), default=0)
        if concentrated >= 0.8:
            top_channel = max(p["channel_mix"], key=p["channel_mix"].get)
            insights.append(
                f"sales are concentrated in the {top_channel} channel ({concentrated:.0%} of transactions)"
            )
        if not insights:
            insights.append("merchant behaviour is broadly in line with category peers")
        return {"insights": insights}


class ReportWriterAgent(BaseAgent):
    role = "Report Writer"
    goal = "Produce a clear merchant intelligence report and persist it."

    def __init__(self, session: Session, llm: LLMClient) -> None:
        self.session = session
        self.llm = llm

    def _recommendation(self, state: dict) -> str:
        level = state["risk_level"]
        return {
            "critical": "place the account under enhanced monitoring and review recent chargebacks immediately",
            "high": "schedule a risk review and increase chargeback monitoring frequency",
            "moderate": "continue standard monitoring and re-assess at the next cycle",
            "low": "no action required; continue standard monitoring",
        }[level]

    def __call__(self, state: dict) -> dict:
        p = state["profile"]
        recommendation = self._recommendation(state)
        context = {
            "kind": "merchant_report",
            "merchant_name": p["merchant_name"],
            "category": p["category"],
            "region": p["region"],
            "risk_level": state["risk_level"],
            "risk_score": state["risk_score"],
            "transaction_count": p["transaction_count"],
            "total_volume": p["total_volume"],
            "chargeback_rate": p["chargeback_rate"],
            "volume_trend_pct": p["volume_trend_pct"],
            "risk_flags": state["risk_flags"],
            "insights": state.get("insights", []),
            "recommendation": recommendation,
        }
        summary = self.llm.complete(
            prompt=(
                "Write a merchant intelligence report from these findings: "
                f"{json.dumps(context, default=str)}"
            ),
            system="You are a payments risk analyst writing concise merchant reports.",
            context=context,
        )
        findings = {
            "profile": p,
            "risk_flags": state["risk_flags"],
            "insights": state.get("insights", []),
            "recommendation": recommendation,
            "llm_provider": getattr(self.llm, "name", "unknown"),
        }
        report = Report(
            merchant_id=state["merchant_id"],
            risk_score=state["risk_score"],
            risk_level=state["risk_level"],
            summary=summary,
            findings_json=json.dumps(findings, default=str),
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.session.add(report)
        self.session.commit()
        return {"report": report, "report_id": report.id, "summary": summary}


class InsufficientDataAgent(BaseAgent):
    role = "Report Writer"
    goal = "Record an explicit insufficient-data report instead of guessing."

    def __init__(self, session: Session) -> None:
        self.session = session

    def __call__(self, state: dict) -> dict:
        merchant: Merchant = state["merchant"]
        summary = (
            f"Merchant intelligence report for {merchant.name}: no transactions "
            "were found in the observation window, so no risk assessment could "
            "be performed. Recommendation: verify data ingestion for this "
            "account before the next analysis cycle."
        )
        report = Report(
            merchant_id=merchant.id,
            risk_score=0.0,
            risk_level="unknown",
            summary=summary,
            findings_json=json.dumps({"reason": "no_transactions"}),
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.session.add(report)
        self.session.commit()
        return {
            "report": report,
            "report_id": report.id,
            "summary": summary,
            "risk_score": 0.0,
            "risk_level": "unknown",
            "risk_flags": ["no transaction data available"],
            "insights": [],
        }
