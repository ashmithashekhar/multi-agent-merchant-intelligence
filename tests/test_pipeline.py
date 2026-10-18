"""End-to-end pipeline tests on a throwaway SQLite database."""

import pytest

from app.data import seed_database
from app.models import Merchant, Report, init_db
from app.pipeline import run_analysis


@pytest.fixture()
def session(tmp_path):
    factory = init_db(f"sqlite:///{tmp_path}/test.db")
    sess = factory()
    seed_database(sess, merchant_count=12, seed=42)
    yield sess
    sess.close()


def _merchant_by_archetype(session, archetype):
    return (
        session.query(Merchant)
        .filter(Merchant.archetype == archetype)
        .first()
    )


def test_full_pipeline_produces_and_persists_report(session):
    merchant = _merchant_by_archetype(session, "stable")
    state = run_analysis(merchant.id, session)

    assert state["_trace"] == ["collect", "risk", "insights", "write"]
    assert state["report_id"] is not None
    assert 0 <= state["risk_score"] <= 100
    assert state["risk_level"] in {"low", "moderate", "high", "critical"}
    assert state["summary"]  # non-empty narrative

    persisted = session.get(Report, state["report_id"])
    assert persisted is not None
    assert persisted.merchant_id == merchant.id
    assert persisted.summary == state["summary"]


def test_chargeback_spike_scores_higher_than_stable(session):
    spike = _merchant_by_archetype(session, "chargeback_spike")
    stable = _merchant_by_archetype(session, "stable")

    spike_state = run_analysis(spike.id, session)
    stable_state = run_analysis(stable.id, session)

    assert spike_state["risk_score"] > stable_state["risk_score"]
    assert spike_state["profile"]["chargeback_rate"] > 0.015
    assert any("chargeback" in flag for flag in spike_state["risk_flags"])


def test_merchant_without_transactions_gets_insufficient_data_report(session):
    empty = Merchant(
        name="Empty Test Merchant",
        category="retail",
        region="US-East",
        archetype="stable",
    )
    session.add(empty)
    session.commit()

    state = run_analysis(empty.id, session)
    assert state["_trace"] == ["collect", "insufficient_data"]
    assert state["risk_level"] == "unknown"
    assert "no transactions" in state["summary"]


def test_unknown_merchant_raises(session):
    with pytest.raises(LookupError):
        run_analysis(999_999, session)
