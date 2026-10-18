"""FastAPI service for the Multi-Agent Merchant Intelligence Platform.

Run locally:

    uvicorn app.main:app --reload

On startup the database is initialised and, if empty, seeded with the
synthetic merchant dataset so the API is immediately usable.
"""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from .config import settings
from .data import seed_database
from .models import Merchant, Report, get_session_factory, init_db
from .pipeline import run_analysis

app = FastAPI(
    title="Multi-Agent Merchant Intelligence Platform",
    description=(
        "Runs a LangGraph-style multi-agent pipeline (Data Collector, "
        "Risk Analyst, Market Insights, Report Writer) over synthetic "
        "merchant payment data and returns intelligence reports."
    ),
    version="0.1.0",
)


# -- dependencies -----------------------------------------------------------


def get_db() -> Session:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@app.on_event("startup")
def _startup() -> None:
    factory = init_db()
    session = factory()
    try:
        if session.query(Merchant).count() == 0:
            seed_database(
                session,
                merchant_count=settings.seed_merchant_count,
                seed=settings.data_seed,
            )
    finally:
        session.close()


# -- schemas ------------------------------------------------------------------


class MerchantOut(BaseModel):
    id: int
    name: str
    category: str
    region: str
    archetype: str

    model_config = ConfigDict(from_attributes=True)


class ReportOut(BaseModel):
    id: int
    merchant_id: int
    risk_score: float
    risk_level: str
    summary: str
    findings: dict
    created_at: Optional[str] = None


def _report_out(report: Report) -> ReportOut:
    try:
        findings = json.loads(report.findings_json or "{}")
    except json.JSONDecodeError:
        findings = {}
    return ReportOut(
        id=report.id,
        merchant_id=report.merchant_id,
        risk_score=report.risk_score,
        risk_level=report.risk_level,
        summary=report.summary,
        findings=findings,
        created_at=report.created_at.isoformat() if report.created_at else None,
    )


# -- endpoints ------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.get("/merchants", response_model=List[MerchantOut])
def list_merchants(db: Session = Depends(get_db)) -> List[Merchant]:
    return db.query(Merchant).order_by(Merchant.id).all()


@app.get("/merchants/{merchant_id}", response_model=MerchantOut)
def get_merchant(merchant_id: int, db: Session = Depends(get_db)) -> Merchant:
    merchant = db.get(Merchant, merchant_id)
    if merchant is None:
        raise HTTPException(status_code=404, detail="merchant not found")
    return merchant


@app.post("/merchants/{merchant_id}/analyze", response_model=ReportOut)
def analyze_merchant(merchant_id: int, db: Session = Depends(get_db)) -> ReportOut:
    """Run the multi-agent pipeline for a merchant and return its report."""
    if db.get(Merchant, merchant_id) is None:
        raise HTTPException(status_code=404, detail="merchant not found")
    state = run_analysis(merchant_id, db)
    report = db.get(Report, state["report_id"])
    return _report_out(report)


@app.get("/merchants/{merchant_id}/reports", response_model=List[ReportOut])
def list_merchant_reports(
    merchant_id: int, db: Session = Depends(get_db)
) -> List[ReportOut]:
    if db.get(Merchant, merchant_id) is None:
        raise HTTPException(status_code=404, detail="merchant not found")
    reports = (
        db.query(Report)
        .filter(Report.merchant_id == merchant_id)
        .order_by(Report.id.desc())
        .all()
    )
    return [_report_out(r) for r in reports]


@app.get("/reports/{report_id}", response_model=ReportOut)
def get_report(report_id: int, db: Session = Depends(get_db)) -> ReportOut:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="report not found")
    return _report_out(report)
