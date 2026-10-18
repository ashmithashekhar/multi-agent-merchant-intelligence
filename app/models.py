"""Persistence layer (SQLAlchemy).

Three tables:

* ``merchants``    — synthetic merchant accounts under analysis.
* ``transactions`` — synthetic payment transactions per merchant.
* ``reports``      — intelligence reports produced by the agent pipeline.

SQLite is the default (zero setup); point ``DATABASE_URL`` at PostgreSQL
(see docker-compose.yml) and the same models are used unchanged.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Iterator, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import settings


def _utcnow() -> datetime:
    """Timezone-aware UTC now, stored naive for SQLite/Postgres parity."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    category: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    region: Mapped[str] = mapped_column(String(60), nullable=False)
    # Behavioural archetype used by the *generator* only (stable,
    # chargeback_spike, volume_decline, growth). Analysts never read it —
    # risk is derived from the transactions themselves.
    archetype: Mapped[str] = mapped_column(String(40), nullable=False, default="stable")
    onboarded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="merchant", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(
        back_populates="merchant", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Merchant id={self.id} name={self.name!r} category={self.category!r}>"


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(
        ForeignKey("merchants.id"), nullable=False, index=True
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    channel: Mapped[str] = mapped_column(String(20), default="online")
    status: Mapped[str] = mapped_column(String(20), default="settled")
    is_chargeback: Mapped[bool] = mapped_column(default=False)
    is_refund: Mapped[bool] = mapped_column(default=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    merchant: Mapped[Merchant] = relationship(back_populates="transactions")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(
        ForeignKey("merchants.id"), nullable=False, index=True
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON-encoded list/dict payloads (kept as text for portability).
    findings_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    merchant: Mapped[Merchant] = relationship(back_populates="reports")


# -- engine / session helpers ----------------------------------------------

_engine = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine(database_url: Optional[str] = None):
    global _engine
    if _engine is None or database_url is not None:
        url = database_url or settings.database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


def init_db(database_url: Optional[str] = None) -> sessionmaker:
    """Create tables (idempotent) and return a session factory."""
    global _SessionFactory
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory


def get_session_factory() -> sessionmaker:
    if _SessionFactory is None:
        return init_db()
    return _SessionFactory


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
