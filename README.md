# Multi-Agent Merchant Intelligence Platform

A multi-agent system that analyses merchant payment behaviour and produces
structured intelligence reports — risk scores, flags, peer benchmarks, and a
written narrative — through a pipeline of specialised agents orchestrated as
a **LangGraph-style state graph**.

The project is a self-contained reference implementation: it ships with a
synthetic merchant dataset generator, runs **entirely offline with no API
keys**, and persists everything to SQLite by default (PostgreSQL supported
via `docker-compose.yml`).

> **Data note:** every merchant and transaction in this project is synthetic,
> generated deterministically from a seed. No real or employer data is used.

## Agent architecture

Agents follow the CrewAI pattern — each has an explicit *role* and *goal* —
and communicate through shared graph state, in the style of LangGraph nodes:

```
                 ┌───────────────┐
                 │ Data Collector│  loads transactions, computes the
                 │               │  merchant's activity profile
                 └──────┬────────┘
                        │  conditional edge
              ┌─────────┴──────────┐
          has data             no data
              │                    │
              ▼                    ▼
     ┌────────────────┐   ┌────────────────────┐
     │  Risk Analyst  │   │ Insufficient-Data  │
     │ explainable    │   │ report (no guessing)│
     │ rule scoring   │   └────────────────────┘
     └──────┬─────────┘
            ▼
     ┌──────────────────┐
     │ Market Insights  │  benchmarks vs same-category peers
     └──────┬───────────┘
            ▼
     ┌──────────────────┐
     │  Report Writer   │  narrative via pluggable LLM client,
     │                  │  persists the report
     └──────────────────┘
```

| Agent | Responsibility |
|---|---|
| **Data Collector** | Loads the merchant + transactions; computes volume, average ticket, chargeback/refund rates, volume trend, channel mix |
| **Risk Analyst** | Transparent, explainable rule-based scoring (0–100) with a written reason for every point |
| **Market Insights** | Benchmarks the merchant against same-category peers (volume percentile, chargeback ratio, growth phase, channel concentration) |
| **Report Writer** | Renders findings into a narrative via the LLM interface and stores a `Report` row |

The graph framework itself (`app/graph.py`) is a small pure-Python
implementation of the LangGraph model — nodes, edges, conditional routing,
state merging, execution tracing — with no external dependencies, so it can
be unit-tested and reused independently of any LLM provider.

## Features

- **Offline-first:** deterministic LLM fallback renders reports from
  structured findings; identical input → identical output (test-friendly)
- **Pluggable LLM:** set `MERCHANT_LLM_PROVIDER=openai` + `OPENAI_API_KEY`
  to swap in a real model behind the same interface
- **Synthetic data with ground-truth behaviours:** merchants are generated
  with stable / growth / volume-decline / chargeback-spike archetypes; the
  agents must *detect* the behaviour from transactions, never from the label
- **Explainable risk scoring:** every score decomposes into named flags
- **Persistence:** SQLAlchemy models for merchants, transactions, reports;
  SQLite default, PostgreSQL via one environment variable
- **REST API:** FastAPI service to list merchants, run an analysis, and
  fetch generated reports

## Tech stack

Python 3.12 · FastAPI · SQLAlchemy 2 · Pydantic 2 · pytest ·
LangGraph-style state graph (implemented in-repo) · CrewAI-style agent roles ·
OpenAI Agents SDK concepts (agent handoffs via state, tool-like node functions) ·
SQLite / PostgreSQL · Docker Compose (optional, Postgres only)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No other configuration is required for the default offline/SQLite mode.

### Optional: PostgreSQL

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg://merchant:merchant@localhost:5432/merchant_intel
pip install "psycopg[binary]"
```

### Optional: real LLM

```bash
export MERCHANT_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
pip install openai
```

## Usage

### Run the API

```bash
uvicorn app.main:app --reload
```

On first startup the database is created and seeded with synthetic
merchants (24 by default; override with `MERCHANT_SEED_COUNT`).

### Try it

```bash
curl http://localhost:8000/merchants
curl -X POST http://localhost:8000/merchants/4/analyze
curl http://localhost:8000/merchants/4/reports
```

Example report produced by the offline pipeline for a synthetic merchant
with a late-window chargeback spike (generated data, illustrative only):

```
Merchant intelligence report for Vertex Essentials (digital_goods, US-South).
Overall risk is assessed as moderate (score 30.0/100).
Activity summary: 5294 transactions totalling $207,510.19; a chargeback rate
of 2.44%; volume declined by 3.0% across the observation window.
Risk flag: chargeback rate 2.44% exceeds the 1.0% monitoring threshold.
Recommendation: continue standard monitoring and re-assess at the next cycle.
```

### Run the pipeline without the API

```bash
python -m scripts.smoke_run --merchants 4
```

Seeds a throwaway database, runs the full agent pipeline for several
merchants, and prints each report with its agent execution trace.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status and active LLM provider |
| GET | `/merchants` | List all merchants |
| GET | `/merchants/{id}` | Get one merchant |
| POST | `/merchants/{id}/analyze` | Run the multi-agent pipeline; returns the new report |
| GET | `/merchants/{id}/reports` | All reports for a merchant, newest first |
| GET | `/reports/{id}` | Fetch a single report |

Interactive docs are available at `/docs` when the server is running.

## Project structure

```
app/
  graph.py     LangGraph-style state graph framework (nodes/edges/routing)
  llm.py       Pluggable LLM client + deterministic offline fallback
  agents.py    Data Collector, Risk Analyst, Market Insights, Report Writer
  pipeline.py  Graph assembly and run_analysis()
  data.py      Synthetic merchant/transaction generator + DB seeding
  models.py    SQLAlchemy models and session management
  main.py      FastAPI application
  config.py    Environment-driven settings
scripts/
  smoke_run.py End-to-end offline demo run
tests/
  test_graph.py     Graph framework unit tests
  test_data.py      Generator determinism and archetype behaviour tests
  test_pipeline.py  End-to-end pipeline tests (incl. insufficient-data path)
  test_api.py       FastAPI endpoint tests
docker-compose.yml  Optional PostgreSQL
```

## Tests

```bash
pytest
```

The suite covers the graph framework, generator determinism and archetype
behaviour, the full pipeline (including the chargeback-spike and
no-transaction paths), and the HTTP API.

## Design notes & limitations

- Risk scoring is **rule-based by design** in this reference build — the
  weights are visible in `RiskAnalystAgent` so every score is auditable.
  A production system would calibrate or learn these from labelled outcomes.
- The offline LLM renders narratives from templates and structured context;
  it does not generate free-form text. The OpenAI backend can be swapped in
  behind the same interface without touching agent code.
- The graph framework intentionally implements only the subset of LangGraph
  semantics this pipeline needs (sequential execution, conditional edges,
  state merge, step guard). It is not a drop-in LangGraph replacement.
- Synthetic archetypes are simplifications of real merchant behaviour and
  are used only to exercise the detection logic.

## License

MIT
