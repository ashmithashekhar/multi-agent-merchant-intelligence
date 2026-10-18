"""End-to-end smoke run: seed a throwaway SQLite DB and analyse merchants.

Usage:

    python -m scripts.smoke_run [--merchants 4]

Runs the full agent pipeline offline (no API keys) and prints the
resulting reports, so you can see the system work in under a minute.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data import seed_database  # noqa: E402
from app.models import init_db  # noqa: E402
from app.pipeline import run_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merchants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    factory = init_db(f"sqlite:///{tmp.name}")
    session = factory()
    try:
        merchants = seed_database(session, merchant_count=24, seed=args.seed)
        print(f"Seeded {len(merchants)} synthetic merchants into {tmp.name}\n")
        for merchant in merchants[: args.merchants]:
            state = run_analysis(merchant.id, session)
            print("=" * 72)
            print(f"Merchant #{merchant.id}: {merchant.name} [{merchant.category}]")
            print(f"Agent trace : {' -> '.join(state['_trace'])}")
            print(f"Risk        : {state['risk_level']} ({state['risk_score']}/100)")
            print(f"Report ID   : {state['report_id']}")
            print("-" * 72)
            print(state["summary"])
            print()
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
