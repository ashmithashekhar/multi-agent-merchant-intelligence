"""Synthetic merchant dataset generator.

All data in this project is fabricated for demonstration and testing.
Merchants are generated with one of four behavioural *archetypes* that
shape their transaction streams:

* ``stable``            — steady volume, low chargebacks.
* ``growth``            — steadily rising volume.
* ``volume_decline``    — steadily falling volume.
* ``chargeback_spike``  — chargeback rate jumps in the second half of the
                          observation window.

Generation is fully deterministic for a given seed, which keeps tests
reproducible. Analysts downstream never look at the archetype label; they
infer behaviour from the transactions, exactly as they would with real
data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

CATEGORIES = [
    "retail",
    "food_and_beverage",
    "travel",
    "digital_goods",
    "healthcare",
    "professional_services",
    "entertainment",
    "home_goods",
]

REGIONS = ["US-Central", "US-East", "US-West", "US-South", "EU-West", "APAC"]

ARCHETYPES = ["stable", "growth", "volume_decline", "chargeback_spike"]

_NAME_PARTS_A = [
    "Northwind", "Bluepeak", "Cedar", "Harbor", "Summit", "Willow",
    "Vertex", "Meadow", "Ironwood", "Lakeside", "Golden", "Redwood",
    "Silverline", "Oakmont", "Brightpath", "Stonebridge",
]
_NAME_PARTS_B = [
    "Trading Co", "Goods", "Market", "Supply", "Retail Group", "Stores",
    "Commerce", "Outfitters", "Essentials", "Direct",
]

# Typical average ticket (USD) and daily transaction counts per category.
_CATEGORY_PROFILE = {
    "retail": (65.0, 55),
    "food_and_beverage": (28.0, 120),
    "travel": (340.0, 22),
    "digital_goods": (45.0, 80),
    "healthcare": (120.0, 30),
    "professional_services": (210.0, 14),
    "entertainment": (55.0, 65),
    "home_goods": (95.0, 40),
}

CHANNELS = ["online", "in_store", "mobile"]


@dataclass
class SyntheticMerchant:
    name: str
    category: str
    region: str
    archetype: str
    avg_ticket: float
    daily_txns: int


@dataclass
class SyntheticTransaction:
    amount: float
    channel: str
    status: str
    is_chargeback: bool
    is_refund: bool
    occurred_at: datetime
    merchant_index: int = field(default=0)


def generate_merchants(
    count: int = 12, seed: int = 42
) -> List[SyntheticMerchant]:
    """Return a deterministic list of synthetic merchant profiles."""
    rng = random.Random(seed)
    merchants: List[SyntheticMerchant] = []
    used_names: set[str] = set()
    for i in range(count):
        while True:
            name = f"{rng.choice(_NAME_PARTS_A)} {rng.choice(_NAME_PARTS_B)}"
            if name not in used_names:
                used_names.add(name)
                break
        category = CATEGORIES[i % len(CATEGORIES)]
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        base_ticket, base_daily = _CATEGORY_PROFILE[category]
        merchants.append(
            SyntheticMerchant(
                name=name,
                category=category,
                region=rng.choice(REGIONS),
                archetype=archetype,
                avg_ticket=round(base_ticket * rng.uniform(0.75, 1.3), 2),
                daily_txns=max(3, int(base_daily * rng.uniform(0.6, 1.4))),
            )
        )
    return merchants


def generate_transactions(
    merchant: SyntheticMerchant,
    merchant_index: int = 0,
    days: int = 60,
    seed: int = 42,
    end_date: Optional[datetime] = None,
) -> List[SyntheticTransaction]:
    """Generate a deterministic transaction stream for one merchant."""
    rng = random.Random(seed * 100003 + merchant_index * 7919)
    end = end_date or datetime(2026, 9, 30, 23, 59)
    txns: List[SyntheticTransaction] = []
    for day_offset in range(days):
        day = end - timedelta(days=days - 1 - day_offset)
        progress = day_offset / max(days - 1, 1)  # 0.0 -> 1.0 over window

        # Volume multiplier driven by the archetype.
        if merchant.archetype == "growth":
            volume_mult = 0.7 + 0.7 * progress
        elif merchant.archetype == "volume_decline":
            volume_mult = 1.3 - 0.7 * progress
        else:
            volume_mult = 1.0
        # Mild weekly seasonality.
        volume_mult *= 1.15 if day.weekday() >= 5 else 1.0

        count = max(1, int(merchant.daily_txns * volume_mult * rng.uniform(0.8, 1.2)))

        # Chargeback probability: low baseline, spiking late for the
        # chargeback_spike archetype.
        cb_rate = 0.004
        if merchant.archetype == "chargeback_spike" and progress > 0.5:
            cb_rate = 0.045
        refund_rate = 0.012

        for _ in range(count):
            amount = max(
                1.0,
                rng.gauss(merchant.avg_ticket, merchant.avg_ticket * 0.35),
            )
            is_cb = rng.random() < cb_rate
            is_refund = (not is_cb) and rng.random() < refund_rate
            occurred = day.replace(
                hour=rng.randint(0, 23),
                minute=rng.randint(0, 59),
                second=rng.randint(0, 59),
            )
            txns.append(
                SyntheticTransaction(
                    amount=round(amount, 2),
                    channel=rng.choices(
                        CHANNELS, weights=[0.55, 0.25, 0.20]
                    )[0],
                    status="chargeback" if is_cb else "settled",
                    is_chargeback=is_cb,
                    is_refund=is_refund,
                    occurred_at=occurred,
                    merchant_index=merchant_index,
                )
            )
    txns.sort(key=lambda t: t.occurred_at)
    return txns


def seed_database(session, merchant_count: int = 12, seed: int = 42, days: int = 60):
    """Populate the database with synthetic merchants and transactions.

    Existing rows are cleared first so re-seeding is idempotent.
    Returns the list of persisted Merchant ORM objects.
    """
    from .models import Merchant, Report, Transaction

    session.query(Report).delete()
    session.query(Transaction).delete()
    session.query(Merchant).delete()
    session.flush()

    profiles = generate_merchants(count=merchant_count, seed=seed)
    persisted = []
    for idx, profile in enumerate(profiles):
        merchant = Merchant(
            name=profile.name,
            category=profile.category,
            region=profile.region,
            archetype=profile.archetype,
        )
        session.add(merchant)
        session.flush()  # assign merchant.id
        for txn in generate_transactions(profile, idx, days=days, seed=seed):
            session.add(
                Transaction(
                    merchant_id=merchant.id,
                    amount=txn.amount,
                    channel=txn.channel,
                    status=txn.status,
                    is_chargeback=txn.is_chargeback,
                    is_refund=txn.is_refund,
                    occurred_at=txn.occurred_at,
                )
            )
        persisted.append(merchant)
    session.commit()
    return persisted
