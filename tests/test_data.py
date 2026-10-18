"""Tests for the synthetic dataset generator."""

from app.data import generate_merchants, generate_transactions


def test_generation_is_deterministic():
    a = generate_merchants(count=6, seed=7)
    b = generate_merchants(count=6, seed=7)
    assert [m.name for m in a] == [m.name for m in b]
    assert [m.archetype for m in a] == [m.archetype for m in b]

    tx_a = generate_transactions(a[0], 0, days=10, seed=7)
    tx_b = generate_transactions(b[0], 0, days=10, seed=7)
    assert len(tx_a) == len(tx_b)
    assert [t.amount for t in tx_a] == [t.amount for t in tx_b]


def test_all_archetypes_represented():
    merchants = generate_merchants(count=12, seed=42)
    assert {m.archetype for m in merchants} == {
        "stable",
        "growth",
        "volume_decline",
        "chargeback_spike",
    }


def test_chargeback_spike_archetype_has_elevated_chargebacks():
    merchants = generate_merchants(count=12, seed=42)
    by_archetype = {m.archetype: (i, m) for i, m in enumerate(merchants)}

    def cb_rate(archetype):
        idx, merchant = by_archetype[archetype]
        txns = generate_transactions(merchant, idx, days=60, seed=42)
        return sum(1 for t in txns if t.is_chargeback) / len(txns)

    assert cb_rate("chargeback_spike") > cb_rate("stable")
    assert cb_rate("chargeback_spike") > 0.015  # clearly elevated vs ~0.4% baseline


def test_growth_and_decline_trends():
    merchants = generate_merchants(count=12, seed=42)
    by_archetype = {m.archetype: (i, m) for i, m in enumerate(merchants)}

    def second_half_share(archetype):
        idx, merchant = by_archetype[archetype]
        txns = generate_transactions(merchant, idx, days=60, seed=42)
        total = sum(t.amount for t in txns)
        start, end = txns[0].occurred_at, txns[-1].occurred_at
        midpoint = start + (end - start) / 2  # split on the mid-window date
        second = sum(t.amount for t in txns if t.occurred_at >= midpoint)
        return second / total

    assert second_half_share("growth") > 0.52
    assert second_half_share("volume_decline") < 0.48
