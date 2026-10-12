import pytest

import feed
from engine import Network
from eval_feed import score
from pipeline import DATA


@pytest.fixture(scope="module")
def net():
    return Network.load(DATA / "network.json")


def sig(i, node, mult=2.0, conf=0.9, relevant=True, ttr=None):
    return feed.Signal(headline_index=i, relevant=relevant, node_id=node, risk_multiplier=mult,
                       ttr_days=ttr, confidence=conf, rationale="test")


def test_offline_sample_parses():
    signals = feed.extract_signals_offline(DATA / "feed_sample_output.json")
    assert len(signals) == 6
    assert any(not s.relevant for s in signals)


def test_overrides_apply_and_clamp(net):
    overrides, _ = feed.signals_to_overrides(net, [sig(0, "K1", mult=50.0, ttr=[30, 10, 20])])
    assert overrides["K1"]["annual_prob"] == pytest.approx(0.07 * 5)   # multiplier capped at 5x
    assert overrides["K1"]["ttr_days"] == [10, 20, 30]                  # sorted min/mode/max


def test_probability_capped(net):
    signals = [sig(i, "O1", mult=5.0) for i in range(4)]
    overrides, _ = feed.signals_to_overrides(net, signals)
    assert overrides["O1"]["annual_prob"] == 0.95


def test_low_confidence_irrelevant_and_unknown_nodes_ignored(net):
    signals = [sig(0, "H1", conf=0.3), sig(1, None, relevant=False), sig(2, "ZZ9")]
    overrides, log = feed.signals_to_overrides(net, signals, min_confidence=0.6)
    assert overrides == {}
    assert [entry[2] for entry in log] == ["skipped (low confidence)"]


def test_eval_scoring():
    labels = [
        {"expect_node": "F1", "expect_direction": "up"},
        {"expect_node": "H2", "expect_direction": "down"},
        {"expect_node": None, "expect_direction": "none"},
    ]
    signals = [sig(0, "F1", mult=2.0), sig(1, "H1", mult=0.8), sig(2, None, relevant=False)]
    r = score(labels, signals)
    assert r["node_accuracy"] == pytest.approx(2 / 3)
    assert r["direction_accuracy"] == 1.0
    assert r["irrelevant_recall"] == 1.0
