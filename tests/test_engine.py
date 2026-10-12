import numpy as np
import pytest

from engine import Network, reference_losses, simulate, triangular_icdf
from pipeline import DATA


@pytest.fixture(scope="module")
def net():
    return Network.load(DATA / "network.json")


def test_vectorised_matches_reference_loop(net):
    r = simulate(net, 300, np, return_samples=True)
    ref = reference_losses(net, r["ttr_samples"].astype(np.float64))
    np.testing.assert_allclose(r["loss_samples"], ref, rtol=1e-4, atol=1.0)


def test_same_seed_is_deterministic(net):
    a, b = simulate(net, 2000, np, seed=3), simulate(net, 2000, np, seed=3)
    np.testing.assert_array_equal(a["eal"], b["eal"])


def test_triangular_samples_stay_in_bounds():
    u = np.random.default_rng(0).random(100_000)
    x = triangular_icdf(np, u, 10.0, 20.0, 60.0)
    assert x.min() >= 10 and x.max() <= 60
    assert abs(x.mean() - (10 + 20 + 60) / 3) < 0.3  # triangular mean


def test_tts_is_weakest_downstream_buffer(net):
    tts = dict(zip(net.ids, net.tts()))
    assert tts["F1"] == 5 + 5 + 12   # K1 -> M2 -> D2 is the thinnest route
    assert tts["T1"] == 0            # site-level: no spare transformer
    assert tts["O1"] == 10           # its own spares, not staged GPUs


def test_second_source_reduces_loss(net):
    i = net.index["C1"]
    without = net.with_overrides({"C1": {"alternate": {"capacity": 0.0, "ramp_days": 1e9}}})
    assert simulate(net, 5000, np)["eal"][i] < simulate(without, 5000, np)["eal"][i]


def test_staged_gpus_do_not_protect_site_level_inputs(net):
    base = simulate(net, 5000, np)
    more = simulate(net, 5000, np, extra_cover={"D1": 30, "D2": 30, "D3": 30})
    t1, f1 = net.index["T1"], net.index["F1"]
    assert more["eal"][t1] == pytest.approx(base["eal"][t1])
    assert more["eal"][f1] < base["eal"][f1]


def test_more_buffer_never_increases_total_loss(net):
    totals = [simulate(net, 5000, np, extra_cover={"D1": d, "D2": d, "D3": d})["eal"].sum()
              for d in (0, 10, 30)]
    assert totals[0] >= totals[1] >= totals[2]
