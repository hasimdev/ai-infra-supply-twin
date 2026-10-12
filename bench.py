"""Correctness check + throughput benchmark on a synthetic large network.

  python bench.py                         # 1,000-node network, NumPy (+ CuPy if present)
  python bench.py --nodes 3000 --runs 200000

1. Verifies the vectorised engine against a plain-Python loop on the same samples.
2. Times plain Python vs NumPy vs CuPy (when a CUDA GPU is available).
"""
import argparse
import random
import time

import numpy as np

from engine import Network, get_xp, reference_losses, simulate


def synthetic_network(n_nodes, seed=1):
    """Tiered DAG: suppliers -> component plants -> integrators -> data centres."""
    rng = random.Random(seed)
    n_dc = max(4, n_nodes // 50)
    n_int = max(4, n_nodes // 10)
    n_comp = max(8, n_nodes // 5)
    n_sup = n_nodes - n_dc - n_int - n_comp
    tiers = [("S", n_sup, "supplier"), ("C", n_comp, "plant"), ("I", n_int, "plant"), ("D", n_dc, "dc")]
    nodes, ids = [], []
    for prefix, count, kind in tiers:
        tier = []
        for k in range(count):
            lo = rng.uniform(3, 20)
            node = {"id": f"{prefix}{k}", "name": f"{prefix}{k}", "type": kind,
                    "ttr_days": [lo, lo * rng.uniform(1.5, 3), lo * rng.uniform(4, 8)],
                    "annual_prob": rng.uniform(0.02, 0.2)}
            if kind != "supplier":
                node["cover_days"] = rng.uniform(2, 15)
            if kind == "dc":
                node.update(daily_demand=rng.randint(100, 2000), unit_value=1440)
            if kind == "supplier" and rng.random() < 0.3:
                node["alternate"] = {"capacity": rng.uniform(0.3, 0.7), "ramp_days": rng.uniform(7, 30)}
            nodes.append(node)
            tier.append(node["id"])
        ids.append(tier)

    edges = []
    # Each supplier is a critical input to 1-2 component plants.
    for s in ids[0]:
        for c in rng.sample(ids[1], rng.randint(1, 2)):
            edges.append({"from": s, "to": c})
    # Integrators and DCs blend 2-3 upstream sources (shares sum to 1).
    for up, down in ((ids[1], ids[2]), (ids[2], ids[3])):
        for d in down:
            srcs = rng.sample(up, rng.randint(2, 3))
            for s in srcs:
                edges.append({"from": s, "to": d, "share": 1 / len(srcs)})
    return Network(nodes, edges)


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=100_000)
    ap.add_argument("--ref-runs", type=int, default=200, help="runs for the slow Python baseline")
    args = ap.parse_args()

    net = synthetic_network(args.nodes)
    print(f"\nSynthetic network: {len(net.nodes):,} nodes, {len(net.edges):,} edges, "
          f"{len(net.paths):,} node->DC paths\n")

    # 1. Correctness: same TTR samples through both implementations.
    small = simulate(net, args.ref_runs, np, return_samples=True)
    ref, t_ref = timed(lambda: reference_losses(net, small["ttr_samples"].astype(np.float64)))
    rel_err = np.abs(small["loss_samples"] - ref).max() / max(ref.max(), 1)
    print(f"Correctness vs plain-Python loop: max relative error {rel_err:.1e} "
          f"({'OK' if rel_err < 1e-4 else 'MISMATCH'})")

    scenarios = len(net.nodes) * args.runs
    py_rate = len(net.nodes) * args.ref_runs / t_ref
    print(f"\n{'Backend':<34} {'Scenarios':>14} {'Time':>9} {'Scenarios/s':>14} {'Speed-up':>9}")
    print(f"{'plain Python (measured on subset)':<34} {len(net.nodes) * args.ref_runs:>14,} "
          f"{t_ref:>8.2f}s {py_rate:>14,.0f} {'1x':>9}")

    backends = [("numpy", np, "numpy (CPU)")]
    try:
        cp, label = get_xp("cupy")
        backends.append(("cupy", cp, label))
    except Exception:
        print(f"{'cupy (GPU)':<34} {'skipped: no CUDA device / CuPy not installed':>48}")

    for name, xp, label in backends:
        if name == "cupy":
            simulate(net, 1000, xp)  # warm-up: kernel compilation
        _, t = timed(lambda: simulate(net, args.runs, xp))
        if name == "cupy":
            xp.cuda.Device().synchronize()
        rate = scenarios / t
        print(f"{label[:34]:<34} {scenarios:>14,} {t:>8.2f}s {rate:>14,.0f} {rate / py_rate:>8,.0f}x")
    print()


if __name__ == "__main__":
    main()
