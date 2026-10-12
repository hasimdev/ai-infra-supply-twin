"""End-to-end analysis shared by the CLI (run.py) and the API (api.py)."""
import json
import time
from pathlib import Path

import numpy as np

import feed
from engine import Network, get_xp, recommend_buffers, simulate

DATA = Path(__file__).parent / "data"


def analyze(runs=200_000, backend="auto", live=False, min_confidence=0.6):
    """news -> risk overrides -> Monte Carlo before/after -> recommendations.

    Returns a JSON-serialisable dict.
    """
    xp, label = get_xp(backend)
    base_net = Network.load(DATA / "network.json")
    headlines = json.loads((DATA / "news.json").read_text(encoding="utf-8"))

    if live:
        signals = feed.extract_signals_live(base_net, headlines)
        (DATA / "feed_live_output.json").write_text(
            feed.SignalReport(signals=signals).model_dump_json(indent=2), encoding="utf-8")
        source = f"live ({feed.MODEL})"
    else:
        signals = feed.extract_signals_offline(DATA / "feed_sample_output.json")
        source = "offline sample"
    overrides, log = feed.signals_to_overrides(base_net, signals, min_confidence)
    net = base_net.with_overrides(overrides)

    start = time.perf_counter()
    before = simulate(base_net, runs, xp)
    after = simulate(net, runs, xp)
    elapsed = time.perf_counter() - start

    risks = []
    for i in np.argsort(-after["eal"]):
        tts, t50, t90 = (float(after[k][i]) for k in ("tts", "ttr_p50", "ttr_p90"))
        node = net.nodes[i]
        risks.append({
            "id": net.ids[i], "name": node["name"], "type": node["type"],
            "tts": tts, "ttr_p50": t50, "ttr_p90": t90,
            "annual_prob": node["annual_prob"],
            "gpus_delayed_per_year": float(after["gpus_delayed_per_year"][i]),
            "eal": float(after["eal"][i]),
            "eal_before_news": float(before["eal"][i]),
            "p95_event_loss": float(after["p95_event_loss"][i]),
            "flag": "critical" if t50 > tts else ("watch" if t90 > tts else "ok"),
        })

    recommendations = []
    for rec in recommend_buffers(net, after, runs, xp):
        if not rec["extra_days"]:
            rec.update(roi=None, verdict="spares or redundant source (staged GPUs can't cover it)")
        else:
            rec["roi"] = rec["annual_benefit"] / rec["annual_cost"]
            rec["verdict"] = ("stage the inventory" if rec["roi"] >= 1
                              else "dual-source / contract capacity instead")
        recommendations.append(rec)

    return {
        "backend": label,
        "feed_source": source,
        "runs_per_node": runs,
        "scenarios": 2 * runs * len(net.nodes),
        "seconds": elapsed,
        "total_eal": float(after["eal"].sum()),
        "total_eal_before_news": float(before["eal"].sum()),
        "signals": [{
            **s.model_dump(),
            "node_name": node["name"] if node else None,
            "change": change,
        } for s, node, change in log],
        "risks": risks,
        "recommendations": recommendations,
        "network": {"nodes": net.nodes, "edges": net.edges},
    }
