"""Measure how accurately the LLM feed maps headlines to network nodes.

  python eval_feed.py                     # live: calls Claude on data/eval_headlines.json
  python eval_feed.py --signals out.json  # score a saved SignalReport instead

Metrics: node accuracy (right node, or correctly "irrelevant"), direction
accuracy on relevant headlines (risk up vs down), and irrelevant-headline recall.
"""
import argparse
import json
from pathlib import Path

import feed
from engine import Network

DATA = Path(__file__).parent / "data"


def direction(signal):
    if not signal.relevant:
        return "none"
    return "up" if signal.risk_multiplier > 1 else "down" if signal.risk_multiplier < 1 else "none"


def score(labels, signals):
    by_index = {s.headline_index: s for s in signals}
    rows, node_ok, dir_ok, n_rel, irr_ok, n_irr = [], 0, 0, 0, 0, 0
    for i, lab in enumerate(labels):
        s = by_index.get(i)
        got_node = s.node_id if s and s.relevant else None
        got_dir = direction(s) if s else "none"
        hit = got_node == lab["expect_node"]
        node_ok += hit
        if lab["expect_node"] is None:
            n_irr += 1
            irr_ok += got_node is None
        else:
            n_rel += 1
            dir_ok += got_dir == lab["expect_direction"]
        rows.append((i, lab["expect_node"], got_node, lab["expect_direction"], got_dir, hit))
    return {
        "node_accuracy": node_ok / len(labels),
        "direction_accuracy": dir_ok / n_rel if n_rel else 1.0,
        "irrelevant_recall": irr_ok / n_irr if n_irr else 1.0,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals", help="saved SignalReport JSON to score instead of calling Claude")
    args = ap.parse_args()

    labels = json.loads((DATA / "eval_headlines.json").read_text(encoding="utf-8"))
    if args.signals:
        signals = feed.extract_signals_offline(args.signals)
    else:
        net = Network.load(DATA / "network.json")
        signals = feed.extract_signals_live(net, labels)
        (DATA / "eval_live_output.json").write_text(
            feed.SignalReport(signals=signals).model_dump_json(indent=2), encoding="utf-8")

    r = score(labels, signals)
    print(f"\n{'#':>2}  {'expected':<9} {'got':<9} {'dir exp':<8} {'dir got':<8}")
    for i, en, gn, ed, gd, hit in r["rows"]:
        print(f"{i:>2}  {str(en):<9} {str(gn):<9} {ed:<8} {gd:<8} {'' if hit else '<- miss'}")
    print(f"\nNode accuracy       {r['node_accuracy']:.0%}")
    print(f"Direction accuracy  {r['direction_accuracy']:.0%}  (relevant headlines)")
    print(f"Irrelevant recall   {r['irrelevant_recall']:.0%}  (noise correctly ignored)\n")


if __name__ == "__main__":
    main()
