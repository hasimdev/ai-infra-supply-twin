"""AI Infra Supply Twin: news -> risk updates -> GPU Monte Carlo -> decisions.

  python run.py                    # offline feed, auto-detect GPU
  python run.py --live             # Claude reads data/news.json
  python run.py --runs 1000000 --backend cupy
"""
import argparse

from pipeline import analyze


def money(x):
    return f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}k"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=200_000)
    ap.add_argument("--backend", choices=["auto", "cupy", "numpy"], default="auto")
    ap.add_argument("--live", action="store_true", help="call Claude instead of the cached sample")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    args = ap.parse_args()

    r = analyze(args.runs, args.backend, args.live, args.min_confidence)

    print(f"\nAI INFRA SUPPLY TWIN  |  backend: {r['backend']}  |  feed: {r['feed_source']}  |  "
          f"{r['runs_per_node']:,} runs/node\n")
    print("DISRUPTION FEED")
    for s in r["signals"]:
        print(f"  [{s['headline_index']}] {s['node_id']} {s['node_name']}: {s['change']}  "
              f"(conf {s['confidence']:.2f})")
        print(f"      {s['rationale']}")

    print(f"\nRISK AFTER NEWS  ({r['scenarios']:,} scenarios in {r['seconds']:.2f}s)\n")
    print(f"{'Node':<4} {'Name':<40} {'TTS':>4} {'TTR50':>6} {'TTR90':>6} "
          f"{'GPUs late/yr':>12} {'Exp.loss/yr':>12} {'Before news':>12}  Flag")
    for x in r["risks"]:
        flag = {"critical": "CRITICAL", "watch": "watch"}.get(x["flag"], "")
        print(f"{x['id']:<4} {x['name'][:40]:<40} {x['tts']:>4.0f} {x['ttr_p50']:>6.1f} "
              f"{x['ttr_p90']:>6.1f} {x['gpus_delayed_per_year']:>12,.0f} {money(x['eal']):>12} "
              f"{money(x['eal_before_news']):>12}  {flag}")
    a, b = r["total_eal"], r["total_eal_before_news"]
    print(f"\nTotal expected annual loss: {money(a)} (was {money(b)} before news, {(a - b) / b:+.0%})")
    print("CRITICAL = median recovery outlasts downstream buffers (TTR50 > TTS).\n")

    print("BUFFER vs FIX-THE-SOURCE (extra staged GPUs sized for each risk's P90 outage)\n")
    for rec in r["recommendations"]:
        if rec["roi"] is None:
            print(f"  vs {rec['for']}: {rec['verdict']}")
            continue
        adds = ", ".join(f"+{d}d at {dc}" for dc, d in rec["extra_days"].items())
        print(f"  vs {rec['for']}: {adds}")
        print(f"     carry {money(rec['annual_cost'])}/yr, avoids {money(rec['annual_benefit'])}/yr, "
              f"ROI {rec['roi']:.2f}x -> {rec['verdict']}")
    print()


if __name__ == "__main__":
    main()
