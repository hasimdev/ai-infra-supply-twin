"""LLM disruption feed: turns news headlines into risk updates for the network.

Claude reads the headlines alongside the node catalogue and returns structured
signals (which node, how much riskier, implied recovery time, confidence).
Signals above a confidence threshold are applied as overrides to the network.

Live mode needs Anthropic credentials (ANTHROPIC_API_KEY or `ant auth login`).
Offline mode replays data/feed_sample_output.json so the pipeline runs anywhere.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

MODEL = "claude-opus-5-5"
log = logging.getLogger(__name__)

SYSTEM = """You are a supply-chain risk analyst for AI data-centre build-outs.
You receive a catalogue of nodes in a supply network and a list of news headlines.
For each headline, decide whether it changes the disruption risk of exactly one node.

- relevant=false (node_id=null) when the headline does not bear on any node.
- risk_multiplier scales that node's 12-month disruption probability:
  1.0 = no change, 2.0 = twice as likely, 0.7 = less likely (e.g. new capacity).
  Stay within 0.5-5.0.
- ttr_days = [min, most likely, max] recovery days, only when the headline
  implies how long an outage would last; otherwise null.
- confidence in 0-1: how sure you are about the node match and the direction.
Ground every judgement in the headline text; do not invent facts."""


class Signal(BaseModel):
    headline_index: int
    relevant: bool
    node_id: Optional[str]
    risk_multiplier: float
    ttr_days: Optional[list[float]]
    confidence: float
    rationale: str


class SignalReport(BaseModel):
    signals: list[Signal]


def extract_signals_live(net, headlines):
    import anthropic

    catalogue = "\n".join(f"{n['id']}: {n['name']} ({n['type']})" for n in net.nodes)
    news = "\n".join(f"[{i}] {h['date']} - {h['headline']}" for i, h in enumerate(headlines))
    # The SDK retries 408/409/429/5xx and connection errors with backoff.
    client = anthropic.Anthropic(max_retries=4, timeout=120.0)
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"<nodes>\n{catalogue}\n</nodes>\n\n<headlines>\n{news}\n</headlines>"}],
        output_format=SignalReport,
        # Server-side refusal fallback: reroutes to another model if this one declines.
        extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        extra_body={"fallbacks": "default"},
    )
    usage = response.usage
    log.info("feed: %d headlines, model=%s, in=%d out=%d tokens, request_id=%s",
             len(headlines), response.model, usage.input_tokens, usage.output_tokens,
             response._request_id)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"No usable output (stop_reason={response.stop_reason})")
    return response.parsed_output.signals


def extract_signals_offline(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SignalReport.model_validate(data).signals


def signals_to_overrides(net, signals, min_confidence=0.6):
    """Convert signals into {node_id: {field: value}} plus a printable change log."""
    overrides, log = {}, []
    for s in signals:
        if not s.relevant or s.node_id not in net.index:
            continue
        if s.confidence < min_confidence:
            log.append((s, None, "skipped (low confidence)"))
            continue
        node = net.node(s.node_id)
        current = overrides.setdefault(s.node_id, {})
        p0 = current.get("annual_prob", node["annual_prob"])
        mult = min(5.0, max(0.5, s.risk_multiplier))
        current["annual_prob"] = min(0.95, p0 * mult)
        if s.ttr_days and len(s.ttr_days) == 3 and min(s.ttr_days) > 0:
            current["ttr_days"] = sorted(s.ttr_days)
        log.append((s, node, f"p {p0:.0%} -> {current['annual_prob']:.0%}"
                    + (f", TTR -> {current['ttr_days']}" if "ttr_days" in current else "")))
    return overrides, log
