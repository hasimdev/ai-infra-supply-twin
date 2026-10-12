# AI Infra Supply Twin

**Where does the AI hardware supply chain break, and what does it cost?**

This is a risk model of the chain behind GPU clusters: foundry, HBM, advanced packaging,
substrates, rack integrators, optics, switching and grid power. It runs millions of
disruption scenarios on a GPU and uses Claude to turn news headlines into risk updates.

<!-- Add a screenshot: docs/dashboard.png -->

```
news headlines ──► Claude (structured output) ──► per-node risk updates
                                                        │
network.json ──────────────────────────────────────────►├──► Monte Carlo on GPU (CuPy) or CPU (NumPy)
                                                        │      time-to-survive vs time-to-recover per node
                                                        ▼
        FastAPI ──► dashboard: risk map · GPUs delayed/yr · $ exposure · stock-vs-dual-source verdict
```

## Quick start

```bash
pip install -r requirements.txt
uvicorn api:app --reload          # dashboard at http://localhost:8000
python run.py                     # same analysis in the terminal
pytest -q                         # 16 tests
```

Or with Docker:

```bash
docker build -t supply-twin . && docker run -p 8000:8000 supply-twin
```

| Option | How |
|---|---|
| GPU backend | `pip install cupy-cuda12x`. It's detected automatically; force it with `--backend cupy`. |
| Live Claude feed | Set `ANTHROPIC_API_KEY`, then run `python run.py --live` or pick "Live Claude feed" in the dashboard. |
| Benchmark | `python bench.py --nodes 1000 --runs 100000` |
| LLM accuracy test | `python eval_feed.py` (needs an API key) |

## How it works

**1. Model (time-to-survive vs time-to-recover).** For every node, *time to survive (TTS)* is
how long the stock downstream of it keeps clusters being built, and *time to recover (TTR)* is
how long the node takes to come back. When TTR is longer than TTS, GPU deployments slip. Each
disruption is valued as GPUs delayed × the value of a GPU-month. Power and network are
**site-level** inputs: staged GPUs can't be installed without them, so only their own spares
protect the site.

**2. GPU engine** (`engine.py`). The network is flattened into node→data-centre paths. Recovery
times are sampled for every node × run with an inverse-CDF triangular draw, then credited for any
second source. Losses per node come from one **incidence-matrix multiply**
`(nodes × paths) @ (paths × runs)`. All of it is written against the array API that NumPy and
CuPy share, so the identical code runs on CPU or CUDA. A plain-Python reference
implementation checks it to within 1e-7.

**3. LLM disruption feed** (`feed.py`). Claude Opus 5.5 receives the node list and the
headlines and returns a schema-validated `SignalReport` via `messages.parse` + Pydantic. For each
headline it gives the node, a risk multiplier, the implied recovery time, a confidence and its
reasoning. Guardrails: the multiplier is clamped to 0.5–5×, probability is capped at 95%,
low-confidence signals are dropped, unknown node IDs are ignored, the SDK retries with backoff,
token usage is logged, and refusals fall back server-side.

**4. Decision layer.** For each top risk the model sizes the GPU stock needed to cover a P90
outage. It then compares the carrying cost with the expected loss avoided across the whole
network, and recommends either staging inventory or adding a second source.

## Results on the sample network

- The sample news raises expected annual loss from **$44M to $116M (+161%)**, mainly through
  grid transformers, optics export reviews and an HBM yield excursion.
- Covering an HBM outage with staged GPUs costs about **60× the loss it avoids**, so the model
  recommends dual-sourcing.
- On a laptop CPU, NumPy runs about **12M scenarios/s, 17× faster than plain Python**. GPU numbers:
  run `bench.py` on a CUDA machine.

## Testing and quality

- `tests/`: engine correctness against the reference loop, determinism, sampling bounds,
  TTS logic, second-source and buffer behaviour, feed guardrails, accuracy-scoring maths and the
  API contract.
- `eval_feed.py` + `data/eval_headlines.json`: 12 labelled headlines, including noise. It scores
  node accuracy, risk-direction accuracy and how often irrelevant news is correctly ignored.
- GitHub Actions runs the tests, an end-to-end smoke run and a benchmark on Python 3.11 and 3.12.

## Limitations (stated up front)

- All network numbers are illustrative placeholders, and suppliers are described by role and
  region, not by company.
- `news.json` and `feed_sample_output.json` are hand-written samples. `--live` produces real
  extractions.
- The model is first-order: buffers add up along a path and second sources ramp linearly. It is
  not a discrete-event simulation.

## Production roadmap

| Area | Next step |
|---|---|
| Data | Load the network from ERP/procurement data; store runs in Postgres |
| Validation | Back-test against documented disruptions (2021 chip shortage, 2011 Thailand floods) |
| Feed | Scheduled ingestion from news APIs; human approval before overrides go live |
| Optimisation | GPU sweep over dual-sourcing choices within a budget |
| Platform | Auth, monitoring, scheduled runs, deploy on a GPU cloud instance |
