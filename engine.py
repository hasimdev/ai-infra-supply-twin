"""Vectorised TTS/TTR Monte Carlo engine.

Written against the array API shared by NumPy and CuPy, so the same code runs
on a CUDA GPU (CuPy) or on the CPU (NumPy) with no changes.

The network is flattened into "paths" (disrupted node -> ... -> data centre).
For R Monte Carlo runs the engine builds:

    ttr       (nodes, R)  recovery time per node, sampled from a triangular law
    outage    (nodes, R)  full-outage-equivalent days after second-source credit
    lost_days (paths, R)  max(0, outage[src] - buffer)
    loss      (nodes, R)  = incidence (nodes, paths) @ (value * lost_days)

The last step is a dense matrix multiply, which is what makes it GPU-friendly.
"""
import copy
import json
from pathlib import Path

import numpy as np

NO_RAMP = 1e9  # "no second source": ramp never completes


def get_xp(backend="auto"):
    """Return (array_module, label). backend: auto | cupy | numpy."""
    if backend in ("auto", "cupy"):
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return cp, f"cupy on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
        except Exception:
            if backend == "cupy":
                raise
    return np, "numpy (CPU)"


def to_host(xp, a):
    return a if xp is np else xp.asnumpy(a)


class Network:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.ids = [n["id"] for n in nodes]
        self.index = {nid: i for i, nid in enumerate(self.ids)}
        self.children = {nid: [] for nid in self.ids}
        for e in edges:
            self.children[e["from"]].append((e["to"], e.get("share", 1.0)))
        self.dcs = [i for i, n in enumerate(nodes) if n["type"] == "dc"]
        # (src, dc, supply_fraction, buffer_days, gpu_buffer_applies) per path
        self.paths = [(self.index[nid], self.index[dc], f, b, ok)
                      for nid in self.ids for dc, f, b, ok in self._paths_from(nid)]

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["nodes"], data["edges"])

    def node(self, nid):
        return self.nodes[self.index[nid]]

    def with_overrides(self, overrides):
        """Copy of the network with {node_id: {field: value}} applied."""
        nodes = copy.deepcopy(self.nodes)
        for n in nodes:
            n.update(overrides.get(n["id"], {}))
        return Network(nodes, self.edges)

    def _paths_from(self, start):
        """Yield (dc_id, fraction, buffer_days, gpu_buffer_applies).

        The failed node's own stock is treated as stranded, so only buffers
        strictly downstream count. Site-level inputs (power, network) set
        `spares_days` instead: staged GPUs can't be installed without them, so
        only their own spares protect the site.
        """
        node = self.node(start)
        if node["type"] == "dc":
            yield start, 1.0, 0.0, False
            return
        if "spares_days" in node:
            for dc, share in self.children[start]:
                yield dc, share, float(node["spares_days"]), False
            return
        stack = [(start, 1.0, 0.0)]
        while stack:
            nid, frac, buf = stack.pop()
            for child, share in self.children[nid]:
                c = self.node(child)
                f, b = frac * share, buf + c.get("cover_days", 0)
                if c["type"] == "dc":
                    yield child, f, b, True
                else:
                    stack.append((child, f, b))

    def tts(self, extra_cover=None):
        """Time-to-survive per node: weakest buffer over all its paths."""
        extra_cover = extra_cover or {}
        out = [float("inf")] * len(self.nodes)
        for src, dc, _, buf, ok in self.paths:
            if ok:
                buf += extra_cover.get(self.ids[dc], 0)
            out[src] = min(out[src], buf)
        return out


def triangular_icdf(xp, u, lo, mode, hi):
    """Inverse CDF of the triangular distribution (vectorised)."""
    span = hi - lo
    c = (mode - lo) / span
    left = lo + xp.sqrt(u * span * (mode - lo))
    right = hi - xp.sqrt((1 - u) * span * (hi - mode))
    return xp.where(u < c, left, right)


def effective_outage(xp, ttr, ramp, cap):
    """Full-outage days: 100% lost until the second source ramps, then (1-cap)."""
    return xp.minimum(ttr, ramp) + (1 - cap) * xp.maximum(0, ttr - ramp)


def simulate(net, runs=100_000, xp=np, seed=7, extra_cover=None, chunk=25_000,
             return_samples=False):
    """Run the Monte Carlo for every node. Returns host-side NumPy results."""
    extra_cover = extra_cover or {}
    f32 = xp.float32
    n = len(net.nodes)
    col = lambda key, default=0.0: xp.asarray([nd.get(key, default) for nd in net.nodes], dtype=f32)

    lo, mode, hi = (xp.asarray([nd["ttr_days"][k] for nd in net.nodes], dtype=f32)[:, None]
                    for k in range(3))
    prob = col("annual_prob")
    ramp = xp.asarray([nd.get("alternate", {}).get("ramp_days", NO_RAMP) for nd in net.nodes], dtype=f32)[:, None]
    cap = xp.asarray([nd.get("alternate", {}).get("capacity", 0.0) for nd in net.nodes], dtype=f32)[:, None]

    src = np.array([p[0] for p in net.paths])
    dc = np.array([p[1] for p in net.paths])
    frac = np.array([p[2] for p in net.paths])
    buf = np.array([b + (extra_cover.get(net.ids[d], 0) if ok else 0)
                    for _, d, _, b, ok in net.paths])
    demand = np.array([net.nodes[d]["daily_demand"] for d in dc])
    value = np.array([net.nodes[d]["unit_value"] for d in dc])

    units_w = xp.asarray(frac * demand, dtype=f32)[:, None]   # units lost per path-day
    buf_d = xp.asarray(buf, dtype=f32)[:, None]
    src_d = xp.asarray(src)
    incidence = xp.zeros((n, len(src)), dtype=f32)
    incidence[src_d, xp.arange(len(src))] = 1
    value_incidence = incidence * xp.asarray(value, dtype=f32)[None, :]

    rng = xp.random.default_rng(seed)
    ttr = xp.empty((n, runs), dtype=f32)
    units = xp.empty((n, runs), dtype=f32)
    loss = xp.empty((n, runs), dtype=f32)
    for s in range(0, runs, chunk):
        e = min(runs, s + chunk)
        u = rng.random((n, e - s), dtype=f32)
        t = triangular_icdf(xp, u, lo, mode, hi)
        outage = effective_outage(xp, t, ramp, cap)
        path_units = units_w * xp.maximum(0, outage[src_d] - buf_d)   # (paths, chunk)
        ttr[:, s:e] = t
        units[:, s:e] = incidence @ path_units
        loss[:, s:e] = value_incidence @ path_units

    result = {
        "ttr_p50": xp.quantile(ttr, 0.5, axis=1),
        "ttr_p90": xp.quantile(ttr, 0.9, axis=1),
        "eal": prob * loss.mean(axis=1),
        "gpus_delayed_per_year": prob * units.mean(axis=1),
        "p95_event_loss": xp.quantile(loss, 0.95, axis=1),
    }
    result = {k: to_host(xp, v) for k, v in result.items()}
    result["tts"] = np.array(net.tts(extra_cover))
    if return_samples:
        result["ttr_samples"] = to_host(xp, ttr)
        result["loss_samples"] = to_host(xp, loss)
    return result


def reference_losses(net, ttr_samples):
    """Plain-Python loop over the same samples: correctness check + CPU baseline."""
    out = []
    for i, node in enumerate(net.nodes):
        alt = node.get("alternate", {})
        ramp, cap = alt.get("ramp_days", NO_RAMP), alt.get("capacity", 0.0)
        my_paths = [p for p in net.paths if p[0] == i]
        row = []
        for t in ttr_samples[i]:
            outage = min(t, ramp) + (1 - cap) * max(0.0, t - ramp)
            total = 0.0
            for _, d, f, b, _ in my_paths:
                dc = net.nodes[d]
                total += f * dc["daily_demand"] * max(0.0, outage - b) * dc["unit_value"]
            row.append(total)
        out.append(row)
    return np.array(out)


def recommend_buffers(net, base, runs, xp, top_n=3):
    """Size extra DC stock to cover each top risk's P90 outage; price it."""
    baseline = float(base["eal"].sum())
    order = np.argsort(-base["eal"])[:top_n]
    recs = []
    for i in order:
        node = net.nodes[i]
        alt = node.get("alternate", {})
        t90 = float(base["ttr_p90"][i])
        ramp, cap = alt.get("ramp_days", NO_RAMP), alt.get("capacity", 0.0)
        p90_outage = min(t90, ramp) + (1 - cap) * max(0.0, t90 - ramp)
        extra = {}
        for s, d, _, b, ok in net.paths:
            if s == i and ok:
                need = round(max(0.0, p90_outage - b))
                if need > 0:
                    extra[net.ids[d]] = max(extra.get(net.ids[d], 0), need)
        if not extra:
            recs.append({"for": net.ids[i], "extra_days": {}, "annual_cost": 0.0,
                         "annual_benefit": 0.0})
            continue
        cost = sum(days * net.node(d)["daily_demand"] * net.node(d)["holding_cost_per_unit_year"]
                   for d, days in extra.items())
        after = float(simulate(net, runs, xp, extra_cover=extra)["eal"].sum())
        recs.append({"for": net.ids[i], "extra_days": extra, "annual_cost": cost,
                     "annual_benefit": baseline - after})
    return recs
