"""Experiment 1 — Greedy Baseline Characterization.

Matches the setup described in the paper draft:
  - 20 assets, 5 Indo-Pacific theater locations, capacity c=5 per location
  - Fixed degradation rate delta=0.08 per step, T=10 steps
  - N_seed=10 independent seeds; results averaged with std dev
  - Compares greedy vs random placement on 4 posture metrics
  - Evaluates SWR under uniform and skewed threat distributions (S=20 scenarios)

Run:
    python scripts/experiment1_greedy.py
"""
from __future__ import annotations

import copy
import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import PostureOptimizer, ReplenishmentPolicy, SustainmentAction
from postureopt.data import make_posture_state, simulate_degradation
from postureopt.evaluate import (
    coverage_score,
    posture_efficiency,
    readiness_score,
    scenario_weighted_readiness,
    sustainment_cost,
)

# ── Experiment parameters ──────────────────────────────────────────────────
N_ASSETS      = 20
CAPACITY      = 5
N_LOCATIONS   = 5
DELTA         = 0.08
T_STEPS       = 10
N_SEEDS       = 10
N_SCENARIOS   = 20
BASE_SEED     = 42


# ── Threat scenario generators ─────────────────────────────────────────────

def _uniform_threats(locations, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        {loc.location_id: rng.uniform(0.10, 0.30) for loc in locations}
        for _ in range(n)
    ]


def _skewed_threats(locations, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        {loc.location_id: min(0.9, loc.strategic_value * rng.uniform(0.50, 1.00))
         for loc in locations}
        for _ in range(n)
    ]


# ── Per-seed simulation ────────────────────────────────────────────────────

def run_seed(seed: int) -> dict:
    state   = make_posture_state(n_assets=N_ASSETS, seed=seed, capacity=CAPACITY)
    assets  = state.assets
    locs    = state.locations
    opt     = PostureOptimizer(seed=seed)

    greedy_assign = opt.greedy_placement(assets, locs)
    random_assign = opt.random_placement(assets, locs)

    threat_seed = BASE_SEED  # fixed so scenario variance doesn't pollute seed variance
    uniform_scenarios = _uniform_threats(locs, N_SCENARIOS, threat_seed)
    skewed_scenarios  = _skewed_threats(locs, N_SCENARIOS, threat_seed)

    def simulate(assignment: dict) -> dict:
        placed = [
            copy.copy(a).__class__(
                asset_id=a.asset_id,
                asset_type=a.asset_type,
                location_id=assignment.get(a.asset_id, a.location_id),
                quantity=a.quantity,
                readiness_rate=a.readiness_rate,
                maintenance_days_remaining=a.maintenance_days_remaining,
            )
            for a in assets
        ]
        from postureopt.core import Asset as _Asset, PostureState as _PS
        start = _PS(assets=placed, locations=locs, time_step=0)
        history = simulate_degradation(
            start, n_steps=T_STEPS, seed=seed, degradation_rate=DELTA
        )

        steps = []
        for t, s in enumerate(history):
            policy = ReplenishmentPolicy()
            actions = [policy.decide(a) for a in s.assets]
            cost = sustainment_cost(actions)
            r    = readiness_score(s.assets)
            c    = coverage_score(s.assets, locs)
            e    = posture_efficiency(r, c, cost)
            swr_u = scenario_weighted_readiness(s.assets, assignment, uniform_scenarios)
            swr_s = scenario_weighted_readiness(s.assets, assignment, skewed_scenarios)
            steps.append(dict(t=t+1, readiness=r, coverage=c, cost=cost,
                               efficiency=e, swr_uniform=swr_u, swr_skewed=swr_s))
        return steps

    return {"greedy": simulate(greedy_assign), "random": simulate(random_assign)}


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(all_runs: list[list[dict]]) -> list[dict]:
    """Average metric dicts across seeds for each time step."""
    n = len(all_runs)
    T = len(all_runs[0])
    result = []
    for t in range(T):
        row = {"t": all_runs[0][t]["t"]}
        for key in ("readiness", "coverage", "cost", "efficiency", "swr_uniform", "swr_skewed"):
            vals = [all_runs[s][t][key] for s in range(n)]
            mean = sum(vals) / n
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
            row[f"{key}_mean"] = mean
            row[f"{key}_std"]  = std
        result.append(row)
    return result


# ── Output helpers ─────────────────────────────────────────────────────────

def print_metrics_table(rows: list[dict], label: str) -> None:
    header = (f"{'t':>3}  {'Readiness':>14} {'Coverage':>10} "
              f"{'Cost':>10} {'Efficiency':>12} {'SWR-Unif':>12} {'SWR-Skew':>12}")
    sep = "-" * len(header)
    print(f"\n{label}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['t']:>3}  "
            f"{r['readiness_mean']:>6.3f}±{r['readiness_std']:.3f}  "
            f"{r['coverage_mean']:>6.3f}±{r['coverage_std']:.3f}  "
            f"{r['cost_mean']:>6.1f}±{r['cost_std']:.1f}  "
            f"{r['efficiency_mean']:>6.3f}±{r['efficiency_std']:.3f}  "
            f"{r['swr_uniform_mean']:>6.3f}±{r['swr_uniform_std']:.3f}  "
            f"{r['swr_skewed_mean']:>6.3f}±{r['swr_skewed_std']:.3f}"
        )
    print(sep)


def main() -> None:
    print(f"Experiment 1 — Greedy Baseline Characterization")
    print(f"Setup: {N_ASSETS} assets, {N_LOCATIONS} locations, "
          f"capacity={CAPACITY}, delta={DELTA}, T={T_STEPS}, N_seed={N_SEEDS}\n")

    seeds = list(range(BASE_SEED, BASE_SEED + N_SEEDS))
    greedy_runs, random_runs = [], []
    for seed in seeds:
        result = run_seed(seed)
        greedy_runs.append(result["greedy"])
        random_runs.append(result["random"])

    greedy_agg = aggregate(greedy_runs)
    random_agg = aggregate(random_runs)

    print_metrics_table(greedy_agg, "Greedy Placement")
    print_metrics_table(random_agg, "Random Placement")

    # Summary: efficiency gap at t=10
    g10 = greedy_agg[-1]
    r10 = random_agg[-1]
    gap_pct = (r10["efficiency_mean"] - g10["efficiency_mean"]) / g10["efficiency_mean"] * 100
    swr_drop = (g10["swr_skewed_mean"] - g10["swr_uniform_mean"]) / g10["swr_uniform_mean"] * 100

    print(f"\nSummary at t={T_STEPS}:")
    print(f"  Greedy efficiency : {g10['efficiency_mean']:.3f} ± {g10['efficiency_std']:.3f}")
    print(f"  Random efficiency : {r10['efficiency_mean']:.3f} ± {r10['efficiency_std']:.3f}")
    print(f"  Efficiency gap    : {gap_pct:+.1f}% (random over greedy)")
    print(f"  Greedy SWR uniform: {g10['swr_uniform_mean']:.3f} ± {g10['swr_uniform_std']:.3f}")
    print(f"  Greedy SWR skewed : {g10['swr_skewed_mean']:.3f} ± {g10['swr_skewed_std']:.3f}")
    print(f"  SWR drop (skewed) : {swr_drop:.1f}% relative to uniform")


if __name__ == "__main__":
    main()
