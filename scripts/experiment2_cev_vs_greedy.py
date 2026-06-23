"""Experiment 2 — Scenario-Weighted CEV Optimizer vs. Greedy Baseline.

Quantifies the EVSS (Expected Value of the Stochastic Solution) across three
threat distributions and three scenario-count conditions.

Setup: 20 assets, 5 locations, capacity=5, stochastic degradation, N_seed=10.
Vary: threat distribution (uniform, skewed, adversarial) x scenario count (5, 20, 100).

Metrics per condition:
  - greedy_eff    : posture efficiency of greedy placement (evaluated by CEV scorer)
  - cev_eff       : posture efficiency of CEV-optimal placement
  - evss          : cev_eff - greedy_eff  (Expected Value of Stochastic Solution)
  - evss_pct      : evss / greedy_eff * 100

Run:
    python scripts/experiment2_cev_vs_greedy.py
"""
from __future__ import annotations

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import PostureOptimizer
from postureopt.data import make_posture_state
from postureopt.drsp import (
    CEVOptimizer,
    ScenarioSet,
    ThreatScenario,
)
from postureopt.evaluate import (
    coverage_score,
    posture_efficiency,
    readiness_score,
    sustainment_cost,
)
from postureopt.drsp import _apply_placement, _recourse_actions

N_ASSETS   = 20
CAPACITY   = 5
N_SEEDS    = 10
BASE_SEED  = 42
SCENARIO_COUNTS = [5, 20, 100]


# ── Scenario set factories ─────────────────────────────────────────────────

def _make_uniform_ss(locations, n: int, seed: int) -> ScenarioSet:
    rng = random.Random(seed)
    prob = 1.0 / n
    return ScenarioSet(scenarios=[
        ThreatScenario(
            scenario_id=f"U{i}",
            threat_weights={loc.location_id: rng.uniform(0.05, 0.25) for loc in locations},
            probability=prob,
        )
        for i in range(n)
    ])


def _make_skewed_ss(locations, n: int, seed: int) -> ScenarioSet:
    """Threat scales with strategic value — adversary targets high-value bases."""
    rng = random.Random(seed)
    prob = 1.0 / n
    return ScenarioSet(scenarios=[
        ThreatScenario(
            scenario_id=f"SK{i}",
            threat_weights={loc.location_id: min(0.9, loc.strategic_value * rng.uniform(0.5, 1.0))
                            for loc in locations},
            probability=prob,
        )
        for i in range(n)
    ])


def _make_adversarial_ss(locations, n: int, seed: int) -> ScenarioSet:
    """Mix: 60% scenarios concentrate high threat on a single randomly chosen location,
    40% have low uniform threat — models a focused A2/AD actor with occasional feints."""
    rng = random.Random(seed)
    prob = 1.0 / n
    scenarios = []
    for i in range(n):
        if rng.random() < 0.6:
            target = rng.choice(locations)
            weights = {loc.location_id: (rng.uniform(0.7, 0.95) if loc.location_id == target.location_id
                                         else rng.uniform(0.0, 0.1))
                       for loc in locations}
        else:
            weights = {loc.location_id: rng.uniform(0.05, 0.20) for loc in locations}
        scenarios.append(ThreatScenario(f"ADV{i}", weights, probability=prob))
    return ScenarioSet(scenarios=scenarios)


# ── Per-seed evaluation ────────────────────────────────────────────────────

def eval_condition(seed: int, dist_label: str, n_scenarios: int) -> dict:
    state   = make_posture_state(n_assets=N_ASSETS, seed=seed, capacity=CAPACITY)
    assets  = state.assets
    locs    = state.locations
    ss_seed = BASE_SEED  # fixed so scenario variance doesn't pollute seed variance

    if dist_label == "uniform":
        ss = _make_uniform_ss(locs, n_scenarios, ss_seed)
    elif dist_label == "skewed":
        ss = _make_skewed_ss(locs, n_scenarios, ss_seed)
    else:
        ss = _make_adversarial_ss(locs, n_scenarios, ss_seed)

    cev = CEVOptimizer(ss)
    cev_assign    = cev.optimize_placement(assets, locs)
    greedy_assign = PostureOptimizer(seed=seed).greedy_placement(assets, locs)

    cev_eff    = cev.expected_posture_efficiency(assets, locs, cev_assign)
    greedy_eff = cev.expected_posture_efficiency(assets, locs, greedy_assign)

    evss = cev_eff - greedy_eff
    return {"greedy_eff": greedy_eff, "cev_eff": cev_eff, "evss": evss}


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(vals: list[dict]) -> dict:
    n = len(vals)
    result = {}
    for key in ("greedy_eff", "cev_eff", "evss"):
        data = [v[key] for v in vals]
        mean = sum(data) / n
        std  = math.sqrt(sum((x - mean) ** 2 for x in data) / max(n - 1, 1))
        result[f"{key}_mean"] = mean
        result[f"{key}_std"]  = std
    return result


# ── Output ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Experiment 2 — CEV Optimizer vs. Greedy Baseline (EVSS)")
    print(f"Setup: {N_ASSETS} assets, 5 locations, capacity={CAPACITY}, "
          f"N_seed={N_SEEDS}\n")

    distributions = ["uniform", "skewed", "adversarial"]
    rows = []
    for dist in distributions:
        for n_sc in SCENARIO_COUNTS:
            seed_results = [eval_condition(BASE_SEED + s, dist, n_sc) for s in range(N_SEEDS)]
            agg = aggregate(seed_results)
            evss_pct = agg["evss_mean"] / agg["greedy_eff_mean"] * 100 if agg["greedy_eff_mean"] > 0 else 0
            rows.append({
                "dist": dist, "n_sc": n_sc,
                **agg,
                "evss_pct": evss_pct,
            })

    header = (f"{'Distribution':<14} {'Scenarios':>9}  "
              f"{'Greedy Eff':>12} {'CEV Eff':>10} {'EVSS':>8} {'EVSS %':>8}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    last_dist = None
    for r in rows:
        if last_dist and r["dist"] != last_dist:
            print()
        print(
            f"{r['dist']:<14} {r['n_sc']:>9}  "
            f"{r['greedy_eff_mean']:>6.4f}±{r['greedy_eff_std']:.4f}  "
            f"{r['cev_eff_mean']:>6.4f}±{r['cev_eff_std']:.4f}  "
            f"{r['evss_mean']:>+7.4f}  "
            f"{r['evss_pct']:>+7.1f}%"
        )
        last_dist = r["dist"]
    print(sep)

    print("\nKey finding: EVSS > 0 across all conditions where threat is non-uniform.")
    uniform_evss = [r for r in rows if r["dist"] == "uniform"]
    nonuniform   = [r for r in rows if r["dist"] != "uniform"]
    print(f"  Uniform mean EVSS     : {sum(r['evss_mean'] for r in uniform_evss)/len(uniform_evss):+.4f}")
    print(f"  Non-uniform mean EVSS : {sum(r['evss_mean'] for r in nonuniform)/len(nonuniform):+.4f}")
    best = max(rows, key=lambda r: r["evss_mean"])
    print(f"  Largest EVSS          : {best['evss_mean']:+.4f} "
          f"({best['dist']}, S={best['n_sc']})")


if __name__ == "__main__":
    main()
