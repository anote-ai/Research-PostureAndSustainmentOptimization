"""Experiment 3 — Adversarial Robustness (Bayesian Counter-Move Model).

Sweeps observation probability p_obs ∈ {0, 0.25, 0.5, 0.75, 1.0} across four
threat distributions and two adversary rationality modes.

Metrics reported per row:
  - naive_eff   : CEV optimized on prior, evaluated under adversarially-updated dist
  - robust_eff  : RobustCEV iterated to convergence, evaluated under its adv dist
  - regret      : robust_eff − naive_eff  (positive = robust wins)

Key findings the table should surface:
  - Random adversary (rationality=0): regret always 0; blend=p_obs*0=0 so the
    adversary never updates.  This is the theoretical null result.
  - Uniform/skewed distributions: CEV already avoids the threatened locations,
    so the adversary has nothing to exploit.  Regret ≈ 0 (expected).
  - Deceptive distribution: naive CEV is lured into concentrating at the highest-
    value location by a mostly-safe prior.  At high p_obs the rational adversary
    collapses naive efficiency.  Robust CEV iterates away from the lure.

Run:
    python scripts/experiment3_adversarial.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.data import (
    make_adversarial_scenario_set,
    make_deceptive_scenario_set,
    make_posture_state,
    make_skewed_scenario_set,
    make_uniform_scenario_set,
)
from postureopt.drsp import (
    AdversarialModel,
    CEVOptimizer,
    RobustCEVOptimizer,
    ScenarioSet,
)

P_OBS_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
RATIONALITY_MODES = [
    ("random",   0.0),
    ("Bayesian", 1.0),
]
N_ASSETS = 20
SEED = 42


def _eval_naive(assets, locations, prior: ScenarioSet, adversary: AdversarialModel) -> float:
    assignment = CEVOptimizer(prior).optimize_placement(assets, locations)
    adv_ss = adversary.update_scenarios(prior, assignment, locations)
    return CEVOptimizer(adv_ss).expected_posture_efficiency(assets, locations, assignment)


def _eval_robust(assets, locations, prior: ScenarioSet, adversary: AdversarialModel) -> float:
    robust = RobustCEVOptimizer(prior, adversary)
    assignment, adv_ss = robust.optimize_placement(assets, locations)
    return robust.expected_posture_efficiency(assets, locations, assignment, adv_ss)


def run_sweep(label: str, prior: ScenarioSet, assets, locations) -> list[dict]:
    rows = []
    for rat_label, rationality in RATIONALITY_MODES:
        for p_obs in P_OBS_VALUES:
            adversary = AdversarialModel(p_obs=p_obs, rationality=rationality)
            naive_eff  = _eval_naive(assets, locations, prior, adversary)
            robust_eff = _eval_robust(assets, locations, prior, adversary)
            rows.append({
                "distribution": label,
                "rationality":  rat_label,
                "p_obs":        p_obs,
                "naive_eff":    naive_eff,
                "robust_eff":   robust_eff,
                "regret":       robust_eff - naive_eff,
            })
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Distribution':<14} {'Rationality':<10} {'p_obs':>6}  "
        f"{'Naive Eff':>10} {'Robust Eff':>11} {'Regret':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    last_key = None
    for r in rows:
        key = (r["distribution"], r["rationality"])
        if last_key and key != last_key:
            print()
        print(
            f"{r['distribution']:<14} {r['rationality']:<10} {r['p_obs']:>6.2f}  "
            f"{r['naive_eff']:>10.4f} {r['robust_eff']:>11.4f} {r['regret']:>+8.4f}"
        )
        last_key = key
    print(sep)


def print_summary(rows: list[dict]) -> None:
    print("\nInterpretation:")

    # Random adversary
    random_rows = [r for r in rows if r["rationality"] == "random"]
    if all(abs(r["regret"]) < 1e-9 for r in random_rows):
        print("  [random adversary]  Regret=0 for all p_obs — expected. "
              "rationality=0 → blend=0, adversary never updates.")

    # Uniform / skewed under Bayesian
    for dist in ("uniform", "skewed"):
        dist_rows = [r for r in rows if r["distribution"] == dist and r["rationality"] == "Bayesian"]
        max_regret = max(abs(r["regret"]) for r in dist_rows)
        if max_regret < 0.005:
            print(f"  [{dist:<12}]  Regret≈0 — CEV already avoids the threatened "
                  "location; adversary finds no exploitable concentration.")

    # Deceptive under Bayesian — the key result
    dec_rows = [r for r in rows if r["distribution"] == "deceptive" and r["rationality"] == "Bayesian"]
    max_row = max(dec_rows, key=lambda r: r["regret"])
    min_row = min(dec_rows, key=lambda r: r["regret"])
    if max_row["regret"] > 0.001:
        print(f"  [deceptive]         Robust beats naive: max regret={max_row['regret']:+.4f} "
              f"at p_obs={max_row['p_obs']:.2f}.")
    if min_row["regret"] < -0.001:
        print(f"  [deceptive]         NOTE: at p_obs={min_row['p_obs']:.2f} robust converges to "
              f"a worse equilibrium (regret={min_row['regret']:+.4f}). "
              "Iteration may not converge at high obs.")

    # Adversarial under Bayesian
    adv_rows = [r for r in rows if r["distribution"] == "adversarial" and r["rationality"] == "Bayesian"]
    notable = [r for r in adv_rows if abs(r["regret"]) > 0.001]
    if notable:
        for r in notable:
            print(f"  [adversarial]       p_obs={r['p_obs']:.2f}: regret={r['regret']:+.4f}")


def main() -> None:
    state = make_posture_state(n_assets=N_ASSETS, seed=SEED)
    assets, locations = state.assets, state.locations

    distributions = [
        ("uniform",     make_uniform_scenario_set(locations)),
        ("skewed",      make_skewed_scenario_set(locations)),
        ("adversarial", make_adversarial_scenario_set(locations)),
        ("deceptive",   make_deceptive_scenario_set(locations)),
    ]

    all_rows: list[dict] = []
    for label, prior in distributions:
        all_rows.extend(run_sweep(label, prior, assets, locations))

    print(f"\nExperiment 3 — Adversarial Robustness")
    print(f"Setup: {N_ASSETS} assets, {len(locations)} locations, seed={SEED}\n")
    print_table(all_rows)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
