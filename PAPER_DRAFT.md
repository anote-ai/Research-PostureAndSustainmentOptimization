# Posture and Sustainment Optimization: Paper Draft Skeleton

> Status note: This is a skeleton/outline that consolidates what is *actually* implemented and measured in this repository, distinguished explicitly from what DESIGN_DOC.md envisions but has not yet been built or run. Numbers marked **(measured)** come directly from `results/tables/exp1_metrics.tex` and `paper/main.tex`, which are the only files in the repo containing real experimental output. Numbers for Experiments 2–4 are marked **(TBD — requires running experiment script and capturing output to results/)**.

## Abstract

We study placement and sustainment policies for distributed military assets under uncertainty. We implement and empirically characterize a greedy value-maximizing placement baseline, a random placement baseline, a scenario-weighted ("Composite Expected Value", CEV) optimizer, and an adversarially-robust optimizer that models a Bayesian-updating adversary. Using a 20-asset, 5-location simulation environment with stochastic degradation and rule-based replenishment, we find (measured, Experiment 1) that greedy placement achieves no readiness advantage over random placement (both converge to readiness 0.570 ± 0.041 by step 10) but is **25.1% less efficient** on a composite posture-efficiency metric due to incomplete geographic coverage (0.80 vs. 1.00), and is structurally vulnerable to value-correlated adversarial threats, suffering a constant **57.3% reduction** in scenario-weighted readiness under a skewed threat distribution relative to a uniform one. Experiments 2–4 (scenario-weighted optimizer vs. greedy under varying threat distributions, adversarial Bayesian counter-move convergence, and cost/robustness tradeoffs at scale) have runnable scripts in this repository but have not yet been executed end-to-end with results captured; we report their hypotheses and designed protocols only, pending verified runs. This is a substantially narrower scope than the original design document's vision of a 200-scenario "SustainBench" benchmark comparing DRSP, SAA, and RL recourse policies against commercial-solver baselines, which remains unimplemented.

## 1. Introduction

Military posture and sustainment planning requires deciding where to place limited assets (aircraft, fuel, maintenance crews, munitions, medical units) across bases, and how to schedule maintenance/resupply actions over time, all under uncertainty about future threats and demand. The original design for this project (`DESIGN_DOC.md`) proposed a large benchmark, "SustainBench" (200 scenarios across 5 categories), to rigorously compare deterministic linear programming, sample-average approximation (SAA), distributionally robust stochastic programming (DRSP), and reinforcement-learning-based recourse policies, targeting a ≥20% out-of-sample cost improvement claim suitable for an INFORMS/MORS submission.

What has actually been implemented (`src/postureopt/`) is a smaller but real and runnable simulation: a greedy placement policy, a random baseline, a scenario-weighted CEV optimizer, and an adversarial Bayesian counter-move model (`drsp.py`), evaluated via readiness, coverage, sustainment cost, posture efficiency, and scenario-weighted readiness (SWR) metrics. This draft documents what has been measured (Experiment 1 only, as written up in `paper/main.tex`) and what remains to be run (Experiments 2–4).

## 2. Methods

### 2.1 Environment
20 assets across 5 named Indo-Pacific bases, location capacity 5, strategic values in [0.78, 0.95], per-step degradation rate 0.08, 10-step horizon, 10 random seeds. (Implemented: `src/postureopt/data.py`, `core.py`.)

### 2.2 Policies under comparison
- **Greedy placement** (`PostureOptimizer.greedy_placement`): assigns assets to the highest-value location with remaining capacity. Implemented and measured.
- **Random placement**: uniform random assignment subject to capacity. Implemented and measured.
- **Scenario-weighted optimizer (CEV)** (`ScenarioWeightedOptimizer`): places assets accounting for a weighted set of threat scenarios. Implemented; comparison script exists (`scripts/experiment2.py`) but has not been run with captured output.
- **Adversarially-robust optimizer** (`drsp.py: RobustCEVOptimizer` + `AdversarialModel`): iterates placement against a Bayesian-updating adversary that shifts scenario weight toward whichever threat pattern most damages the observed placement. Implemented; no captured experiment output yet (`scripts/experiment3.py`, `experiment4.py`).
- **DRSP (distributionally robust stochastic programming, per the original design doc's LP/SAA/DRSP formulation), SAA, Robust LP, and RL recourse (constrained PPO)**: described in `DESIGN_DOC.md` but **not implemented** in `src/postureopt/`. No solver code (Gurobi/CPLEX), no RL training loop, and no SustainBench dataset of 200 scenarios exists in this repository as of this audit.

### 2.3 Metrics
Readiness score, coverage score, sustainment cost, posture efficiency (= readiness × coverage / log(2+cost)), and scenario-weighted readiness (SWR) — all implemented in `src/postureopt/evaluate.py` and unit-tested in `tests/test_evaluate.py`.

## 3. Results

### 3.1 Experiment 1 — Greedy Baseline Characterization (measured)

From `results/tables/exp1_metrics.tex` (10 seeds, 95% CI):

| Step | Readiness | Coverage | Sustainment Cost | Posture Efficiency |
|---|---|---|---|---|
| 0  | 0.712 (0.671, 0.753) | 0.80 | 13.10 (8.78, 17.42) | 0.221 |
| 5  | 0.554 (0.526, 0.582) | 0.80 | 5.20 (3.27, 7.13)   | 0.239 |
| 10 | 0.570 (0.541, 0.599) | 0.80 | 6.80 (4.54, 9.06)   | 0.222 |

Key measured findings (from `paper/main.tex`):
- Greedy and random placement converge to **identical readiness** (0.570 ± 0.041) by step 10 — placement does not affect readiness under the rule-based ReplenishmentPolicy.
- Random placement achieves **25.1% higher posture efficiency** than greedy (0.278 vs. 0.222 at step 10), entirely attributable to full coverage (1.00 vs. 0.80).
- Under a skewed (value-correlated) threat distribution, greedy's scenario-weighted readiness is **57.3% lower** than under a uniform threat distribution — a gap that is constant across all 10 time steps and all 10 seeds, indicating a structural (not statistical) vulnerability.

### 3.2 Experiment 2 — CEV vs. Greedy (TBD — requires running `scripts/experiment2.py` and capturing output to `results/`)

Script is implemented and sweeps threat distribution (uniform/skewed/adversarial) × scenario count (5/20/100) × weight distribution (uniform/peaked), computing Expected Value of the Stochastic Solution (EVSS = CEV SWR − greedy SWR). No run has been captured; hypothesis (per design doc framing in the script's docstring) is that EVSS > 0, with the largest gains under adversarial/peaked-weight conditions.

### 3.3 Experiment 3 — Adversarial Robustness (TBD — requires running `scripts/experiment3.py`)

Tests convergence of `RobustCEVOptimizer` against a Bayesian counter-move adversary at varying observation probability and rationality. Implemented; not yet run with captured results.

### 3.4 Experiment 4 — Scale / distribution shift (TBD — requires running `scripts/experiment4.py`)

Script exists; no captured results. The original design doc's specific numeric projections (e.g., "DRSP ROE 0.76", "DRSP solve time >12hr at 500 vars") refer to a DRSP/SAA/LP solver stack that is not implemented here and should not be cited as results of this codebase.

## 4. Discussion

The one experiment that has been run end-to-end with real measured data (Experiment 1) already yields an actionable, non-trivial finding: naive value-maximizing placement is both less efficient than random placement and structurally exposed to adversaries who target high-value sites. This motivates — but does not yet validate — the scenario-weighted and adversarially-robust optimizers that are implemented in code but not yet benchmarked.

The gap between `DESIGN_DOC.md`'s vision (SustainBench, DRSP vs. SAA vs. RL, INFORMS-grade claims) and the current implementation is large: there is no stochastic-programming solver, no RL recourse agent, and no 200-scenario benchmark dataset. Any paper submission should either (a) scope the contribution down to the implemented greedy/CEV/adversarial-robust comparison, reporting only measured results, or (b) treat Experiments 2-4 and the full DRSP/SAA/RL comparison as future work requiring substantial additional implementation.

## 5. Limitations

- Single environment size (20 assets, 5 locations) — no scaling experiments have been run.
- Experiments 2–4 scripts are unexecuted; no results files exist for them in `results/`.
- DRSP, SAA, Robust LP, and RL recourse policy from the design doc are not implemented.
- No human expert evaluation component exists, despite being listed in the design doc's evaluation protocol.

## 6. Reproducibility

All Experiment 1 numbers can be reproduced via `python scripts/experiment1.py` (see `README.md` for setup). Experiments 2–4 can be run via `python scripts/experiment2.py` / `experiment3.py` / `experiment4.py`, but their output has not yet been captured into `results/` as of this draft.
