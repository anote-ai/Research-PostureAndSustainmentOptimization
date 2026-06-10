# PostureAndSustainmentOptimization — Research Design Document

## Goal

Develop and evaluate a distributionally robust optimization framework for joint force posture and sustainment planning in contested (A2/AD) environments — demonstrating that robust solutions outperform deterministic baselines on out-of-sample scenarios, and that deep RL can solve the recourse problem faster than branch-and-bound at operational scale.

## Objective

1. Formulate the force posture problem as a two-stage distributionally robust stochastic program (DRSP) with adversarial interdiction of sustainment routes
2. Implement and compare 3 solution approaches: deterministic LP, stochastic programming, and DRSP + RL recourse
3. Demonstrate on publicly-available synthetic theater scenarios that DRSP produces better worst-case performance and out-of-sample robustness than deterministic planning

## Background / Motivation

Force posture decisions are made under deep uncertainty about adversary behavior, weather, and terrain access. Current planning tools use deterministic optimization (optimize against a single "most likely" scenario) or simple stochastic programming. Neither is robust to adversarial scenario selection — a sophisticated adversary will probe exactly the scenarios the plan is weakest against.

Distributionally robust optimization (DRO) explicitly hedges against the worst case over a family of plausible distributions — the right model for contested-environment planning.

## Experimental Design

### Baseline Experiment

**Solve 10 synthetic theater posture instances using deterministic LP (optimize against the mean scenario)**

- Metric: total logistics cost; feasibility rate when scenarios other than the mean are realized
- Purpose: establish the deterministic planning baseline and confirm that mean-scenario optimization is brittle
- Expected result: deterministic solutions are infeasible or very costly on 30–50% of off-mean scenarios

### Test Experiment 1: DRSP vs. Stochastic Programming

Solve 10 instances with 3 methods: (1) deterministic LP, (2) two-stage stochastic program with SAA (100 scenarios), (3) two-stage DRSP with Wasserstein-ball uncertainty set. Metrics: in-sample objective, out-of-sample worst-case cost (across 1000 held-out scenarios), solution time.

**Expected result:** DRSP produces 15–25% lower worst-case cost than SAA at 5–10% higher in-sample cost — the price of robustness

### Test Experiment 2: Deep RL for Recourse Decisions

Train a deep RL agent to solve the recourse problem (real-time sustainment flow allocation). Compare RL vs. branch-and-bound on: solution quality (% of optimal), solution time (ms).

**Expected result:** RL achieves 95%+ of B&B optimality in 100x less time — sufficient for operational real-time planning

### Test Experiment 3: Out-of-Sample Robustness Generalization

Train/optimize all methods on distribution A; test on distribution B (different adversary strategy, different access constraints). Measure which method's solutions generalize best.

**Expected result:** DRSP + RL recourse outperforms both LP and SAA on out-of-sample robustness, especially under adversarially-selected scenarios

## Expected Results

1. A publicly-available synthetic theater scenario generator and 20+ benchmark instances
2. Comparison table: deterministic LP vs. SAA vs. DRSP on in-sample cost, worst-case cost, and solution time
3. RL recourse policy: 95%+ of optimal quality in 100x less time than B&B
4. **Key finding:** "Distributionally robust posture planning reduces worst-case sustainment cost by 20% vs. mean-scenario optimization — at only 8% in-sample cost premium"
5. Practical recommendation: when to use DRSP vs. SAA, and when RL recourse is worth the training cost

## Why This Matters / Why People Would Care

- **Defense planners and combatant commands:** worst-case robustness improvement is directly operationally relevant
- **RAND, CNA, and TRAC:** defense OR analysts will want to evaluate and extend this framework
- **INFORMS/OR community:** DRSP with RL recourse is methodologically novel; logistics under adversarial disruption is broadly applicable
- **Logistics AI broadly:** the methodology applies to supply chain optimization under adversarial disruption (port strikes, natural disasters)

## Timeline

| Month | Milestone |
|---|---|
| 1 | Problem formulation + scenario generator implementation |
| 2 | Deterministic LP and SAA baselines |
| 3 | DRSP implementation and comparison |
| 4 | Deep RL recourse policy training + evaluation |
| 5 | Out-of-sample robustness experiments + Monte Carlo CIs |
| 6 | Submission to Operations Research (INFORMS) or MORS Symposium |

## Related Issues

- Design doc GitHub issue: #19
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
