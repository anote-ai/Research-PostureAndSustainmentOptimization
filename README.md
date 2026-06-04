# PostureOpt: Posture & Sustainment Optimization

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Decision framework and multi-agent RL formulation for optimal asset positioning and sustainment, tied to the **Air Force DASH-5 Posture and Sustainment Problem**.

## DASH-5 Problem Context

The Air Force DASH-5 problem addresses a fundamental operational research challenge:

> *Given M assets of K types and N candidate locations, determine the optimal placement policy and replenishment strategy to maximize force readiness under resource constraints.*

Key considerations:
- **Where** to pre-position assets (forward vs. rear basing)
- **How many** of each type at each location
- **When and how** to replenish/reposition based on consumption and threat
- **Sustainment costs** vs. operational readiness trade-offs

## Problem Formulation

### Decision Variables

| Variable | Description |
|----------|-------------|
| x[i,j] | Quantity of asset type i at location j |
| a[i,t] | Sustainment action for asset i at time t |
| r[j,t] | Readiness rate at location j at time t |

### Objective

Maximize: `sum(readiness[j] * priority[j] for j in locations)` subject to:
- Capacity constraints: `sum(x[i,j]) <= capacity[j]`
- Budget constraints: `sum(cost(a[i,t])) <= budget`
- Replenishment lead times

## Optimization Approach

1. **Combinatorial baseline**: Greedy placement by location priority score
2. **Multi-agent RL**: Each asset type is an agent; locations are environment nodes
3. **Uncertainty quantification**: Monte Carlo rollouts over demand scenarios

## Metrics

| Metric | Description |
|--------|-------------|
| Readiness Score | Weighted mean readiness rate across assets |
| Coverage Score | Fraction of locations with ≥1 asset |
| Sustainment Cost | Sum of action costs (REPOSITION=10, RESUPPLY=5, MAINTAIN=2, HOLD=0) |
| Posture Efficiency | (readiness × coverage) / log1p(cost) |

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
from postureopt.core import Asset, AssetType, Location, PostureOptimizer
from postureopt.evaluate import readiness_score, coverage_score, posture_efficiency

locations = [
    Location("LOC01", "Ramstein AB", lat=49.44, lon=7.60, capacity=20),
    Location("LOC02", "Al Udeid AB", lat=25.12, lon=51.31, capacity=15),
]
assets = [
    Asset("A001", AssetType.AIRCRAFT, "LOC01", quantity=8, readiness_rate=0.90),
    Asset("A002", AssetType.FUEL_DEPOT, "LOC02", quantity=4, readiness_rate=0.75),
]

opt = PostureOptimizer()
placement = opt.optimize_placement(assets, locations, demand={"LOC01": 2, "LOC02": 1})

print(f"Readiness: {readiness_score(assets):.2f}")
print(f"Coverage: {coverage_score(assets, locations):.2f}")
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Government Use Disclaimer

This repository is research software developed for educational and research purposes. It is not affiliated with, endorsed by, or intended for direct operational use by the United States Air Force or Department of Defense.

## Citation

```bibtex
@misc{anoteai2025postureopt,
  title        = {PostureOpt: Posture and Sustainment Optimization Framework},
  author       = {Anote AI},
  year         = {2025},
  howpublished = {\url{https://github.com/anote-ai/research-postureandsustainmentoptimization}},
  note         = {Air Force DASH-5 Research}
}
```

## License

Apache 2.0
