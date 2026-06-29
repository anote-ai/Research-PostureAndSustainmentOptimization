# Can AI Place Military Assets More Wisely Than a Simple Rule of Thumb?

*A plain-language summary of the Posture and Sustainment Optimization research project.*

## The problem in one sentence

If you have a limited number of planes, fuel depots, repair crews, and supplies, and a handful of bases to put them at, where should you put them — and how should you keep them running — so that you stay "ready" without overspending, even when something unexpected happens?

This is the **posture and sustainment** problem the military logistics community has wrestled with for decades. This project asks a narrower, testable version of it: does a smarter, threat-aware placement algorithm actually beat the obvious "put your best stuff at your best bases" strategy?

## The setup

The team built a small simulated world: 20 assets (aircraft, fuel depots, maintenance crews, munitions, medical units), 5 real-world-style bases (e.g., Kadena, Andersen, Diego Garcia) with limited capacity, and a 10-step timeline over which everything slowly degrades and has to be resupplied or maintained.

Three placement strategies were compared:

1. **Greedy** — always send assets to the highest-value base that still has room.
2. **Random** — spread assets across bases with no strategy at all.
3. **Scenario-weighted (CEV) and adversarially-robust optimizers** — place assets while accounting for a range of possible threat patterns, including a simulated adversary that reacts to where you put things.

## What was actually measured (not just hoped for)

Running the greedy-vs-random comparison for real (10 random seeds, with confidence intervals) turned up a genuinely useful, slightly counterintuitive result:

- **Readiness ends up the same either way.** Whether you place assets greedily or randomly, the day-to-day maintenance/resupply routine equalizes asset readiness by step 10 (~0.57 in both cases). Placement doesn't change how "fixed up" your gear is.
- **But greedy is worse on overall efficiency.** Because greedy always fills the four best bases and leaves the worst one empty, it gets only 80% geographic coverage versus random's 100%. That gap alone makes greedy **25% less efficient** on the project's composite "posture efficiency" score.
- **Greedy has a hidden, structural weakness.** If an adversary specifically targets high-value bases (the kind greedy loves to fill), the assets concentrated there see their effective readiness collapse by **57%** compared to a world with uniformly spread-out threats. This isn't noise — it shows up identically across every simulated run and every time step, because it's baked into *where* the assets are sitting, not into bad luck.

In short: a simple "put the best stuff in the best spot" rule looks reasonable but quietly creates a single point of failure. It also wastes coverage of cheaper, lower-value sites for no readiness benefit.

## What's still aspirational

The project's original design doc set out a much bigger goal: build a 200-scenario benchmark ("SustainBench") and rigorously compare distributionally robust stochastic programming, sample-average approximation, and reinforcement learning against classical optimization, with claims like "20% lower cost than plain linear programming." That larger comparison — including the RL recourse agent and the full SustainBench dataset — has not been built yet. What exists today is a smaller, real, working simulation (greedy / random / scenario-weighted / adversarial placement) with one solid, measured experiment (Experiment 1) and partially-built follow-on experiment scripts whose outputs have not yet been run end-to-end and verified.

## Why it matters anyway

Even at this smaller scale, the finding is real and useful: optimizing for "what looks valuable" without thinking about coverage or adversarial behavior can leave you both less efficient and more exposed. That is a genuinely actionable insight for anyone designing a placement heuristic, even before the bigger benchmark is built.

## Where this is headed

The next steps are to actually run and verify Experiments 2–4 (scenario-weighted optimizer vs. greedy, the adversarial Bayesian counter-move model, and the cost/robustness tradeoffs at larger problem sizes), record real measured numbers for each, and only then write the full paper's results section. Until those experiments are run and their outputs captured in `results/`, any larger claims should be treated as hypotheses, not findings.
