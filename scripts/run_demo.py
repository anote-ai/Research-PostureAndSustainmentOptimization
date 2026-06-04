"""Demo: posture simulation and summary."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.data import make_posture_state, simulate_degradation
from postureopt.evaluate import simulation_summary, readiness_score


def main() -> None:
    print("Creating initial posture state...")
    state = make_posture_state(n_assets=20, seed=42)
    print(f"Initial readiness: {state.total_readiness():.3f}")

    print("Running 5-step simulation...")
    history = simulate_degradation(state, n_steps=5, seed=42)

    summary = simulation_summary(history)
    print(f"Simulation summary: {summary}")
    print(f"Final readiness: {summary['final_readiness']:.3f}")


if __name__ == "__main__":
    main()
