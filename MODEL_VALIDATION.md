# Model Validation Guide

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Validation Tests

```bash
python3 prove_model.py
```

This will:
1. Run all mathematical validation tests
2. Execute a full Pod simulation (20 cycles)
3. Generate visualization plots
4. Export results to CSV

## What Gets Validated

### Test 1: Foresight Efficiency (A1)
Validates: \(F_t = \frac{V_t}{T_t \cdot C_t}\)

### Test 2: Trust-Leverage Relationship (A3)
Validates: \(L_t = L_0(1 + k(\tau_t - \tau_{eq}))\)

### Test 3: Network Relativity Principle (A4)
Validates: \(\frac{T_B}{T_A} = \frac{1}{1 - v_{rel}/c_t}\)

### Test 4: Stability Conditions (A8)
Checks convergence requirements:
- \(\alpha_t + \beta_t > \gamma_t\)
- \(\tau_t > \tau_{eq}\)
- \(F_t > F_{min}\)

### Test 5: Recursive Compounding (A2)
Validates: \(F_{t+1} = F_t(1 + \lambda(\alpha_t + \beta_t - \gamma_t))\)

### Test 6: Full Simulation
Runs complete Pod evolution over multiple cycles and validates:
- Temporal Alpha generation
- Pod Coherence metrics
- Time Violence scoring
- Total Compounded Understanding

## Output Files

- `pod_simulation_results.png` - Visualization plots
- `pod_simulation_results.csv` - Time series data

## Custom Simulations

```python
from prove_model import PodState, simulate_pod

# Create custom initial state
initial_state = PodState(
    t=0,
    V_t=200.0,      # Initial value
    T_t=8.0,        # Initial verification time
    C_t=2000.0,     # Initial capital
    alpha_t=0.15,   # Initial temporal alpha
    beta_t=0.25,    # Initial trust reinforcement
    gamma_t=0.03,   # Initial entropy
    tau_t=0.7,      # Initial trust
    lambda_val=0.12,  # Learning rate
    L_0=1.2,        # Baseline leverage
    k=2.5,          # Sensitivity constant
    tau_eq=0.55     # Equilibrium trust threshold
)

# Run simulation
model, df = simulate_pod(n_cycles=50, initial_state=initial_state)

# Access results
print(f"Final Foresight Efficiency: {df['F_t'].iloc[-1]}")
print(f"Total Compounded Understanding: {model.total_compounded_understanding(len(df))}")
```

## Key Metrics Explained

- **F_t (Foresight Efficiency)**: Value created per unit time and capital
- **L_t (Leverage)**: Trust-weighted scaling capacity
- **α_t (Temporal Alpha)**: Performance beyond collective time expectation
- **Φ_t (Coherence)**: Alignment between Pod subsystems
- **TV_t (Time Violence)**: Operational entropy score
- **U_n (Compounded Understanding)**: Total capacity growth over n cycles

## Model Assumptions

1. Trust changes proportionally to performance feedback
2. Entropy (γ_t) can increase or decrease based on coordination
3. Verification time decreases with better foresight
4. Capital increases with trust and efficiency
5. Beta (trust reinforcement) is proportional to trust level

These assumptions can be adjusted in the `evolve()` method for different scenarios.

