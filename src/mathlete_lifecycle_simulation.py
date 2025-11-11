"""
Mathlete Token Lifecycle Simulation

Simulates the evolution of a Mathlete Token (MLT) over time using the
Temporal Compounding Model from the Proximity to the Future Fund.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prove_model import PodState, TemporalCompoundingModel
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def simulate_mathlete_token_lifecycle(n_epochs: int = 10) -> pd.DataFrame:
    """
    Simulate MLT-001 lifecycle based on Appendix A specifications.
    
    Args:
        n_epochs: Number of epochs to simulate
    
    Returns:
        DataFrame with token evolution over time
    """
    # A1. Initial Conditions
    initial_state = PodState(
        t=0,
        V_t=100.0,      # Verified value creation
        T_t=10.0,       # Verification time (days)
        C_t=1000.0,     # Capital deployed (relative)
        alpha_t=0.10,   # Temporal alpha
        beta_t=0.20,    # Trust reinforcement
        gamma_t=0.05,   # Entropy rate
        tau_t=0.60,     # Trust level
        lambda_val=0.1, # Learning rate
        L_0=1.0,        # Leverage baseline
        k=2.0,          # Sensitivity constant
        tau_eq=0.5      # Trust equilibrium
    )
    
    model = TemporalCompoundingModel(initial_state)
    results = []
    
    # Calculate initial metrics
    F_t = model.foresight_efficiency(initial_state)
    L_t = model.trust_leverage(initial_state)
    U_n = 1.0  # Initial compounded understanding
    TV_t = model.time_violence(initial_state)
    
    results.append({
        'epoch': 0,
        'alpha_t': initial_state.alpha_t,
        'beta_t': initial_state.beta_t,
        'gamma_t': initial_state.gamma_t,
        'tau_t': initial_state.tau_t,
        'F_t': F_t,
        'L_t': L_t,
        'U_n': U_n,
        'TV_t': TV_t,
        'V_t': initial_state.V_t,
        'T_t': initial_state.T_t,
        'C_t': initial_state.C_t
    })
    
    # Simulate evolution over epochs
    for epoch in range(1, n_epochs + 1):
        # Simulate validation events
        # Success rate increases with trust
        success_rate = 0.7 + (initial_state.tau_t + (epoch - 1) * 0.03) * 0.3
        success = np.random.random() < success_rate
        
        # Value creation increases with successful validations
        delta_V = np.random.normal(10, 2) if success else np.random.normal(5, 2)
        delta_T_network = np.random.normal(1, 0.2)
        
        # Trust improves with successful validations
        trust_change = np.random.normal(0.03, 0.01) if success else np.random.normal(-0.01, 0.01)
        
        # Entropy decreases with successful validations
        entropy_change = np.random.normal(-0.01, 0.005) if success else np.random.normal(0.01, 0.005)
        
        # Evolve state
        model.evolve(
            delta_V=delta_V,
            delta_T_network=delta_T_network,
            trust_change=trust_change,
            entropy_change=entropy_change
        )
        
        # Get current state
        state = model.state
        
        # Calculate metrics
        F_t = model.foresight_efficiency(state)
        L_t = model.trust_leverage(state)
        TV_t = model.time_violence(state)
        
        # Update compounded understanding
        # U_n = U_0 * ∏(1 + F_i * L_i)
        prev_U = results[-1]['U_n']
        U_n = prev_U * (1 + F_t * L_t)
        
        results.append({
            'epoch': epoch,
            'alpha_t': state.alpha_t,
            'beta_t': state.beta_t,
            'gamma_t': state.gamma_t,
            'tau_t': state.tau_t,
            'F_t': F_t,
            'L_t': L_t,
            'U_n': U_n,
            'TV_t': TV_t,
            'V_t': state.V_t,
            'T_t': state.T_t,
            'C_t': state.C_t
        })
    
    df = pd.DataFrame(results)
    return df


def plot_mathlete_lifecycle(df: pd.DataFrame, save_path: str = None):
    """
    Visualize Mathlete Token lifecycle evolution.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mathlete Token (MLT-001) Lifecycle Simulation', fontsize=16, y=1.02)
    
    # Plot 1: Foresight Efficiency (F_t)
    axes[0, 0].plot(df['epoch'], df['F_t'], 'b-o', linewidth=2, markersize=8, label='F_t')
    axes[0, 0].set_title('Foresight Efficiency (F_t)')
    axes[0, 0].set_xlabel('Epoch (t)')
    axes[0, 0].set_ylabel('Foresight Efficiency')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Trust Level (τ_t)
    axes[0, 1].plot(df['epoch'], df['tau_t'], 'g-o', linewidth=2, markersize=8, label='τ_t')
    axes[0, 1].axhline(y=df['tau_t'].iloc[0], color='r', linestyle='--', alpha=0.5, label='τ_eq')
    axes[0, 1].set_title('Trust Level (τ_t)')
    axes[0, 1].set_xlabel('Epoch (t)')
    axes[0, 1].set_ylabel('Trust Level')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Time Violence (TV_t)
    axes[0, 2].plot(df['epoch'], df['TV_t'], 'r-o', linewidth=2, markersize=8, label='TV_t')
    axes[0, 2].set_title('Time Violence (TV_t)')
    axes[0, 2].set_xlabel('Epoch (t)')
    axes[0, 2].set_ylabel('Time Violence')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Plot 4: Compounded Understanding (U_n)
    axes[1, 0].plot(df['epoch'], df['U_n'], 'purple', marker='o', linewidth=2, markersize=8, label='U_n')
    axes[1, 0].set_title('Compounded Understanding (U_n)')
    axes[1, 0].set_xlabel('Epoch (t)')
    axes[1, 0].set_ylabel('Compounded Understanding')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 5: Trust Leverage (L_t)
    axes[1, 1].plot(df['epoch'], df['L_t'], 'orange', marker='o', linewidth=2, markersize=8, label='L_t')
    axes[1, 1].set_title('Trust Leverage (L_t)')
    axes[1, 1].set_xlabel('Epoch (t)')
    axes[1, 1].set_ylabel('Trust Leverage')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Alpha, Beta, Gamma Components
    axes[1, 2].plot(df['epoch'], df['alpha_t'], 'b-', label='α_t', linewidth=2)
    axes[1, 2].plot(df['epoch'], df['beta_t'], 'g-', label='β_t', linewidth=2)
    axes[1, 2].plot(df['epoch'], df['gamma_t'], 'r-', label='γ_t', linewidth=2)
    axes[1, 2].set_title('Temporal Components (α, β, γ)')
    axes[1, 2].set_xlabel('Epoch (t)')
    axes[1, 2].set_ylabel('Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Lifecycle plot saved to {save_path}")
    else:
        plt.show()


def print_lifecycle_table(df: pd.DataFrame):
    """
    Print formatted lifecycle table matching Appendix A format.
    """
    print("\n" + "=" * 80)
    print("A3. SIMULATED PROGRESSION")
    print("=" * 80)
    print()
    print(f"{'Epoch (t)':<12} {'α_t':<8} {'β_t':<8} {'γ_t':<8} {'τ_t':<8} {'F_t':<10} {'L_t':<8} {'U_n':<10} {'TV_t':<8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{int(row['epoch']):<12} "
              f"{row['alpha_t']:<8.2f} "
              f"{row['beta_t']:<8.2f} "
              f"{row['gamma_t']:<8.2f} "
              f"{row['tau_t']:<8.2f} "
              f"{row['F_t']:<10.4f} "
              f"{row['L_t']:<8.2f} "
              f"{row['U_n']:<10.3f} "
              f"{row['TV_t']:<8.3f}")
    
    print()


def print_visualization_ascii(df: pd.DataFrame):
    """
    Print ASCII visualization matching Appendix A format.
    """
    print("\n" + "=" * 80)
    print("A5. VISUALIZATION")
    print("=" * 80)
    print()
    
    # Normalize values for visualization (0-8 scale)
    def normalize(values, min_val=None, max_val=None):
        if min_val is None:
            min_val = values.min()
        if max_val is None:
            max_val = values.max()
        if max_val == min_val:
            return [4] * len(values)
        normalized = ((values - min_val) / (max_val - min_val)) * 8
        return [int(max(0, min(8, v))) for v in normalized]
    
    # Unicode block characters
    blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    F_t_norm = normalize(df['F_t'])
    tau_t_norm = normalize(df['tau_t'])
    TV_t_norm = normalize(df['TV_t'], min_val=df['TV_t'].max(), max_val=df['TV_t'].min())  # Inverted
    U_n_norm = normalize(df['U_n'])
    
    print("Epoch →   ", end="")
    for i in range(len(df)):
        print(f"{int(df.iloc[i]['epoch']):>4}", end="")
    print()
    
    print("F_t     → ", end="")
    for v in F_t_norm:
        print(blocks[v], end="")
    print()
    
    print("τ_t     → ", end="")
    for v in tau_t_norm:
        print(blocks[v], end="")
    print()
    
    print("TV_t    → ", end="")
    for v in TV_t_norm:
        print(blocks[v], end="")
    print()
    
    print("U_n     → ", end="")
    for v in U_n_norm:
        print(blocks[v], end="")
    print()
    print()


def generate_mlt_metadata(df: pd.DataFrame, claim_hash: str = "0x5c3f2e") -> dict:
    """
    Generate MLT token metadata matching Appendix A format.
    
    Args:
        df: Lifecycle simulation results
        claim_hash: Claim hash identifier
    
    Returns:
        Dictionary with token metadata
    """
    final_row = df.iloc[-1]
    
    metadata = {
        "name": "Mathlete Token 001 — Temporal Compounding Simulation",
        "description": "Represents the evolving trust curve of the Proximity to the Future Fund model.",
        "simulation_seed": claim_hash,
        "alpha_ref": 15.0,
        "capital_max_growth": 0.5,
        "a4_soft_link_rho": 0.2,
        "F_t_final": round(final_row['F_t'], 4),
        "U_n_final": round(final_row['U_n'], 3),
        "TV_t_final": round(final_row['TV_t'], 3),
        "trust_tau_final": round(final_row['tau_t'], 2),
        "pod_signature": "sig_pod_zero_0xabc",
        "epoch_count": len(df) - 1
    }
    
    return metadata


if __name__ == "__main__":
    print("=" * 80)
    print("MATHLETE TOKEN LIFECYCLE SIMULATION")
    print("Appendix A — Simulation of a Mathlete Token Lifecycle")
    print("=" * 80)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # A1. Initial Conditions
    print("A1. INITIAL CONDITIONS")
    print("-" * 80)
    print("MLT-001: 'Verification time in decentralized networks scales inversely with trust velocity'")
    print()
    print("Initial State:")
    print("  V_0 = 100 (Information yield from initial work)")
    print("  T_0 = 10 (Days to replicate baseline)")
    print("  C_0 = 1000 (Research resources, relative)")
    print("  α_0 = 0.10 (Early foresight efficiency)")
    print("  β_0 = 0.20 (Peer support / credibility)")
    print("  γ_0 = 0.05 (Confusion / drift)")
    print("  τ_0 = 0.60 (Initial community trust)")
    print("  λ = 0.1 (Responsiveness of system)")
    print("  L_0 = 1.0 (Default multiplier)")
    print("  k = 2.0 (Trust leverage sensitivity)")
    print("  τ_eq = 0.5 (Neutral trust baseline)")
    print()
    
    # A2. Evolution Functions
    print("A2. EVOLUTION FUNCTIONS")
    print("-" * 80)
    print("1. Foresight Efficiency: F_t = V_t / (T_t * C_t)")
    print("2. Recursive Compounding: F_{t+1} = F_t(1 + λ(α_t + β_t - γ_t))")
    print("3. Trust-Leverage: L_t = L_0(1 + k(τ_t - τ_eq))")
    print("4. Compounded Understanding: U_n = U_0 ∏(1 + F_i * L_i)")
    print("5. Time Violence: TV_t = γ_t * T_t * (1 - τ_t)")
    print()
    
    # Simulate lifecycle
    print("Simulating MLT-001 lifecycle over 10 epochs...")
    df = simulate_mathlete_token_lifecycle(n_epochs=10)
    
    # Print table
    print_lifecycle_table(df)
    
    # Print ASCII visualization
    print_visualization_ascii(df)
    
    # A4. Interpretation
    print("=" * 80)
    print("A4. INTERPRETATION")
    print("=" * 80)
    print()
    print("1. Foresight Efficiency (F_t):")
    print(f"   Increases from {df['F_t'].iloc[0]:.4f} to {df['F_t'].iloc[-1]:.4f}")
    print("   Gradually increases as replication reduces uncertainty.")
    print("   Represents the epistemic yield per unit of time and capital.")
    print()
    print("2. Trust (τ_t):")
    print(f"   Increases from {df['tau_t'].iloc[0]:.2f} to {df['tau_t'].iloc[-1]:.2f}")
    print("   Moves toward unity as validators confirm the research.")
    print("   When τ_t > τ_eq, leverage L_t amplifies compounding.")
    print()
    print("3. Time Violence (TV_t):")
    print(f"   Decreases from {df['TV_t'].iloc[0]:.3f} to {df['TV_t'].iloc[-1]:.3f}")
    print("   Declines as the system approaches stability.")
    print("   Meaning less coordination waste per cycle.")
    print()
    print("4. Compounded Understanding (U_n):")
    print(f"   Grows from {df['U_n'].iloc[0]:.3f} to {df['U_n'].iloc[-1]:.3f}")
    coherence_gain = (df['U_n'].iloc[-1] - df['U_n'].iloc[0]) / df['U_n'].iloc[0] * 100
    print(f"   Net coherence gain: {coherence_gain:.1f}% from compounding trust and foresight.")
    print()
    
    # A6. Network Relativity
    print("=" * 80)
    print("A6. NETWORK RELATIVITY INTERPRETATION")
    print("=" * 80)
    print()
    print("If Pod B verifies faster than Pod A, their relative foresight velocity")
    print("(v_rel) introduces temporal dilation:")
    print()
    print("  T_B = T_A / (1 - v_rel / c_t)")
    print()
    print("If v_rel / c_t = 0.3, then:")
    print("  T_B ≈ 1.43 * T_A")
    print()
    print("This means Pod B experiences time dilation of verification—")
    print("it compresses uncertainty faster than its peers, giving it")
    print("predictive leverage across the network.")
    print()
    print("This is the Network Relativity Principle in epistemic space:")
    print("  'Faster verification warps collective time.'")
    print()
    
    # A7. Emergent Takeaways
    print("=" * 80)
    print("A7. EMERGENT TAKEAWAYS")
    print("=" * 80)
    print()
    print("Uncertainty = Fuel.")
    print("  Volatility in early research is productive when properly capitalized.")
    print()
    print("Trust = Compression.")
    print("  Each new validation collapses the network's information entropy.")
    print()
    print("Time Violence = Drag.")
    print("  The cost of misaligned attention diminishes as coherence increases.")
    print()
    print("Understanding = Yield.")
    print("  Measured in compounding foresight, not speculation.")
    print()
    
    # A8. Token Metadata
    print("=" * 80)
    print("A8. EXAMPLE TOKEN METADATA (MLT-001)")
    print("=" * 80)
    print()
    metadata = generate_mlt_metadata(df)
    import json
    print(json.dumps(metadata, indent=2))
    print()
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
    
    df.to_csv(os.path.join(results_dir, "data", "mathlete_lifecycle_simulation.csv"), index=False)
    print("Lifecycle simulation results saved to results/data/mathlete_lifecycle_simulation.csv")
    
    # Generate visualization
    plot_mathlete_lifecycle(df, save_path=os.path.join(results_dir, "figures", "mathlete_lifecycle_simulation.png"))
    
    # A9. Closing Remark
    print()
    print("=" * 80)
    print("A9. CLOSING REMARK")
    print("=" * 80)
    print()
    print("This appendix is not a financial forecast—it's a simulation of")
    print("epistemic value formation.")
    print()
    print("Each variable in the model is a reflection of how real human")
    print("collaboration, trust, and iteration reduce uncertainty.")
    print()
    print("In this framing, understanding itself becomes the yield curve.")
    print()
    print("'In the Mathlete Chain, the market moves not toward profit,")
    print(" but toward truth.'")
    print()

