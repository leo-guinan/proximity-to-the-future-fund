"""
Proximity Token ($PROX) Simulation

Simulates the tokenomics of $PROX based on the mathematical framework
from the Proximity to the Future Fund model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prove_model import (
    PodState, TemporalCompoundingModel, simulate_pod,
    ALPHA_REF, CAPITAL_MAX_GROWTH, A4_SOFT_LINK_RHO,
    RANDOM_SEED, F_MIN_STABILITY
)
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class ProximityToken:
    """
    Simulates $PROX token mechanics based on Pod state.
    """
    
    def __init__(self, initial_supply: float = 1_000_000.0):
        """
        Initialize $PROX token.
        
        Args:
            initial_supply: Initial circulating supply
        """
        self.initial_supply = initial_supply
        self.supply_history = [initial_supply]
        self.mint_history = [0.0]
        self.burn_history = [0.0]
        self.net_supply_change_history = [0.0]
    
    def calculate_mint(self, supply: float, F_t: float, L_t: float) -> float:
        """
        Calculate minting amount based on foresight efficiency and trust leverage.
        
        Mint_{t+1} = S_t * F_t * L_t
        """
        return supply * F_t * L_t
    
    def calculate_burn(self, supply: float, gamma_t: float, T_t: float, tau_t: float) -> float:
        """
        Calculate burn amount based on time violence.
        
        Burn_{t+1} = S_t * TV_t = S_t * γ_t * T_t * (1 - τ_t)
        """
        TV_t = gamma_t * T_t * (1 - tau_t)
        return supply * TV_t
    
    def update_supply(self, F_t: float, L_t: float, gamma_t: float, 
                     T_t: float, tau_t: float) -> tuple:
        """
        Update token supply based on minting and burning.
        
        S_{t+1} = S_t + S_t(F_t * L_t - γ_t * T_t * (1 - τ_t))
        
        Returns:
            (new_supply, mint_amount, burn_amount, net_change)
        """
        current_supply = self.supply_history[-1]
        
        # Calculate mint and burn
        mint_amount = self.calculate_mint(current_supply, F_t, L_t)
        burn_amount = self.calculate_burn(current_supply, gamma_t, T_t, tau_t)
        
        # Net supply change
        net_change = mint_amount - burn_amount
        new_supply = current_supply + net_change
        
        # Ensure supply doesn't go negative
        new_supply = max(0.0, new_supply)
        
        # Store history
        self.supply_history.append(new_supply)
        self.mint_history.append(mint_amount)
        self.burn_history.append(burn_amount)
        self.net_supply_change_history.append(net_change)
        
        return new_supply, mint_amount, burn_amount, net_change
    
    def calculate_staking_reward(self, staked_amount: float, L_t: float, 
                                L_0: float = 1.0, base_apy: float = 0.05,
                                epochs_per_year: int = 365) -> float:
        """
        Calculate staking rewards based on trust leverage.
        
        R_i ∝ L_t = L_0(1 + k(τ_t - τ_eq))
        
        Args:
            staked_amount: Amount staked
            L_t: Trust leverage
            L_0: Baseline leverage
            base_apy: Base annual percentage yield
            epochs_per_year: Number of epochs per year
        
        Returns:
            Reward amount per epoch
        """
        # Reward scales with leverage relative to baseline
        leverage_multiplier = max(0.0, L_t / L_0)  # Ensure non-negative
        epoch_reward = staked_amount * base_apy * leverage_multiplier / epochs_per_year
        return epoch_reward
    
    def get_supply_forecast(self, n_epochs: int, F_t: float, L_t: float,
                           gamma_t: float, T_t: float, tau_t: float) -> np.ndarray:
        """
        Forecast token supply over n_epochs given constant parameters.
        
        E[S_t] = S_0 * ∏(1 + F_i * L_i - γ_i * T_i * (1 - τ_i))
        """
        current_supply = self.supply_history[-1]
        forecast = [current_supply]
        
        for _ in range(n_epochs):
            net_rate = F_t * L_t - gamma_t * T_t * (1 - tau_t)
            new_supply = forecast[-1] * (1 + net_rate)
            forecast.append(max(0.0, new_supply))
        
        return np.array(forecast)


def simulate_prox_tokenomics(pod_model: TemporalCompoundingModel, 
                            initial_supply: float = 1_000_000.0) -> pd.DataFrame:
    """
    Simulate $PROX tokenomics based on Pod evolution.
    
    Args:
        pod_model: TemporalCompoundingModel with history
        initial_supply: Initial token supply
    
    Returns:
        DataFrame with token metrics over time
    """
    token = ProximityToken(initial_supply)
    results = []
    
    # Record initial state (epoch 0)
    initial_state = pod_model.history[0]
    F_t_0 = pod_model.foresight_efficiency(initial_state)
    L_t_0 = pod_model.trust_leverage(initial_state)
    TV_t_0 = pod_model.time_violence(initial_state)
    
    # Initial staking calculation
    staking_participation_0 = initial_state.tau_t * 0.5
    staked_amount_0 = initial_supply * staking_participation_0
    staking_reward_0 = token.calculate_staking_reward(staked_amount_0, L_t_0, initial_state.L_0)
    staking_apy_0 = (staking_reward_0 * 365 / staked_amount_0 * 100) if staked_amount_0 > 0 else 0.0
    
    results.append({
        'epoch': 0,
        'supply': initial_supply,
        'mint': 0.0,
        'burn': 0.0,
        'net_change': 0.0,
        'inflation_rate': 0.0,
        'F_t': F_t_0,
        'L_t': L_t_0,
        'tau_t': initial_state.tau_t,
        'gamma_t': initial_state.gamma_t,
        'TV_t': TV_t_0,
        'staking_apy': staking_apy_0,
        'time_violence': TV_t_0
    })
    
    # Process remaining states
    for i, state in enumerate(pod_model.history[1:], start=1):
        # Calculate metrics
        F_t = pod_model.foresight_efficiency(state)
        L_t = pod_model.trust_leverage(state)
        TV_t = pod_model.time_violence(state)
        
        # Update token supply
        new_supply, mint_amount, burn_amount, net_change = token.update_supply(
            F_t, L_t, state.gamma_t, state.T_t, state.tau_t
        )
        
        # Calculate inflation/deflation rate
        prev_supply = token.supply_history[i]  # Previous supply
        inflation_rate = (net_change / prev_supply) * 100 if prev_supply > 0 else 0.0
        
        # Calculate staking reward (example: staking participation based on tau_t)
        # Higher trust = higher staking participation
        staking_participation = state.tau_t * 0.5  # Up to 50% staking at full trust
        staked_amount = new_supply * staking_participation
        staking_reward = token.calculate_staking_reward(staked_amount, L_t, state.L_0)
        staking_apy = (staking_reward * 365 / staked_amount * 100) if staked_amount > 0 else 0.0
        
        results.append({
            'epoch': state.t,
            'supply': new_supply,
            'mint': mint_amount,
            'burn': burn_amount,
            'net_change': net_change,
            'inflation_rate': inflation_rate,
            'F_t': F_t,
            'L_t': L_t,
            'tau_t': state.tau_t,
            'gamma_t': state.gamma_t,
            'TV_t': TV_t,
            'staking_apy': staking_apy,
            'time_violence': TV_t
        })
    
    return pd.DataFrame(results)


def plot_token_supply_dynamics(df: pd.DataFrame, save_path: str = None):
    """
    Visualize $PROX token supply dynamics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('$PROX Token Supply Dynamics', fontsize=16, y=1.02)
    
    # Plot 1: Token Supply Over Time
    axes[0, 0].plot(df['epoch'], df['supply'], 'b-', linewidth=2, label='Circulating Supply')
    axes[0, 0].set_title('Token Supply Evolution')
    axes[0, 0].set_xlabel('Epoch (t)')
    axes[0, 0].set_ylabel('Supply ($PROX)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Minting and Burning
    axes[0, 1].plot(df['epoch'], df['mint'], 'g-', linewidth=2, label='Mint', alpha=0.7)
    axes[0, 1].plot(df['epoch'], df['burn'], 'r-', linewidth=2, label='Burn', alpha=0.7)
    axes[0, 1].set_title('Minting vs Burning')
    axes[0, 1].set_xlabel('Epoch (t)')
    axes[0, 1].set_ylabel('Amount ($PROX)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Net Supply Change
    axes[0, 2].plot(df['epoch'], df['net_change'], 'purple', linewidth=2, label='Net Change')
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_title('Net Supply Change (Mint - Burn)')
    axes[0, 2].set_xlabel('Epoch (t)')
    axes[0, 2].set_ylabel('Net Change ($PROX)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Inflation/Deflation Rate
    axes[1, 0].plot(df['epoch'], df['inflation_rate'], 'orange', linewidth=2, label='Inflation Rate')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Inflation/Deflation Rate (%)')
    axes[1, 0].set_xlabel('Epoch (t)')
    axes[1, 0].set_ylabel('Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Staking APY
    axes[1, 1].plot(df['epoch'], df['staking_apy'], 'darkgreen', linewidth=2, label='Staking APY')
    axes[1, 1].set_title('Staking APY (Trust-Weighted)')
    axes[1, 1].set_xlabel('Epoch (t)')
    axes[1, 1].set_ylabel('APY (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Foresight Efficiency and Trust Leverage
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    line1 = ax6.plot(df['epoch'], df['F_t'], 'blue', label='Foresight Efficiency (F_t)', linewidth=2)
    line2 = ax6_twin.plot(df['epoch'], df['L_t'], 'red', label='Trust Leverage (L_t)', linewidth=2)
    ax6.set_xlabel('Epoch (t)')
    ax6.set_ylabel('Foresight Efficiency (F_t)', color='blue')
    ax6_twin.set_ylabel('Trust Leverage (L_t)', color='red')
    ax6.set_title('Productivity Metrics')
    ax6.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Token dynamics plot saved to {save_path}")
    else:
        plt.show()


def analyze_token_predictability(df: pd.DataFrame) -> dict:
    """
    Analyze token supply predictability based on mathematical framework.
    """
    # Calculate expected supply path
    initial_supply = df['supply'].iloc[0]
    final_supply = df['supply'].iloc[-1]
    
    # Calculate average rates
    avg_F_t = df['F_t'].mean()
    avg_L_t = df['L_t'].mean()
    avg_gamma_t = df['gamma_t'].mean()
    avg_T_t = df['time_violence'].mean() / (avg_gamma_t * (1 - df['tau_t'].mean())) if avg_gamma_t > 0 else 0
    avg_tau_t = df['tau_t'].mean()
    
    # Calculate expected net rate
    avg_net_rate = avg_F_t * avg_L_t - avg_gamma_t * avg_T_t * (1 - avg_tau_t)
    
    # Forecast vs actual
    n_epochs = len(df) - 1
    expected_supply = initial_supply * ((1 + avg_net_rate) ** n_epochs)
    actual_supply = final_supply
    
    # Predictability metrics
    supply_volatility = df['supply'].std() / df['supply'].mean()
    inflation_volatility = df['inflation_rate'].std()
    
    return {
        'initial_supply': initial_supply,
        'final_supply': final_supply,
        'expected_supply': expected_supply,
        'supply_error': abs(actual_supply - expected_supply) / expected_supply * 100,
        'avg_net_rate': avg_net_rate,
        'supply_volatility': supply_volatility,
        'inflation_volatility': inflation_volatility,
        'total_minted': df['mint'].sum(),
        'total_burned': df['burn'].sum(),
        'net_supply_change': final_supply - initial_supply,
        'avg_staking_apy': df['staking_apy'].mean()
    }


if __name__ == "__main__":
    print("=" * 60)
    print("$PROX TOKEN SIMULATION")
    print("=" * 60)
    print()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Simulate Pod evolution
    print("Simulating Pod evolution...")
    model, pod_df = simulate_pod(n_cycles=50, alpha_ref=ALPHA_REF)
    
    # Simulate tokenomics
    print("Simulating $PROX tokenomics...")
    token_df = simulate_prox_tokenomics(model, initial_supply=1_000_000.0)
    
    # Analyze predictability
    print("\nAnalyzing token predictability...")
    predictability = analyze_token_predictability(token_df)
    
    # Print results
    print("\n" + "=" * 60)
    print("TOKEN PREDICTABILITY ANALYSIS")
    print("=" * 60)
    print(f"Initial Supply: {predictability['initial_supply']:,.0f} $PROX")
    print(f"Final Supply: {predictability['final_supply']:,.0f} $PROX")
    print(f"Expected Supply: {predictability['expected_supply']:,.0f} $PROX")
    print(f"Supply Error: {predictability['supply_error']:.2f}%")
    print(f"Average Net Rate: {predictability['avg_net_rate']:.6f}")
    print(f"Supply Volatility: {predictability['supply_volatility']:.4f}")
    print(f"Inflation Volatility: {predictability['inflation_volatility']:.4f}%")
    print(f"\nTotal Minted: {predictability['total_minted']:,.0f} $PROX")
    print(f"Total Burned: {predictability['total_burned']:,.0f} $PROX")
    print(f"Net Supply Change: {predictability['net_supply_change']:+,.0f} $PROX")
    print(f"Average Staking APY: {predictability['avg_staking_apy']:.2f}%")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
    
    token_df.to_csv(os.path.join(results_dir, "data", "prox_token_simulation.csv"), index=False)
    print(f"\nToken simulation results saved to results/data/prox_token_simulation.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_token_supply_dynamics(token_df, save_path=os.path.join(results_dir, "figures", "prox_token_dynamics.png"))
    
    print("\nSimulation complete!")

