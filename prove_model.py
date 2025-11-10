"""
Proximity to the Future Fund - Model Validation and Simulation

This module implements the mathematical foundations from Appendix A
and provides simulations to validate the temporal compounding model.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

# Constants
F_MIN_STABILITY = 0.03  # Consistent stability threshold (0.02-0.05 range)
ALPHA_REF = 15.0  # Reference for alpha normalization (targets α in ~0.1-0.6 range)
CAPITAL_MAX_GROWTH = 0.5  # Maximum capital growth per step (prevents explosive growth)
# Note: If ALPHA_REF is raised or F_MIN_STABILITY is lowered, this cap may need adjustment.
# Monitor for hockey-stick behavior when changing scales.
COHERENCE_MAX = 100.0  # Maximum coherence value (prevents spikes)
A4_SOFT_LINK_RHO = 0.2  # Soft link coefficient for A4 Network Relativity influence
RANDOM_SEED = 42  # Seed for reproducibility

# Time-violence control parameters
GAMMA_EQUILIBRIUM = 0.01  # Target equilibrium entropy (small, positive)
GAMMA_MEAN_REVERSION_ETA = 0.1  # Mean reversion rate for gamma (0 < eta < 1)
TRUST_F_IMPROVEMENT_COEFF = 0.05  # Coefficient for trust increase when F_t improves

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


@dataclass
class PodState:
    """Represents the state of a Pod at time t"""
    t: int
    V_t: float  # verified value creation
    T_t: float  # mean verification time
    C_t: float  # capital deployed
    alpha_t: float  # temporal alpha
    beta_t: float  # trust reinforcement coefficient
    gamma_t: float  # entropy rate
    tau_t: float  # trust level [0, 1]
    lambda_val: float  # learning rate
    L_0: float  # baseline leverage
    k: float  # sensitivity constant
    tau_eq: float  # equilibrium trust threshold


class TemporalCompoundingModel:
    """
    Implements the mathematical foundations of temporal compounding
    as defined in Appendix A.
    """
    
    def __init__(self, initial_state: PodState):
        self.state = initial_state
        self.history: List[PodState] = [initial_state]
    
    def foresight_efficiency(self, state: PodState) -> float:
        """
        A1. Foresight Efficiency
        
        F_t = V_t / (T_t * C_t)
        """
        if state.T_t == 0 or state.C_t == 0:
            return 0.0
        return state.V_t / (state.T_t * state.C_t)
    
    def recursive_compounding(self, state: PodState) -> float:
        """
        A2. Recursive Compounding Function
        
        F_{t+1} = F_t(1 + λ(α_t + β_t - γ_t))
        """
        F_t = self.foresight_efficiency(state)
        return F_t * (1 + state.lambda_val * (state.alpha_t + state.beta_t - state.gamma_t))
    
    def trust_leverage(self, state: PodState) -> float:
        """
        A3. Trust-Leverage Relationship
        
        L_t = L_0(1 + k(τ_t - τ_eq))
        """
        return state.L_0 * (1 + state.k * (state.tau_t - state.tau_eq))
    
    def network_relativity(self, T_local: float, v_rel: float, c_t: float, 
                          use_lorentz: bool = False) -> float:
        """
        A4. Network Relativity Principle
        
        Calculates network-dilated verification time from local time.
        
        Args:
            T_local: Local verification time
            v_rel: Relative foresight velocity between Pods
            c_t: Speed of trust propagation in the network
            use_lorentz: If True, use Lorentz-style γ = 1/√(1-(v/c)²), 
                        else use γ = 1/(1-v/c) (current analog)
        
        Returns:
            T_network: Network-observed (dilated) verification time
        """
        if v_rel >= c_t:
            return float('inf')  # Faster than trust propagation
        
        if use_lorentz:
            # Lorentz-style: symmetric around zero
            gamma = 1.0 / np.sqrt(1 - (v_rel / c_t) ** 2)
        else:
            # Current analog: γ = 1/(1-v/c)
            gamma = 1.0 / (1 - v_rel / c_t)
        
        return gamma * T_local
    
    def total_compounded_understanding(self, n: int) -> float:
        """
        A5. Total Compounded Understanding
        
        U_n = U_0 ∏(1 + F_i * L_i)
        """
        U_0 = 1.0  # Initial understanding
        result = U_0
        
        for i in range(min(n, len(self.history))):
            state = self.history[i]
            F_i = self.foresight_efficiency(state)
            L_i = self.trust_leverage(state)
            result *= (1 + F_i * L_i)
        
        return result
    
    def temporal_alpha(self, delta_V: float, delta_T_network: float) -> float:
        """
        A6. Temporal Alpha Calculation
        
        α_t = ΔV / ΔT_network
        """
        if delta_T_network == 0:
            return 0.0
        return delta_V / delta_T_network
    
    def pod_coherence(self, state: PodState, epsilon: float = 0.001) -> float:
        """
        A7. Pod Coherence Metric
        
        Φ_t = (α_t * β_t) / (γ_t + ε)
        Clipped to prevent spikes when γ → 0
        """
        denominator = state.gamma_t + epsilon
        if denominator == 0:
            return 0.0
        coherence = (state.alpha_t * state.beta_t) / denominator
        return min(coherence, COHERENCE_MAX)  # Clip to prevent spikes
    
    def time_violence(self, state: PodState) -> float:
        """
        A10. Time Violence Score
        
        TV_t = γ_t * T_t * (1 - τ_t)
        """
        return state.gamma_t * state.T_t * (1 - state.tau_t)
    
    def is_stable(self, state: PodState, F_min: float = None) -> bool:
        """
        A8. Convergence Conditions
        
        Checks if Pod meets stability requirements:
        - α_t + β_t > γ_t
        - τ_t > τ_eq
        - F_t > F_min
        """
        if F_min is None:
            F_min = F_MIN_STABILITY
        condition1 = state.alpha_t + state.beta_t > state.gamma_t
        condition2 = state.tau_t > state.tau_eq
        F_t = self.foresight_efficiency(state)
        condition3 = F_t > F_min
        
        return condition1 and condition2 and condition3
    
    def evolve(self, 
               delta_V: float,
               delta_T_network: float,
               trust_change: float = 0.0,
               entropy_change: float = 0.0,
               k_beta: float = 1.0,
               alpha_ref: float = None,
               T_from_A4: float = None,
               rho: float = None,
               F_other: float = None,
               capital_max_growth: float = None) -> PodState:
        """
        Evolve Pod state to next time step based on feedback loop.
        A2 (Recursive Compounding) drives the state evolution.
        
        Args:
            delta_V: Value creation increment
            delta_T_network: Network verification time
            trust_change: Change in trust level
            entropy_change: Change in entropy (gamma)
            k_beta: Beta coefficient multiplier
            alpha_ref: Reference for alpha normalization (None uses global ALPHA_REF)
            T_from_A4: A4 Network Relativity prediction for T (optional, for soft link)
            rho: Soft link coefficient for A4 influence (None uses global A4_SOFT_LINK_RHO)
            F_other: Foresight efficiency of other Pod (for conditional A4 influence)
            capital_max_growth: Maximum capital growth per step (None uses global CAPITAL_MAX_GROWTH)
        """
        current = self.history[-1]
        
        # B. Normalize alpha to bounded scale [0, 1]
        # Use global ALPHA_REF for consistent calibration (targets α in ~0.1-0.6 range)
        if alpha_ref is None:
            alpha_ref = ALPHA_REF
        raw_alpha = self.temporal_alpha(delta_V, delta_T_network)
        # Squash to (0, 1) using saturation function
        new_alpha = raw_alpha / (raw_alpha + alpha_ref)
        
        # A. Make A2 drive the state
        # Calculate F_next from A2: F_{t+1} = F_t(1 + λ(α_t + β_t - γ_t))
        F_t = self.foresight_efficiency(current)
        
        # Time-violence control: Nudge trust upward when F_t improves
        # Calculate F_next temporarily to get delta_F
        F_next_prelim = F_t * (1 + current.lambda_val * (new_alpha + max(0.0, (current.tau_t - current.tau_eq)) * k_beta - current.gamma_t))
        delta_F = max(0.0, F_next_prelim - F_t)  # Only positive changes
        trust_improvement_bonus = TRUST_F_IMPROVEMENT_COEFF * delta_F
        
        # Update trust (with bounds) - includes improvement bonus
        new_tau = max(0.0, min(1.0, current.tau_t + trust_change + trust_improvement_bonus))
        
        # Time-violence control: Let gamma mean-revert
        # gamma_t ← (1-η)·gamma_t + η·gamma_eq
        new_gamma = (1 - GAMMA_MEAN_REVERSION_ETA) * current.gamma_t + GAMMA_MEAN_REVERSION_ETA * GAMMA_EQUILIBRIUM
        # Then add entropy change
        new_gamma = max(0.0, new_gamma + entropy_change)
        
        # C. Tie beta to trust dynamics more explicitly
        # β_t = k_β·(τ_t − τ_eq)_+ so reinforcement only grows above equilibrium
        new_beta = max(0.0, (new_tau - current.tau_eq)) * k_beta
        
        # F. Enforce stability before stepping
        # Check if current state is stable, damp if not
        is_current_stable = self.is_stable(current, F_min=F_MIN_STABILITY)
        damp = 1.0
        if not is_current_stable:
            # Reduce learning rate or skip risky update
            damp = 0.5
            new_alpha *= damp
            new_beta *= damp
        
        # Recalculate F_next with updated values
        F_next = F_t * (1 + current.lambda_val * (new_alpha + new_beta - new_gamma))
        F_next = max(F_next, 0.0)  # Ensure non-negative
        
        # E. Use A5 directly for capital evolution
        # Capital-first update via A5: C_{t+1} = C_t * (1 + F_t * L_t)
        # Guard explosive capital growth with cap or sigmoid
        L_t = self.trust_leverage(current)
        if capital_max_growth is None:
            capital_max_growth = CAPITAL_MAX_GROWTH
        growth_rate = min(F_t * L_t, capital_max_growth)  # Cap growth to prevent hockey-stick
        new_C = current.C_t * (1 + growth_rate)
        
        # A. Solve for T_{t+1} to satisfy F_next
        # F_next = V_{t+1} / (T_{t+1} * C_{t+1})
        # Therefore: T_{t+1} = V_{t+1} / (F_next * C_{t+1})
        new_V = current.V_t + delta_V
        if F_next > 0 and new_C > 0:
            new_T = max(0.1, new_V / (F_next * new_C))
        else:
            # Fallback: D. Update T with gentle rule if F_next is invalid
            s_T = 0.1
            new_T = current.T_t * (1 - s_T * (new_alpha - new_gamma))
            new_T = np.clip(new_T, 0.5 * current.T_t, 2.0 * current.T_t)
        
        # G. Use A4 to influence T (soft link for Network Relativity)
        # Bound the A4 soft-link influence to avoid local T being pulled too high
        if T_from_A4 is not None and T_from_A4 > 0:
            if rho is None:
                # Adaptive rho: smaller when T_local is large, or only when F_B > F_A
                if F_other is not None:
                    # Only apply A4 influence when this Pod is actually outperforming
                    if F_next > F_other:
                        rho_base = A4_SOFT_LINK_RHO
                    else:
                        rho_base = 0.0  # No A4 influence if not outperforming
                else:
                    rho_base = A4_SOFT_LINK_RHO
                
                # Cap rho based on T_local: rho_t = min(base, 0.05 + 0.2/(1+T_local))
                rho = min(rho_base, 0.05 + 0.2 / (1 + new_T))
            new_T = (1 - rho) * new_T + rho * T_from_A4
        
        # Create new state
        new_state = PodState(
            t=current.t + 1,
            V_t=new_V,
            T_t=new_T,
            C_t=new_C,
            alpha_t=new_alpha,
            beta_t=new_beta,
            gamma_t=new_gamma,
            tau_t=new_tau,
            lambda_val=current.lambda_val,
            L_0=current.L_0,
            k=current.k,
            tau_eq=current.tau_eq
        )
        
        # Invariance test: Assert nonnegativity and finiteness
        assert np.isfinite(new_T) and new_T > 0, f"T must be finite and positive, got {new_T}"
        assert np.isfinite(new_C) and new_C > 0, f"C must be finite and positive, got {new_C}"
        assert 0 <= new_tau <= 1, f"tau must be in [0,1], got {new_tau}"
        assert np.isfinite(new_alpha), f"alpha must be finite, got {new_alpha}"
        assert np.isfinite(new_beta), f"beta must be finite, got {new_beta}"
        assert np.isfinite(new_gamma), f"gamma must be finite, got {new_gamma}"
        assert np.isfinite(new_V) and new_V >= 0, f"V must be finite and non-negative, got {new_V}"
        
        self.history.append(new_state)
        self.state = new_state
        
        return new_state


def simulate_pod(n_cycles: int = 50, 
                 initial_state: PodState = None,
                 k_beta: float = 1.0,
                 alpha_ref: float = None) -> Tuple[TemporalCompoundingModel, pd.DataFrame]:
    """
    Simulate a Pod over n cycles and return results.
    """
    if initial_state is None:
        # Default initial state
        initial_state = PodState(
            t=0,
            V_t=100.0,  # Initial value
            T_t=10.0,   # Initial verification time
            C_t=1000.0, # Initial capital
            alpha_t=0.1,  # Initial temporal alpha
            beta_t=0.2,   # Initial trust reinforcement
            gamma_t=0.05, # Initial entropy
            tau_t=0.6,    # Initial trust
            lambda_val=0.1,  # Learning rate
            L_0=1.0,     # Baseline leverage
            k=2.0,       # Sensitivity constant
            tau_eq=0.5   # Equilibrium trust threshold
        )
    
    model = TemporalCompoundingModel(initial_state)
    
    # Simulate cycles
    for cycle in range(n_cycles):
        # Simulate feedback loop with some randomness
        delta_V = np.random.normal(50, 10)  # Value creation per cycle
        delta_T_network = np.random.normal(2, 0.5)  # Network verification time
        
        # Trust changes based on performance
        trust_change = np.random.normal(0.01, 0.02)
        
        # Entropy changes (can increase or decrease)
        entropy_change = np.random.normal(0, 0.01)
        
        model.evolve(delta_V, delta_T_network, trust_change, entropy_change, k_beta, alpha_ref)
    
    # Build results dataframe
    results = []
    for i, state in enumerate(model.history):
        F_t = model.foresight_efficiency(state)
        L_t = model.trust_leverage(state)
        coherence = model.pod_coherence(state)
        time_violence = model.time_violence(state)
        is_stable = model.is_stable(state, F_min=F_MIN_STABILITY)
        
        # Calculate TV trend (slope over last 10 cycles if available)
        if i >= 10:
            recent_states = model.history[i-9:i+1]  # Last 10 states
            tv_values = [model.time_violence(s) for s in recent_states]
            tv_trend = np.polyfit(range(len(tv_values)), tv_values, 1)[0]
        else:
            tv_trend = 0.0
        
        results.append({
            't': state.t,
            'V_t': state.V_t,
            'T_t': state.T_t,
            'C_t': state.C_t,
            'F_t': F_t,
            'alpha_t': state.alpha_t,
            'beta_t': state.beta_t,
            'gamma_t': state.gamma_t,
            'tau_t': state.tau_t,
            'L_t': L_t,
            'coherence': coherence,
            'time_violence': time_violence,
            'TV_trend': tv_trend,
            'is_stable': is_stable
        })
    
    df = pd.DataFrame(results)
    
    return model, df


def prove_model():
    """
    Run validation tests to prove the model works correctly.
    """
    print("=" * 60)
    print("PROXIMITY TO THE FUTURE FUND - MODEL VALIDATION")
    print("=" * 60)
    print()
    
    # Test 1: Basic Foresight Efficiency
    print("Test 1: Foresight Efficiency Calculation")
    print("-" * 60)
    state = PodState(
        t=0, V_t=100, T_t=10, C_t=1000,
        alpha_t=0.1, beta_t=0.2, gamma_t=0.05,
        tau_t=0.6, lambda_val=0.1, L_0=1.0, k=2.0, tau_eq=0.5
    )
    model = TemporalCompoundingModel(state)
    F_t = model.foresight_efficiency(state)
    print(f"F_t = V_t / (T_t * C_t) = {state.V_t} / ({state.T_t} * {state.C_t}) = {F_t:.6f}")
    assert F_t > 0, "Foresight efficiency must be positive"
    print("✓ PASSED\n")
    
    # Test 2: Trust-Leverage Relationship
    print("Test 2: Trust-Leverage Relationship")
    print("-" * 60)
    L_t = model.trust_leverage(state)
    print(f"L_t = L_0(1 + k(τ_t - τ_eq))")
    print(f"L_t = {state.L_0}(1 + {state.k}({state.tau_t} - {state.tau_eq})) = {L_t:.4f}")
    assert L_t > 0, "Leverage must be positive"
    print("✓ PASSED\n")
    
    # Test 3: Network Relativity
    print("Test 3: Network Relativity Principle")
    print("-" * 60)
    T_local = 10.0
    v_rel = 0.5
    c_t = 1.0
    T_network = model.network_relativity(T_local, v_rel, c_t)
    print(f"T_network = γ * T_local where γ = 1 / (1 - v_rel / c_t)")
    print(f"T_network = {T_local} / (1 - {v_rel} / {c_t}) = {T_network:.4f}")
    assert T_network > T_local, "Network-observed time should be dilated (longer) than local time"
    print("✓ PASSED\n")
    
    # Test 4: Stability Conditions (with crafted stable state)
    print("Test 4: Stability Conditions")
    print("-" * 60)
    # Test with unstable state (current)
    is_stable_current = model.is_stable(state, F_min=F_MIN_STABILITY)
    print(f"Current state: α_t={state.alpha_t}, β_t={state.beta_t}, γ_t={state.gamma_t}")
    print(f"α_t + β_t > γ_t: {state.alpha_t} + {state.beta_t} > {state.gamma_t} = {state.alpha_t + state.beta_t > state.gamma_t}")
    print(f"τ_t > τ_eq: {state.tau_t} > {state.tau_eq} = {state.tau_t > state.tau_eq}")
    print(f"F_t > F_min ({F_MIN_STABILITY}): {F_t} > {F_MIN_STABILITY} = {F_t > F_MIN_STABILITY}")
    print(f"Pod is stable: {is_stable_current}")
    
    # Create a crafted stable state for rigorous test
    stable_state = PodState(
        t=0, V_t=200.0, T_t=5.0, C_t=1000.0,  # Higher F_t
        alpha_t=0.2, beta_t=0.3, gamma_t=0.05,  # α + β > γ
        tau_t=0.7, lambda_val=0.1, L_0=1.0, k=2.0, tau_eq=0.5  # τ > τ_eq
    )
    F_stable = model.foresight_efficiency(stable_state)
    is_stable_crafted = model.is_stable(stable_state, F_min=F_MIN_STABILITY)
    print(f"\nCrafted stable state: F_t={F_stable:.6f}, α_t={stable_state.alpha_t}, β_t={stable_state.beta_t}, γ_t={stable_state.gamma_t}")
    print(f"Stability check: {is_stable_crafted}")
    assert is_stable_crafted, f"Crafted stable state must pass stability check (F_t={F_stable:.6f} > {F_MIN_STABILITY})"
    assert is_stable_current or F_t <= F_MIN_STABILITY, "Stability check must be consistent"
    print("✓ PASSED\n")
    
    # Test 5: Recursive Compounding
    print("Test 5: Recursive Compounding Function")
    print("-" * 60)
    F_next = model.recursive_compounding(state)
    print(f"F_{{t+1}} = F_t(1 + λ(α_t + β_t - γ_t))")
    print(f"F_{{t+1}} = {F_t:.6f}(1 + {state.lambda_val}({state.alpha_t} + {state.beta_t} - {state.gamma_t}))")
    print(f"F_{{t+1}} = {F_next:.6f}")
    assert F_next > F_t, "Compounding should increase efficiency"
    print("✓ PASSED\n")
    
    # Test 6: Full Simulation
    print("Test 6: Full Pod Simulation (20 cycles)")
    print("-" * 60)
    model_sim, df = simulate_pod(n_cycles=20)
    print(f"Simulated {len(df)} time steps")
    print(f"Final Foresight Efficiency: {df['F_t'].iloc[-1]:.6f}")
    print(f"Final Trust: {df['tau_t'].iloc[-1]:.4f}")
    print(f"Final Leverage: {df['L_t'].iloc[-1]:.4f}")
    print(f"Final Coherence: {df['coherence'].iloc[-1]:.4f}")
    print(f"Final Time Violence: {df['time_violence'].iloc[-1]:.4f}")
    print(f"Pod Stability: {df['is_stable'].iloc[-1]}")
    
    # Calculate compounded understanding
    U_n = model_sim.total_compounded_understanding(len(df))
    print(f"Total Compounded Understanding (U_n): {U_n:.4f}")
    assert U_n > 1.0, "Understanding should compound over time"
    print("✓ PASSED\n")
    
    print("=" * 60)
    print("ALL TESTS PASSED - MODEL VALIDATED")
    print("=" * 60)
    
    return model_sim, df


def simulate_two_pods_comparison(n_cycles: int = 50,
                                  v_rel_A: float = 0.3,
                                  v_rel_B: float = 0.7,
                                  c_t: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    G. Actually use A4 in a comparison experiment
    
    Simulates two Pods (A and B) with different relative foresight velocities
    and computes temporal dilation effects using A4 (Network Relativity Principle).
    """
    # Pod A: Slower foresight velocity
    initial_A = PodState(
        t=0, V_t=100.0, T_t=10.0, C_t=1000.0,
        alpha_t=0.1, beta_t=0.2, gamma_t=0.05,
        tau_t=0.6, lambda_val=0.1, L_0=1.0, k=2.0, tau_eq=0.5
    )
    
    # Pod B: Faster foresight velocity
    initial_B = PodState(
        t=0, V_t=100.0, T_t=10.0, C_t=1000.0,
        alpha_t=0.15, beta_t=0.25, gamma_t=0.03,  # Better initial conditions
        tau_t=0.7, lambda_val=0.12, L_0=1.2, k=2.5, tau_eq=0.55
    )
    
    model_A = TemporalCompoundingModel(initial_A)
    model_B = TemporalCompoundingModel(initial_B)
    
    results_A = []
    results_B = []
    
    for cycle in range(n_cycles):
        # Simulate feedback loop for both Pods
        delta_V_A = np.random.normal(50, 10)
        delta_V_B = np.random.normal(60, 10)  # Pod B creates more value
        
        delta_T_network = np.random.normal(2, 0.5)
        
        trust_change_A = np.random.normal(0.01, 0.02)
        trust_change_B = np.random.normal(0.015, 0.02)
        
        entropy_change_A = np.random.normal(0, 0.01)
        entropy_change_B = np.random.normal(-0.005, 0.01)  # Pod B reduces entropy faster
        
        # Get current states before evolution
        state_A_current = model_A.state
        state_B_current = model_B.state
        
        # Calculate A4: Network Relativity BEFORE evolution
        # T_B_local is what we'll use for soft link
        # Network-dilated time is calculated from B's local time
        T_A_current = state_A_current.T_t
        T_B_current_local = state_B_current.T_t
        # Calculate what network would observe for B (dilated from B's local time)
        T_B_network_predicted = model_A.network_relativity(T_B_current_local, v_rel_B - v_rel_A, c_t)
        
        # Evolve both Pods
        # Calculate F values for conditional A4 influence
        F_A_current = model_A.foresight_efficiency(state_A_current)
        F_B_current = model_B.foresight_efficiency(state_B_current)
        
        # Pod A evolves normally
        model_A.evolve(delta_V_A, delta_T_network, trust_change_A, entropy_change_A,
                      F_other=F_B_current)
        
        # Pod B evolves with A4 soft link influence
        # Use network-dilated prediction to nudge B's local time
        # Pass F_A_current so A4 influence only applies when F_B > F_A
        model_B.evolve(delta_V_B, delta_T_network, trust_change_B, entropy_change_B,
                      T_from_A4=T_B_network_predicted, rho=None, F_other=F_A_current)
        
        # Get current states after evolution
        state_A = model_A.state
        state_B = model_B.state
        
        # Calculate A4: Network Relativity AFTER evolution (for tracking)
        # T_t is local verification time
        T_A_local = state_A.T_t
        T_B_local = state_B.T_t
        
        # Calculate network-dilated times using A4
        # Network observes B's time as dilated relative to A
        T_B_network = model_A.network_relativity(T_B_local, v_rel_B - v_rel_A, c_t)
        
        # Calculate temporal alpha spread
        alpha_spread = state_B.alpha_t - state_A.alpha_t
        
        # Calculate temporal dilation delta (local times)
        temporal_dilation_delta_local = T_B_local - T_A_local
        # Network-observed dilation
        temporal_dilation_delta_network = T_B_network - T_A_local
        
        # Calculate foresight efficiency difference
        F_A = model_A.foresight_efficiency(state_A)
        F_B = model_B.foresight_efficiency(state_B)
        F_spread = F_B - F_A
        
        # Store results
        results_A.append({
            't': state_A.t,
            'T_local': T_A_local,  # Local verification time
            'alpha_t': state_A.alpha_t,
            'F_t': F_A,
            'tau_t': state_A.tau_t,
            'V_t': state_A.V_t,
            'C_t': state_A.C_t
        })
        
        results_B.append({
            't': state_B.t,
            'T_local': T_B_local,  # Local verification time
            'T_network': T_B_network,  # Network-dilated verification time (A4)
            'alpha_t': state_B.alpha_t,
            'F_t': F_B,
            'tau_t': state_B.tau_t,
            'V_t': state_B.V_t,
            'C_t': state_B.C_t,
            'alpha_spread': alpha_spread,
            'temporal_dilation_delta_local': temporal_dilation_delta_local,
            'temporal_dilation_delta_network': temporal_dilation_delta_network,
            'F_spread': F_spread
        })
    
    df_A = pd.DataFrame(results_A)
    df_B = pd.DataFrame(results_B)
    
    return df_A, df_B


def plot_two_pods_comparison(df_A: pd.DataFrame, df_B: pd.DataFrame, save_path: str = None):
    """
    Plot comparison between two Pods showing A4 Network Relativity effects.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('A4 Network Relativity: Two-Pod Comparison', fontsize=16)
    
    # Plot 1: Verification Time Comparison (local vs network-dilated)
    axes[0, 0].plot(df_A['t'], df_A['T_local'], 'b-', label='Pod A (local)', linewidth=2)
    axes[0, 0].plot(df_B['t'], df_B['T_local'], 'r-', label='Pod B (local, faster)', linewidth=2)
    axes[0, 0].plot(df_B['t'], df_B['T_network'], 'r--', label='B (network-dilated, A4)', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Verification Time (local vs network-dilated)')
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('Verification Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Temporal Alpha Spread
    axes[0, 1].plot(df_A['t'], df_A['alpha_t'], 'b-', label='Pod A α_t', linewidth=2)
    axes[0, 1].plot(df_B['t'], df_B['alpha_t'], 'r-', label='Pod B α_t', linewidth=2)
    axes[0, 1].plot(df_B['t'], df_B['alpha_spread'], 'g--', label='Alpha Spread (B - A)', linewidth=2)
    axes[0, 1].set_title('Temporal Alpha: Performance Beyond Time Expectation')
    axes[0, 1].set_xlabel('Time (t)')
    axes[0, 1].set_ylabel('Temporal Alpha (α_t)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Foresight Efficiency Spread
    axes[1, 0].plot(df_A['t'], df_A['F_t'], 'b-', label='Pod A F_t', linewidth=2)
    axes[1, 0].plot(df_B['t'], df_B['F_t'], 'r-', label='Pod B F_t', linewidth=2)
    axes[1, 0].plot(df_B['t'], df_B['F_spread'], 'g--', label='F Spread (B - A)', linewidth=2)
    axes[1, 0].set_title('Foresight Efficiency: Value per Time×Capital')
    axes[1, 0].set_xlabel('Time (t)')
    axes[1, 0].set_ylabel('Foresight Efficiency (F_t)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Temporal Dilation Delta (local and network)
    axes[1, 1].plot(df_B['t'], df_B['temporal_dilation_delta_local'], 'purple', 
                    label='Local Δ(T_B - T_A)', linewidth=2)
    axes[1, 1].plot(df_B['t'], df_B['temporal_dilation_delta_network'], 'orange', 
                    label='Network Δ(T_B_network - T_A)', linewidth=2, linestyle='--')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Temporal Dilation: Local vs Network')
    axes[1, 1].set_xlabel('Time (t)')
    axes[1, 1].set_ylabel('Dilation Delta')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Two-Pod comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_simulation_results(df: pd.DataFrame, save_path: str = None):
    """
    Create visualization plots of simulation results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Proximity to the Future Fund - Pod Simulation Results', fontsize=16)
    
    # Plot 1: Foresight Efficiency over time
    axes[0, 0].plot(df['t'], df['F_t'], 'b-', linewidth=2)
    axes[0, 0].set_title('Foresight Efficiency (F_t)')
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('F_t')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trust and Leverage
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(df['t'], df['tau_t'], 'g-', label='Trust (τ_t)', linewidth=2)
    line2 = ax2_twin.plot(df['t'], df['L_t'], 'r-', label='Leverage (L_t)', linewidth=2)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Trust (τ_t)', color='g')
    ax2_twin.set_ylabel('Leverage (L_t)', color='r')
    ax2.set_title('Trust-Leverage Relationship')
    ax2.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    # Plot 3: Temporal Alpha and Coherence
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(df['t'], df['alpha_t'], 'purple', label='Temporal Alpha (α_t)', linewidth=2)
    line2 = ax3_twin.plot(df['t'], df['coherence'], 'orange', label='Coherence (Φ_t)', linewidth=2)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Temporal Alpha (α_t)', color='purple')
    ax3_twin.set_ylabel('Coherence (Φ_t)', color='orange')
    ax3.set_title('Temporal Alpha and Pod Coherence')
    ax3.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # Plot 4: Value Creation and Capital
    ax4 = axes[1, 0]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(df['t'], df['V_t'], 'b-', label='Value (V_t)', linewidth=2)
    line2 = ax4_twin.plot(df['t'], df['C_t'], 'r--', label='Capital (C_t)', linewidth=2)
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Value (V_t)', color='b')
    ax4_twin.set_ylabel('Capital (C_t)', color='r')
    ax4.set_title('Value Creation and Capital Deployment')
    ax4.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    # Plot 5: Time Violence and Trend
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(df['t'], df['time_violence'], 'r-', label='Time Violence', linewidth=2)
    line2 = ax5_twin.plot(df['t'], df['TV_trend'], 'g--', label='TV Trend (slope)', linewidth=2, alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5_twin.axhline(y=0, color='g', linestyle='--', alpha=0.3)
    ax5.set_title('Time Violence Score and Trend')
    ax5.set_xlabel('Time (t)')
    ax5.set_ylabel('Time Violence', color='r')
    ax5_twin.set_ylabel('TV Trend (negative = improving)', color='g')
    ax5.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    
    # Plot 6: Entropy Components
    axes[1, 2].plot(df['t'], df['gamma_t'], 'r-', label='Entropy (γ_t)', linewidth=2)
    axes[1, 2].plot(df['t'], df['beta_t'], 'g-', label='Trust Reinforcement (β_t)', linewidth=2)
    axes[1, 2].set_title('Entropy and Trust Reinforcement')
    axes[1, 2].set_xlabel('Time (t)')
    axes[1, 2].set_ylabel('Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Run validation tests
    model, df = prove_model()
    
    # Generate visualization
    print("\nGenerating visualization plots...")
    plot_simulation_results(df, save_path="pod_simulation_results.png")
    
    # Export results to CSV with metadata
    df.to_csv("pod_simulation_results.csv", index=False)
    print("Results exported to pod_simulation_results.csv")
    print(f"Simulation metadata: seed={RANDOM_SEED}, F_min={F_MIN_STABILITY}, alpha_ref={ALPHA_REF}, capital_max_growth={CAPITAL_MAX_GROWTH}")
    
    # G. Run A4 Network Relativity comparison experiment
    print("\n" + "=" * 60)
    print("A4 NETWORK RELATIVITY: TWO-POD COMPARISON EXPERIMENT")
    print("=" * 60)
    print("\nSimulating two Pods with different foresight velocities...")
    v_rel_A = 0.3
    v_rel_B = 0.7
    c_t = 1.0
    df_A, df_B = simulate_two_pods_comparison(n_cycles=50, v_rel_A=v_rel_A, v_rel_B=v_rel_B, c_t=c_t)
    
    print(f"\nPod A (slower): Final T_local = {df_A['T_local'].iloc[-1]:.4f}, Final α_t = {df_A['alpha_t'].iloc[-1]:.4f}")
    print(f"Pod B (faster): Final T_local = {df_B['T_local'].iloc[-1]:.4f}, Final T_network = {df_B['T_network'].iloc[-1]:.4f}, Final α_t = {df_B['alpha_t'].iloc[-1]:.4f}")
    print(f"Alpha Spread: {df_B['alpha_spread'].iloc[-1]:.4f}")
    print(f"Temporal Dilation (local): Δ(T_B_local - T_A_local) = {df_B['temporal_dilation_delta_local'].iloc[-1]:.4f}")
    print(f"Temporal Dilation (network): Δ(T_B_network - T_A_local) = {df_B['temporal_dilation_delta_network'].iloc[-1]:.4f}")
    
    # Generate comparison plots
    plot_two_pods_comparison(df_A, df_B, save_path="two_pods_comparison.png")
    
    # Export comparison results with metadata
    df_A.to_csv("pod_A_results.csv", index=False)
    df_B.to_csv("pod_B_results.csv", index=False)
    print("\nTwo-Pod comparison results exported to CSV files")
    print(f"Simulation metadata: seed={RANDOM_SEED}, v_rel_A={v_rel_A}, v_rel_B={v_rel_B}, c_t={c_t}")

