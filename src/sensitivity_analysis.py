"""
Sensitivity Analysis for Proximity to the Future Fund Model

Sweeps key parameters across a grid and analyzes their impact on model outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prove_model import (
    PodState, TemporalCompoundingModel, simulate_pod,
    simulate_two_pods_comparison,
    ALPHA_REF, CAPITAL_MAX_GROWTH, A4_SOFT_LINK_RHO,
    F_MIN_STABILITY, RANDOM_SEED
)
import itertools

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def run_sensitivity_analysis(
    alpha_ref_range: list = None,
    capital_max_growth_range: list = None,
    a4_soft_link_rho_range: list = None,
    n_cycles: int = 50,
    n_replicates: int = 3
) -> pd.DataFrame:
    """
    Sweep parameters across a grid and save key outputs.
    
    Args:
        alpha_ref_range: List of ALPHA_REF values to test
        capital_max_growth_range: List of CAPITAL_MAX_GROWTH values to test
        a4_soft_link_rho_range: List of A4_SOFT_LINK_RHO values to test
        n_cycles: Number of simulation cycles
        n_replicates: Number of replicates per parameter combination
    
    Returns:
        DataFrame with results for each parameter combination
    """
    # Default ranges if not provided
    if alpha_ref_range is None:
        alpha_ref_range = [10.0, 15.0, 20.0, 25.0]
    if capital_max_growth_range is None:
        capital_max_growth_range = [0.3, 0.5, 0.7, 1.0]
    if a4_soft_link_rho_range is None:
        a4_soft_link_rho_range = [0.1, 0.2, 0.3, 0.4]
    
    results = []
    
    # Create parameter grid
    param_grid = list(itertools.product(
        alpha_ref_range,
        capital_max_growth_range,
        a4_soft_link_rho_range
    ))
    
    print(f"Running sensitivity analysis: {len(param_grid)} parameter combinations")
    print(f"Replicates per combination: {n_replicates}")
    print(f"Total simulations: {len(param_grid) * n_replicates}")
    print()
    
    for i, (alpha_ref, capital_max_growth, a4_rho) in enumerate(param_grid):
        print(f"Progress: {i+1}/{len(param_grid)} - "
              f"alpha_ref={alpha_ref}, capital_max={capital_max_growth}, rho={a4_rho}")
        
        # Store original constants
        original_alpha_ref = ALPHA_REF
        original_capital_max = CAPITAL_MAX_GROWTH
        original_rho = A4_SOFT_LINK_RHO
        
        # Temporarily modify constants (note: this is a workaround)
        # In practice, you'd pass these as parameters to evolve()
        # For now, we'll modify the constants and restore them
        
        # Run replicates
        for replicate in range(n_replicates):
            # Set random seed for reproducibility (with replicate offset)
            np.random.seed(RANDOM_SEED + replicate)
            
            # Create initial state
            initial_state = PodState(
                t=0, V_t=100.0, T_t=10.0, C_t=1000.0,
                alpha_t=0.1, beta_t=0.2, gamma_t=0.05,
                tau_t=0.6, lambda_val=0.1, L_0=1.0, k=2.0, tau_eq=0.5
            )
            
            # Create model
            model = TemporalCompoundingModel(initial_state)
            
            # Simulate with custom parameters
            # Note: We need to pass these through evolve(), so we'll use a wrapper
            # For simplicity, we'll modify the simulation to accept parameters
            try:
                # Run simulation cycles manually to pass custom parameters
                for cycle in range(n_cycles):
                    delta_V = np.random.normal(50, 10)
                    delta_T_network = np.random.normal(2, 0.5)
                    trust_change = np.random.normal(0.01, 0.02)
                    entropy_change = np.random.normal(0, 0.01)
                    
                    # Evolve with custom parameters
                    model.evolve(
                        delta_V, delta_T_network, trust_change, entropy_change,
                        alpha_ref=alpha_ref,
                        capital_max_growth=capital_max_growth
                    )
                
                # Calculate metrics
                final_state = model.state
                F_t_final = model.foresight_efficiency(final_state)
                U_n = model.total_compounded_understanding(n_cycles)
                time_violence_final = model.time_violence(final_state)
                
                # Calculate TV trend (slope over last 10 cycles)
                if len(model.history) >= 10:
                    recent_states = model.history[-10:]
                    tv_values = [model.time_violence(s) for s in recent_states]
                    tv_trend = np.polyfit(range(len(tv_values)), tv_values, 1)[0]
                else:
                    tv_trend = 0.0
                
                results.append({
                    'alpha_ref': alpha_ref,
                    'capital_max_growth': capital_max_growth,
                    'a4_soft_link_rho': a4_rho,
                    'replicate': replicate,
                    'final_F_t': F_t_final,
                    'final_U_n': U_n,
                    'final_TV': time_violence_final,
                    'TV_trend': tv_trend,
                    'final_tau': final_state.tau_t,
                    'final_gamma': final_state.gamma_t,
                    'final_T': final_state.T_t,
                    'final_C': final_state.C_t
                })
                
            except Exception as e:
                print(f"  Error in replicate {replicate}: {e}")
                results.append({
                    'alpha_ref': alpha_ref,
                    'capital_max_growth': capital_max_growth,
                    'a4_soft_link_rho': a4_rho,
                    'replicate': replicate,
                    'final_F_t': np.nan,
                    'final_U_n': np.nan,
                    'final_TV': np.nan,
                    'TV_trend': np.nan,
                    'final_tau': np.nan,
                    'final_gamma': np.nan,
                    'final_T': np.nan,
                    'final_C': np.nan,
                    'error': str(e)
                })
    
    df = pd.DataFrame(results)
    return df


def run_ablation_study(
    alpha_ref: float = 15.0,
    capital_max_growth: float = 0.5,
    rho_values: list = None,
    n_cycles: int = 50,
    n_replicates: int = 5,
    v_rel_A: float = 0.3,
    v_rel_B: float = 0.7,
    c_t: float = 1.0
) -> pd.DataFrame:
    """
    Run ablation study comparing A4 influence (rho) vs no A4 influence (rho=0).
    
    Args:
        alpha_ref: Alpha reference value
        capital_max_growth: Capital max growth value
        rho_values: List of rho values to test (must include 0.0 for baseline)
        n_cycles: Number of simulation cycles
        n_replicates: Number of replicates per rho value
        v_rel_A: Relative velocity of Pod A
        v_rel_B: Relative velocity of Pod B
        c_t: Speed of trust propagation
    
    Returns:
        DataFrame with ablation study results
    """
    if rho_values is None:
        rho_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    if 0.0 not in rho_values:
        rho_values = [0.0] + rho_values
    
    results = []
    
    print(f"Running ablation study: {len(rho_values)} rho values")
    print(f"Replicates per rho: {n_replicates}")
    print(f"Total simulations: {len(rho_values) * n_replicates * 2} (2 Pods each)")
    print()
    
    for rho in rho_values:
        print(f"Testing rho={rho:.2f}")
        
        for replicate in range(n_replicates):
            # Set random seed for reproducibility
            np.random.seed(RANDOM_SEED + replicate)
            
            try:
                # Run two-pod comparison with specific rho
                # Note: We need to modify simulate_two_pods_comparison to accept rho
                # For now, we'll create a custom version
                initial_A = PodState(
                    t=0, V_t=100.0, T_t=10.0, C_t=1000.0,
                    alpha_t=0.1, beta_t=0.2, gamma_t=0.05,
                    tau_t=0.6, lambda_val=0.1, L_0=1.0, k=2.0, tau_eq=0.5
                )
                
                initial_B = PodState(
                    t=0, V_t=100.0, T_t=10.0, C_t=1000.0,
                    alpha_t=0.15, beta_t=0.25, gamma_t=0.03,
                    tau_t=0.7, lambda_val=0.12, L_0=1.2, k=2.5, tau_eq=0.55
                )
                
                model_A = TemporalCompoundingModel(initial_A)
                model_B = TemporalCompoundingModel(initial_B)
                
                for cycle in range(n_cycles):
                    delta_V_A = np.random.normal(50, 10)
                    delta_V_B = np.random.normal(60, 10)
                    delta_T_network = np.random.normal(2, 0.5)
                    trust_change_A = np.random.normal(0.01, 0.02)
                    trust_change_B = np.random.normal(0.015, 0.02)
                    entropy_change_A = np.random.normal(0, 0.01)
                    entropy_change_B = np.random.normal(-0.005, 0.01)
                    
                    # Get current states
                    state_A_current = model_A.state
                    state_B_current = model_B.state
                    
                    # Calculate A4 prediction
                    T_A_current = state_A_current.T_t
                    T_B_current_local = state_B_current.T_t
                    T_B_network_predicted = model_A.network_relativity(
                        T_B_current_local, v_rel_B - v_rel_A, c_t
                    )
                    
                    # Calculate F values for conditional A4
                    F_A_current = model_A.foresight_efficiency(state_A_current)
                    F_B_current = model_B.foresight_efficiency(state_B_current)
                    
                    # Evolve Pods
                    model_A.evolve(
                        delta_V_A, delta_T_network, trust_change_A, entropy_change_A,
                        alpha_ref=alpha_ref,
                        capital_max_growth=capital_max_growth,
                        F_other=F_B_current
                    )
                    
                    # For Pod B, use specified rho (0 for ablation baseline)
                    if rho == 0.0:
                        # No A4 influence
                        model_B.evolve(
                            delta_V_B, delta_T_network, trust_change_B, entropy_change_B,
                            alpha_ref=alpha_ref,
                            capital_max_growth=capital_max_growth,
                            T_from_A4=None,  # No A4 influence
                            F_other=F_A_current
                        )
                    else:
                        # With A4 influence
                        model_B.evolve(
                            delta_V_B, delta_T_network, trust_change_B, entropy_change_B,
                            alpha_ref=alpha_ref,
                            capital_max_growth=capital_max_growth,
                            T_from_A4=T_B_network_predicted,
                            rho=rho,
                            F_other=F_A_current
                        )
                
                # Calculate final metrics
                state_A_final = model_A.state
                state_B_final = model_B.state
                
                F_A_final = model_A.foresight_efficiency(state_A_final)
                F_B_final = model_B.foresight_efficiency(state_B_final)
                
                # Calculate temporal dilation
                T_A_final = state_A_final.T_t
                T_B_final_local = state_B_final.T_t
                T_B_final_network = model_A.network_relativity(
                    T_B_final_local, v_rel_B - v_rel_A, c_t
                )
                
                alpha_spread = state_B_final.alpha_t - state_A_final.alpha_t
                F_spread = F_B_final - F_A_final
                temporal_dilation_local = T_B_final_local - T_A_final
                temporal_dilation_network = T_B_final_network - T_A_final
                
                results.append({
                    'rho': rho,
                    'replicate': replicate,
                    'F_A_final': F_A_final,
                    'F_B_final': F_B_final,
                    'F_spread': F_spread,
                    'alpha_spread': alpha_spread,
                    'T_A_final': T_A_final,
                    'T_B_local_final': T_B_final_local,
                    'T_B_network_final': T_B_final_network,
                    'temporal_dilation_local': temporal_dilation_local,
                    'temporal_dilation_network': temporal_dilation_network,
                    'tau_A_final': state_A_final.tau_t,
                    'tau_B_final': state_B_final.tau_t,
                    'gamma_A_final': state_A_final.gamma_t,
                    'gamma_B_final': state_B_final.gamma_t
                })
                
            except Exception as e:
                print(f"  Error in replicate {replicate}: {e}")
                results.append({
                    'rho': rho,
                    'replicate': replicate,
                    'error': str(e)
                })
    
    df = pd.DataFrame(results)
    return df


def analyze_sensitivity_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sensitivity analysis results and compute statistics.
    
    Returns:
        DataFrame with aggregated statistics per parameter combination
    """
    # Group by parameters and compute statistics
    grouped = df.groupby(['alpha_ref', 'capital_max_growth', 'a4_soft_link_rho'])
    
    summary = grouped.agg({
        'final_F_t': ['mean', 'std', 'min', 'max'],
        'final_U_n': ['mean', 'std', 'min', 'max'],
        'final_TV': ['mean', 'std', 'min', 'max'],
        'TV_trend': ['mean', 'std', 'min', 'max'],
        'final_tau': ['mean', 'std'],
        'final_gamma': ['mean', 'std'],
        'final_T': ['mean', 'std'],
        'final_C': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in summary.columns.values]
    
    return summary


if __name__ == "__main__":
    from src.visualize_sensitivity import (
        plot_sensitivity_heatmaps,
        plot_parameter_sensitivity,
        plot_ablation_study,
        analyze_ablation_results
    )
    
    # Setup results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
    
    print("=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    print()
    
    # Run sensitivity analysis
    results_df = run_sensitivity_analysis(
        alpha_ref_range=[10.0, 15.0, 20.0],
        capital_max_growth_range=[0.3, 0.5, 0.7],
        a4_soft_link_rho_range=[0.1, 0.2, 0.3],
        n_cycles=30,
        n_replicates=2
    )
    
    # Save raw results
    results_df.to_csv(os.path.join(results_dir, "data", "sensitivity_analysis_raw.csv"), index=False)
    print(f"\nRaw results saved to results/data/sensitivity_analysis_raw.csv")
    print(f"Total parameter combinations: {len(results_df)}")
    
    # Analyze and save summary
    summary_df = analyze_sensitivity_results(results_df)
    summary_df.to_csv(os.path.join(results_dir, "data", "sensitivity_analysis_summary.csv"), index=False)
    print(f"Summary saved to results/data/sensitivity_analysis_summary.csv")
    
    # Generate visualizations
    print("\nGenerating sensitivity analysis visualizations...")
    plot_sensitivity_heatmaps(summary_df, save_path=os.path.join(results_dir, "figures", "sensitivity_heatmaps.png"))
    plot_parameter_sensitivity(summary_df, save_path=os.path.join(results_dir, "figures", "parameter_sensitivity.png"))
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"\nBest F_t (highest mean):")
    best_F = summary_df.loc[summary_df['final_F_t_mean'].idxmax()]
    print(f"  alpha_ref={best_F['alpha_ref']:.1f}, "
          f"capital_max={best_F['capital_max_growth']:.2f}, "
          f"rho={best_F['a4_soft_link_rho']:.2f}")
    print(f"  Mean F_t: {best_F['final_F_t_mean']:.6f}")
    
    print(f"\nLowest TV trend (most negative, best for time-violence reduction):")
    best_TV_trend = summary_df.loc[summary_df['TV_trend_mean'].idxmin()]
    print(f"  alpha_ref={best_TV_trend['alpha_ref']:.1f}, "
          f"capital_max={best_TV_trend['capital_max_growth']:.2f}, "
          f"rho={best_TV_trend['a4_soft_link_rho']:.2f}")
    print(f"  Mean TV trend: {best_TV_trend['TV_trend_mean']:.6f}")
    
    print(f"\nHighest U_n (compounded understanding):")
    best_U = summary_df.loc[summary_df['final_U_n_mean'].idxmax()]
    print(f"  alpha_ref={best_U['alpha_ref']:.1f}, "
          f"capital_max={best_U['capital_max_growth']:.2f}, "
          f"rho={best_U['a4_soft_link_rho']:.2f}")
    print(f"  Mean U_n: {best_U['final_U_n_mean']:.2e}")
    
    # Run ablation study
    print("\n" + "=" * 60)
    print("ABLATION STUDY: A4 CAUSAL LIFT")
    print("=" * 60)
    print()
    
    ablation_df = run_ablation_study(
        alpha_ref=15.0,
        capital_max_growth=0.5,
        rho_values=[0.0, 0.1, 0.2, 0.3, 0.4],
        n_cycles=50,
        n_replicates=5
    )
    
    # Save ablation results
    ablation_df.to_csv(os.path.join(results_dir, "data", "ablation_study_raw.csv"), index=False)
    print(f"\nAblation study results saved to results/data/ablation_study_raw.csv")
    
    # Analyze ablation results
    ablation_summary = analyze_ablation_results(ablation_df)
    ablation_summary.to_csv(os.path.join(results_dir, "data", "ablation_study_summary.csv"), index=False)
    print(f"Ablation summary saved to results/data/ablation_study_summary.csv")
    
    # Generate ablation visualizations
    print("\nGenerating ablation study visualizations...")
    plot_ablation_study(ablation_df, save_path=os.path.join(results_dir, "figures", "ablation_study.png"))
    
    # Print causal lift summary
    print("\n" + "=" * 60)
    print("A4 CAUSAL LIFT SUMMARY")
    print("=" * 60)
    baseline_F = ablation_summary[ablation_summary['rho'] == 0.0]['F_B_final_mean'].values[0]
    for _, row in ablation_summary.iterrows():
        if row['rho'] > 0:
            lift_pct = row['F_B_causal_lift_pct']
            print(f"ρ = {row['rho']:.2f}: {lift_pct:+.2f}% improvement over baseline (ρ=0)")
            print(f"  F_B: {baseline_F:.6f} → {row['F_B_final_mean']:.6f}")
            print(f"  F_spread lift: {row['F_spread_causal_lift']:+.6f}")
    print()
    print(f"Baseline (ρ=0) F_B: {baseline_F:.6f}")
    print(f"Best improvement at ρ={ablation_summary.loc[ablation_summary['F_B_causal_lift_pct'].idxmax(), 'rho']:.2f}")
    print(f"  Causal lift: {ablation_summary['F_B_causal_lift_pct'].max():+.2f}%")

