"""
Visualization functions for sensitivity analysis and ablation studies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def plot_sensitivity_heatmaps(summary_df: pd.DataFrame, save_path: str = None):
    """
    Create heatmaps showing sensitivity of key metrics to parameters.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sensitivity Analysis: Parameter Impact on Key Metrics', fontsize=16, y=1.02)
    
    metrics = [
        ('final_F_t_mean', 'Final Foresight Efficiency (F_t)'),
        ('final_U_n_mean', 'Final Compounded Understanding (U_n)'),
        ('final_TV_mean', 'Final Time Violence'),
        ('TV_trend_mean', 'TV Trend (negative = improving)'),
        ('final_tau_mean', 'Final Trust (τ)'),
        ('final_gamma_mean', 'Final Entropy (γ)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Pivot table for heatmap
        if idx < 4:  # For main metrics, show interaction of alpha_ref and capital_max_growth
            pivot = summary_df.pivot_table(
                values=metric,
                index='capital_max_growth',
                columns='alpha_ref',
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', ax=ax, cbar_kws={'label': metric})
            ax.set_title(title)
            ax.set_xlabel('Alpha Reference (α_ref)')
            ax.set_ylabel('Capital Max Growth')
        else:  # For trust and gamma, show interaction with rho
            pivot = summary_df.pivot_table(
                values=metric,
                index='a4_soft_link_rho',
                columns='alpha_ref',
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='coolwarm', ax=ax, center=0, cbar_kws={'label': metric})
            ax.set_title(title)
            ax.set_xlabel('Alpha Reference (α_ref)')
            ax.set_ylabel('A4 Soft Link Rho (ρ)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity heatmaps saved to {save_path}")
    else:
        plt.show()


def plot_parameter_sensitivity(summary_df: pd.DataFrame, save_path: str = None):
    """
    Create line plots showing how metrics change with each parameter.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Sensitivity: Metrics vs Parameter Values', fontsize=16, y=1.02)
    
    metrics = [
        ('final_F_t_mean', 'Final Foresight Efficiency (F_t)'),
        ('final_U_n_mean', 'Final Compounded Understanding (U_n)'),
        ('final_TV_mean', 'Final Time Violence'),
        ('TV_trend_mean', 'TV Trend'),
        ('final_tau_mean', 'Final Trust (τ)'),
        ('final_gamma_mean', 'Final Entropy (γ)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Plot sensitivity to alpha_ref
        alpha_sensitivity = summary_df.groupby('alpha_ref')[metric].mean()
        ax.plot(alpha_sensitivity.index, alpha_sensitivity.values, 'o-', label='α_ref', linewidth=2, markersize=8)
        
        # Plot sensitivity to capital_max_growth
        capital_sensitivity = summary_df.groupby('capital_max_growth')[metric].mean()
        ax2 = ax.twinx()
        ax2.plot(capital_sensitivity.index, capital_sensitivity.values, 's-', 
                label='capital_max', color='orange', linewidth=2, markersize=8)
        
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel(f'{title} (α_ref)', color='blue')
        ax2.set_ylabel(f'{title} (capital_max)', color='orange')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter sensitivity plots saved to {save_path}")
    else:
        plt.show()


def plot_ablation_study(ablation_df: pd.DataFrame, save_path: str = None):
    """
    Visualize ablation study results showing A4 causal lift.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('A4 Ablation Study: Causal Lift from Network Relativity', fontsize=16, y=1.02)
    
    # Group by rho and calculate statistics
    ablation_summary = ablation_df.groupby('rho').agg({
        'F_B_final': ['mean', 'std'],
        'F_spread': ['mean', 'std'],
        'alpha_spread': ['mean', 'std'],
        'temporal_dilation_network': ['mean', 'std'],
        'T_B_local_final': ['mean', 'std'],
        'tau_B_final': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    ablation_summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in ablation_summary.columns.values]
    
    # Plot 1: Foresight Efficiency (F_B) vs rho
    ax = axes[0, 0]
    rho_values = ablation_summary['rho']
    F_B_mean = ablation_summary['F_B_final_mean']
    F_B_std = ablation_summary['F_B_final_std']
    ax.errorbar(rho_values, F_B_mean, yerr=F_B_std, marker='o', linewidth=2, 
                markersize=8, label='Pod B F_t', color='blue')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Baseline (no A4)')
    ax.set_xlabel('A4 Soft Link Rho (ρ)')
    ax.set_ylabel('Final Foresight Efficiency (F_B)')
    ax.set_title('A4 Impact on Foresight Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: F Spread (F_B - F_A) vs rho
    ax = axes[0, 1]
    F_spread_mean = ablation_summary['F_spread_mean']
    F_spread_std = ablation_summary['F_spread_std']
    ax.errorbar(rho_values, F_spread_mean, yerr=F_spread_std, marker='s', linewidth=2,
                markersize=8, label='F Spread (B - A)', color='green')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Baseline (no A4)')
    ax.set_xlabel('A4 Soft Link Rho (ρ)')
    ax.set_ylabel('F Spread (F_B - F_A)')
    ax.set_title('A4 Impact on Foresight Efficiency Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Alpha Spread vs rho
    ax = axes[0, 2]
    alpha_spread_mean = ablation_summary['alpha_spread_mean']
    alpha_spread_std = ablation_summary['alpha_spread_std']
    ax.errorbar(rho_values, alpha_spread_mean, yerr=alpha_spread_std, marker='^', linewidth=2,
                markersize=8, label='Alpha Spread (B - A)', color='purple')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Baseline (no A4)')
    ax.set_xlabel('A4 Soft Link Rho (ρ)')
    ax.set_ylabel('Alpha Spread (α_B - α_A)')
    ax.set_title('A4 Impact on Temporal Alpha Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Temporal Dilation (Network) vs rho
    ax = axes[1, 0]
    temp_dil_mean = ablation_summary['temporal_dilation_network_mean']
    temp_dil_std = ablation_summary['temporal_dilation_network_std']
    ax.errorbar(rho_values, temp_dil_mean, yerr=temp_dil_std, marker='d', linewidth=2,
                markersize=8, label='Network Dilation', color='orange')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Baseline (no A4)')
    ax.set_xlabel('A4 Soft Link Rho (ρ)')
    ax.set_ylabel('Temporal Dilation (Network)')
    ax.set_title('A4 Impact on Network Temporal Dilation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Local Verification Time (T_B) vs rho
    ax = axes[1, 1]
    T_B_mean = ablation_summary['T_B_local_final_mean']
    T_B_std = ablation_summary['T_B_local_final_std']
    ax.errorbar(rho_values, T_B_mean, yerr=T_B_std, marker='*', linewidth=2,
                markersize=10, label='T_B (local)', color='red')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Baseline (no A4)')
    ax.set_xlabel('A4 Soft Link Rho (ρ)')
    ax.set_ylabel('Local Verification Time (T_B)')
    ax.set_title('A4 Impact on Local Verification Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Causal Lift (relative to rho=0)
    ax = axes[1, 2]
    baseline_mask = ablation_summary['rho'] == 0.0
    if baseline_mask.any():
        baseline_idx = ablation_summary[baseline_mask].index[0]
        F_B_baseline = ablation_summary.loc[baseline_idx, 'F_B_final_mean']
        
        causal_lift = ((F_B_mean.values - F_B_baseline) / F_B_baseline) * 100
        ax.plot(rho_values.values, causal_lift, marker='o', linewidth=2, markersize=8, 
                label='Causal Lift (%)', color='darkgreen')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Baseline (no A4)')
        ax.set_xlabel('A4 Soft Link Rho (ρ)')
        ax.set_ylabel('Causal Lift (%)')
        ax.set_title('A4 Causal Lift: % Improvement over Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No baseline (rho=0) found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('A4 Causal Lift (no baseline)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation study plots saved to {save_path}")
    else:
        plt.show()
    
    # Print causal lift summary
    print("\n" + "=" * 60)
    print("A4 CAUSAL LIFT SUMMARY")
    print("=" * 60)
    baseline_mask = ablation_summary['rho'] == 0.0
    if baseline_mask.any():
        baseline_idx = ablation_summary[baseline_mask].index[0]
        F_B_baseline = ablation_summary.loc[baseline_idx, 'F_B_final_mean']
        causal_lift = ((F_B_mean.values - F_B_baseline) / F_B_baseline) * 100
        
        for i, rho in enumerate(rho_values.values):
            if rho > 0 and i < len(causal_lift):
                lift = causal_lift[i]
                print(f"ρ = {rho:.2f}: {lift:+.2f}% improvement over baseline (ρ=0)")
    else:
        print("No baseline (rho=0) found for comparison")


def analyze_ablation_results(ablation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ablation study results and compute causal lift statistics.
    """
    # Get baseline (rho=0) statistics
    baseline = ablation_df[ablation_df['rho'] == 0.0].groupby('replicate').agg({
        'F_B_final': 'mean',
        'F_spread': 'mean',
        'alpha_spread': 'mean',
        'temporal_dilation_network': 'mean'
    }).mean()
    
    # Group by rho and calculate statistics
    ablation_summary = ablation_df.groupby('rho').agg({
        'F_B_final': ['mean', 'std', 'min', 'max'],
        'F_spread': ['mean', 'std'],
        'alpha_spread': ['mean', 'std'],
        'temporal_dilation_network': ['mean', 'std'],
        'T_B_local_final': ['mean', 'std'],
        'tau_B_final': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    ablation_summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in ablation_summary.columns.values]
    
    # Calculate causal lift
    F_B_baseline = baseline['F_B_final']
    ablation_summary['F_B_causal_lift_pct'] = (
        (ablation_summary['F_B_final_mean'] - F_B_baseline) / F_B_baseline * 100
    )
    
    F_spread_baseline = baseline['F_spread']
    ablation_summary['F_spread_causal_lift'] = (
        ablation_summary['F_spread_mean'] - F_spread_baseline
    )
    
    return ablation_summary


if __name__ == "__main__":
    # Example usage
    print("Visualization functions for sensitivity analysis and ablation studies")
    print("Import these functions and use with your analysis results.")

