# Sensitivity Analysis and Ablation Study

## Overview

This module provides comprehensive sensitivity analysis and ablation studies for the Proximity to the Future Fund model, with visualization capabilities to understand parameter impacts and quantify the causal lift from A4 Network Relativity.

## Files

- `sensitivity_analysis.py`: Main analysis functions
- `visualize_sensitivity.py`: Visualization functions
- `ablation_study_raw.csv`: Raw ablation study results
- `ablation_study_summary.csv`: Aggregated ablation statistics
- `sensitivity_analysis_raw.csv`: Raw sensitivity analysis results
- `sensitivity_analysis_summary.csv`: Aggregated sensitivity statistics

## Features

### 1. Sensitivity Analysis

Sweeps key parameters across a grid:
- `alpha_ref`: Alpha reference values (default: [10.0, 15.0, 20.0, 25.0])
- `capital_max_growth`: Capital growth caps (default: [0.3, 0.5, 0.7, 1.0])
- `a4_soft_link_rho`: A4 influence coefficients (default: [0.1, 0.2, 0.3, 0.4])

**Outputs:**
- Final Foresight Efficiency (F_t)
- Final Compounded Understanding (U_n)
- Final Time Violence (TV)
- TV Trend (negative = improving)
- Final Trust (τ)
- Final Entropy (γ)

### 2. Ablation Study

Quantifies the causal lift from A4 Network Relativity by comparing:
- **Baseline (ρ=0)**: No A4 influence
- **With A4 (ρ>0)**: A4 soft-link influence active

**Metrics tracked:**
- Foresight Efficiency (F_B)
- F Spread (F_B - F_A)
- Alpha Spread (α_B - α_A)
- Temporal Dilation (Network)
- Local Verification Time (T_B)
- Causal Lift (% improvement over baseline)

### 3. Visualizations

#### Sensitivity Heatmaps
- Shows parameter interactions via heatmaps
- Highlights optimal parameter combinations
- Identifies sensitive regions

#### Parameter Sensitivity Plots
- Line plots showing metric changes vs parameter values
- Dual y-axes for multi-parameter comparison
- Identifies key parameter sensitivities

#### Ablation Study Plots
- 6-panel visualization showing A4 impact
- Causal lift quantification
- Statistical significance with error bars

## Usage

### Run Full Analysis

```bash
python3 sensitivity_analysis.py
```

This will:
1. Run sensitivity analysis across parameter grid
2. Generate sensitivity visualizations
3. Run ablation study (rho=0 vs rho>0)
4. Generate ablation visualizations
5. Print causal lift summary
6. Save all results to CSV files

### Run Ablation Study Only

```python
from sensitivity_analysis import run_ablation_study
from visualize_sensitivity import plot_ablation_study, analyze_ablation_results

# Run ablation study
ablation_df = run_ablation_study(
    alpha_ref=15.0,
    capital_max_growth=0.5,
    rho_values=[0.0, 0.1, 0.2, 0.3, 0.4],
    n_cycles=50,
    n_replicates=5
)

# Analyze results
ablation_summary = analyze_ablation_results(ablation_df)

# Visualize
plot_ablation_study(ablation_df, save_path="ablation_study.png")
```

### Run Sensitivity Analysis Only

```python
from sensitivity_analysis import run_sensitivity_analysis, analyze_sensitivity_results
from visualize_sensitivity import plot_sensitivity_heatmaps, plot_parameter_sensitivity

# Run sensitivity analysis
results_df = run_sensitivity_analysis(
    alpha_ref_range=[10.0, 15.0, 20.0],
    capital_max_growth_range=[0.3, 0.5, 0.7],
    a4_soft_link_rho_range=[0.1, 0.2, 0.3],
    n_cycles=30,
    n_replicates=2
)

# Analyze results
summary_df = analyze_sensitivity_results(results_df)

# Visualize
plot_sensitivity_heatmaps(summary_df, save_path="sensitivity_heatmaps.png")
plot_parameter_sensitivity(summary_df, save_path="parameter_sensitivity.png")
```

## Interpreting Results

### Sensitivity Analysis

- **Heatmaps**: Darker colors indicate higher values. Look for patterns showing parameter interactions.
- **Parameter Sensitivity**: Steeper slopes indicate higher sensitivity to that parameter.
- **Optimal Parameters**: Parameters that maximize F_t, minimize TV trend, or maximize U_n.

### Ablation Study

- **Causal Lift**: Percentage improvement over baseline (ρ=0). Positive values indicate A4 improves performance.
- **F Spread**: Difference in foresight efficiency between Pods. Positive values indicate Pod B outperforms Pod A.
- **Temporal Dilation**: Network-observed time dilation. Should increase with rho if A4 is working correctly.

## Key Findings

The ablation study quantifies:
1. **A4 Causal Lift**: How much A4 Network Relativity improves foresight efficiency
2. **Optimal Rho**: The rho value that maximizes performance
3. **Parameter Interactions**: How A4 interacts with other model parameters

## Notes

- Ablation study uses two-Pod comparison to test A4 influence
- Baseline (ρ=0) represents no A4 influence
- Higher rho values increase A4 influence but may have diminishing returns
- Results are reproducible with fixed random seed (RANDOM_SEED=42)

## Output Files

- `sensitivity_analysis_raw.csv`: Raw sensitivity results
- `sensitivity_analysis_summary.csv`: Aggregated sensitivity statistics
- `ablation_study_raw.csv`: Raw ablation results
- `ablation_study_summary.csv`: Aggregated ablation statistics
- `sensitivity_heatmaps.png`: Sensitivity heatmap visualizations
- `parameter_sensitivity.png`: Parameter sensitivity plots
- `ablation_study.png`: Ablation study visualizations

