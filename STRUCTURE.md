# Repository Structure

Complete directory structure of the Proximity to the Future Fund repository.

```
proximity-to-the-future-fund/
├── README.md                    # Main entry point
├── STRUCTURE.md                 # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── docs/                        # Documentation
│   ├── INDEX.md                 # Documentation index
│   ├── core/                    # Core documentation
│   │   ├── PURPOSE.md
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── PROX_TOKENOMICS.md
│   │   └── MATHLETE_CHAIN.md
│   ├── appendices/              # Mathematical appendices
│   │   └── APPENDIX_A_MATHEMATICAL_FOUNDATIONS.md
│   └── guides/                  # User guides
│       ├── MODEL_VALIDATION.md
│       └── SENSITIVITY_ANALYSIS_README.md
│
├── src/                         # Source code
│   ├── prove_model.py           # Core model implementation
│   ├── prox_token_simulation.py  # $PROX tokenomics simulation
│   ├── mathlete_token.py        # Mathlete Token (MLT) implementation
│   ├── mathlete_lifecycle_simulation.py  # MLT lifecycle simulation
│   ├── sensitivity_analysis.py # Parameter sensitivity analysis
│   └── visualize_sensitivity.py # Visualization functions
│
└── results/                     # Output files
    ├── figures/                 # Visualization plots (PNG)
    │   ├── pod_simulation_results.png
    │   ├── two_pods_comparison.png
    │   ├── prox_token_dynamics.png
    │   ├── mathlete_lifecycle_simulation.png
    │   ├── sensitivity_heatmaps.png
    │   ├── parameter_sensitivity.png
    │   └── ablation_study.png
    └── data/                    # Time series data (CSV)
        ├── pod_simulation_results.csv
        ├── pod_A_results.csv
        ├── pod_B_results.csv
        ├── prox_token_simulation.csv
        ├── mathlete_lifecycle_simulation.csv
        ├── sensitivity_analysis_raw.csv
        ├── sensitivity_analysis_summary.csv
        ├── ablation_study_raw.csv
        └── ablation_study_summary.csv
```

## Quick Navigation

- **Getting Started**: [README.md](README.md)
- **Documentation Index**: [docs/INDEX.md](docs/INDEX.md)
- **Core Concepts**: [docs/core/PURPOSE.md](docs/core/PURPOSE.md)
- **Mathematical Foundations**: [docs/appendices/APPENDIX_A_MATHEMATICAL_FOUNDATIONS.md](docs/appendices/APPENDIX_A_MATHEMATICAL_FOUNDATIONS.md)
- **Code**: [src/](src/)
- **Results**: [results/](results/)

