# The Proximity Token ($PROX): A Predictable Crypto-Economic Model

## I. Overview

The Proximity Token ($PROX) is a self-regulating crypto asset derived from the mathematical framework of the Proximity to the Future Fund. It integrates the principles of temporal compounding, trust-weighted leverage, and entropy-based burn mechanics to produce a predictable, equilibrium-seeking token economy.

The $PROX model is designed to measure and reward proximity to verified understanding—that is, how efficiently a network converts foresight and trust into lasting value without inducing time violence (instability caused by excessive acceleration).

---

## II. Core Principles

| Economic Variable | Symbol | Description | Token Equivalent |
|-------------------|--------|-------------|------------------|
| Foresight Efficiency | F_t | Verified value per capital × time | Productivity yield |
| Trust Leverage | L_t | Multiplier on capital efficiency from collective trust | Staking multiplier |
| Entropy | γ_t | Rate of systemic uncertainty | Volatility cost |
| Trust Level | τ_t | Confidence in network's coherence (0–1) | Staking participation ratio |
| Time Violence | TV_t | Energy lost to misalignment or over-acceleration | Automatic burn rate |
| Compounded Understanding | U_n | Total accumulated network insight | Treasury growth / intrinsic value |

These principles mirror the economic structure already validated in simulation—providing a closed, measurable system for crypto token dynamics.

---

## III. Token Mechanics

### 1. Minting Rule (Growth Phase)

At each epoch \(t\):

\[
\text{Mint}_{t+1} = S_t \cdot F_t \cdot L_t
\]

where \(S_t\) is circulating supply.

This reflects the verified productivity of the network. The more foresight efficiency and trust leverage increase, the more supply is minted. However, this growth is bounded by capital and coherence conditions defined in Appendix A of the parent model.

### 2. Burn Rule (Stabilization Phase)

Burns are triggered automatically when systemic entropy or trust instability rises:

\[
\text{Burn}_{t+1} = S_t \cdot TV_t = S_t \cdot \gamma_t T_t (1 - \tau_t)
\]

This acts as a self-correcting feedback loop, ensuring that expansion never outpaces understanding.

### 3. Net Supply Equation

\[
S_{t+1} = S_t + S_t(F_t L_t - \gamma_t T_t(1 - \tau_t))
\]

The total token supply evolves endogenously—increasing during stable foresight and contracting during volatile phases.

---

## IV. Trust-Weighted Staking

Validator or pod rewards are determined by the Trust-Leverage relationship:

\[
R_i \propto L_t = L_0(1 + k(\tau_t - \tau_{eq}))
\]

Validators maintaining high coherence (\(\tau_t > \tau_{eq}\)) receive increased yield.

Low-trust or misaligned pods experience diminishing returns, reducing incentive for reckless acceleration.

This creates a trust gradient that naturally stabilizes network behavior.

---

## V. Temporal Policy Layer

The learning rate (\(\lambda\)) and sensitivity constant (\(k\)) dynamically adjust based on time-violence trends:

- If \(TV_t\) rises: \(\lambda \downarrow\) (network slows learning rate)
- If \(F_t\) and \(\tau_t\) rise while \(TV_t \downarrow\): \(\lambda \uparrow\)

This forms an adaptive monetary policy: speed is modulated to maintain coherent progress.

---

## VI. Predictability and Stability

Unlike conventional crypto systems with arbitrary issuance or halving schedules, $PROX derives all monetary behavior from observable state variables. The supply path is mathematically forecastable:

\[
E[S_t] = S_0 \prod_{i=0}^{t} (1 + F_i L_i - \gamma_i T_i(1 - \tau_i))
\]

Because \(F_t\), \(L_t\), \(\tau_t\), and \(\gamma_t\) are bounded within empirically measured ranges, the token's long-term inflation and deflation curves are predictable.

---

## VII. Implementation Layers

| Layer | Mechanism | Example Technology |
|-------|-----------|-------------------|
| **Core Logic** | On-chain Temporal Compounding Engine (A1–A10 equations) | Solidity / Move module |
| **Trust Oracle Layer** | Off-chain and on-chain data aggregation for \(\tau_t, \gamma_t\) | Chainlink, subDAO attestation |
| **Pod Architecture** | Each pod = subDAO validating its foresight efficiency | Cosmos SDK / Optimism Superchain |
| **Treasury Module** | Tracks \(U_n\); funds long-term R&D or retroactive public goods | Superfluid, Sablier, Gnosis Safe |
| **Governance** | Weighted by \(\tau_t^2\) to privilege coherence over capital | Snapshot + reputation staking |

---

## VIII. Simulation Insights for Token Policy

Sensitivity analysis results (20+ parameter sweeps) demonstrated:

- **High foresight efficiency** with low volatility at \(\alpha_{ref} = 10\), capital cap \(= 0.3\), and \(\rho = 0.1\).
- **Lowest time violence** (best stability) at \(\alpha_{ref} = 20\), same capital cap.
- **A balanced operating range**: \(\alpha_{ref} \approx 15\), capital cap 0.3–0.5, \(\rho \approx 0.1\text{–}0.2\).

These findings directly translate to token policy parameters for minting, leverage multipliers, and adaptive burn rates.

---

## IX. Economic Interpretation

$PROX functions as a temporal stability token:

- **Inflation** is a function of foresight productivity.
- **Deflation** is a function of entropy and distrust.
- **Capital compounding** tracks understanding, not speculation.

This creates a trust-to-yield economy—a new form of predictive capital market where investors stake belief in verified foresight rather than speculative hype.

---

## X. Conclusion

The Proximity Token defines a predictable crypto-economic framework rooted in the mathematics of temporal compounding and network relativity. It replaces arbitrary tokenomics with an equation of understanding, aligning supply growth with verified foresight and trust.

As each pod within the network maintains coherence, $PROX serves as both measurement and medium—the first token whose value is a direct function of how close a network is to the future it understands.

---

## XI. Token Supply Simulation

See `prox_token_simulation.py` for a complete simulation of $PROX token mechanics, including:
- Minting and burning dynamics
- Trust-weighted staking rewards
- Supply evolution over time
- Predictability analysis

---

**Author:** Leo Guinan  
**Affiliation:** Idea Nexus Ventures / MetaSPN  
**Date:** November 2025

