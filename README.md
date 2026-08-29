# physics-informed-cholera-forecast

# Cholera Outbreak Forecasting with Physics-Informed Neural Networks (PINNs)

## Overview

This project develops a **Physics-Informed Neural Network (PINN)** framework for forecasting weekly cholera cases and deaths, by combining a mechanistic epidemiological compartmental model with deep learning. The goal is to produce forecasts that are both **data-driven** (learn patterns from historical surveillance data) and **epidemiologically grounded** (respect the underlying disease transmission dynamics), and to show that this hybrid approach outperforms standard machine learning / deep learning baselines — especially at longer forecast horizons.

The work is being developed as an academic paper targeting venues such as **IEEE ICHI**, **ACM CHIL**, or **IEEE EMBC**.

---

## Core Idea: SIRB / SIRWB Model + Neural Network

Cholera is a waterborne disease, so standard SIR-style models are extended with a **Bacteria/Water compartment (B)** to capture indirect transmission through contaminated water — giving the **SIRB** (or **SIRWB**) compartmental model:

- **S** — Susceptible
- **I** — Infected
- **R** — Recovered
- **B** — Bacterial/waterborne reservoir concentration

Rather than solving this system with fixed parameters, a neural network is trained to fit the observed weekly case/death data **while being constrained by the SIRB differential equations** (the "physics-informed" part). This lets the model:

- Learn from data like a standard neural forecaster, and
- Stay consistent with plausible disease dynamics, improving generalization — particularly useful when case data is noisy or sparse.

A **SIR-ablation** variant (dropping the waterborne compartment) is also evaluated to test whether the waterborne pathway is actually contributing meaningfully to forecast accuracy — and this contribution turns out to be **country-specific** (see below).

---

## Case Studies

After iterating through several WHO datasets (annual data, then weekly data for DRC, Tanzania, Nigeria, Cameroon, Mozambique, and Kenya), the project settled on two primary country case studies, using weekly surveillance data from the Cholera Taxonomy / public surveillance database:

| Country | Weeks of Data | Notes |
|---|---|---|
| **South Sudan** | 82 weeks | High case-fatality rate (~59.9% aggregate CFR) |
| **Somalia** | 122 weeks | 23 weeks with missing death data (handled via masked loss); low CFR (~0.62% aggregate) |

The large CFR difference between the two countries required **country-specific, CFR-anchored bounds** on the model's death-rate parameter (`p_d`) — using a generic bound (e.g. 1–10%) badly degrades death forecasts for a high-CFR setting like South Sudan.

---

## Evaluation Framework

### Multi-Horizon Forecasting
Early versions of the project evaluated only **1-week-ahead** forecasts, but this setup structurally favors autoregressive ML baselines (they can "cheat" using strong lag correlations). The framework was redesigned to evaluate forecasts at **1, 4, 8, and 12 weeks ahead**, which is a fairer and more clinically meaningful test — most baseline models' accuracy (measured by R²) collapses at longer horizons, while the PINN's mechanistic structure helps it degrade more gracefully.

### Models Compared
- RNN, LSTM, GRU, Transformer (standard DL baselines)
- Pure NN (no physics constraint)
- **SIRB-PINN** (the proposed model)
- SIR-ablation (PINN without the waterborne compartment)
- PINNsformer (Transformer-based PINN, per Zhao et al. 2023)

### Metrics
- A corrected **MASE (Mean Absolute Scaled Error)** implementation (`compute_mase_v2`) was built to match the scoring logic used in Qian et al.'s reference implementation, after discovering a bug in the original version (incorrect naive-reference denominator for deaths, plus an indexing misalignment). The fix was verified with sanity checks (a naive forecast scored against itself correctly returns MASE = 1.000 at every horizon).
- **SMAPE** is used in place of MAPE where zero-case weeks occur, since MAPE is undefined/unstable in that case.

---

## Key Findings So Far

- **Waterborne compartment matters differently by country**: including the B compartment improved short-horizon case forecasts by ~22% for Somalia, but made negligible difference for South Sudan — this is a country-specific effect, not a general rule.
- **Rainfall was not a strong predictor** of case counts at any lag for either country (correlations were non-significant), though in Somalia, wet weeks did show significantly more cases in a direct comparison (t-test, p = 0.008).
- **CFR-anchored death-rate bounds are essential** for realistic death forecasting — mismatched bounds severely bias predictions.
- **Multi-horizon evaluation reveals the PINN's real advantage**: baselines look competitive at H=1 but fail (often negative R²) by H=8–12, whereas the PINN remains stable.

---

## Technical Notes

- PINNsformer required a specific PyTorch attention backend fix (`torch.backends.cuda.enable_flash_sdp(False)` + `enable_math_sdp(True)`) to allow double-backpropagation through attention layers, which standard flash attention doesn't support.
- Training uses **rolling-window** retraining (e.g., `PRED_FROM=20`, `STRIDE=1`) to simulate realistic sequential forecasting.
- A fixed random seed (42) is used throughout for reproducibility.
- Best SIRB-PINN result for Somalia (v4) used a two-stage optimizer: Adam (3000 iterations) followed by L-BFGS (300 iterations).
- Results are checkpointed to pickle files after major runs (e.g. `south_sudan_session_state_day2.pkl`) to survive Kaggle/Colab session disconnects.

---

## Open Items / Next Steps

- **Baseline fairness check**: baselines were originally trained for 2000 iterations vs. 5000 for the PINN; since baselines (e.g. RNN) still show meaningful loss improvement between 2000–5000 iterations, this convergence gap needs to be resolved or explicitly justified before submission.
- **Sensitivity analysis** on fixed biological parameters (γ, δ) under ±5% / ±10% / ±20% perturbations.
- **Publication-quality figures** matching the style of Qian et al.
- A separate, fully corrected re-analysis using `compute_mase_v2` across both countries (kept apart from the already-submitted paper version) is planned as future work.

---

## References

- Qian et al. (2025) — rolling-window PINN retraining and scoring methodology
- Raissi, Ramezani & Seshaiyer (2019) — PINNs with learnable epidemiological parameters
- Shaier, Raissi & Seshaiyer (2022) — Disease-Informed Neural Networks (DINNs)
- Zhao et al. (2023) — PINNsformer (Transformer-based PINN architecture)
