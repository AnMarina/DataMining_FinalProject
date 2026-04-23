# Uncertainty-Filtered Semi-Supervised Learning (UF-SSL)

Experiments comparing pseudo-labeling strategies for the Wisconsin Breast Cancer Diagnostic dataset, with a focus on reducing false-negative rate (FNR) in a safety-critical setting.

## Methods compared

| Method | Key idea |
|---|---|
| Supervised only | MLP / Logistic Regression trained on labeled set only |
| Vanilla PL (Lee, 2013) | Fixed confidence threshold `τ = 0.95` |
| UPS (Rizve et al., 2021) | Vanilla PL + MC-Dropout uncertainty gate |
| Adaptive UF-SSL | Confidence + linearly increasing entropy threshold |
| Adaptive UF-SSL + Weighted Loss | Stage 2: entropy-weighted sample loss |
| Cost-Sensitive Asymmetric UF-SSL | Stage 6: asymmetric class penalty + asymmetric gate |
| Risk Aware Defer Gating| Stage 7: FNR feedback loop with adaptive benign gate |

## Project structure

```
.
├── run_experiments.ipynb   # End-to-end runner — produces results table + plots
├── requirements.txt
└── src/
    ├── data.py          # Dataset loading and SSL splits
    ├── metrics.py       # Accuracy, F1, AUC, FNR, Brier, ECE, MCE
    ├── model.py         # MLP, training routines, MC-Dropout inference
    ├── ssl_methods.py   # All six SSL algorithms
    └── plots.py         # Training dynamics, calibration, entropy plots
```

## Quick start

```bash
pip install -r requirements.txt
python run_experiments.py
```

## Data split

| Subset | Fraction |
|---|---|
| Labeled | 10 % |
| Unlabeled | 40 % |
| Test | 50 % |

All splits are stratified. The scaler is fit **only** on the labeled set to prevent leakage.

## Key metric: FNR

False Negative Rate (missed malignant cases) is the primary safety metric. Lower is better.
