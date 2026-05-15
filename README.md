# PFE + τ-Calibration: Cross-Domain Face Recognition

## Domain-Adaptive Probabilistic Face Embeddings via Temperature Scaling

---

## Overview

**PFE** (Shi & Jain, ICCV 2019) models face embeddings as Gaussian distributions `(μ, σ)` and uses **Mutual Likelihood Score (MLS)** for matching. It achieves **~98.9% verification** on standard LFW.

However, when matching across domains (clean mugshot → degraded CCTV), the σ values become miscalibrated — the model was never trained on CCTV-quality images.

**Our contribution**: A lightweight **per-domain temperature head τ** that recalibrates σ across domains using **Strategy A (Scalar τ)**, showing that domain-adaptive calibration significantly improves cross-domain face verification.

```
σ_calibrated = σ_raw / τ(domain)
```

---

## Pipeline

| Step | Script | What It Does |
|------|--------|-------------|
| **Step 0** | `align_dataset.py` | MTCNN-based face alignment on raw LFW images |
| **Generate** | `generate_dataset.py` | Create 3-camera synthetic CCTV from LFW |
| **Step 1** | `step1_extract_features.py` | Extract μ, σ features using PFE checkpoint |
| **Step 2** | `step2_eval_baseline.py` | Evaluate vanilla PFE baseline (MLS + fusion) |
| **Step 3** | `step3_train_tau.py` | Train τ-head (Strategy A: Scalar τ) |
| **Step 4** | `step4_eval_proposed.py` | Evaluate Strategy A (camera-wise + fusion) |
| **Step 5** | `step5_compare_results.py` | Generate final comparison table |

---

## Dataset: 3-Camera Synthetic CCTV

We take LFW subjects with ≥2 images:
- **Image 1** → Clean mugshot (Domain 0)
- **Image 2** → Degraded 3 ways to simulate different CCTV cameras

| Camera | Type | Blur Kernel | Noise σ | Simulates |
|--------|------|-------------|---------|-----------|
| Cam 1 | Very Mild | (3,3) | 3 | High-quality indoor CCTV |
| Cam 2 | Mild | (3,3) | 8 | Standard indoor CCTV |
| Cam 3 | Medium Mild | (5,5) | 12 | Moderate outdoor CCTV |

**Split**: Subjects 1–1000 (train τ), Subjects 1001–1680 (test)

---

## τ-Calibration Strategy

| ID | Strategy | τ Shape | Description |
|----|----------|---------|-------------|
| A | Scalar τ | 1 per domain | Single scalar uncertainty scale per domain |

---

## Comparison Table (Output)

The final output is a single comparison table (`results/comparison_table.txt`) containing:

| Row | Description |
|-----|-------------|
| **MLS + fusion (vanilla)** | PFE baseline with multi-image fusion |
| **Scalar-tau + Fusion** | fused result |
| **Scalar-tau (per camera)** | Cam1-VeryMild, Cam2-Mild, Cam3-MedMild |

Metrics: **Accuracy**, **TAR@FAR=1%**, **TAR@FAR=0.1%**

---

## Why Vanilla PFE Fails Cross-Domain

The σ module was trained on clean web photos (CASIA-WebFace). It learned *within-domain* quality but was never exposed to *domain shift* between mugshots and CCTV.

- Mugshot σ_raw ≈ 0.009 (high confidence)
- CCTV σ_raw ≈ 0.025 (moderate — **can't distinguish quality levels!**)

These σ values are not comparable across domains → MLS scores are biased.

### Our Fix: Temperature Head τ

```
τ(domain) = exp(W_τ · one_hot(domain) + b_τ)
σ_calibrated = σ_raw / τ(domain)
```

- For CCTV domains: τ adapts to correct miscalibrated σ
- For Mugshot: τ ≈ 1 (already well-calibrated)

### Training

- **Data**: Genuine pairs (mugshot ↔ same-person CCTV) from training subjects
- **Loss**: MLS loss (maximize mutual likelihood for genuine pairs)
- **Optimizer**: Adam, lr=0.01, cosine annealing, 300 epochs
- **All CNN weights frozen** — only τ-head parameters are trained

---

## Project Structure

```
Updated_Tau_LFW_Dataset/
├── config.py                    ← Configuration (paths, strategy, hyperparams)
├── align_dataset.py             ← MTCNN-based face alignment
├── generate_dataset.py          ← Create 3-camera synthetic CCTV from LFW
├── step1_extract_features.py    ← Extract μ, σ from PFE checkpoint
├── step2_eval_baseline.py       ← Baseline evaluation (MLS + fusion)
├── step3_train_tau.py           ← Train Strategy A (Scalar τ)
├── step4_eval_proposed.py       ← Evaluate Strategy A
├── step5_compare_results.py     ← Generate comparison table
├── requirements.txt
├── HOW_TO_RUN.md                ← Step-by-step execution guide
├── utils/
│   ├── __init__.py
│   ├── dataset.py               ← Image loading + preprocessing
│   └── metrics.py               ← MLS, cosine, TAR@FAR, fusion
├── features/                    ← Extracted feature .npz files
├── results/
│   └── comparison_table.txt     ← Final comparison table
└── saved_models/                ← Trained τ-head checkpoint
```

---

## References

1. Shi, Y., & Jain, A. K. (2019). *Probabilistic Face Embeddings*. ICCV 2019.
2. Huang, G. B., et al. (2007). *Labeled Faces in the Wild*. UMass Amherst.
