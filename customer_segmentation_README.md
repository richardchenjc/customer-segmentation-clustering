# Customer Segmentation & Behavioral Profiling

End-to-end unsupervised segmentation of 1.3M credit-card transactions into three actionable customer personas, with a rule-based surrogate for production deployment without an ML runtime.

**Author:** Richard Chen ([@richardchenjc](https://github.com/richardchenjc))
**Course context:** DSA5102 (Introduction to Data Science & Machine Learning), NUS — individual notebook within a group project scoped to Fraud Detection, Next-Purchase Prediction, and Segmentation. This repo contains my solo segmentation submission only.

---

## TL;DR

- **Input:** ~1.3M credit-card transactions (Kaggle `priyamchoksi/credit-card-transactions-dataset`), filtered to non-fraud.
- **Output:** 983 unique customer profiles assigned to one of three behavioral personas, plus a depth-3 Decision Tree surrogate translating the clustering into two-threshold production rules.
- **Best model:** K-Means, `k = 3`, Silhouette **0.2905** — tied with GMM at the same `k`, cross-validated by HDBSCAN density analysis.

| Cluster | Persona | Defining Traits | Strategy |
|:--:|:--|:--|:--|
| **0** | Wealthy VIP | High total spend, travel-heavy, low essentials engagement | Premium upsell |
| **1** | Budget Night Owl | Active off-hours, low total spend | Reactivation |
| **2** | Mainstream Family | High frequency, low ticket size, essentials-focused | Habit retention |

---

## Why this project

Conventional RFM (Recency–Frequency–Monetary) segmentations bias toward wealth — a high-net-worth user buying only essentials is indistinguishable from a student with identical purchasing habits. This project decouples **Wealth (magnitude)** from **Lifestyle (behavior)** through engineered features so the resulting personas reflect intrinsic consumption habits rather than financial capacity alone.

---

## Methodology

**1. Feature engineering.** 15 behavioral features across four families:
- **Geospatial mobility:** Haversine distance-from-home + **Radius of Gyration** (physics-based dispersion metric separating Locals from Travelers).
- **Complexity / loyalty:** Shannon entropy of spend categories (`cat_entropy`) — measures predictability independent of volume.
- **Temporal ratios:** weekend / night / grocery / gas / dining / travel share of wallet.
- **Magnitude:** log-transformed totals, means, max, std of `amt` and `dist_km`.

**2. Validation.** Dual-metric selection — Silhouette Score for separation quality, Elbow Method on a 50k sample for variance-drop inflection. Both agreed at `k = 3`.

**3. Multi-model benchmark.**
- **K-Means** (partitioning) — selected for stability and interpretable boundaries.
- **GMM** (probabilistic) — converged on the same global max at k=3, confirming structural stability; degraded at k>3 due to covariance overfitting.
- **HDBSCAN** (density, Manhattan distance for high-dim robustness) — flagged 30.2% noise and collapsed to 2 cores, revealing that Wealth is a *continuous spectrum*; K-Means imposes a useful business cutoff on top.

**4. Interpretability.**
- **PCA loadings** — named the two principal axes "Velocity" (txn_count / entropy vs. amt_mean) and "Mobility" (dist_km / RoG).
- **Heatmap + Radar** — persona DNA across the 7 story-telling features.
- **Decision Tree surrogate** (depth=3) — extracts production-grade rules with two thresholds: ATV ≈ \$94, Travel ratio ≈ 11%.

**5. Temporal stability.** Q1→Q4 Transition Matrix over 908 customers active in both quarters — surfaces the "Sticky Extremes" pattern (83.7% of Low and 77.5% of V.High retain quartile) and identifies the High-Value tier as the marketing battleground (59% retention, 19.8% upgrade vs. 19.8% downgrade).

**6. Geospatial validation.** Folium maps (K-Means + HDBSCAN overlays) confirm the segments are **geographically agnostic** — differentiation is driven by spending power and day-night habits, not region.

---

## Key findings

- **k = 3 is the structural truth.** Both K-Means and GMM hit the same global-max silhouette (0.2905) at k=3 — a rare convergence between a geometric and a probabilistic model, and strong evidence that three archetypes are the real structure in this data.
- **Mobility is not a differentiator.** PC2 ("Mobility") explains variance, but personas are distributed across the same geographic regions. Strategy: deploy nationally, don't segment by region.
- **The "Aspiring" tier is the battleground.** 59% quartile-retention in the High-Value segment vs. 77–83% at the extremes — market resources should be concentrated here.
- **Two thresholds collapse the model into a rule.** The Decision Tree surrogate shows that *Average Transaction Value* and *Travel Spend Percentage* alone reproduce the clustering logic, enabling production deployment without an ML runtime.

---

## Repository contents

```
.
├── customer_segmentation.ipynb   # The full analysis (load → features → cluster → validate → rules)
├── README.md                     # This file
└── figures/                      # (optional) exported key plots
```

---

## Reproducing

**Requirements:** Python 3.10+, with:

```
pandas numpy seaborn matplotlib folium scipy scikit-learn
```

The notebook uses the `HDBSCAN` class from `sklearn.cluster` (scikit-learn ≥ 1.3).

**Data:** the notebook pulls `priyamchoksi/credit-card-transactions-dataset` via the Kaggle API. Configure your Kaggle credentials first (`~/.kaggle/kaggle.json`), then run all cells top-to-bottom. Runtime on the full 1.3M-row dataset: ~3–5 minutes on a modern laptop (feature engineering is vectorized; the Elbow Method uses a 50k sample).

---

## Notes on scope

This notebook is my individual submission. The broader DSA5102 group project covered Fraud Detection and Next-Purchase Prediction as separate modules; those are not included here. The segmentation output (`Cluster_Label`) was designed to feed the Next-Purchase module as a behavioral context feature, but the supervised pipelines themselves are not in this repo.
