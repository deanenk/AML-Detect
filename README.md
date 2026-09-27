# AML-Detect

**A research prototype for detecting adversarial machine-learning activity and assessing its escalation in critical national infrastructure (CNI).**

> ⚠️ **Status: Research prototype — NOT a deployable detector.**
> The artefact runs end-to-end, and the ensemble demonstrates baseline detection validity at a balanced operating point. However, escalation validity is **not** established: the chaos-aware risk score performs at chance (AUC = 0.5011). This repository documents both positive and negative results, as required by the study's reproducibility commitment (Chapter 3, §3.7.4)
---

## 📋 Overview

AML-Detect is a layered anomaly-detection system built for a design-science research project. It combines:

- **Three heterogeneous ML detectors** — an autoencoder, a graph neural network (GNN), and a one-class SVM (OCSVM) — combined into a weighted ensemble.
- **A chaos-aware risk engine** that aggregates ensemble evidence, local trajectory divergence, detector confidence, and persistence into an escalation index.
- **A moving-target defence (MTD) coordinator** that makes reversible, threshold-based adaptations when risk is elevated.
- **A multi-agent layer** (reactive, deliberative, supervisory agents) for distributed analysis.
- **A FastAPI service** exposing health, status, and prediction endpoints.

The system is evaluated on the **Secure Water Treatment (SWaT)** dataset — specifically a smaller Kaggle-hosted derivative, because the full iTrust archive (~104 GB) exceeded the development workstation's storage.

---

## 🔬 Key Findings (Summary)

| Component | Result | Interpretation |
|---|---|---|
| **Full ensemble** | F1 = 0.8246, ROC-AUC = 0.8609, PR-AUC = 0.8526 | Baseline detection validity at 50% prevalence |
| **Autoencoder alone** | F1 = 0.7931, PR-AUC = 0.8963, ROC-AUC = 0.8709 | **Beats the full ensemble** on threshold-independent metrics |
| **GNN alone** | ROC-AUC = 0.5000 | Performs at chance; **detrimental** to the ensemble |
| **OCSVM alone** | F1 = 0.0882 | Effectively **redundant**; removing it leaves F1 unchanged |
| **Risk engine** | AUC = 0.5011, Cohen's d = 0.0456, KS = 0.0021 | **No class separation** — escalation validity not established |
| **Operational precision** | ≈ 0.411 at 14.0% prevalence | Fewer than half of alerts would be true attacks in real conditions |
| **FGSM robustness** | 0% attack success at ε ≤ 0.05 | Bounded to FGSM only; PGD/Carlini-Wagner not tested |
| **Detection delay** | Not measured | Dataset lacks attack-onset timestamps |

**Bottom line:** The autoencoder is the primary contributing detector. The GNN and OCSVM add complexity without value. The risk layer destroys the ensemble's ranking signal. Fundamental redesign is required before any operational claim.

---

## 📂 Repository Structure
aml-detect/
├── configs/ # Configuration files
├── data/
│ ├── raw/ # Raw SWaT Kaggle derivative (not tracked)
│ └── processed/ # Processed arrays, thresholds (not tracked)
├── results/
│ ├── figures/ # Evaluation plots (ROC, PR, confusion matrix, etc.)
│ ├── metrics/ # CSV/JSON metrics and logs
│ └── models/ # Trained model artifacts (.pt, .pkl)
├── src/
│ ├── agents/ # SPADE-based multi-agent layer
│ ├── api/ # FastAPI service
│ ├── ensemble/ # autoencoder.py, gnn.py, ocsvm.py, ensemble.py
│ ├── mtd_coordinator/ # Moving-target defence logic
│ ├── preprocessing/ # preprocess.py
│ └── risk_engine/ # risk_engine.py, lyapunov.py
├── tests/ # Unit, integration, adversarial tests
├── README.md
└── requirements.txt

text

> **Note:** `data/raw/` and `data/processed/` are excluded via `.gitignore`. The SWaT Kaggle derivative is ~1.3 GB and exceeds GitHub's file size limits. See the **Data** section below to reproduce it.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13.7
- 32 GB RAM recommended (development environment: Dell G15 5515, Windows 11)
- pip

### Installation

```bash
git clone https://github.com/deanenk/AML-Detect.git
cd AML-Detect
pip install -r requirements.txt
Data Setup
The SWaT Kaggle derivative is not included in this repository. To reproduce:

Download normal.csv, attack.csv, and merged.csv from the Kaggle SWaT derivative.

Place them in data/raw/.

Run the preprocessing pipeline:

bash
python src/preprocessing/preprocess.py
This produces X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy, and scaler_params.json in data/processed/.

Dataset provenance:

Source: iTrust, Singapore University of Technology and Design (Goh et al., 2016)

Derivative: Kaggle-hosted subset

Access date: 29 August 2026

Schema: 53 columns (timestamp, 51 numeric features, binary label)

Licence: CC0 1.0 Universal

Full archive: ~104 GB (not used due to workstation storage constraints)

🏃 Running the Pipeline
Step 1 — Preprocess
bash
python src/preprocessing/preprocess.py
Loads raw CSVs, validates timestamps and labels, splits chronologically (70/15/15), fits StandardScaler on training data only, and saves processed arrays.

Step 2 — Train Detectors
bash
python src/ensemble/autoencoder.py
python src/ensemble/gnn.py
python src/ensemble/ocsvm.py
Autoencoder: Symmetric encoder-decoder with 128-dim latent space. Threshold = 95th percentile of reconstruction error on normal validation data only.

GNN: Correlation-based graph (threshold = 0.5 on training data). Early stopping (patience = 10).

OCSVM: SGDOneClassSVM with nu sensitivity analysis (nu = 0.01, 0.05, 0.1, 0.2).

Step 3 — Run the Ensemble
bash
python src/ensemble/ensemble.py
Loads trained models, normalises detector scores to [0, 1], applies confidence-weighted dynamic weights, and produces the ensemble score. Saves ensemble_weight_comparison.csv.

Step 4 — Compute Risk Scores
bash
python src/risk_engine/risk_engine.py
Combines ensemble evidence (0.35), local divergence (0.25), detector confidence (0.25), and persistence (0.15) into a bounded risk score. Calibrated on normal validation data only.

Step 5 — Run MTD Coordinator
bash
python src/mtd_coordinator/mtd.py
Activates when risk > 0.70, rolls back when risk < 0.30 (0.40 hysteresis gap). Decreases the detection threshold during elevated risk to increase sensitivity.

Step 6 — Launch the API
bash
python src/api/main.py
Or with Uvicorn:

bash
uvicorn main:app --host 0.0.0.0 --port 8000
Endpoints:

GET / — root/health message

GET /health — healthy/unhealthy

GET /status — thresholds, available models, weights, MTD state

POST /predict — takes a feature vector, returns ensemble score, risk score, weights, component scores, MTD state

GET /metrics — thresholds, model count, MTD active flag

Step 7 — Evaluation
bash
python src/evaluation/evaluate_metrics.py
python tests/test_ablation.py
python tests/test_adversarial.py
python tests/test_api_predict.py
python tests/test_agents_simple.py
Evaluation scripts:

Script	Purpose
evaluate_metrics.py	Balanced 5,000-sample held-out evaluation
test_ablation.py	Remove one detector at a time and compare metrics
test_adversarial.py	FGSM perturbations at increasing epsilon
test_api_predict.py	Validate end-to-end API prediction
test_agents_simple.py	Simulate alert, analysis, escalation
test_agents.py	Run SPADE agents through XMPP (infrastructure required)
📊 Results
Classification Performance (Balanced 5,000-Sample Test: 2,500 Normal, 2,500 Attack)
Metric	Value
Accuracy	0.8216
Precision	0.8109
Recall	0.8388
F1-score	0.8246
ROC-AUC	0.8609
PR-AUC	0.8526
False-positive rate	0.1956
False-negative rate	0.1612
Inference time	1.53 ms/sample
Detection delay	Not measured
Confusion matrix: TN = 2,011; FP = 489; TP = 2,097; FN = 403.

Ablation Study
Configuration	F1	PR-AUC	ROC-AUC
Full ensemble	0.8246	0.8526	0.8609
Without autoencoder	0.0000	0.3481	0.1135
Without GNN	0.7931	0.8525	0.8609
Without OCSVM	0.8246	0.8963	0.8709
Autoencoder alone	0.7931	0.8963	0.8709
GNN alone	0.0000	0.5000	0.5000
OCSVM alone	0.0882	0.3481	0.1135
Risk-Score Separation
Metric	Value
AUC (risk score alone)	0.5011
Cohen's d	0.0456
KS statistic	0.0021
The risk score is effectively random with respect to class membership.

FGSM Adversarial Evaluation
Epsilon	Adversarial Accuracy	Attack Success Rate
0.01	0.8740	0.0000
0.03	0.8740	0.0000
0.05	0.8740	0.0000
0.10	0.8720	0.0046
Caveat: Only FGSM tested, only 500 samples, epsilon capped at 0.10. PGD, Carlini-Wagner, and transfer-based attacks were not tested.

🛠️ Technology Stack
Component	Technology
Agents	Python, SPADE
Machine learning	PyTorch, PyTorch Geometric, scikit-learn
Interfaces	FastAPI
Deployment (planned)	Docker, Kubernetes
⚠️ Known Limitations
Risk engine is non-discriminating (AUC = 0.5011). Escalation validity is not established.

GNN performs at chance and is detrimental to the ensemble in its current static-correlation form.

OCSVM is redundant — removing it leaves F1 unchanged.

Dynamic weighting does not beat fixed weights on this baseline.

MTD never activates in the agent simulation (risk never reaches 0.70).

Threshold mismatch across layers (ensemble threshold 1360.6691 vs MTD threshold 0.5).

Adversarial robustness is bounded to FGSM at ε ≤ 0.10.

Detection delay is not measured (no attack-onset timestamps in the Kaggle derivative).

Single evaluation run — no repeated runs, confidence intervals, or significance tests.

Kaggle derivative limits generalisability vs. the full iTrust archive.

Operational precision ≈ 0.411 at 14.0% prevalence — fewer than half of alerts would be true attacks.

🔭 Future Work
Recalibrate the risk engine with normal-only recalibration and grid-search reweighting.

Replace persistence counts with a trend feature.

Redesign the GNN with temporal graph construction and a less degenerate edge rule.

Conduct an OCSVM nu sweep and kernel comparison.

Test stronger adversarial attacks (PGD, Carlini-Wagner, transfer-based).

Measure detection delay with attack-onset timestamps.

Run repeated evaluations with bootstrap confidence intervals.

Implement the AML-CNI ontology and knowledge graph (currently future work only).

📜 Licence
MIT Open-Source License.

👤 Author
Deane Nkuna

GitHub: @deanenk

📚 References
Goh, J., Adepu, S., Junejo, K.N. and Mathur, A. (2016) A dataset to support research in the design of secure water treatment systems. CRITIS.

Kapoor, S. and Narayanan, A. (2023) Leakage and the reproducibility crisis in machine-learning-based science. Patterns.

Kim et al. (2023) A comprehensive survey of anomaly detection in industrial control systems. (Ensemble diversity rationale.)

Sun et al. (2023) Moving-target defence in industrial control systems: A review.

Tuunanen, T., Winter, R. and vom Brocke, J. (2024) Dealing with complexity in design science research. MIS Quarterly.

Hevner, A. et al. (2024) Design science research transparency and validity.

Reproducibility note: Fixed seeds, dataset provenance, configuration files, and code versions are retained. Negative and inconclusive findings are reported alongside positive ones, as required by Chapter 3 (§3.7.4; §3.8.2).
