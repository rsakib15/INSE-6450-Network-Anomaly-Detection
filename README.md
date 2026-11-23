# Network Traffic Anomaly Detection Using Machine Learning

**Course:** INSE 6450: AI in Systems Engineering
**Semester:** Fall 2026
**Team:** Md Sakib Rahman · Daniel Adonteng Sarpong · Karim Al-Sharara

---

## 1. Project Overview

Supervised **binary** network anomaly detection on **CICIDS2017** flow features.
Each flow is classified as **BENIGN** or **Attack** to support SOC alert prioritisation.

| Phase | Focus |
|-------|--------|
| **Milestone 1** | Problem definition, cleaning, EDA, imbalance, feature ranking, stratified split |
| **Milestone 2** | Model selection (RF primary), training, efficiency, deployment design |
| **Milestone 3** | Failure analysis, robustness tests, PSI monitoring, drift adaptation |
| **Milestone 4** | Continual learning (replay + RF retrain), HITL / active learning, system integration |

**Primary model:** Random Forest (`n_estimators=100`, `max_depth=20`, `random_state=42`)
**Comparison:** RF without class weights; MLP (PyTorch)

### Headline results

| Setting | Result |
|---------|--------|
| Clean RF (test) | F1 ≈ **0.996**, PR-AUC ≈ **0.999** |
| Noise σ=0.1 | F1 ≈ **0.010** (brittleness; mitigations in M3) |
| Adapted on drift (M3) | F1 **0 → 0.986** |
| Continual learning (M4) | F1 before=0 → after ≈ **0.985–0.987**; clean stays ≈ **0.992** |
| Active learning (M4) | F1 **0.956** (2k labels) → **0.992** (6k labels) |

---

## 2. Dataset

**CICIDS2017** — [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

Place CSV files in `data/raw/` (not shipped in this repo):

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

Optional demo data (no full CICIDS download):

```bash
python src/create_demo_data.py
```

---

## 3. Prerequisites

- Python **3.10+**
- `pip`, Git (optional)

---

## 4. Installation

```bash
git clone <repository-url>
cd INSE-6450-Network-Anomaly-Detection

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install psutil                # optional: inference memory
```

All paths are **relative** to the project root. Seed: **`RANDOM_STATE = 42`**.

---

## 5. How to Run

### Milestone 1 — Data

```bash
python src/preprocessing.py
python src/split_data.py
python src/eda.py
python src/feature_selection.py
```

### Milestone 2 — Models

```bash
python src/train_rf_baseline.py
python src/train_mlp.py                 # optional comparison
python src/compare_ablation.py
PYTHONPATH=src python src/measure_rf_inference.py
```

### Milestone 3 — Robustness, monitoring, adaptation

```bash
python src/milestone3_robustness.py
python src/milestone3_monitoring.py
python src/milestone3_adaptation.py
```

### Milestone 4 — Continual learning & HITL

```bash
python src/milestone4_continual.py
python src/milestone4_hitl_active.py
# optional:
# python src/milestone4_optimisation.py --quick
```

### End-to-end demo (final submission)

```bash
python src/demo_end_to_end.py
```

**Demo flow:** load → infer → PSI drift detect → uncertainty query (human/oracle) → replay retrain → save artefact.

### Full sequence

```bash
python src/preprocessing.py
python src/split_data.py
python src/eda.py
python src/feature_selection.py
python src/train_rf_baseline.py
python src/milestone3_robustness.py
python src/milestone3_monitoring.py
python src/milestone3_adaptation.py
python src/milestone4_continual.py
python src/milestone4_hitl_active.py
python src/demo_end_to_end.py
```

---

## 6. Project Structure

```text
INSE-6450-Network-Anomaly-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── raw/                 # CICIDS2017 CSVs (user-provided)
│   └── processed/           # all_data, train, test, attacks/
├── src/
│   ├── utils.py
│   ├── create_demo_data.py
│   ├── preprocessing.py              # M1
│   ├── split_data.py                 # M1
│   ├── eda.py                        # M1
│   ├── feature_selection.py          # M1
│   ├── data_loader.py                # shared load / split
│   ├── train_rf_baseline.py          # M2 primary
│   ├── train_mlp.py                  # M2 comparison
│   ├── compare_ablation.py
│   ├── measure_rf_inference.py
│   ├── measure_inference.py
│   ├── robustness.py                 # M3
│   ├── monitoring.py                 # M3
│   ├── adaptation.py                 # M3
│   ├── continual.py                  # M4 CL
│   ├── hitl_active.py                # M4 HITL / AL
│   ├── optimisation.py               # M4 optional
│   └── demo_end_to_end.py            # Final E2E demo
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/              # rf_baseline, rf_adapted, rf_continual_final, ...
└── reports/
    ├── Milestone1–4 (.tex / PDF)
    └── Final_Bullet_Points.md
```

---

## 7. Expected Outputs

### Models (`outputs/models/`)

| File | Description |
|------|-------------|
| `rf_baseline.joblib` | Milestone 2 primary RF |
| `rf_adapted.joblib` | M3 drift-adapted RF |
| `rf_continual_final.joblib` | M4 continual learning final |
| `rf_demo_updated.joblib` | E2E demo updated model |
| `mlp_binary.pt` | Optional MLP |

### Key tables & figures

| Artefact | Milestone |
|----------|-----------|
| `class_distribution.*`, EDA plots, feature importance | M1 |
| `rf_metrics.csv`, `rf_comparison.csv`, learning curves, confusion matrix | M2 |
| `m3_robustness_metrics.csv`, `m3_drift_*.csv`, `m3_adaptation_metrics.csv` | M3 |
| `m3_robustness_curves.png`, `m3_drift_dashboard.png`, `m3_adaptation_before_after.png` | M3 |
| `m4_continual_metrics.csv`, `m4_continual_trajectory.png` | M4 |
| `m4_active_learning_metrics.csv`, `m4_active_learning_curve.png` | M4 |
| `demo_end_to_end_summary.csv`, `demo_end_to_end.png` | Final |

---

## 8. Design Summary

| Topic | Choice |
|-------|--------|
| Task | Binary flow classification (Benign vs Attack) |
| Features | Numeric flow stats; **no** IP/port/Flow ID leakage |
| Split | Stratified 80/20, seed 42; val held out inside training scripts |
| Primary model | Random Forest |
| Imbalance | Stratified metrics (F1, PR-AUC); class-weight ablation |
| Monitoring | Schema, null rate, **PSI** (alert if mean PSI > 0.2) |
| Continual learning | Experience **replay buffer + full RF retrain** |
| HITL / AL | Abstain + tickets + drift validation; **margin** uncertainty sampling |
| Deployment | Online CPU inference; offline retrain; versioned `joblib`; rollback |

---

## 9. Reproducibility

- Fixed seed `42`
- Relative paths only (no absolute paths)
- Test set reserved for final metrics
- M3/M4 shifts are **synthetic stress tests** (documented in reports)
- GitHub repo shared with the course POD

---

## 10. License / Data

Course project code for INSE 6450.
CICIDS2017 must be obtained from the CIC under their research terms.
