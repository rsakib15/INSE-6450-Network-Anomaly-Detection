# Network Traffic Anomaly Detection Using Machine Learning

**Course:** INSE 6450: AI in Systems Engineering
**Semester:** Fall 2026

---

## 1. Project Overview

An AI-based network intrusion detection system that analyses flow-level network traffic to identify suspicious activity and potential cyberattacks. The system classifies each flow as **benign** or **attack** using supervised machine learning on the **CICIDS2017** dataset.

| Phase | Focus |
| ------- | -------- |
| **Milestone 1** | Data cleaning, EDA, class imbalance, feature ranking, stratified train/test split |
| **Milestone 2** | Model selection, training, evaluation, efficiency metrics, deployment design |

**Primary model (Milestone 2):** Random Forest classifier
**Comparison models:** Random Forest without class weights; MLP (PyTorch) with/without class-weighted loss

---

## 2. Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity (University of New Brunswick)
Official page: <https://www.unb.ca/cic/datasets/ids-2017.html>

### Setup

Download the CICIDS2017 **CSV** files and place them in:

```text
data/raw/
```

Typical files:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

File names may vary slightly by download source. The original dataset is **not** stored in this repository.

**Quick test without the full dataset:**

```bash
python src/create_demo_data.py
```

---

## 3. Prerequisites

- Python **3.10+**
- `pip`
- Git (optional)

---

## 4. Installation

```bash
git clone <repository-url>
cd INSE-6450-Network-Anomaly-Detection
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```powershell
python -m venv venv
venv\Scripts\activate
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (inference memory measurement):

```bash
pip install psutil
```

CPU-only PyTorch (if needed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. How to Run

Run **all commands from the project root**. Later steps depend on earlier outputs.

### Milestone 1 — Data pipeline

```bash
# 1) Clean and merge raw CSVs → data/processed/all_data.csv
python src/preprocessing.py

# 2) Stratified 80/20 split → train.csv, test.csv + split figure
python src/split_data.py

# 3) EDA plots and tables
python src/eda.py

# 4) Feature importance + balanced attack subsets
python src/feature_selection.py
```

Optional fast preprocessing test:

```bash
python src/preprocessing.py --quick
```

### Milestone 2 — Models and evaluation

```bash
# Primary model: Random Forest
python src/train_rf_baseline.py
python src/train_mlp.py

# Optional unweighted MLP ablation
python src/train_mlp.py --no-class-weights
# mv outputs/tables/mlp_metrics.csv outputs/tables/mlp_metrics_unweighted.csv
# python src/train_mlp.py    # re-train weighted as main MLP metrics file

# Merge ablation table (uses rf_comparison.csv when available)
python src/compare_ablation.py
python src/measure_rf_inference.py

# Optional MLP inference latency
python src/measure_inference.py
```

### Recommended full sequence

```bash
python src/preprocessing.py
python src/split_data.py
python src/eda.py
python src/feature_selection.py
python src/train_rf_baseline.py
python src/train_mlp.py
python src/compare_ablation.py
PYTHONPATH=src python src/measure_rf_inference.py
```

---

## 6. Project Structure

```text
INSE-6450-Network-Anomaly-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── mlp_binary_v1.json
├── data/
│   ├── raw/                      # Original CICIDS2017 CSVs (user-provided)
│   └── processed/
│       ├── all_data.csv
│       ├── train.csv
│       ├── test.csv
│       └── attacks/              # Balanced per-attack subsets
├── src/
│   ├── utils.py
│   ├── create_demo_data.py
│   ├── preprocessing.py          # Milestone 1
│   ├── split_data.py             # Milestone 1
│   ├── eda.py                    # Milestone 1
│   ├── feature_selection.py      # Milestone 1
│   ├── data_loader.py            # Milestone 2 shared loading
│   ├── train_rf.py      # Milestone 2 primary model
│   ├── train_mlp.py              # Milestone 2 MLP comparison
│   ├── compare_ablation.py
│   ├── measure_rf_inference.py
│   └── measure_inference.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/                   # rf_baseline.joblib, mlp_binary.pt, scaler
└── reports/
    ├── Milestone1.pdf
    └── Milestone2.pdf
```

All paths are **relative** to the project root (no hard-coded absolute paths).
Random seed: **`RANDOM_STATE = 42`**.

---

## 7. Expected Outputs

### Milestone 1

| Output | Location |
| -------- | ---------- |
| Cleaned full dataset | `data/processed/all_data.csv` |
| Train / test split | `data/processed/train.csv`, `test.csv` |
| Class distribution | `outputs/tables/class_distribution.csv`, `figures/class_distribution.png` |
| Split figure | `outputs/figures/train_test_split.png` |
| EDA plots | `flow_duration_hist.png`, `bwd_pkt_len_boxplot.png`, `correlation_heatmap.png` |
| Feature importance | `outputs/tables/feature_importance_*.csv`, `figures/feature_importance_*.png` |
| Attack subsets | `data/processed/attacks/*.csv` |

### Milestone 2

| Output | Location |
| -------- | ---------- |
| Trained Random Forest | `outputs/models/rf_baseline.joblib` |
| RF metrics (val/test) | `outputs/tables/rf_metrics.csv` |
| RF learning curve | `outputs/figures/rf_learning_curve.png`, `tables/rf_learning_curve.csv` |
| RF confusion matrix | `outputs/figures/confusion_matrix_rf.png` |
| Model comparison | `outputs/tables/rf_comparison.csv`, `figures/rf_comparison_bars.png` |
| RF train efficiency | `outputs/tables/rf_efficiency.csv` |
| RF inference efficiency | `outputs/tables/rf_inference_efficiency.csv` |
| MLP model (optional) | `outputs/models/mlp_binary.pt`, `scaler.joblib` |
| MLP metrics / curves | `outputs/tables/mlp_*.csv`, `figures/learning_curves.png` |
| Ablation table | `outputs/tables/ablation_comparison.csv` |

---

## 8. Model Summary (Milestone 2)

| Model | Role | Notes |
| ------- | ------ | -------- |
| **Random Forest** (`class_weight` ablation) | **Primary** | Best test F1 / precision on tabular CICIDS flows |
| MLP (PyTorch) | Comparison | Weighted loss raises recall, lowers precision vs unweighted |

**Default RF hyperparameters:** `n_estimators=100`, `max_depth=20`, `random_state=42`.

**Split:** stratified 80% train / 20% test; 15% of train held out as validation inside training scripts.
