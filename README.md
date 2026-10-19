
# Network Traffic Anomaly Detection Using Machine Learning

**Course:** INSE 6450: AI in Systems Engineering
**Semester:** Fall 2026

## 1. Project Overview

An AI-based network intrusion detection system that analyzes network traffic to identify suspicious activity and potential cyberattacks. Machine learning is used to classify network traffic as benign or malicious and distinguish between different attack types.

## Milestones

### Milestone 1 Objectives

- Clean and preprocess the CICIDS2017 dataset.
- Perform exploratory data analysis (EDA) and investigate class imbalance.
- Rank feature importance using a Random Forest model.
- Generate balanced, attack-specific datasets for further experimentation.
- Produce reproducible outputs, including cleaned CSV files, summary tables, and visualizations.

## 2. Dataset

This project uses the **CICIDS2017 dataset**, published by the Canadian Institute for Cybersecurity (CIC).

**Official dataset:** <https://www.unb.ca/cic/datasets/ids-2017.html>

### Dataset Setup

Download the CICIDS2017 CSV files and place all eight daily traffic CSV files in the following directory:

```text
data/raw/
```

Expected files include:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

*Note: File names may vary slightly depending on the downloaded version.*

The original dataset is not included in this repository. Users must download it separately and place the files in `data/raw/`.

## 3. Prerequisites

Ensure the following software is installed:

- Python 3.10 or later
- pip
- Git (optional, for cloning the repository)

## 4. Installation

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd INSE-6450-Network-Anomaly-Detection
```

Create and activate a virtual environment.

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

Install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. How to Run

Run all commands from the project root directory.

### Step 1: Prepare the Dataset

Place the original CICIDS2017 CSV files in `data/raw/`.

### Step 2: Preprocess the Data

```bash
python src/preprocessing.py
```

This step cleans and combines the input CSV files and prepares the processed dataset.

### Step 3: Perform Exploratory Data Analysis

```bash
python src/eda.py
```

This step generates descriptive statistics, data-quality summaries, class distributions, and visualizations to help investigate the dataset and its class imbalance.

### Step 4: Analyze Feature Importance

```bash
python src/feature_selection.py
```

This step ranks the input features using a Random Forest model and generates feature-importance outputs for further analysis.

**Important:** Run the scripts in the order shown above because later stages may depend on outputs produced by earlier stages.

## 6. Project Structure

```text
INSE-6450-Network-Anomaly-Detection
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Original CICIDS2017 CSV files
│   └── processed/            # Cleaned and processed datasets
│       └── attacks/          # Attack-specific balanced subsets
├── src/
│   ├── utils.py              # Shared utility functions
│   ├── create_demo_data.py   # Demo dataset generation
│   ├── preprocessing.py      # Data cleaning and preprocessing
│   ├── eda.py                # Exploratory data analysis
│   └── feature_selection.py  # Feature-importance analysis
├── outputs/
│   ├── figures/              # Generated plots and visualizations
│   └── tables/               # CSV summaries and analysis tables
└── reports/
    └── Milestone1.pdf        # Milestone 1 report
```

## 7. Expected Outputs

After running the scripts successfully, the project produces the following types of artifacts:

| Output | Description |
| --- | --- |
| Cleaned dataset | Combined and preprocessed network traffic data |
| Data-quality summaries | Missing values, duplicates, and other quality checks |
| Class distribution analysis | Traffic and attack-category distributions |
| Visualizations | Plots for exploratory data analysis |
| Feature-importance tables | Ranked features from the Random Forest model |
| Attack-specific subsets | Balanced datasets for selected attack categories |

Generated files are stored under `data/processed/` and `outputs/`, according to the implementation of each script.
