
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
