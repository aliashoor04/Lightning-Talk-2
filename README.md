# Motor Fault Detection Using Motor Current Signature Analysis (MCSA)

**Author:** Ali Ashoor  
**Program:** General Assembly Data Science Bootcamp — Bahrain, June 2026 Cohort  
**Submission:** Lightning Talk 2 — Capstone Part 2 · July 2026

---

## Project Overview

Induction motors account for nearly 70% of industrial electricity consumption worldwide. When they fail unexpectedly, the cost of downtime frequently exceeds the cost of the motor itself. This project builds a machine learning fault detection system on top of Motor Current Signature Analysis (MCSA) — a non-invasive technique that identifies fault-specific frequency signatures directly from stator current measurements, using existing current sensors with no additional hardware.

The project extends a prior senior thesis (Bahrain Polytechnic, Best Research Award 2025) that built a rule-based MCSA detection system using FFT and CWT analysis. This capstone adds a machine learning layer to address the generalization limitations of rule-based approaches.

**Research Question:** Can a machine learning classifier trained on engineered time-domain and FFT-based features from three-phase motor current signals match or outperform the existing rule-based MCSA fault detection system — and which feature extraction approach (time-domain, FFT, wavelet, or hybrid) provides the strongest classification performance?

---

## Dataset

**Source:** Bruinsma et al. (2024) — *Electrical signals of an induction motor with different types of faults*  
4TU.ResearchData. https://doi.org/10.4121/ae694be5-b76b-4693-a9e7-ba616c82b875

| Property | Value |
|----------|-------|
| Motor | Motor 2 only |
| Conditions | healthy · brb (Broken Rotor Bar) · swf (Stator Winding Fault) |
| Speeds | 50% · 75% · 100% of rated load |
| Phases | A · B · C |
| Measurement replicates | 5 independent runs per group |
| Total rows | 2,700,000 |
| Total columns | 18 |
| File format | Parquet |
| File size | ~217 MB |
| Class balance | Perfectly balanced — 900,000 rows per condition |

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `time` | float32 | Timestamp in seconds relative to recording start |
| `A_m1` … `A_m5` | float32 | Phase A current (Amperes) — replicates 1–5 |
| `B_m1` … `B_m5` | float32 | Phase B current (Amperes) — replicates 1–5 |
| `C_m1` … `C_m5` | float32 | Phase C current (Amperes) — replicates 1–5 |
| `speed` | int64 | Motor load speed: 50, 75, or 100 |
| `condition` | str | Target variable: healthy · brb · swf |

---

## Repository Structure

```
motor-fault-detection/
│
├── data/
│   └── mcsa_master_dataset.parquet     # Master dataset (not tracked in git — download separately)
│
├── lightning_talk_2_mcsa.ipynb         # Main EDA notebook (Lightning Talk 2 deliverable)
├── lt2_v3.pptx                         # Slide presentation (Lightning Talk 2 deliverable)
├── lightning-talk#2-guideline.md.      # Project guidelines 
└── README.md                           # This file
```

---

## Notebook Structure

The main notebook `lightning_talk_2_mcsa.ipynb` follows this structure:

| Section | Contents |
|---------|----------|
| Problem Statement | Research question, citations, success metrics |
| Objectives | 6 guiding questions for analysis and modeling |
| Data Inspection | Data dictionary, `.head()`, `.info()`, `.describe()`, balance checks |
| Data Cleaning | 4 signal integrity checks — null, flat signal, group counts, amplitude range |
| EDA | Waveforms, amplitude distributions, RMS/kurtosis, inter-measurement consistency, correlation analysis |
| Preprocessing Outline | Sliding window segmentation, 4 feature pipelines, encoding and scaling |
| Modeling Outline | 12 models across 4 pipelines, evaluation metrics, baseline comparison |
| PPT Graph Exports | Graph generation cells sized for PowerPoint placeholders |

---

## Key EDA Finding

Maximum Pearson correlation between raw signal amplitude and the condition label is **≈ 0.007** — effectively zero across all 15 signal columns. Fault signatures (BRB sidebands at `f ± 2sf`, SWF harmonics and rush harmonics ratio) manifest exclusively in the frequency domain. This confirms that FFT-based feature extraction is a prerequisite before any ML classifier can work.

---

## Modeling Plan (Capstone Phase)

### Preprocessing Pipeline
1. Recording-level train/test split (80/20) — before windowing to prevent data leakage
2. Sliding window segmentation — 1,024 samples, 50% overlap → ~64,000 labeled samples
3. Four feature pipelines: Time-Domain · FFT · Wavelet · Hybrid
4. StandardScaler fit on train set only · speed included as a feature

### Models (12 total)

| Pipeline | Dummy | Random Forest | SVM |
|----------|-------|---------------|-----|
| TD — Time-Domain | ✓ | ✓ | ✓ |
| FFT — Frequency-Domain | ✓ | ✓ | ✓ |
| WT — Wavelet | ✓ | ✓ | ✓ |
| HYB — Hybrid | ✓ | ✓ | ✓ |

### Evaluation Metrics
- **Weighted F1-score** ≥ 0.90 — primary metric
- **Per-class recall** ≥ 0.85 for BRB and SWF
- Confusion matrix · Precision · ROC-AUC (one-vs-rest)

### Baseline Comparison
All models will be benchmarked against the rule-based MCSA system from the senior thesis (Bahrain Polytechnic, 2024), which uses FFT sideband ratio thresholds to classify BRB and SWF.

---

## Requirements

```
python >= 3.10
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
pyarrow
```
---

## References

- Bruinsma et al. (2024) — 4TU.ResearchData. https://doi.org/10.4121/ae694be5-b76b-4693-a9e7-ba616c82b875
- Maraaba et al. (2018) — Energies, MDPI. https://www.mdpi.com/1996-1073/11/3/653
- IEA (2020) — Global Efficiency Intelligence. https://www.globalefficiencyintel.com/new-blog/2017/infographic-energy-industrial-motor-systems
- Senior Thesis: Ali Ashoor & Sanad Sanad (2025) — Rule-based MCSA Fault Detection System, Bahrain Polytechnic (Best Research Award)
