# Machine Learning-Based Prediction and Causal Evaluation of Spontaneous Reabsorption in Lumbar Disc Herniation

This repository contains the code and results for the manuscript submitted to *The Spine Journal*.

## Overview

We developed a machine learning pipeline to predict spontaneous reabsorption of lumbar disc herniation (LDH) using clinical and imaging features. The champion model is an **LDA+TabPFN Voting** ensemble, combining Linear Discriminant Analysis with TabPFN (a Transformer-based Prior-data Fitted Network, [Nature 2024](https://www.nature.com/articles/s41586-024-08328-6)).

### Key Results

| Metric | OOF Internal Validation (n=608) | External Prospective Validation (n=56) |
|--------|--------------------------------|---------------------------------------|
| AUC | 0.727 | 0.914 |
| Sensitivity | 0.497 | 0.912 |
| Specificity | 0.854 | 0.773 |
| F1 Score | 0.537 | 0.886 |

## Repository Structure

```
.
├── 代码/                          # Source code (all scripts)
│   ├── 1_double_entry_consistency_check.R     # Data quality: double entry validation
│   ├── 2_Data_analysis_...py                  # Main pipeline: statistics + ML + SHAP
│   ├── 2a_Bull_eye_sensitivity_analysis.py    # Sensitivity analysis: Bull_eye masking
│   ├── 3_Gender_difference_analysis.py        # Sex-stratified subgroup analysis
│   ├── 4_app.py                               # Streamlit web calculator
│   ├── 5_lmm_vas_odi_joa_analysis.R           # Linear mixed model: VAS/ODI/JOA
│   ├── 6_IPTW_WEIGHTED_LOGISTIC.R             # Causal inference: IPTW analysis
│   ├── manuscript_ml_upgrade_core.py          # ML engine (models, training, evaluation)
│   └── manuscript_ml_upgrade_explain.py       # SHAP explainability module
│
├── Results/                       # Output from latest run
│   └── Manuscript_v2/
│       └── run_20260402_092359/
│           ├── 01_Descriptive/
│           ├── 02_Correlation/
│           ├── 03_Univariate/
│           ├── 04_ML_ModelDevelopment/
│           ├── 05_SHAP/
│           ├── 06_Calculator_Deployment/
│           └── 07_Sensitivity_Analysis_BullEye/
│
├── TSJ_Submission/                # Journal submission materials
│   ├── Manuscript/
│   ├── Figures/
│   ├── Tables/
│   └── Supplementary/
│
├── requirements.txt               # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- **Python 3.12+** (tested with 3.12.10)
- **R 4.x** (for scripts 1, 5, 6)
- **GPU** (optional): NVIDIA GPU with CUDA 12.x for accelerated training

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexHangggg/Manuscript-The-Spine-Journal.git
cd Manuscript-The-Spine-Journal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support (NVIDIA CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Data

Patient-level data is not included in this repository due to privacy regulations. Data is available from the corresponding author upon reasonable request.

**Expected data files** (place in `文件/` directory):
- `Retrospective data.xlsx` — Retrospective cohort (n=608), sheet: "Train"
- `Prospective data.xlsx` — Prospective cohort (n=56), sheet: "Train_Pors"

### Running the Pipeline

```bash
cd 代码

# Step 1: Data quality check (R)
Rscript 1_double_entry_consistency_check.R

# Step 2: Main ML pipeline (statistics + model training + SHAP)
python 2_Data_analysis___Model_construction___SHAP_analysis.py

# Step 2a: Bull_eye sensitivity analysis
python 2a_Bull_eye_sensitivity_analysis.py

# Step 3: Sex-stratified analysis
python 3_Gender_difference_analysis.py

# Step 4: Launch web calculator
streamlit run 4_app.py

# Step 5: Linear mixed model analysis (R)
Rscript 5_lmm_vas_odi_joa_analysis.R

# Step 6: IPTW causal inference (R)
Rscript 6_IPTW_WEIGHTED_LOGISTIC.R
```

## Models Evaluated

14 candidate models were trained and evaluated via 3-fold out-of-fold (OOF) cross-validation with BayesSearchCV hyperparameter tuning:

| Category | Models |
|----------|--------|
| Traditional | LDA, Logistic Regression, SVM, KNN |
| Tree-based | Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost, Extra Trees, Gradient Boosting |
| Deep tabular | TabPFN |
| Ensemble | Soft Voting, **LDA+TabPFN Voting** (champion) |

## Web Calculator

A Streamlit-based clinical decision support tool is included. After running the pipeline, launch it with:

```bash
# Windows
代码\Streamlit.bat

# Or directly
streamlit run 代码/4_app.py
```

## License

This project is for academic research purposes. Please cite the manuscript if you use this code.

## Contact

For data access requests or questions, please contact the corresponding author.
