# ==================== 1 Global setup & reproducibility ====================
# Purpose:
#   - Configure core libraries, plotting/style options, and display defaults
#   - Define target column, paths, and global settings for outputs
#   - Prepare a consistent environment prior to data loading and analysis

import os
import sys
import random
import argparse
from datetime import datetime
from pathlib import Path

# ---------- Reproducibility (set BEFORE importing numpy/scipy/sklearn) ----------
# Note: PYTHONHASHSEED is only fully effective when set before Python starts.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # disable interactive backend — must be called before pyplot import
import matplotlib.pyplot as plt
import numpy as np
random.seed(SEED)
np.random.seed(SEED)
import pandas as pd
import seaborn as sns
import json
import warnings
import joblib
from joblib import dump
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, shapiro, spearmanr, pearsonr, kruskal


# Target column naming
TARGET_COL = "Reabsorption"
EXCLUDE_FROM_MAIN_ANALYSES = {"Months_of_Review"}

def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def _prob_from_estimator(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1], "predict_proba"
    if hasattr(est, "decision_function"):
        print("[WARN] Using sigmoid(decision_function) as a pseudo-probability (not calibrated).")
        logits = est.decision_function(X)
        return _sigmoid(logits), "pseudo_probability_sigmoid(decision_function)"
    print("[WARN] Model has no probability output; using predict() labels.")
    return est.predict(X), "class_label_only"

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Set fonts: use Arial (remove Chinese fonts)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly

# TSJ figure export settings
FIG_EXT = "tiff"
FIG_DPI = 300
MIN_FIG_WIDTH_IN = 3.5
FIG_SAVE_KW = {
    "format": FIG_EXT,
    "dpi": FIG_DPI,
    "bbox_inches": "tight",
    "pil_kwargs": {"compression": "tiff_lzw"},
}


def _fig_path(dir_path, stem):
    return os.path.join(dir_path, f"{stem}.{FIG_EXT}")


def _save_fig(fig, path, *args, **kwargs):
    # Enforce TSJ minimum width requirement for code-generated figures.
    width_in = float(fig.get_size_inches()[0])
    if width_in < MIN_FIG_WIDTH_IN:
        raise ValueError(f"Figure width {width_in:.2f}in is below TSJ minimum {MIN_FIG_WIDTH_IN:.1f}in: {path}")

    # FIG_SAVE_KW provides defaults; explicit caller kwargs take precedence.
    save_kwargs = dict(FIG_SAVE_KW)
    save_kwargs.update(kwargs)
    try:
        fig.savefig(path, **save_kwargs)
    except TypeError:
        save_kwargs.pop("pil_kwargs", None)
        fig.savefig(path, **save_kwargs)

# ---------------------------------------------------------------------------
# Input/output path resolution — anchor to the project root, not the shell cwd.
# Directory layout assumed:
#   <project_root>\代码\     <- this script lives here
#   <project_root>\文件\     <- source data files
#   <project_root>\Results\  <- analysis outputs
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve() if "__file__" in globals() else None
SCRIPT_DIR = SCRIPT_PATH.parent if SCRIPT_PATH else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_PATH else Path.cwd()
BASE_DIR = str(PROJECT_ROOT)

def _resolve_data_file(filename: str) -> str:
    """Resolve a data file relative to the project root, regardless of launch cwd."""
    candidates = [
        PROJECT_ROOT / "文件" / filename,
        Path.cwd() / "文件" / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"Data file '{filename}' not found. Paths tried:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )

DATA_PATH  = _resolve_data_file("Retrospective data.xlsx")
DATA_SHEET = "Train"
print(f"[INFO] Script directory: {SCRIPT_DIR}")
print(f"[INFO] Project root: {PROJECT_ROOT}")
print(f"[INFO] Retrospective data: {DATA_PATH} | sheet={DATA_SHEET}")

RESULTS_ROOT = os.path.join(BASE_DIR, "Results")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = os.path.join(RESULTS_ROOT, "Manuscript_v2", f"run_{RUN_ID}")
SAVE_PATH = RUN_ROOT

BASELINE_DIR = os.path.join(RUN_ROOT, "01_Baseline")
BASELINE_FIG_DIR = os.path.join(BASELINE_DIR, "figures")
CORR_DIR = os.path.join(RUN_ROOT, "02_Correlation")
CORR_FIG_DIR = os.path.join(CORR_DIR, "figures")
CORR_DATA_DIR = os.path.join(CORR_DIR, "data")
ML_DIR = os.path.join(RUN_ROOT, "04_ML_ModelDevelopment")
ML_FIG_DIR = os.path.join(ML_DIR, "figures")
ML_METRICS_DIR = os.path.join(ML_DIR, "metrics")
ML_MODELS_DIR = os.path.join(ML_DIR, "models")
ML_AUDIT_DIR = os.path.join(ML_DIR, "audit")
SHAP_DIR = os.path.join(RUN_ROOT, "05_SHAP", "figures")
DEPLOY_DIR = os.path.join(RUN_ROOT, "06_Calculator_Deployment", "exported_model")
LOG_DIR = os.path.join(RUN_ROOT, "99_Logs_and_Metadata")

for d in [
    SAVE_PATH,
    BASELINE_DIR, BASELINE_FIG_DIR,
    CORR_DIR, CORR_FIG_DIR, CORR_DATA_DIR,
    ML_DIR, ML_FIG_DIR, ML_METRICS_DIR, ML_MODELS_DIR, ML_AUDIT_DIR,
    SHAP_DIR, DEPLOY_DIR, LOG_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ── Load retrospective data (modelling only) ─────────────────────────────────
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Input file not found: {DATA_PATH}")
data_retro = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)

if TARGET_COL not in data_retro.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found. Please confirm it exists in the Excel file.")
data_retro[TARGET_COL] = pd.to_numeric(data_retro[TARGET_COL], errors="coerce")
data_retro = data_retro[pd.notna(data_retro[TARGET_COL])].copy()
data_retro["Cohort"] = "Retrospective"

# ── Load prospective data and merge (baseline & EDA only) ─────────────────────
_PROS_PATH  = _resolve_data_file("Prospective data.xlsx")
_PROS_SHEET = "Train_Pors"
print(f"[INFO] Prospective data: {_PROS_PATH} | sheet={_PROS_SHEET}")
data_pros = pd.read_excel(_PROS_PATH, sheet_name=_PROS_SHEET)
data_pros[TARGET_COL] = pd.to_numeric(data_pros[TARGET_COL], errors="coerce")
data_pros = data_pros[pd.notna(data_pros[TARGET_COL])].copy()
data_pros["Cohort"] = "Prospective"

# Combine all records without deduplication.
# Same-person records across cohorts are treated as distinct longitudinal
# observations, consistent with scripts 3 and 6.
data_combined = pd.concat([data_retro, data_pros], ignore_index=True)
print(f"[INFO] Bidirectional combined dataset: "
      f"Retrospective n={len(data_retro)}, Prospective n={len(data_pros)}, "
      f"Total n={len(data_combined)}")

# Sections 2-3 (Baseline, EDA, Correlation) use the combined cohort.
# Section 4 (ML modelling) will reassign 'data' back to retrospective only.
data = data_combined.copy()

print("=" * 60)
print("[INFO] Reabsorption Dataset - Exploratory Data Analysis Report")
print("       (Bidirectional combined cohort: Retrospective + Prospective)")
print("=" * 60)

print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")


# ==================== 2 Data loading & EDA reporting ====================

# ==================== 2.1 Basic Dataset Information ====================
# Purpose:
#   - Inspect dataset dimensions, column types, and overall structure
#   - Display head() and info() summaries for data sanity checking
#   - Identify missing values prior to EDA and modeling
print("\n[SECTION] 1. Basic Dataset Information")
print("-" * 40)

print(f"Dataset shape: {data.shape}")
print(f"Number of features (excluding target): {data.shape[1] - 1}")
print(f"Number of samples: {data.shape[0]}")

print("\nDataset info:")
print(data.info())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing value check:")
missing_values = data.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("No missing values in the dataset.")
else:
    print("Missing values detected please handle them before further analysis.")

print(f"\n[INFO] Data type distribution:")
print("\nData type distribution:")

out_dir = CORR_DIR
os.makedirs(out_dir, exist_ok=True)

# ==================== 2.2 Baseline Characteristics (Table 1) ====================
# Purpose:
#   - Compute group-wise summary statistics and tests for baseline variables
#   - Build Table 1 with grouped rows, p-values, and test methods
#   - Export baseline characteristics to CSV/XLSX for reporting
print("\n[INFO] Baseline characteristics (Overall vs Resorption vs Non-resorption)")
print("-" * 70)


# --- 1) Variable groups (consistent with earlier sections; keep only columns that exist in the data) ---
continuous_vars_all = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM", "Initial_volume", "RSI", "DHI"
]
ordinal_vars_all = ["Pfirrmann", "Komori", "MSU"]
nominal_vars_all = ["Gender", "Herniated_Level", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Bull_eye", "Cohort"]

continuous_vars = [c for c in continuous_vars_all if c in data.columns]
ordinal_vars    = [c for c in ordinal_vars_all    if c in data.columns]
nominal_vars    = [c for c in nominal_vars_all    if c in data.columns]

# Exclude follow-up duration from main analyses
continuous_vars = [c for c in continuous_vars if c not in EXCLUDE_FROM_MAIN_ANALYSES]
ordinal_vars    = [c for c in ordinal_vars    if c not in EXCLUDE_FROM_MAIN_ANALYSES]
nominal_vars    = [c for c in nominal_vars    if c not in EXCLUDE_FROM_MAIN_ANALYSES]

# In the baseline table, treat "ordinal variables" as categorical (report counts and percentages)
cat_vars = ordinal_vars + nominal_vars
all_vars = continuous_vars + ordinal_vars + nominal_vars
assert "Months_of_Review" not in all_vars

# Group sizes
if TARGET_COL not in data.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found.")
n_total = len(data)
n_res   = int((data[TARGET_COL] == 1).sum())
n_non   = int((data[TARGET_COL] == 0).sum())

col_overall = "Overall (n=%d)" % n_total
col_res     = "Resorption (n=%d)" % n_res
col_non     = "Non-resorption (n=%d)" % n_non

# --- 2) Utility functions ---
def fmt_mean_sd(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    return f"{s.mean():.2f} +/- {s.std(ddof=1):.2f}"

def format_n_pct(n, denom):
    if denom <= 0:
        return "0 (0.0%)"
    return f"{n} ({(n/denom)*100:.1f}%)"

def format_pval(p):
    if p is None or pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def need_mwu(x0, x1):
    # Very small samples / zero variance / too few unique values -> fall back to MWU
    return (len(x0) < 8 or len(x1) < 8 or
            x0.var(ddof=1) == 0 or x1.var(ddof=1) == 0 or
            x0.nunique() < 3 or x1.nunique() < 3)

def p_continuous_with_test(x, g):
    """Continuous variables: Welch's t-test; fall back to Mann-Whitney U when needed.
       Returns (p-value, test name)."""
    x0 = pd.to_numeric(x[g==0], errors="coerce").dropna()
    x1 = pd.to_numeric(x[g==1], errors="coerce").dropna()
    if len(x0) < 2 or len(x1) < 2:
        return np.nan, "Insufficient data"
    if need_mwu(x0, x1):
        try:
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided", method="auto")
        except TypeError:
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
        return p, "Mann-Whitney U"
    else:
        _, p = ttest_ind(x0, x1, equal_var=False)
        return p, "Welch's t-test"

def p_categorical_with_test(x, g):
    """
    Categorical variables:
    - 2x2 -> use Pearson Chi-square if expected counts >=5; otherwise use Fisher's exact
    - Multi-level -> use Chi-square; flag small expected counts
    Returns (p-value, test name)
    """
    x = pd.Series(x)
    tab = pd.crosstab(g, x)  # rows: Reabsorption(0/1), cols: category levels
    if tab.shape[1] == 0:
        return np.nan, "NA"

    try:
        chi2, p_chi2, _, expected = chi2_contingency(tab)
        if (expected < 5).any():
            if tab.shape == (2, 2):
                try:
                    _, p_f = fisher_exact(tab.values)
                    return p_f, "Fisher's exact (expected<5)"
                except Exception:
                    return p_chi2, "Chi-square (fallback)"
            return p_chi2, "Chi-square (expected<5; Fisher not applicable)"
        return p_chi2, "Chi-square"
    except Exception:
        return np.nan, "Chi-square (error)"

def ordered_levels(series: pd.Series):
    """Return levels in an expected order:
       - If all values are numeric/numeric strings -> ascending numeric order
       - If categorical dtype -> category order
       - Else -> alphabetical order
    """
    s = series.dropna()
    # Try numeric ordering
    try:
        vals = pd.to_numeric(s.astype(str), errors="raise")
        lvls = sorted(pd.unique(vals))
        out = []
        for v in lvls:
            if isinstance(v, (int, np.integer)) or float(v).is_integer():
                out.append(str(int(v)))
            else:
                out.append(str(v))
        return out
    except Exception:
        pass
    # Category order
    if pd.api.types.is_categorical_dtype(series):
        return [str(v) for v in series.cat.categories.tolist()]
    # Alphabetical
    return [str(v) for v in sorted(pd.unique(s.astype(str)))]

# --- 3) Build "Table 1" by blocks: Characteristic header rows + p-value & test on the header row ---
rows = []

# 3.1 Continuous variables: one header row per variable (Mean +/- SD), show p-value & test
for var in continuous_vars:
    s_all  = data[var]
    s_res  = data.loc[data[TARGET_COL] == 1, var]
    s_non  = data.loc[data[TARGET_COL] == 0, var]
    p_val, test_name = p_continuous_with_test(data[var], data[TARGET_COL])

    rows.append({
        "Characteristic": var,  # header row
        col_overall: fmt_mean_sd(s_all),
        col_res:     fmt_mean_sd(s_res),
        col_non:     fmt_mean_sd(s_non),
        "P-value":   format_pval(p_val),
        "Test":      test_name,
        "_grp": 0, "_var": var, "_ord": -1  # keep continuous block first
    })

# 3.2 Categorical variables: variable name as a header row + indented sub-level rows; p-value & test only on the header
for var in cat_vars:
    p_val, test_name = p_categorical_with_test(data[var], data[TARGET_COL])

    # Header row (shows p-value & test)
    rows.append({
        "Characteristic": var,
        col_overall: "",
        col_res:     "",
        col_non:     "",
        "P-value":   format_pval(p_val),
        "Test":      test_name,
        "_grp": 1, "_var": var, "_ord": -1
    })

    # Order levels
    lvls = ordered_levels(data[var])

    # Sub-level rows (do not repeat p-value & test)
    for lvl in lvls:
        # Robust level matching: numeric compare if possible, else string compare
        s_num = pd.to_numeric(data[var], errors="coerce")
        try:
            lvl_num = float(lvl)
            mask_all = (s_num == lvl_num)
        except Exception:
            mask_all = (data[var].astype(str) == str(lvl))
        n_all = int(mask_all.sum())
        mask_res = (data[TARGET_COL] == 1) & mask_all
        mask_non = (data[TARGET_COL] == 0) & mask_all
        n_r = int(mask_res.sum())
        n_n = int(mask_non.sum())

        # numeric key to sort numeric-looking levels
        try:
            ord_key = float(lvl)
        except:
            ord_key = np.inf

        rows.append({
            "Characteristic": f"    - {lvl}",
            col_overall: format_n_pct(n_all, n_total),
            col_res:     format_n_pct(n_r,   n_res),
            col_non:     format_n_pct(n_n,   n_non),
            "P-value":   "",
            "Test":      "",
            "_grp": 1, "_var": var, "_ord": ord_key
        })

# Assemble the table
baseline_df = pd.DataFrame(rows)

# Sort: continuous block (_grp=0) first; within block by variable name; sub-rows (_ord>=0) after header rows
baseline_df.sort_values(by=["_grp", "_var", "_ord"], inplace=True)
baseline_df = baseline_df.drop(columns=["_grp", "_var", "_ord"])

# Column order
baseline_df = baseline_df[["Characteristic", col_overall, col_res, col_non, "P-value", "Test"]]

# Preview
print("\n[INFO] Baseline table (first 20 rows):")
print(baseline_df.head(20))

# --- 4) Save to working directory ---
base_dir = SAVE_PATH if 'SAVE_PATH' in globals() else os.getcwd()
baseline_dir = BASELINE_DIR if 'BASELINE_DIR' in globals() else os.path.join(base_dir, "01_Baseline")
os.makedirs(baseline_dir, exist_ok=True)
csv_path  = os.path.join(baseline_dir, "baseline_table.csv")
xlsx_path = os.path.join(baseline_dir, "baseline_table.xlsx")

baseline_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# Excel: bold for header rows, indentation for sub-level rows; center numeric columns
try:
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        baseline_df.to_excel(writer, index=False, sheet_name="Baseline")
        wb  = writer.book
        ws  = writer.sheets["Baseline"]

        # Column widths
        ws.set_column(0, 0, 36)  # Characteristic
        ws.set_column(1, 5, 22)  # stats + p + Test

        # Formats
        fmt_bold   = wb.add_format({'bold': True})
        fmt_lvl    = wb.add_format({'indent': 1})
        fmt_center = wb.add_format({'align': 'center'})

        # Header row: center all but first column
        for col in range(1, 5 + 1):
            ws.set_column(col, col, 22, fmt_center)

        # Row styles: bold for header rows, indent for sub-level rows
        for r in range(1, len(baseline_df) + 1):  # skip header row
            text = str(baseline_df.iloc[r-1, 0])
            if text.startswith("    - "):
                ws.write(r, 0, text, fmt_lvl)
            else:
                ws.write(r, 0, text, fmt_bold)
except Exception as e:
    print(f"Excel export encountered an issue, but CSV was saved: {e}")

print(f"\n[OK] Baseline table saved to:\n  - {csv_path}\n  - {xlsx_path}")


# ==================== 2.3 Target Variable Analysis ====================
# Purpose:
#   - Profile target distribution with fixed class order
#   - Summarize counts/percentages and basic visual checks
#   - Provide baseline context for downstream comparisons and models
print("\n[SECTION] Part 2: Target Variable Analysis")
print("-" * 40)

# Fixed order: 0 = Non-reabsorption, 1 = Reabsorption
label_order = [0, 1]
labels_en = ['Non-reabsorption', 'Reabsorption']
colors = ['#3498db', '#e74c3c']  # Blue = Non-reabsorption, Red = Reabsorption

# Distribution of the target variable
target_counts = data[TARGET_COL].value_counts().reindex(label_order, fill_value=0)
target_pct = data[TARGET_COL].value_counts(normalize=True).reindex(label_order, fill_value=0) * 100

print("Target variable distribution:")
for k, label in zip(label_order, labels_en):
    print(f"  {label} ({TARGET_COL}={k}): {target_counts.loc[k]} samples ({target_pct.loc[k]:.1f}%)")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=300)  # TSJ-ready: >=300 dpi

# Bar chart
bars = axes[0].bar(labels_en, target_counts.values, color=colors, alpha=0.8)
axes[0].set_title(f'{TARGET_COL} Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sample Count', fontsize=12)
for bar, count in zip(bars, target_counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
        str(count), ha='center', va='bottom', fontweight='bold'
    )

# Pie chart
axes[1].pie(
    target_counts.values,
    labels=labels_en,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=(0.05, 0),
)
axes[1].set_title(f'{TARGET_COL} Proportion', fontsize=14, fontweight='bold')

# Donut chart
center_circle = plt.Circle((0, 0), 0.70, fc='white')
axes[2].pie(
    target_counts.values,
    labels=labels_en,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85,
)
axes[2].add_artist(center_circle)
axes[2].set_title(f'{TARGET_COL} Donut Chart', fontsize=14, fontweight='bold')

plt.tight_layout()

# === Define figure paths (baseline + correlation) ===
SAVE_PATH = SAVE_PATH if 'SAVE_PATH' in globals() else os.path.join(BASE_DIR if 'BASE_DIR' in globals() else os.getcwd(), "Results")
baseline_fig_dir = BASELINE_FIG_DIR if 'BASELINE_FIG_DIR' in globals() else os.path.join(SAVE_PATH, "01_Baseline", "figures")
corr_fig_dir = CORR_FIG_DIR if 'CORR_FIG_DIR' in globals() else os.path.join(SAVE_PATH, "02_Correlation", "figures")
corr_data_dir = CORR_DATA_DIR if 'CORR_DATA_DIR' in globals() else os.path.join(SAVE_PATH, "02_Correlation", "data")
os.makedirs(baseline_fig_dir, exist_ok=True)
os.makedirs(corr_fig_dir, exist_ok=True)
os.makedirs(corr_data_dir, exist_ok=True)

# Keep legacy name for downstream sections that reuse this variable
supp_fig_dir = corr_fig_dir

# Save current figure to baseline figure folder
supp_fig_path = os.path.join(baseline_fig_dir, f"target_distribution_{TARGET_COL.lower()}.tiff")
_save_fig(fig, 
    supp_fig_path,
    format='tiff',
    dpi=300,
    bbox_inches='tight'
)

print(f"[OK] Supplementary figure saved: {supp_fig_path}")

# ==================== 2.4 Feature Variable Distribution Analysis ====================
# Purpose:
#   - Summarize numeric feature distributions and descriptive statistics
#   - Visualize distributions to spot skewness/outliers
#   - Prepare understanding of features before relational analyses
print("\n[SECTION] Part 3: Feature Variable Distribution Analysis")
print("-" * 40)

# Select numeric features (exclude target 'Reabsorption'; if 'ID' is numeric, exclude it as well)
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
for col_to_drop in [TARGET_COL, "ID"]:
    if col_to_drop in numeric_features:
        numeric_features.remove(col_to_drop)

print(f"Numeric features: {numeric_features}")

# Summary statistics
print("\n[INFO] Summary statistics for numeric features:")
print(data[numeric_features].describe().round(2))

# Visualization of feature distributions
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols if n_features > 0 else 1

# TSJ standard: high-resolution figure (>=300 dpi, here 300 dpi)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=300)

# Ensure axes are indexable
if n_rows == 1 and n_cols == 1:
    axes = [axes]
else:
    axes = np.array(axes).flatten()

for i, feature in enumerate(numeric_features):
    # Histogram with KDE
    sns.histplot(data[feature].dropna(), kde=True, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'{feature} Distribution', fontweight='bold')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

    # Add mean and median reference lines
    mean_val = data[feature].mean()
    median_val = data[feature].median()
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    axes[i].legend()

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# Save as TIFF (raster, 300 dpi) in baseline figures
_save_fig(fig, 
    os.path.join(baseline_fig_dir, "feature_distributions.tiff"),
    format='tiff',
    dpi=300,
    bbox_inches='tight'
)

print(f"[OK] Figure saved: {os.path.join(baseline_fig_dir, 'feature_distributions.tiff')}")

# ==================== 3 Statistical analysis & association ====================

# ==================== 3.1 Feature-target Relationship Analysis ====================
# Purpose:
#   - Compute feature-target associations by data type (continuous/ordinal/nominal)
#   - Apply appropriate tests/correlations (Pearson/Spearman/Cramer's V, etc.)
#   - Highlight variables related to the target for later modeling
print("\n[SECTION] Part 4: Feature-target Relationship Analysis")
print("-" * 40)

# Recompute grid based on the number of numeric features
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols if n_features > 0 else 1

# Compare feature distributions across Reabsorption groups
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), dpi=300)

# Ensure axes are indexable
if isinstance(axes, np.ndarray):
    axes = axes.reshape(-1)
else:
    axes = np.array([axes])

for i, feature in enumerate(numeric_features):
    # Grouped boxplots
    sns.boxplot(data=data, x=TARGET_COL, y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} by {TARGET_COL} Group', fontweight='bold')
    axes[i].set_xlabel(f'{TARGET_COL} (0 = Non-reabsorption, 1 = Reabsorption)')
    axes[i].set_ylabel(feature)

    # Add mean markers
    for j, group in enumerate([0, 1]):
        subset = data.loc[data[TARGET_COL] == group, feature].dropna()
        if len(subset) > 0:
            mean_val = subset.mean()
            axes[i].plot(j, mean_val, marker='D', markersize=8, color='red')

# Hide unused subplots
for j in range(len(numeric_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# Save to the supplementary figure directory defined in Part 2
supp_fig_path = os.path.join(supp_fig_dir, f"feature_target_relationship_{TARGET_COL.lower()}.tiff")
_save_fig(fig, 
    supp_fig_path,
    format='tiff',
    dpi=300,
    bbox_inches='tight'
)

print(f"[OK] Supplementary figure saved: {supp_fig_path}")

# ==================== 3.1.1 Feature-target Relationship Analysis (Continuous Variables) ====================
# Purpose:
#   - Focus on continuous variables with tailored plots/tests
#   - Recompute grids/visuals based on available numeric features
#   - Provide detailed continuous-feature insights vs target
print("\n[SECTION] Part 4: Feature-target Relationship Analysis (Continuous Variables)")
print("-" * 40)

# Continuous variables (select only those existing in the dataset)
continuous_vars = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
    "Months_of_Review", "Initial_volume", "RSI", "DHI"
]
continuous_vars = [c for c in continuous_vars if c in data.columns]

# Layout configuration
n_features = len(continuous_vars)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols if n_features > 0 else 1

# TSJ-ready: raster TIFF, high resolution 300 dpi
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), dpi=300)
axes = np.array(axes).flatten() if n_rows * n_cols > 1 else np.array([axes])

for i, feature in enumerate(continuous_vars):
    # Boxplot grouped by Reabsorption
    sns.boxplot(
        data=data,
        x=TARGET_COL,
        y=feature,
        ax=axes[i],
        palette=['#3498db', '#e74c3c']  # Blue = Non-reabsorption, Red = Reabsorption
    )
    axes[i].set_title(f'{feature} by {TARGET_COL} Group', fontweight='bold')
    axes[i].set_xlabel(f'{TARGET_COL} (0 = Non-reabsorption, 1 = Reabsorption)')
    axes[i].set_ylabel(feature)

    # Add mean markers
    for j, group in enumerate([0, 1]):
        subset = data.loc[data[TARGET_COL] == group, feature].dropna()
        if len(subset) > 0:
            mean_val = subset.mean()
            axes[i].plot(j, mean_val, marker='D', markersize=8, color='red')

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# === Save figure to supplementary figure directory (defined in Part 2) ===
supp_fig_path = os.path.join(supp_fig_dir, f"feature_target_relationship_{TARGET_COL.lower()}_continuous.tiff")
_save_fig(fig, 
    supp_fig_path,
    format='tiff',
    dpi=300,
    bbox_inches='tight'
)

print(f"[OK] Supplementary figure saved: {supp_fig_path}")

# ==================== 3.2 Comprehensive Association Analysis (Single Lower-Triangle Heatmap) ====================
# Purpose:
#   - Build unified association matrices across feature types
#   - Visualize associations via lower-triangle heatmaps with masking
#   - Offer a global view of inter-feature and feature-target relationships
print("\n[SECTION] Part 5: Comprehensive Association Analysis (type-specific measures, unified to 0-1)")
print("-" * 40)

from scipy.stats import shapiro, pearsonr

# 1) Variable groups - keep only existing columns; explicitly exclude some variables
continuous_vars_all = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM", "Initial_volume", "RSI", "DHI"
]
# Exclude Last_volume, Absorption_rate
continuous_vars = [c for c in continuous_vars_all if c in data.columns and c not in ["Last_volume", "Absorption_rate"]]

ordinal_vars = [c for c in ["Pfirrmann", "Komori", "MSU"] if c in data.columns]

# Nominal variables (excluding Absorption_type)
nominal_vars = [
    "Gender", "Herniated_Level", "Iwabuchi", "Modic",
    "Spinal_canal_stenosis", "Bull_eye"
]
nominal_vars = [c for c in nominal_vars if c in data.columns]

continuous_vars = [c for c in continuous_vars if c not in EXCLUDE_FROM_MAIN_ANALYSES]
ordinal_vars    = [c for c in ordinal_vars    if c not in EXCLUDE_FROM_MAIN_ANALYSES]
nominal_vars    = [c for c in nominal_vars    if c not in EXCLUDE_FROM_MAIN_ANALYSES]

if TARGET_COL not in data.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found.")

# Include Reabsorption as a nominal variable in the matrix
if TARGET_COL not in nominal_vars:
    nominal_vars = nominal_vars + [TARGET_COL]

numeric_vars = continuous_vars + ordinal_vars
all_vars = numeric_vars + nominal_vars
assert "Months_of_Review" not in all_vars

# ---------- Helper functions ----------
def _pairwise_clean(x, y):
    """Drop pairwise missing values; return aligned Series."""
    x = pd.Series(x)
    y = pd.Series(y)
    mask = (~x.isna()) & (~y.isna())
    return x[mask], y[mask]

def _is_binary(s: pd.Series):
    vals = pd.unique(pd.Series(s).dropna())
    return len(vals) == 2

def _normal_enough(s: pd.Series):
    """Shapiro-Wilk normality test: p>=0.05 -> approximately normal; fall back for very small/large n."""
    s = pd.Series(s).dropna()
    n = len(s)
    if n < 8:
        return False
    if n > 5000:
        return False
    try:
        _, p = shapiro(s)
        return bool(p >= 0.05)
    except Exception:
        return False

def corr_cont_cont(x, y):
    """Continuous-continuous: Pearson if normal; otherwise Spearman. Return |r| in [0,1]."""
    x, y = _pairwise_clean(x, y)
    if len(x) < 3:
        return 0.0
    if _normal_enough(x) and _normal_enough(y):
        r, _ = pearsonr(x, y)
        return abs(r)
    else:
        r, _ = spearmanr(x, y)
        return abs(r)

def corr_ord_any(x, y):
    """Ordinal-ordinal or Ordinal-continuous: Spearman; return |rho|."""
    x, y = _pairwise_clean(x, y)
    if len(x) < 3:
        return 0.0
    r, _ = spearmanr(x, y)
    return abs(r)

def corr_nom_nom(x, y):
    """Nominal-nominal: Cramer's V in [0,1]."""
    x, y = _pairwise_clean(x, y)
    if len(x) == 0:
        return 0.0
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))

def corr_nom_cont(x_cat, y_cont):
    """
    Nominal-continuous:
    - Binary nominal -> treat as 0/1 and use Pearson (equivalent to point-biserial r), return |r|.
    - Multi-class -> Kruskal-Wallis H, convert to an eta^2-like effect size in [0,1].
    """
    x_cat, y_cont = _pairwise_clean(x_cat, y_cont)
    # Ensure continuous side is numeric
    y_cont = pd.to_numeric(y_cont, errors="coerce")
    mask = ~y_cont.isna()
    x_cat = x_cat[mask]
    y_cont = y_cont[mask]
    if len(x_cat) < 3:
        return 0.0
    if _is_binary(x_cat):
        x01 = pd.Categorical(x_cat).codes.astype(float)  # 0/1 coding by category order
        r, _ = pearsonr(y_cont.values, x01)
        return abs(r)
    else:
        groups = [y_cont[x_cat == g] for g in pd.unique(x_cat)]
        groups = [g.dropna() for g in groups if len(g.dropna()) >= 2]
        if len(groups) < 2:
            return 0.0
        H, _ = kruskal(*groups)
        n = len(y_cont)
        k = len(groups)
        denom = (n - k)
        if denom <= 0:
            return 0.0
        eta2 = (H - k + 1) / denom
        return float(np.clip(eta2, 0, 1))

# ---------- Compute the association matrix (lower triangle; then symmetrize) ----------
assoc = pd.DataFrame(np.eye(len(all_vars)), index=all_vars, columns=all_vars, dtype=float)

for i, vi in enumerate(all_vars):
    for j, vj in enumerate(all_vars):
        if j >= i:
            continue  # only lower triangle
        xi, xj = data[vi], data[vj]

        if (vi in continuous_vars) and (vj in continuous_vars):
            val = corr_cont_cont(xi, xj)

        elif ((vi in ordinal_vars) and (vj in ordinal_vars)) or \
             ((vi in ordinal_vars) and (vj in continuous_vars)) or \
             ((vi in continuous_vars) and (vj in ordinal_vars)):
            val = corr_ord_any(xi, xj)

        elif (vi in nominal_vars) and (vj in nominal_vars):
            val = corr_nom_nom(xi, xj)

        else:
            # Nominal-continuous (either direction)
            if vi in nominal_vars and vj in continuous_vars:
                val = corr_nom_cont(xi, xj)
            elif vj in nominal_vars and vi in continuous_vars:
                val = corr_nom_cont(xj, xi)
            else:
                val = 0.0  # defensive fallback

        assoc.loc[vi, vj] = val
        assoc.loc[vj, vi] = val

# Diagonal = 1
np.fill_diagonal(assoc.values, 1.0)

# Print associations with target (descending)
if TARGET_COL in assoc.columns:
    print(f"\nComprehensive association strength with {TARGET_COL} (0-1, descending):")
    tmp = assoc[TARGET_COL].drop(index=TARGET_COL).sort_values(ascending=False)
    print(tmp.round(3))

# ---------- Plot lower-triangle heatmap (0-1) ----------
fig = plt.figure(figsize=(12, 10), dpi=300)  # display at 300 dpi as well
mask = np.triu(np.ones_like(assoc, dtype=bool))
sns.heatmap(
    assoc,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='RdBu_r',   # colorblind-friendly divergent palette
    vmin=0, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": .8}
)
plt.title(
    'Comprehensive Association Heatmap (Lower Triangle)\n'
    'Cont-Cont: Pearson/Spearman | Ord-(Ord/Cont): Spearman | Nom-Cont: Point-biserial / Kruskal-Wallis (eta^2) | Nom-Nom: Cramer\'s V',
    fontsize=14, fontweight='bold', pad=16
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# ---------- Save as supplementary figure (TIFF, 300 dpi) ----------
# Guard: reuse supp_fig_dir from Part 2; if missing (e.g., running this cell first), create it now.
if 'supp_fig_dir' not in globals():
    base_path = SAVE_PATH if 'SAVE_PATH' in globals() else os.path.join(BASE_DIR if 'BASE_DIR' in globals() else os.getcwd(), "Results")
    supp_fig_dir = os.path.join(base_path, "02_Correlation", "figures")
os.makedirs(supp_fig_dir, exist_ok=True)

supp_fig_path = os.path.join(supp_fig_dir, f"association_heatmap_{TARGET_COL.lower()}.tiff")
_save_fig(fig, 
    supp_fig_path,
    format='tiff',
    dpi=300,
    bbox_inches='tight'
)

print(f"[OK] Supplementary figure saved: {supp_fig_path}")


# ==================== 3.3 Outlier Detection (Continuous Variables, IQR) ====================
# Purpose:
#   - Detect outliers in continuous variables using IQR-based rules
#   - Visualize boxplots and outlier counts to assess distribution tails
#   - Flag features with many outliers for potential handling strategies
print("\n[SECTION] Part 6: Outlier Detection (Continuous Variables, IQR)")
print("-" * 40)

# Continuous variables (keep only those that exist in the dataset)
continuous_vars = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
    "Months_of_Review", "Initial_volume", "RSI", "DHI"
]
continuous_vars = [c for c in continuous_vars if c in data.columns]

# IQR-based outlier detection
def detect_outliers_iqr(df, feature):
    s = pd.to_numeric(df[feature], errors='coerce')  # ensure numeric
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(s < lower_bound) | (s > upper_bound)]
    return outliers, lower_bound, upper_bound

# Summary of outliers
outlier_summary = {}
for feature in continuous_vars:
    outliers, lower, upper = detect_outliers_iqr(data, feature)
    outlier_summary[feature] = {
        'count': int(len(outliers)),
        'percentage': (len(outliers) / len(data) * 100) if len(data) > 0 else 0.0,
        'lower_bound': lower,
        'upper_bound': upper
    }

print("Outlier detection summary (IQR):")
for feature, info in outlier_summary.items():
    print(f"  {feature}: {info['count']} outliers ({info['percentage']:.1f}%)")

# Visualization (boxplot of all continuous features + bar chart of outlier counts)
if len(continuous_vars) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=300)  # TSJ-ready: 300 dpi

    # Boxplot for all continuous features
    melted = pd.melt(
        data[continuous_vars].apply(pd.to_numeric, errors='coerce'),
        var_name='Feature', value_name='Value'
    )
    sns.boxplot(data=melted, x='Feature', y='Value', ax=axes[0])
    axes[0].set_title('Boxplot of Continuous Features - Outlier Detection (IQR)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)

    # Bar chart for outlier counts
    features = list(outlier_summary.keys())
    outlier_counts = [outlier_summary[f]['count'] for f in features]
    bars = axes[1].bar(features, outlier_counts, alpha=0.8)
    axes[1].set_title('Number of Outliers by Continuous Feature', fontweight='bold')
    axes[1].set_ylabel('Number of Outliers')
    axes[1].tick_params(axis='x', rotation=30)

    # Add value labels
    for bar, count in zip(bars, outlier_counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(count), ha='center', va='bottom'
        )

    plt.tight_layout()
    plt.show()  # display before saving

    # === Save as supplementary figure (TIFF, 300 dpi) ===
    # Reuse supp_fig_dir from Part 2; if missing (e.g., this cell runs first), create it now.
    if 'supp_fig_dir' not in globals():
        base_path = SAVE_PATH if 'SAVE_PATH' in globals() else os.path.join(BASE_DIR if 'BASE_DIR' in globals() else os.getcwd(), "Results")
        supp_fig_dir = os.path.join(base_path, "02_Correlation", "figures")
    os.makedirs(supp_fig_dir, exist_ok=True)

    supp_fig_path = os.path.join(supp_fig_dir, "outlier_detection_continuous_iqr.tiff")
    _save_fig(fig, 
        supp_fig_path,
        format='tiff',
        dpi=300,
        bbox_inches='tight'
    )
    print(f"[OK] Supplementary figure saved: {supp_fig_path}")
else:
    print("No continuous variables found - skipping outlier detection.")


# ==================== 3.4 Advanced Visualization (reuse association results) ====================
# Purpose:
#   - Reuse association outputs to generate additional comparative plots
#   - Provide richer visual insights beyond basic distributions
#   - Support interpretation and communication of data patterns
print("\n[SECTION] Part 7: Advanced Visualization (based on the Part 5 association matrix)")
print("-" * 40)


# 1) Use the association matrix 'assoc' computed in Part 5
if 'assoc' not in globals():
    raise RuntimeError("Association matrix 'assoc' not found. Please run Part 5 first to generate 'assoc'.")

if TARGET_COL not in assoc.columns:
    raise RuntimeError(f"Target column {TARGET_COL} not found in the association matrix 'assoc'.")

# 2) Defensive existence filtering (same groups as defined in Part 5)
continuous_vars = [c for c in [
    "Age","SS","Upper_VB_Posterior_Height_CM","Lower_VB_Posterior_Height_CM",
    "Months_of_Review","Initial_volume","RSI","DHI"  # Last_volume and Absorption_rate excluded as requested
] if c in data.columns]

ordinal_vars = [c for c in ["Pfirrmann","Komori","MSU"] if c in data.columns]

nominal_vars = [c for c in [
    "Gender","Herniated_Level","Iwabuchi","Modic","Spinal_canal_stenosis","Bull_eye"
] if c in data.columns]

numord_vars = continuous_vars + ordinal_vars

# 3) Association strength with Reabsorption (0-1), sorted
assoc_to_target = assoc[TARGET_COL].drop(index=TARGET_COL).sort_values(ascending=False)
print(f"Association strength with {TARGET_COL} (from Part 5, top 6):")
print(assoc_to_target.head(6).round(3))

# 4) Select up to 4 numeric/ordinal features for pairwise plots, strictly by assoc ranking
top_any = assoc_to_target.index.tolist()                  # all features ranked by strength
top_numord = [f for f in top_any if f in numord_vars][:4] # take top 4 among numeric/ordinal only
print(f"Numeric/ordinal features for pairplot (from assoc ranking, up to 4): {top_numord}")

if len(top_numord) >= 2:
    g = sns.pairplot(
        data[top_numord + [TARGET_COL]],
        hue=TARGET_COL,
        diag_kind='kde',
        plot_kws={'alpha': 0.6}
    )
    # g.fig.suptitle('Pairplot of Top Features vs Reabsorption (from assoc ranking)',
    #                y=1.02, fontsize=16, fontweight='bold')
    # Display at 300 dpi
    g.fig.set_dpi(300)
    plt.show()  # show first
    # Then save as TIFF (300 dpi) to the existing main output directory (out_dir)
    fig_path = os.path.join(corr_fig_dir, f"pairplot_{TARGET_COL.lower()}_top_from_assoc.tiff")
    _save_fig(g.fig, fig_path, format='tiff', dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
else:
    print("Insufficient numeric/ordinal features (<2) for pairplot - skipping.")

# 5) 'Feature importance' bar chart: show the top 10 association strengths with Reabsorption
plt.figure(figsize=(10, 6))
fi = assoc_to_target.head(10).copy()   # top 10 (keep your original logic)
fi = fi.iloc[::-1]  # draw horizontal bars from small to large

# Colors: distinguish variable types (numeric/ordinal vs nominal)
def _feattype(f):
    if f in nominal_vars: return 'nominal'
    if f in numord_vars:  return 'numeric/ordinal'
    return 'other'

colors = ['#4C78A8' if _feattype(f) == 'nominal' else '#2ecc71' for f in fi.index]

bars = plt.barh(range(len(fi)), fi.values, color=colors, alpha=0.9)
plt.yticks(range(len(fi)), fi.index)
plt.xlabel('Association Strength with Reabsorption (0-1)')
# plt.title('Top 6 Features by Association with Reabsorption (from assoc)',
#           fontweight='bold', fontsize=14)
plt.grid(axis='x', alpha=0.3)

# Value labels
for bar, val in zip(bars, fi.values):
    plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
# Display at 300 dpi
fig = plt.gcf()
fig.set_dpi(300)
# Then save as TIFF (300 dpi) to the correlation figure directory
_save_fig(fig, os.path.join(corr_fig_dir, "feature_importance_top6_from_assoc.tiff"),
            format='tiff', dpi=300, bbox_inches='tight')
print(f"[OK] Figure saved: {os.path.join(corr_fig_dir, 'feature_importance_top6_from_assoc.tiff')}")

# ==================== 3.5 Data Quality Assessment ====================
# Purpose:
#   - Evaluate completeness, duplicates, and type consistency
#   - Surface text/object columns and target distribution diagnostics
#   - Summarize data integrity before downstream analytics
print("\n[SECTION] Part 8: Data Quality Assessment")
print("-" * 40)


# -- Data completeness and duplicate records --
total_cells = data.shape[0] * data.shape[1]
missing_total = int(data.isnull().sum().sum())
integrity = (1 - missing_total / total_cells) * 100 if total_cells > 0 else 0.0
dup_cnt = int(data.duplicated().sum())

print("Data Quality Report:")
print(f"  - Data completeness: {integrity:.1f}%  (missing cells: {missing_total})")
print(f"  - Duplicate records: {dup_cnt}")

# -- Data type consistency --
obj_cols = data.select_dtypes(include=['object']).columns.tolist()
if len(obj_cols) == 0:
    print("  - Data type consistency: [OK] No object/text columns detected.")
else:
    print("  - Data type consistency: [WARN] Text columns detected (consider encoding).")
    print(f"    Text columns: {obj_cols}")

# -- Target distribution (fixed order 0->1) --
target_counts = data[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
target_pct = (data[TARGET_COL].value_counts(normalize=True)
              .reindex([0, 1], fill_value=0) * 100)

# -- Most correlated feature (reuse 'assoc' from Part 5) --
top_feat_str = "N/A"
top_feat_val = np.nan
if 'assoc' in globals() and TARGET_COL in assoc.columns:
    assoc_to_target = assoc[TARGET_COL].drop(index=TARGET_COL).sort_values(ascending=False)
    if len(assoc_to_target) > 0:
        top_feat_str = assoc_to_target.index[0]
        top_feat_val = assoc_to_target.iloc[0]
else:
    print("  - Note: Association matrix 'assoc' not detected from Part 5; cannot extract top correlated feature.")

# -- Feature with most outliers (reuse 'outlier_summary' from Part 6; otherwise compute temporarily) --
def _detect_outliers_iqr(df, feature):
    s = pd.to_numeric(df[feature], errors='coerce')
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return int(((s < lb) | (s > ub)).sum())

if 'outlier_summary' in globals() and isinstance(outlier_summary, dict) and len(outlier_summary) > 0:
    outlier_most_feat = max(outlier_summary.keys(), key=lambda x: outlier_summary[x]['count'])
else:
    cont_candidates = [
        "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
        "Months_of_Review", "Initial_volume"  # Last_volume & Absorption_rate excluded
    ]
    cont_candidates = [c for c in cont_candidates if c in data.columns]
    if len(cont_candidates) > 0:
        counts = {f: _detect_outliers_iqr(data, f) for f in cont_candidates}
        outlier_most_feat = max(counts.keys(), key=lambda x: counts[x])
    else:
        outlier_most_feat = "N/A"

# -- Summary Report --
print("\n" + "=" * 60)
print("[SECTION] EDA Summary")
print("=" * 60)
print(f"[OK] Dataset contains {data.shape[0]} samples and {data.shape[1] - 1} features (excluding the target).")
print(f"[OK] Target variable distribution: Non-resorption {target_pct.loc[0]:.1f}%, Resorption {target_pct.loc[1]:.1f}%")
if top_feat_str != "N/A":
    print(f"[OK] Most correlated feature with the outcome (based on association strength): {top_feat_str}  (strength: {top_feat_val:.3f})")
else:
    print("[OK] Most correlated feature with the outcome (based on association strength): N/A")
print(f"[OK] Feature with the most outliers: {outlier_most_feat}")
print("=" * 60)


# ==================== 3.6 Statistical Analysis Optimization (Assoc + Significance) ====================
# Purpose:
#   - Combine association strengths with significance across feature types
#   - Rank/export significant relationships for reporting
#   - Guide feature prioritization for subsequent modeling
print("\n[SECTION] Part 9: Statistical Analysis Optimization (reuse Part 5 results + significance)")
print("-" * 40)

from scipy.stats import spearmanr, kruskal


# ---- Variable groups (consistent with global settings; auto-filter non-existing; exclude Last_volume/Absorption_rate) ----
continuous_vars = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
    "Months_of_Review", "Initial_volume", "RSI", "DHI"
]
ordinal_vars = ["Pfirrmann", "Komori", "MSU"]
nominal_vars = ["Gender", "Herniated_Level", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Bull_eye"]

continuous_vars = [c for c in continuous_vars if c in data.columns]
ordinal_vars    = [c for c in ordinal_vars    if c in data.columns]
nominal_vars    = [c for c in nominal_vars    if c in data.columns]

continuous_vars = [c for c in continuous_vars if c not in EXCLUDE_FROM_MAIN_ANALYSES]
ordinal_vars    = [c for c in ordinal_vars    if c not in EXCLUDE_FROM_MAIN_ANALYSES]
nominal_vars    = [c for c in nominal_vars    if c not in EXCLUDE_FROM_MAIN_ANALYSES]

numeric_vars = continuous_vars + ordinal_vars

# Treat Reabsorption as a nominal variable
if TARGET_COL not in nominal_vars:
    nominal_vars = nominal_vars + [TARGET_COL]

all_vars = numeric_vars + nominal_vars

# ---- Reuse 'assoc' from Part 5; require it to exist ----
if 'assoc' not in globals():
    raise RuntimeError("Association matrix 'assoc' from Part 5 not detected. Please run Part 5 first.")

# Keep only variables to be displayed/tested this time (preserve order)
assoc = assoc.reindex(index=all_vars, columns=all_vars)

# ---- Compute p-value matrix p_assoc (lower triangle), mirror symmetrically ----
def _dropna_pair(x, y):
    s1 = pd.Series(x); s2 = pd.Series(y)
    m = (~s1.isna()) & (~s2.isna())
    return s1[m], s2[m]

def p_numeric_numeric(x, y):
    x, y = _dropna_pair(x, y)
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    r, p = spearmanr(x, y)
    return p

def p_nominal_nominal(x, y):
    x, y = _dropna_pair(x, y)
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    tab = pd.crosstab(x, y)
    try:
        if tab.shape == (2, 2) and (tab.values < 5).any():
            _, p = fisher_exact(tab)
        else:
            _, p, _, _ = chi2_contingency(tab)
    except Exception:
        p = np.nan
    return p

def p_mixed_kw(cat, meas):
    """Nominal (or discrete) vs numeric/ordinal: Kruskal-Wallis"""
    c, m = _dropna_pair(cat, meas)
    if c.nunique() < 2:
        return np.nan
    groups = [m[c == g].dropna().values for g in pd.unique(c)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    try:
        _, p = kruskal(*groups)
    except Exception:
        p = np.nan
    return p

# p-value matrix
p_assoc = pd.DataFrame(np.nan, index=all_vars, columns=all_vars, dtype=float)

for i, vi in enumerate(all_vars):
    for j, vj in enumerate(all_vars):
        if j >= i:
            continue  # lower triangle only
        xi, xj = data[vi], data[vj]
        if (vi in numeric_vars) and (vj in numeric_vars):
            p = p_numeric_numeric(xi, xj)
        elif (vi in nominal_vars) and (vj in nominal_vars):
            p = p_nominal_nominal(xi, xj)
        else:
            # mixed: nominal vs numeric/ordinal (either direction)
            if vi in nominal_vars and vj in numeric_vars:
                p = p_mixed_kw(xi, xj)
            elif vj in nominal_vars and vi in numeric_vars:
                p = p_mixed_kw(xj, xi)
            else:
                p = np.nan
        p_assoc.loc[vi, vj] = p
        p_assoc.loc[vj, vi] = p

# ---- Overlay significance stars on the association matrix (lower triangle) ----
def p_to_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# Display at 300 dpi as well
fig = plt.figure(figsize=(12, 10), dpi=300)
mask = np.triu(np.ones_like(assoc, dtype=bool))
ax = sns.heatmap(assoc, mask=mask, annot=True, fmt=".2f",
                 cmap='RdBu_r', vmin=0, vmax=1,
                 square=True, linewidths=0.5,
                 cbar_kws={"shrink": .8})
# plt.title('Comprehensive Association Heatmap with Significance (Lower Triangle)\n'
#           'Num->Num: Spearman p | Nom->Nom: Chi-square/Fisher p | Mixed: Kruskal-Wallis p',
#           fontsize=14, fontweight='bold', pad=16)
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)

# Overlay stars
for i, vi in enumerate(assoc.index):
    for j, vj in enumerate(assoc.columns):
        if i > j:
            stars = p_to_stars(p_assoc.loc[vi, vj])
            if stars:
                ax.text(j + 0.25, i + 0.75, stars,
                        ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()

# Save as TIFF (300 dpi) to correlation figures
_save_fig(fig, os.path.join(corr_fig_dir, "association_heatmap_with_significance.tiff"),
            format='tiff', dpi=300, bbox_inches='tight')
print(f"[OK] Figure saved: {os.path.join(corr_fig_dir, 'association_heatmap_with_significance.tiff')}")

# ---- Significance summary vs Reabsorption (all variable types) ----
rows = []
for f in all_vars:
    if f == TARGET_COL:
        continue
    assoc_strength = assoc.loc[f, TARGET_COL] if (f in assoc.index and TARGET_COL in assoc.columns) else np.nan
    # corresponding p-value + test type
    if (f in numeric_vars) and (TARGET_COL in numeric_vars):
        test = "Spearman"
        pval = p_assoc.loc[f, TARGET_COL]
    elif (f in nominal_vars) and (TARGET_COL in nominal_vars):
        test = "Chi-square/Fisher"
        pval = p_assoc.loc[f, TARGET_COL]
    else:
        test = "Kruskal-Wallis"
        pval = p_assoc.loc[f, TARGET_COL]

    rows.append({
        "feature": f,
        "test": test,
        "association_strength(0-1)": assoc_strength,
        "p_value": pval,
        "signif": p_to_stars(pval)
    })

sig_summary = pd.DataFrame(rows).sort_values(
    by=["p_value", "association_strength(0-1)"],
    ascending=[True, False]
).reset_index(drop=True)

print("\n[INFO] Significance summary vs Reabsorption (sorted by p ascending, tie-break by strength descending):")
print(sig_summary.round(3))

# Export to correlation data directory (formal result)
csv_path = os.path.join(corr_data_dir, "significance_summary_vs_absorption.csv")
sig_summary.to_csv(csv_path, index=False)
print(f"\n[OK] Exported: {csv_path}")


# ==================== 3.7 Between-Group Comparison (Reabsorption=0 vs 1) ====================
# Purpose:
#   - Compare numeric/ordinal features between target groups
#   - Run t-tests/MWU, effect sizes, FDR, bootstrap CIs, and visualize top effects
#   - Identify discriminative variables for interpretation and modeling
print("\n[SECTION] Part 10: Between-Group Comparison (numeric/ordinal variables, Reabsorption=0 vs 1)")
print("-" * 40)

from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---- Feature set (continuous + ordinal; exclude Last_volume / Absorption_rate) ----
continuous_vars = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
    "Months_of_Review", "Initial_volume", "RSI", "DHI"
]
ordinal_vars = ["Pfirrmann", "Komori", "MSU"]

continuous_vars = [c for c in continuous_vars if c in data.columns]
ordinal_vars    = [c for c in ordinal_vars    if c in data.columns]
numeric_features = continuous_vars + ordinal_vars

# Ensure numeric types for tests
for col in numeric_features:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# ========== 1. Welch's t-test ==========
print("\n[SECTION] Welch's t-test results:")
t_test_results = {}
for feature in numeric_features:
    g0 = data.loc[data[TARGET_COL]==0, feature].dropna()
    g1 = data.loc[data[TARGET_COL]==1, feature].dropna()
    if len(g0)<2 or len(g1)<2:
        t_stat, p_val = np.nan, np.nan
    else:
        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)  # Welch
    t_test_results[feature] = {"t-statistic": t_stat, "p-value": p_val}
t_test_df = pd.DataFrame(t_test_results).T
print(t_test_df)

# Export CSV to correlation data directory
t_test_csv = os.path.join(corr_data_dir, "ttest_results_absorption.csv")
t_test_df.to_csv(t_test_csv, index=True)
print(f"[OK] Exported: {t_test_csv}")

# ========== 2. Mann-Whitney U ==========
print("\n[SECTION] Mann-Whitney U test results:")
mw_test_results = {}
for feature in numeric_features:
    g0 = data.loc[data[TARGET_COL]==0, feature].dropna()
    g1 = data.loc[data[TARGET_COL]==1, feature].dropna()
    if len(g0)<2 or len(g1)<2:
        stat, p_val = np.nan, np.nan
    else:
        stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    mw_test_results[feature] = {"U-statistic": stat, "p-value": p_val}
mw_test_df = pd.DataFrame(mw_test_results).T
print(mw_test_df)

# Export CSV to correlation data directory
mw_csv = os.path.join(corr_data_dir, "mannwhitneyu_results_absorption.csv")
mw_test_df.to_csv(mw_csv, index=True)
print(f"[OK] Exported: {mw_csv}")

# ========== 3. Cohen's d ==========
def cohens_d(g1, g0):
    g1 = pd.Series(g1).dropna(); g0 = pd.Series(g0).dropna()
    if len(g1)<2 or len(g0)<2: return np.nan
    diff = g1.mean() - g0.mean()
    pooled = np.sqrt((g1.var(ddof=1)+g0.var(ddof=1))/2)
    if pooled==0: return np.nan
    return diff/pooled

print("\n[INFO] Effect size (Cohen's d):")
effect_size = {}
for feature in numeric_features:
    g0 = data.loc[data[TARGET_COL]==0, feature]
    g1 = data.loc[data[TARGET_COL]==1, feature]
    effect_size[feature] = cohens_d(g1, g0)
effect_size_df = pd.DataFrame(effect_size, index=["Cohen's d"]).T
print(effect_size_df)

# Export CSV to correlation data directory
effect_csv = os.path.join(corr_data_dir, "effect_size_cohensd_absorption.csv")
effect_size_df.to_csv(effect_csv, index=True)
print(f"[OK] Exported: {effect_csv}")

# ========== 4. Multiple testing correction (FDR) ==========
p_vals = t_test_df["p-value"].fillna(1.0).values
reject, p_adj, _, _ = multipletests(p_vals, method="fdr_bh")
t_test_df["FDR p-value"] = p_adj

print("\n[INFO] t-test results with FDR correction:")
print(t_test_df)

# Export CSV to correlation data directory
ttest_fdr_csv = os.path.join(corr_data_dir, "ttest_results_with_fdr_absorption.csv")
t_test_df.to_csv(ttest_fdr_csv, index=True)
print(f"[OK] Exported: {ttest_fdr_csv}")

# ========== 5. Bootstrap CI ==========
print("\n[INFO] Bootstrap confidence intervals (mean):")
def bootstrap_ci(arr, statistic=np.mean, n_iter=1000, alpha=0.05, rng=None):
    x = pd.Series(arr).dropna().values
    if len(x)==0: return np.nan,np.nan,np.nan
    if rng is None:
        rng = np.random.default_rng(SEED)
    boot = [statistic(rng.choice(x, size=len(x), replace=True)) for _ in range(n_iter)]
    low = np.percentile(boot, 100*alpha/2)
    high = np.percentile(boot, 100*(1-alpha/2))
    return low, high, np.mean(boot)

bootstrap_results = pd.DataFrame(index=numeric_features,
    columns=["Mean (0)","CI Lower (0)","CI Upper (0)","Mean (1)","CI Lower (1)","CI Upper (1)"])

bootstrap_rng = np.random.default_rng(SEED)
for feature in numeric_features:
    g0 = data.loc[data[TARGET_COL]==0, feature]
    g1 = data.loc[data[TARGET_COL]==1, feature]
    ci0 = bootstrap_ci(g0, rng=bootstrap_rng); ci1 = bootstrap_ci(g1, rng=bootstrap_rng)
    bootstrap_results.loc[feature] = [ci0[2], ci0[0], ci0[1], ci1[2], ci1[0], ci1[1]]
print(bootstrap_results)

# Export CSV to correlation data directory
boot_csv = os.path.join(corr_data_dir, "bootstrap_ci_means_absorption.csv")
bootstrap_results.to_csv(boot_csv, index=True)
print(f"[OK] Exported: {boot_csv}")

# ========== 6. Visualization (top 4 features by Cohen's d) ==========
print("\n[INFO] Visualization of top 4 features by effect size:")
top_features = effect_size_df["Cohen's d"].abs().sort_values(ascending=False).index[:4].tolist()

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)  # display at 300 dpi
axes = axes.flatten()
for i, feat in enumerate(top_features):
    ax = axes[i]
    sns.violinplot(x=TARGET_COL, y=feat, data=data, ax=ax, inner=None,
                   palette=['#3498db','#e74c3c'], alpha=0.5)
    sns.boxplot(x=TARGET_COL, y=feat, data=data, ax=ax, width=0.3,
                showcaps=True, boxprops={'facecolor':'white'}, showfliers=False)
    sns.stripplot(x=TARGET_COL, y=feat, data=data, ax=ax, size=4, jitter=True, alpha=0.6)

    # Add mean and CI
    row = bootstrap_results.loc[feat]
    means = [row["Mean (0)"], row["Mean (1)"]]
    lowers = [row["CI Lower (0)"], row["CI Lower (1)"]]
    uppers = [row["CI Upper (0)"], row["CI Upper (1)"]]
    ax.hlines(y=means[0], xmin=-0.2, xmax=0.2, color="green", linewidth=2)
    ax.hlines(y=means[1], xmin=0.8, xmax=1.2, color="red", linewidth=2)
    ax.errorbar(x=[0,1], y=means,
                yerr=[[means[0]-lowers[0], means[1]-lowers[1]],
                      [uppers[0]-means[0], uppers[1]-means[1]]],
                fmt="none", capsize=5, ecolor="black", elinewidth=2)

    # p-value & d-value annotation
    p_val = t_test_df.loc[feat,"p-value"]
    d_val = effect_size_df.loc[feat,"Cohen's d"]
    sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "ns"
    ax.annotate(f"p = {p_val:.4f} {sig}\nCohen's d = {d_val:.2f}",
                xy=(0.5,0.95), xycoords="axes fraction",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3))
    ax.set_title(f"{feat}", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Non-resorption (0)","Resorption (1)"])

# plt.suptitle("Group Comparison of Top Numeric/Ordinal Features (Reabsorption=0 vs 1)",
#              fontsize=18, fontweight="bold")
plt.tight_layout(); plt.subplots_adjust(top=0.9)

# Save as TIFF (300 dpi) to correlation figures
_save_fig(fig, os.path.join(corr_fig_dir, "group_comparison_absorption.tiff"),
            format="tiff", dpi=300, bbox_inches="tight")
print(f"[OK] Figure saved: {os.path.join(corr_fig_dir, 'group_comparison_absorption.tiff')}")


# # Model Development

# ## Data Segmentation and Standardization

from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# ==================== 4 Preprocessing & modeling pipeline ====================

# ── Reassign 'data' to retrospective-only cohort for ML modelling ─────────────
# Sections 2-3 (Baseline, EDA, Correlation) used the combined bidirectional
# cohort (data_combined, n=Retrospective+Prospective).
# From here onwards, ALL modelling, validation, and SHAP analyses use only the
# retrospective cohort (data_retro), which is the pre-specified training dataset.
data = data_retro.copy()
print(f"[INFO] ML modelling dataset reassigned to retrospective cohort: n={len(data)}")

# ==================== 4.1 Data Splitting & Preprocessing (Reproducible | Train/Val/Test=7/2/1) ====================
print("\n[SECTION] Part 1: Data Splitting & Preprocessing (Reproducible | CT: OHE+Ordinal+Scaler | 7/2/1 split)")
print("-" * 80)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------- Reproducibility ----------
# SEED and global RNGs are initialized at the top of this file.

# 1) Target and excluded columns (fixed)
exclude_cols = [
    "ID", "Name",
    "Last_volume", "Absorption_rate", "Absorption", "Absorption_type",
    "Months_of_Review",
    "Cohort",          # design variable; not a clinical feature for modelling
    "Source_File", "Source_Sheet", "Unified_ID",  # metadata fields if present
]
if TARGET_COL not in data.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found.")

# 2) Variable groups (keep only existing columns)
continuous_vars = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
    "Months_of_Review", "Initial_volume", "RSI", "DHI"
]
ordinal_vars = ["Pfirrmann", "Komori", "MSU"]
nominal_vars = ["Gender", "Herniated_Level", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Bull_eye"]

continuous_vars = [c for c in continuous_vars if c in data.columns]
ordinal_vars    = [c for c in ordinal_vars    if c in data.columns]
nominal_vars    = [c for c in nominal_vars    if c in data.columns]

# 3) Select feature columns (exclude excluded columns)
feature_cols = [c for c in (continuous_vars + ordinal_vars + nominal_vars) if c not in exclude_cols]
X_raw = data[feature_cols].copy()
y_ser = data[TARGET_COL].astype(int).copy()

print(f"[OK] Features ({len(feature_cols)}): {feature_cols}")
print(f"[INFO] Target distribution: {y_ser.value_counts().to_dict()}")

# 4) Coerce continuous/ordinal to numeric
for col in continuous_vars + ordinal_vars:
    if col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

# 4.1) Normalize numeric-coded nominal variables -> string labels (keep NaN)
for _col in ["Iwabuchi", "Modic", "Spinal_canal_stenosis"]:
    if _col in X_raw.columns:
        _tmp = pd.to_numeric(X_raw[_col], errors="coerce")
        X_raw[_col] = _tmp.apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)

# 4.2) Gender cleanup (Female/Male)
# NOTE: Gender will be treated as NOMINAL and OneHot encoded in ColumnTransformer.
if "Gender" in X_raw.columns:
    X_raw["Gender"] = X_raw["Gender"].astype(str).str.strip()
    X_raw["Gender"] = X_raw["Gender"].replace({
        "female": "Female", "FEMALE": "Female", "F": "Female",
        "male": "Male", "MALE": "Male", "M": "Male"
    })

# 4.3) Bull_eye keep numeric for imputation (allow NaN)
if "Bull_eye" in X_raw.columns:
    X_raw["Bull_eye"] = pd.to_numeric(X_raw["Bull_eye"], errors="coerce")

# ==================== 4.1A Upgraded OOF + External Benchmark ====================
print("\n[SECTION] Part 4 Upgrade: OOF + External Validation + TabPFN + Ensemble Benchmark")
print("-" * 80)

from manuscript_ml_upgrade_core import BenchmarkPaths, UpgradeConfig, run_oof_external_benchmark
from manuscript_ml_upgrade_explain import run_champion_shap

UPGRADE_PATHS = BenchmarkPaths(
    ml_dir=Path(ML_DIR),
    metrics_dir=Path(ML_METRICS_DIR),
    models_dir=Path(ML_MODELS_DIR),
    audit_dir=Path(ML_AUDIT_DIR),
    deploy_dir=Path(DEPLOY_DIR),
    shap_dir=Path(RUN_ROOT) / "05_SHAP",
    figures_dir=Path(SHAP_DIR),
)

UPGRADE_CONFIG = UpgradeConfig(
    seed=SEED,
    validation_mode="oof_external",
    oof_folds=5,
    inner_cv_folds=3,
    bayes_n_iter=12,
    enable_tabpfn=True,
    enable_voting=True,
    enable_stacking=True,
    ensemble_base_models=("Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "TabPFN"),
    shap_mode="champion_only",
    device="cpu",
    model_version="v2.5",
    champion_auc_threshold=0.70,
)

upgrade_outputs = run_oof_external_benchmark(
    data_retro=data_retro,
    data_pros=data_pros,
    target_col=TARGET_COL,
    feature_cols=feature_cols,
    continuous_vars=continuous_vars,
    ordinal_vars=ordinal_vars,
    nominal_vars=nominal_vars,
    paths=UPGRADE_PATHS,
    config=UPGRADE_CONFIG,
)

if UPGRADE_CONFIG.shap_mode == "champion_only":
    shap_outputs = run_champion_shap(
        champion_name=upgrade_outputs["champion_name"],
        champion_result=upgrade_outputs["champion_result"],
        X_retro=upgrade_outputs["X_retro"],
        X_external=upgrade_outputs["X_external"],
        paths=UPGRADE_PATHS,
        seed=SEED,
    )
    print(f"[OK] Champion SHAP completed: {shap_outputs['summary_path']}")

print(f"[OK] OOF benchmark summary saved to: {upgrade_outputs['summary_path']}")
print(f"[OK] Champion model: {upgrade_outputs['champion_name']}")
print("[INFO] Upgraded modelling workflow completed.")
