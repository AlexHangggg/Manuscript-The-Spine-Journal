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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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
print("[INFO] Upgraded modelling workflow completed. Legacy Part 4-8 pipeline skipped by design.")
sys.exit(0)

# =====================================================================
# Bull_eye imputation: two-stage strategy (LASSO feature selection + Random Forest prediction)
# =====================================================================
#
# Strategy overview:
# 1. Stage 1: run LASSO-based feature selection
#    - One-hot encode candidate predictors
#    - Select the most informative features
#    - Keep non-zero-coefficient features
#
# 2. Stage 2: train Random Forest on selected features
#    - Reduce dimensionality and overfitting risk
#    - Improve generalization
#
# 3. Diagnostics:
#    - Feature importance visualization
#    - Pre/post imputation distribution comparison
#    - Predicted class distribution
#
# =====================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import ParameterGrid


class BullEyeImputer:
    """
    Two-stage Bull_eye imputer: LASSO feature selection + Random Forest prediction.

    Key ideas:
    - Use LASSO first for feature reduction from all candidates.
    - Train Random Forest on selected encoded features.
    - Prevent leakage by fitting only on training data.
    - Provide diagnostic plots for transparency.

    Parameters:
        predictors: Candidate predictor column names.
        seed: Random seed.
        n_estimators: Number of trees for Random Forest.
        lasso_C: Inverse regularization strength for LASSO logistic model.
        min_features: Minimum number of retained features.
        max_features: Maximum number of retained features.
        output_dir: Output directory for diagnostic figures.
    """

    def __init__(self, predictors: list, seed: int = SEED, n_estimators: int = 100,
                 lasso_C: float = 0.5, min_features: int = 3, max_features: int =5,
                 output_dir: str = None):
        self.predictors = predictors
        self.seed = seed
        self.n_estimators = n_estimators
        self.lasso_C = lasso_C
        self.min_features = min_features
        self.max_features = max_features
        self.output_dir = output_dir

        # Model components
        self.lasso = None
        self.clf = None
        self.fallback_value = None

        # Feature management
        self.all_cols = None  # All candidate predictors
        self.selected_cols = None  # Selected original predictors after LASSO mapping
        self.cat_encoders = {}  # Reserved for compatibility (not used in OHE flow)
        self.encoded_feature_names = None
        self.selected_feature_names = None
        self.selected_feature_indices = None
        self.selected_cols_original = None
        self.num_fill_values = {}

        # Diagnostic artifacts
        self.original_distribution = None  # Observed Bull_eye distribution
        self.imputed_distribution = None  # Imputed Bull_eye distribution
        self.feature_importance = None  # Random Forest feature importances
        self.lasso_coefficients = None  # LASSO coefficient summary

    def _prepare_data_for_lasso(self, X: pd.DataFrame, mask: pd.Series):
        """
        Prepare data for LASSO with one-hot encoding for categoricals.
        """
        X_be = X.loc[mask, self.all_cols].copy()
        y_be = X.loc[mask, "Bull_eye"].copy().astype(int)

        # Split predictors into numeric and categorical columns
        numeric_cols = []
        categorical_cols = []

        for col in self.all_cols:
            if pd.api.types.is_numeric_dtype(X_be[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # Robust numeric fill using training medians
        X_num_df = X_be[numeric_cols].apply(pd.to_numeric, errors="coerce")
        self.num_fill_values = X_num_df.median(numeric_only=True).to_dict()
        X_num_df = X_num_df.fillna(self.num_fill_values)

        # One-hot encode categorical variables
        if categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_df = X_be[categorical_cols].fillna("__MISSING__").astype(str)
            cat_encoded = ohe.fit_transform(cat_df)
            cat_feature_names = ohe.get_feature_names_out(categorical_cols)

            # Store encoder for reuse in transform stage
            self.ohe = ohe
            self.categorical_cols = categorical_cols
            self.numeric_cols = numeric_cols

            # Concatenate numeric and one-hot categorical features
            X_numeric = X_num_df.values
            X_encoded = np.hstack([X_numeric, cat_encoded])
            all_feature_names = numeric_cols + list(cat_feature_names)
        else:
            X_encoded = X_num_df.values
            all_feature_names = numeric_cols
            self.ohe = None
            self.categorical_cols = []
            self.numeric_cols = numeric_cols

        return X_encoded, y_be, all_feature_names

    def _encode_with_fitted_lasso_ohe(self, X: pd.DataFrame) -> np.ndarray:
        """Encode data to the same feature space fitted in LASSO stage."""
        X_part = X.loc[:, self.all_cols].copy()

        if len(self.numeric_cols) > 0:
            X_num_df = X_part[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            if self.num_fill_values:
                X_num_df = X_num_df.fillna(self.num_fill_values)
            X_numeric = X_num_df.values
        else:
            X_numeric = np.empty((len(X_part), 0))

        if self.ohe is not None and len(self.categorical_cols) > 0:
            X_cat_df = X_part[self.categorical_cols].fillna("__MISSING__").astype(str)
            X_cat = self.ohe.transform(X_cat_df)
            return np.hstack([X_numeric, X_cat])

        return X_numeric

    def _select_features_with_lasso(self, X_encoded: np.ndarray, y: pd.Series,
                                    feature_names: list):
        """
        Select features using LASSO logistic regression.
        """
        print(f"\n[INFO] ===== STAGE 1: LASSO Feature Selection (Dimensionality Reduction) =====")
        print(f"[INFO] Input: {len(feature_names)} One-Hot encoded features from {len(self.all_cols)} original variables")
        print(f"[INFO] Training samples: {len(y)}")
        print(f"[INFO] Goal: Select most predictive features for Random Forest")

        # Train LASSO model (multinomial logistic regression with L1 penalty)
        # Note: sklearn LogisticRegression uses saga solver for multinomial + L1
        lasso = LogisticRegression(
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            C=self.lasso_C,
            max_iter=5000,
            random_state=self.seed
        )

        # Evaluate LASSO selection quality with cross-validation
        # Use fixed StratifiedKFold for reproducibility
        cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_scores = cross_val_score(lasso, X_encoded, y, cv=cv_kfold, scoring='accuracy')
        print(f"[INFO] LASSO feature selection quality (5-fold CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Fit LASSO on all available training samples
        lasso.fit(X_encoded, y)
        self.lasso = lasso

        # Extract non-zero coefficient features
        # For multi-class, keep features non-zero in any class
        coef_matrix = lasso.coef_  # shape: (n_classes, n_features)
        non_zero_mask = (coef_matrix != 0).any(axis=0)
        selected_indices = np.where(non_zero_mask)[0]

        # If too few features are selected, relax with top-k by coefficient magnitude
        if len(selected_indices) < self.min_features:
            # Rank by absolute coefficient sum
            coef_abs_sum = np.abs(coef_matrix).sum(axis=0)
            top_indices = np.argsort(coef_abs_sum)[-self.min_features:]
            selected_indices = top_indices
            print(f"[WARN] LASSO selected only {len(selected_indices)} features, expanding to top {self.min_features}")

        # If too many features are selected, truncate to max_features
        if len(selected_indices) > self.max_features:
            # Rank selected features by absolute coefficient sum
            coef_abs_sum = np.abs(coef_matrix).sum(axis=0)
            selected_coef_abs_sum = coef_abs_sum[selected_indices]
            # Keep top max_features within selected set
            top_within_selected = np.argsort(selected_coef_abs_sum)[-self.max_features:]
            selected_indices = selected_indices[top_within_selected]
            print(f"[INFO] LASSO selected {len(np.where(non_zero_mask)[0])} features, limiting to top {self.max_features}")

        selected_feature_names = [feature_names[i] for i in selected_indices]

        # Save coefficient table for diagnostics
        self.lasso_coefficients = pd.DataFrame({
            'feature': feature_names,
            'coef_abs_sum': np.abs(coef_matrix).sum(axis=0),
            'selected': non_zero_mask
        }).sort_values('coef_abs_sum', ascending=False)

        print(f"[OK] LASSO selected {len(selected_feature_names)} features:")
        for feat in selected_feature_names:
            print(f"       - {feat}")

        return selected_feature_names

    def _map_selected_to_original_cols(self, selected_feature_names: list):
        """
        Map selected one-hot feature names back to original predictor columns.
        """
        selected_original_cols = set()

        for feat in selected_feature_names:
            # Check whether a selected feature belongs to a source column
            for col in self.all_cols:
                if feat.startswith(col + '_') or feat == col:
                    selected_original_cols.add(col)
                    break

        return sorted(selected_original_cols)

    def _prepare_data_for_rf(self, X: pd.DataFrame, mask: pd.Series):
        """
        Prepare Random Forest inputs using selected OHE feature indices.
        """
        X_all_encoded = self._encode_with_fitted_lasso_ohe(X.loc[mask, self.all_cols])
        X_be = X_all_encoded[:, self.selected_feature_indices]
        y_be = X.loc[mask, "Bull_eye"].copy().astype(int)
        return X_be, y_be

    def fit(self, X_train: pd.DataFrame, y_train=None) -> 'BullEyeImputer':
        """
        Fit the imputer on training data.
        """
        if "Bull_eye" not in X_train.columns:
            return self

        # Save observed Bull_eye distribution
        self.original_distribution = X_train["Bull_eye"].value_counts().sort_index()

        # Identify available predictor columns
        self.all_cols = [c for c in self.predictors if c in X_train.columns]

        # Keep rows with observed Bull_eye for training
        train_mask = X_train["Bull_eye"].notna()

        # Check minimum data sufficiency
        if len(self.all_cols) == 0 or train_mask.sum() < 20:
            mode = X_train.loc[train_mask, "Bull_eye"].dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            print(f"[WARN] Bull_eye imputer: using fallback mode={self.fallback_value} (insufficient data: {train_mask.sum()} samples)")
            return self

        # ========== STAGE 1: LASSO Feature Selection ==========
        try:
            X_encoded, y_be, feature_names = self._prepare_data_for_lasso(X_train, train_mask)
            self.encoded_feature_names = feature_names
            selected_feature_names = self._select_features_with_lasso(X_encoded, y_be, feature_names)
            self.selected_feature_names = selected_feature_names
            self.selected_cols_original = self._map_selected_to_original_cols(selected_feature_names)
        except Exception as e:
            print(f"[WARN] LASSO feature selection failed: {e}")
            print(f"[INFO] Falling back to using all encoded predictors")
            try:
                X_encoded, y_be, feature_names = self._prepare_data_for_lasso(X_train, train_mask)
                self.encoded_feature_names = feature_names
                self.selected_feature_names = feature_names
                self.selected_cols_original = self.all_cols
            except Exception as e2:
                mode = X_train.loc[train_mask, "Bull_eye"].dropna().mode()
                self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
                print(f"[WARN] Bull_eye imputer: fallback mode={self.fallback_value} (encoding failed: {e2})")
                return self

        if self.selected_feature_names is None or len(self.selected_feature_names) == 0:
            mode = X_train.loc[train_mask, "Bull_eye"].dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            print(f"[WARN] Bull_eye imputer: using fallback mode={self.fallback_value} (no selected features)")
            return self

        self.selected_feature_indices = [
            self.encoded_feature_names.index(f)
            for f in self.selected_feature_names
            if f in self.encoded_feature_names
        ]
        if len(self.selected_feature_indices) == 0:
            self.selected_feature_indices = list(range(len(self.encoded_feature_names)))
            self.selected_feature_names = [self.encoded_feature_names[i] for i in self.selected_feature_indices]
            self.selected_cols_original = self.all_cols

        self.selected_cols = self.selected_cols_original

        # ========== STAGE 2: Random Forest Training ==========
        print(f"\n[INFO] ===== STAGE 2: Random Forest Prediction (using LASSO-selected features) =====")
        print(
            f"[INFO] Input: {len(self.selected_feature_names)} OHE features "
            f"({len(self.selected_cols_original)} original vars) selected from "
            f"{len(self.encoded_feature_names)} OHE features ({len(self.all_cols)} original vars)"
        )
        print(f"[INFO] Training Random Forest for final Bull_eye imputation...")

        X_train_be, y_train_be = self._prepare_data_for_rf(X_train, train_mask)

        # Check class diversity
        if y_train_be.nunique() < 2:
            mode = y_train_be.dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            print(f"[WARN] Bull_eye imputer: using fallback mode={self.fallback_value} (low variance)")
            return self

        # Train Random Forest with hyperparameter search
        from sklearn.model_selection import GridSearchCV

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3],
        }

        print(f"[INFO] Running GridSearchCV to optimize Random Forest hyperparameters...")
        print(f"[INFO] Search space: {len(list(ParameterGrid(param_grid)))} combinations")

        # Fixed StratifiedKFold for reproducibility
        cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        # Grid search for best hyperparameters
        grid_search = GridSearchCV(
            RandomForestClassifier(
                random_state=self.seed,
                n_jobs=1,
                class_weight='balanced'
            ),
            param_grid=param_grid,
            cv=cv_kfold,
            scoring='accuracy',
            n_jobs=1,
            verbose=0
        )

        grid_search.fit(X_train_be, y_train_be)

        # Use best estimator from grid search as final model
        self.clf = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_cv_score = grid_search.best_score_

        print(f"[INFO] Best hyperparameters found:")
        for param, value in self.best_params.items():
            print(f"       - {param}: {value}")
        print(f"[INFO] Best 5-fold CV accuracy during search: {self.best_cv_score:.4f}")

        # Re-evaluate best model with same CV protocol for stable reporting
        cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_scores = cross_val_score(self.clf, X_train_be, y_train_be, cv=cv_kfold, scoring='accuracy')
        print(f"[INFO] Final imputation model (Random Forest) 5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Store feature importances
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_feature_names,
            'importance': self.clf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"[OK] Bull_eye imputer pipeline completed (trained on {train_mask.sum()} samples)")
        print(f"[INFO] Top 5 important features for Bull_eye prediction:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"       - {row['feature']}: {row['importance']:.4f}")

        return self

    def transform(self, X: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
        """
        Impute missing Bull_eye values in a dataset.
        """
        if "Bull_eye" not in X.columns:
            return X

        X_out = X.copy()
        miss_mask = X_out["Bull_eye"].isna()

        if miss_mask.sum() == 0:
            return X_out

        # Use fallback when no trained model is available
        if self.clf is None:
            fillv = self.fallback_value if self.fallback_value is not None else 1
            X_out.loc[miss_mask, "Bull_eye"] = int(np.clip(fillv, 1, 3))
            print(f"[INFO] Bull_eye ({dataset_name}): filled {miss_mask.sum()} missing values with fallback={fillv}")
            return X_out

        # Prepare encoded predictors for missing rows
        X_all_miss = self._encode_with_fitted_lasso_ohe(X_out.loc[miss_mask, self.all_cols])
        X_miss = X_all_miss[:, self.selected_feature_indices]

        # Predict classes
        pred = self.clf.predict(X_miss).astype(int)
        pred = np.clip(pred, 1, 3)

        # Predict probabilities
        pred_proba = self.clf.predict_proba(X_miss)

        X_out.loc[miss_mask, "Bull_eye"] = pred

        # Print imputation summary
        pred_counts = pd.Series(pred).value_counts().sort_index()
        print(f"[INFO] Bull_eye ({dataset_name}): imputed {miss_mask.sum()} missing values")
        print(f"[INFO] Predicted distribution: {dict(pred_counts)}")

        # Save imputed distribution (used in diagnostics)
        if dataset_name == "Train":
            self.imputed_distribution = pred_counts

        return X_out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit imputer on training data and return transformed training set."""
        self.fit(X_train)
        return self.transform(X_train, dataset_name="Train")

    def visualize_diagnostics(self):
        """
        Generate diagnostic visualizations.
        """
        if self.output_dir is None:
            print("[INFO] No output directory specified, skipping visualization")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. LASSO coefficient diagnostics
        if self.lasso_coefficients is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_features = self.lasso_coefficients.head(20)
            colors = ['green' if s else 'gray' for s in top_features['selected']]
            ax.barh(range(len(top_features)), top_features['coef_abs_sum'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Absolute Coefficient Sum')
            ax.set_title('LASSO Feature Selection: Top 20 Features')
            ax.legend([plt.Rectangle((0,0),1,1,color='green'),
                       plt.Rectangle((0,0),1,1,color='gray')],
                      ['Selected', 'Not Selected'])
            plt.tight_layout()
            _save_fig(plt.gcf(), os.path.join(self.output_dir, 'bull_eye_lasso_coefficients.tiff'), dpi=300)
            plt.close()
            print(f"[INFO] Saved: {os.path.join(self.output_dir, 'bull_eye_lasso_coefficients.tiff')}")

        # 2. Random Forest feature importance
        if self.feature_importance is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(self.feature_importance)),
                   self.feature_importance['importance'])
            ax.set_yticks(range(len(self.feature_importance)))
            ax.set_yticklabels(self.feature_importance['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance for Bull_eye Prediction')
            plt.tight_layout()
            _save_fig(plt.gcf(), os.path.join(self.output_dir, 'bull_eye_rf_importance.tiff'), dpi=300)
            plt.close()
            print(f"[INFO] Saved: {os.path.join(self.output_dir, 'bull_eye_rf_importance.tiff')}")

        # 3. Distribution comparison (observed vs imputed)
        if self.original_distribution is not None and self.imputed_distribution is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Observed distribution
            self.original_distribution.plot(kind='bar', ax=axes[0], color='steelblue')
            axes[0].set_title('Original Distribution (Observed)')
            axes[0].set_xlabel('Bull_eye')
            axes[0].set_ylabel('Count')

            # Imputed distribution
            self.imputed_distribution.plot(kind='bar', ax=axes[1], color='coral')
            axes[1].set_title('Imputed Distribution (Predicted)')
            axes[1].set_xlabel('Bull_eye')
            axes[1].set_ylabel('Count')

            # Percentage comparison
            original_pct = self.original_distribution / self.original_distribution.sum() * 100
            imputed_pct = self.imputed_distribution / self.imputed_distribution.sum() * 100

            x = np.arange(len(original_pct))
            width = 0.35
            axes[2].bar(x - width/2, original_pct, width, label='Original', color='steelblue')
            axes[2].bar(x + width/2, imputed_pct, width, label='Imputed', color='coral')
            axes[2].set_title('Percentage Comparison')
            axes[2].set_xlabel('Bull_eye')
            axes[2].set_ylabel('Percentage (%)')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels([f'{i}' for i in range(1, 4)])
            axes[2].legend()

            plt.tight_layout()
            _save_fig(plt.gcf(), os.path.join(self.output_dir, 'bull_eye_distribution_comparison.tiff'), dpi=300)
            plt.close()
            print(f"[INFO] Saved: {os.path.join(self.output_dir, 'bull_eye_distribution_comparison.tiff')}")

        print(f"[INFO] All diagnostic visualizations saved to: {self.output_dir}")

# =====================================================================
# Two-stage workflow: data split followed by Bull_eye imputation
# =====================================================================
#
# Background:
# - Bull_eye is a nominal class variable (1/2/3) with missing values.
# - Missingness is treated as approximately MCAR from examination availability.
# - Other predictor columns are assumed complete here.
#
# Workflow:
# 1. Split full data into Train/Val/Test (72/18/10 effective split).
# 2. Fit Bull_eye imputer on training set only.
# 3. Impute missing Bull_eye in train/val/test.
# 4. Continue main prediction modeling on imputed datasets.
#
# =====================================================================

# 6) Dataset split: 70% Train / 20% Val / 10% Test (effective 72/18/10)
# IMPORTANT: Split BEFORE imputation to prevent data leakage!
print("\n" + "="*70)
print("[STAGE 1] Data Splitting - BEFORE Bull_eye Imputation")
print("="*70)

# Step 6a: Hold out 10% as an independent test set
X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
    X_raw, y_ser,
    test_size=0.10,
    random_state=SEED,
    stratify=y_ser
)

# Step 6b: Split remaining 90% to obtain validation set
# Final ratio: Train(72%) / Val(18%) / Test(10%)
val_ratio_of_remain = 0.20 / 0.90
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_trainval_raw, y_trainval,
    test_size=val_ratio_of_remain,
    random_state=SEED,
    stratify=y_trainval
)

print(f"[INFO] Data split completed (SEED={SEED}):")
print(f"       - Train:      {X_train_raw.shape[0]:3d} samples ({X_train_raw.shape[0]/X_raw.shape[0]*100:.1f}%)")
print(f"       - Validation: {X_val_raw.shape[0]:3d} samples ({X_val_raw.shape[0]/X_raw.shape[0]*100:.1f}%)")
print(f"       - Test:       {X_test_raw.shape[0]:3d} samples ({X_test_raw.shape[0]/X_raw.shape[0]*100:.1f}%)")

# Summarize Bull_eye missingness by split
for name, df in [("Train", X_train_raw), ("Val", X_val_raw), ("Test", X_test_raw)]:
    if "Bull_eye" in df.columns:
        n_total = len(df)
        n_missing = df["Bull_eye"].isna().sum()
        n_valid = n_total - n_missing
        print(f"[INFO] {name:12s} - Bull_eye: {n_valid:3d} valid, {n_missing:3d} missing ({n_missing/n_total*100:.1f}% missing)")

# 7) Bull_eye imputation: two-stage LASSO + Random Forest
# =====================================================================
# Updated strategy:
# 1. Include all available predictors
# 2. Stage 1: LASSO feature reduction
#    - One-hot encode categorical variables
#    - Use L1 regularization to prioritize informative features
#    - Keep 3-10 top features automatically
#
# 3. Stage 2: Random Forest on selected features
#    - Lower dimensionality, lower overfitting risk
#    - Better generalization
#
# 4. Diagnostics:
#    - LASSO coefficient plot
#    - Random Forest importance plot
#    - Distribution comparison before/after imputation
#
# =====================================================================
print("\n" + "="*70)
print("[STAGE 2] Bull_eye Imputation - LASSO + Random Forest")
print("="*70)

# Define candidate predictors for Bull_eye imputation
# - Continuous: Age, SS, Upper/Lower VB posterior heights, Months_of_Review, Initial_volume, RSI, DHI
#   Included variables: Months_of_Review, Initial_volume, RSI, DHI
# - Ordinal: Pfirrmann, Komori, MSU
# - Nominal: Gender, Herniated_Level, Iwabuchi, Modic, Spinal_canal_stenosis
# - Note: Absorption_type is the target and is excluded
bull_eye_predictors = continuous_vars + ordinal_vars + [
    "Gender", "Herniated_Level", "Iwabuchi", "Modic", "Spinal_canal_stenosis"
]
# Filter out predictors not present in current dataset
bull_eye_predictors = [c for c in bull_eye_predictors if c in X_raw.columns]

print(f"[INFO] Total candidate predictors for Bull_eye: {len(bull_eye_predictors)}")
print(f"[INFO] Predictors: {bull_eye_predictors}")

# Configure diagnostic output directory
bulleye_diagnostic_dir = os.path.join(ML_AUDIT_DIR, "bulleye_imputation_diagnostics")
print(f"[INFO] Diagnostic output directory: {bulleye_diagnostic_dir}")

# Create and train the two-stage imputer
bull_eye_imputer = BullEyeImputer(
    predictors=bull_eye_predictors,
    seed=SEED,
    n_estimators=100,       # Number of trees for Random Forest
    lasso_C=0.5,            # LASSO inverse regularization strength (smaller = stronger)
    min_features=3,         # Keep at least 3 selected features
    max_features=10,        # Keep at most 10 selected features
    output_dir=bulleye_diagnostic_dir
)

# Fit on training split, then transform all splits
print(f"\n[INFO] Fitting Bull_eye imputer on training set...")
X_train_raw = bull_eye_imputer.fit_transform(X_train_raw)
X_val_raw = bull_eye_imputer.transform(X_val_raw, dataset_name="Validation")
X_test_raw = bull_eye_imputer.transform(X_test_raw, dataset_name="Test")

# Generate diagnostic plots
print(f"\n[INFO] Generating diagnostic visualizations...")
bull_eye_imputer.visualize_diagnostics()

# 8) Convert imputed Bull_eye to string for downstream categorical encoding
print("\n[INFO] Converting Bull_eye to string format for categorical encoding...")
for df in [X_train_raw, X_val_raw, X_test_raw]:
    if "Bull_eye" in df.columns:
        _be = pd.to_numeric(df["Bull_eye"], errors="coerce").round()
        df["Bull_eye"] = _be.astype("Int64").astype(str).replace("<NA>", np.nan)

print("\n" + "="*70)
print("[STAGE 3] Ready for Main Prediction Model Training")
print("="*70)
print("[INFO] All datasets now have complete Bull_eye values (either observed or imputed)")
print("[INFO] Checking whether any other predictors still contain missing values...")

def _report_remaining_missing(df, split_name):
    remaining_missing = df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if remaining_missing.empty:
        print(f"[INFO] {split_name}: no remaining missing values after Bull_eye imputation.")
    else:
        print(f"[WARN] {split_name}: remaining missing values after Bull_eye imputation:")
        print(remaining_missing)

for _split_name, _df in [("Train", X_train_raw), ("Validation", X_val_raw), ("Test", X_test_raw)]:
    _report_remaining_missing(_df, _split_name)

print("[NOTE] Models that do not support NaN natively will fail unless the remaining missing values are imputed.")
print("[INFO] Proceeding to main model training for LDH reabsorption prediction...\n")

# 8) ColumnTransformer (old-model style): OHE nominal + OrdinalEncoder ordinal + StandardScaler continuous; remainder passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("nominal", OneHotEncoder(handle_unknown="ignore"), [c for c in nominal_vars if c in X_raw.columns]),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [c for c in ordinal_vars if c in X_raw.columns]),
        ("num", StandardScaler(), [c for c in continuous_vars if c in X_raw.columns]),
    ],
    remainder="passthrough"
)

# 9) Fit on TRAIN only, then transform VAL/TEST (no leakage)
X_train_scaled = preprocessor.fit_transform(X_train_raw)
X_val_scaled   = preprocessor.transform(X_val_raw)
X_test_scaled  = preprocessor.transform(X_test_raw)

# 10) Convert y to ndarray for downstream compatibility
y_train = y_train.values
y_val   = y_val.values
y_test  = y_test.values

print(f"\n[INFO] Final matrices: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")
print(f"[CHECK] y counts: train={dict(pd.Series(y_train).value_counts())}, "
      f"val={dict(pd.Series(y_val).value_counts())}, test={dict(pd.Series(y_test).value_counts())}")

# ---- feature names after preprocessing (for SHAP & reports) ----
def get_feature_names_from_preprocessor(prep, X_ref_cols):
    names = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                names.extend(list(trans.get_feature_names_out(cols)))
            except Exception:
                names.extend(list(trans.get_feature_names_out()))
        else:
            if isinstance(cols, (list, tuple, np.ndarray)):
                names.extend([str(c) for c in cols])
            else:
                names.append(str(cols))
    used = set()
    for _, _, cols in prep.transformers_:
        if isinstance(cols, (list, tuple, np.ndarray)):
            used.update(cols)
    passthrough_cols = [c for c in X_ref_cols if c not in used]
    names.extend([str(c) for c in passthrough_cols])
    return names

feature_names_ohe = get_feature_names_from_preprocessor(preprocessor, X_raw.columns)
if len(feature_names_ohe) != X_train_scaled.shape[1]:
    raise RuntimeError(
        f"Preprocessed feature name length mismatch: "
        f"{len(feature_names_ohe)} names vs {X_train_scaled.shape[1]} features."
    )

def _make_scaled_df(X, cols):
    return (
        pd.DataFrame.sparse.from_spmatrix(X, columns=cols)
        if hasattr(X, "tocoo") else pd.DataFrame(X, columns=cols)
    )

X_train_scaled_df = _make_scaled_df(X_train_scaled, feature_names_ohe)
X_val_scaled_df = _make_scaled_df(X_val_scaled, feature_names_ohe)
X_test_scaled_df = _make_scaled_df(X_test_scaled, feature_names_ohe)

# ==================== 4.2 Split Distribution Difference Check (Train vs Val vs Test) ====================
# Purpose:
#   - Check whether TRAIN differs from VAL/TEST in feature distribution after preprocessing
#   - Provide diagnostic report (do NOT abort by default; small samples + multiple tests => false positives)
print("\n[SECTION] Part 2: Checking distribution differences (Train vs Val/Test)")
print("-" * 70)

from scipy.stats import ttest_ind

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

Xtr = _to_dense(X_train_scaled)
Xva = _to_dense(X_val_scaled)
Xte = _to_dense(X_test_scaled)

# feature names: from preprocessing (validated above)
feature_names = feature_names_ohe

rows = []
for j, fname in enumerate(feature_names):
    # Welch's t-test on transformed feature values
    p_tr_te = ttest_ind(Xtr[:, j], Xte[:, j], equal_var=False).pvalue
    p_tr_va = ttest_ind(Xtr[:, j], Xva[:, j], equal_var=False).pvalue

    rows.append({
        "Feature": fname,
        "p_train_vs_test": float(p_tr_te),
        "p_train_vs_val": float(p_tr_va),
    })

diff_df = pd.DataFrame(rows)
diff_df["p_train_vs_test"] = diff_df["p_train_vs_test"].round(6)
diff_df["p_train_vs_val"]  = diff_df["p_train_vs_val"].round(6)

print(diff_df.sort_values("p_train_vs_test").head(20))

# Save
ml_dir = ML_DIR
os.makedirs(ml_dir, exist_ok=True)

diff_df.to_csv(os.path.join(ML_AUDIT_DIR, "feature_distribution_test_result_train_vs_val_test.csv"), index=False)

print(f"\n[INFO] Distribution check saved to: {ML_AUDIT_DIR}")
print("[NOTE] Small samples + many features => expect some small p-values by chance; treat as diagnostic, not hard stop.")

# ==================== 4.3 Model Training, Evaluation, and Comparison (Train/Test + 5-fold CV) ====================
# Purpose:
#   - Train multiple models with BayesSearchCV and cross-validation
#   - Evaluate on train/test metrics and rank by Test AUC
#   - Persist trained models and summary outputs for review
# ==================== Part 4: Model Training, Evaluation, and Comparison ====================
print("\n[SECTION] Part 4: Model Training, Evaluation, and Comparison")
print("-" * 70)

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.base import clone
from joblib import dump

# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, y_proba):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_score = np.asarray(y_proba).ravel()

    # Keep binary confusion matrix shape even if one class is absent.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) < 2:
        auprc = np.nan
    else:
        auprc = average_precision_score(y_true, y_score)

    return {
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "Youden": (tp / (tp + fn) + tn / (tn + fp) - 1) if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'AUC': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0,
        'AUPRC': auprc,
    }

# ---------------------------------------------------------------------
# Models and search spaces
# ---------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=SEED),
        'params': {
            'C': Real(0.01, 100, prior='log-uniform'),
            'max_iter': Integer(100, 1000)
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=SEED, n_jobs=1),
        'params': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=SEED),
        'params': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform')
        }
    },
    'XGBoost': {
        'model': XGBClassifier(
            random_state=SEED,
            eval_metric='logloss',
            n_jobs=1,
            tree_method='hist'
        ),
        'params': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.6, 1.0)
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(
            random_state=SEED,
            verbosity=-1,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True
        ),
        'params': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'num_leaves': Integer(10, 100)
        }
    },
    'SVM': {
        'model': SVC(random_state=SEED, probability=True),
        'params': {
            'C': Real(0.1, 100, prior='log-uniform'),
            'gamma': Real(0.001, 1, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear'])
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': Integer(3, 20),
            'weights': Categorical(['uniform', 'distance']),
            'metric': Categorical(['euclidean', 'manhattan'])
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=SEED),
        'params': {
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        }
    },
    'Extra Trees': {
        'model': ExtraTreesClassifier(random_state=SEED, n_jobs=1),
        'params': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20)
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=SEED),
        'params': {
            'n_estimators': Integer(50, 200),
            'learning_rate': Real(0.01, 2, prior='log-uniform')
        }
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': Real(1e-10, 1e-6, prior='log-uniform')
        }
    },
    'LDA': {
        'model': LinearDiscriminantAnalysis(),
        'params': {
            'solver': Categorical(['lsqr']),
            'shrinkage': Real(0.0, 1.0)
        }
    },
    'QDA': {
        'model': QuadraticDiscriminantAnalysis(),
        'params': {
            'reg_param': Real(0.0, 1.0)
        }
    }
}

# ---------------------------------------------------------------------
# Training on TRAIN, evaluation on VAL
# ---------------------------------------------------------------------
results_val = {}
best_models = {}

print("Starting Bayesian optimization on training set...")

cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

for name, config in models_config.items():
    print(f"\nTraining {name}...")
    try:
        search = BayesSearchCV(
            estimator=config['model'],
            search_spaces=config['params'],
            n_iter=30,
            cv=cv_strat,
            scoring='roc_auc',
            random_state=SEED,
            n_jobs=1,
            refit=True
        )

        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_

        y_val_pred = best_model.predict(X_val_scaled)
        if hasattr(best_model, "predict_proba"):
            y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
        else:
            scores = best_model.decision_function(X_val_scaled)
            y_val_proba = 1.0 / (1.0 + np.exp(-np.asarray(scores).ravel()))

        metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
        results_val[name] = metrics
        best_models[name] = best_model

        print(f"{name} finished | VAL AUC = {metrics['AUC']:.4f}")

    except Exception as e:
        print(f"{name} failed: {e}")
        continue

# ---------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results_val).T.round(4).sort_values('AUC', ascending=False)
print("\nValidation performance:")
print(results_df)

results_df.to_csv(os.path.join(ML_METRICS_DIR, "model_performance_results_VAL.csv"))

# ---------------------------------------------------------------------
# Champion selection: Multi-stage filtering + Weighted comprehensive score
# ---------------------------------------------------------------------

# ========== Stage 1: AUC threshold filtering ==========
# Set minimum AUC threshold to remove weak models
AUC_THRESHOLD = 0.70

print(f"\n{'='*60}")
print(f"STAGE 1: AUC Threshold Filtering (threshold >= {AUC_THRESHOLD})")
print(f"{'='*60}")

qualified_models = results_df[results_df['AUC'] >= AUC_THRESHOLD].copy()
filtered_out = results_df[results_df['AUC'] < AUC_THRESHOLD]

print(f"Total models: {len(results_df)}")
print(f"Qualified models (AUC >= {AUC_THRESHOLD}): {len(qualified_models)}")
print(f"Filtered out models: {len(filtered_out)}")

if len(filtered_out) > 0:
    print(f"\nFiltered out models (AUC < {AUC_THRESHOLD}):")
    print(filtered_out[['AUC', 'Sensitivity', 'Specificity', 'F1']])

if len(qualified_models) == 0:
    print(f"\n[WARN] No model meets AUC threshold {AUC_THRESHOLD}. Using top AUC model instead.")
    qualified_models = results_df.head(1).copy()

# ========== Stage 2: Weighted comprehensive scoring ==========
# Recommended weight setting for clinical screening context
WEIGHTS = {
    'Sensitivity': 0.25,   # Prioritize recall in screening setting
    'AUC': 0.25,           # Overall discrimination
    'Specificity': 0.15,   # Control false positives
    'F1': 0.15,            # Balance precision and recall
    'PPV': 0.10,
    'AUPRC': 0.10
}

weight_sum = float(sum(WEIGHTS.values()))
if not np.isclose(weight_sum, 1.0):
    raise ValueError(f"WEIGHTS must sum to 1.0, got {weight_sum:.6f}")

missing_metrics = [metric for metric in WEIGHTS if metric not in qualified_models.columns]
if missing_metrics:
    raise KeyError(
        f"Missing required metric columns for comprehensive scoring: {missing_metrics}"
    )

if qualified_models["AUPRC"].isna().any():
    bad_models = qualified_models.index[qualified_models["AUPRC"].isna()].tolist()
    raise ValueError(
        "AUPRC contains NaN for some models. This may happen when validation labels have a single class. "
        f"Affected models: {bad_models}"
    )

print(f"\n{'='*60}")
print("STAGE 2: Weighted Comprehensive Score Calculation")
print(f"{'='*60}")
print("Weight configuration:")
for metric, weight in WEIGHTS.items():
    print(f"  - {metric}: {weight:.2f}")

# Compute comprehensive score for each qualified model
for model_name in qualified_models.index:
    comprehensive_score = 0.0
    for metric, weight in WEIGHTS.items():
        comprehensive_score += qualified_models.loc[model_name, metric] * weight
    qualified_models.loc[model_name, 'Comprehensive_Score'] = comprehensive_score

# Rank models by comprehensive score
qualified_models = qualified_models.sort_values('Comprehensive_Score', ascending=False)

print(f"\nQualified models ranked by Comprehensive Score:")
ranking_cols = ['AUC', 'AUPRC', 'Sensitivity', 'Specificity', 'F1', 'PPV', 'NPV', 'Comprehensive_Score']
print(qualified_models[ranking_cols].round(4))

# ========== Final Selection ==========
best_model_name = qualified_models.index[0]
best_model = best_models[best_model_name]
best_comprehensive_score = qualified_models.loc[best_model_name, 'Comprehensive_Score']

print(f"\n{'='*60}")
print(f"FINAL SELECTED MODEL: {best_model_name}")
print(f"{'='*60}")
print(f"Comprehensive Score: {best_comprehensive_score:.4f}")
print(f"  - AUC: {qualified_models.loc[best_model_name, 'AUC']:.4f}")
print(f"  - AUPRC: {qualified_models.loc[best_model_name, 'AUPRC']:.4f}")
print(f"  - Sensitivity: {qualified_models.loc[best_model_name, 'Sensitivity']:.4f}")
print(f"  - Specificity: {qualified_models.loc[best_model_name, 'Specificity']:.4f}")
print(f"  - F1: {qualified_models.loc[best_model_name, 'F1']:.4f}")
print(f"  - PPV: {qualified_models.loc[best_model_name, 'PPV']:.4f}")
print(f"  - NPV: {qualified_models.loc[best_model_name, 'NPV']:.4f}")

# Save comprehensive ranking results
qualified_models_export = qualified_models[ranking_cols].copy()
qualified_models_export.to_csv(os.path.join(ML_METRICS_DIR, "model_comprehensive_score_ranking.csv"))
print(f"\n[OK] Comprehensive score ranking saved to: model_comprehensive_score_ranking.csv")

# ---------------------------------------------------------------------
# Final model (selected by validation AUC; trained on TRAIN only)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Final evaluation on TEST
# ---------------------------------------------------------------------
y_test_pred = best_model.predict(X_test_scaled)
if hasattr(best_model, "predict_proba"):
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    scores = best_model.decision_function(X_test_scaled)
    y_test_proba = 1.0 / (1.0 + np.exp(-np.asarray(scores).ravel()))

test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

print("\nFinal test performance:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# Save final model and test metrics
model_filename = f"best_model_{best_model_name.lower().replace(' ', '_').replace('/', '_')}.joblib"
dump(best_model, os.path.join(ML_MODELS_DIR, model_filename))

pd.DataFrame([{"Model": best_model_name, **{k: round(v, 6) for k, v in test_metrics.items()}}]) \
    .to_csv(os.path.join(ML_METRICS_DIR, "best_model_TEST_metrics.csv"), index=False)

print(f"\nModel saved to: {os.path.join(ML_MODELS_DIR, model_filename)}")

# ==================== 4.4 Visualization ====================
print("\n[INFO] Generating visualization figures...")


from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    auc as sklearn_auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# Expect from 4.3:
# - results_df: validation results table sorted by AUC (descending)
# - best_models: dict of best estimators per model (trained on TRAIN via BayesSearchCV)
# - best_model_name: selected by validation AUC (results_df.index[0])
# - best_model: validation-selected champion trained on TRAIN
# - X_val_scaled, y_val, X_test_scaled, y_test
# - ml_dir

def _get_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s).ravel()
        return 1.0 / (1.0 + np.exp(-s))
    pred = model.predict(X)
    return np.asarray(pred).ravel().astype(float)


def _safe_pr(y_true, y_proba):
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    if y_true.size == 0:
        return np.array([1.0]), np.array([0.0]), np.nan

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length for PR calculation.")

    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_proba = np.clip(y_proba, 0.0, 1.0)
    uniq = np.unique(y_true)

    if uniq.size < 2:
        prevalence = float(np.mean(y_true == 1))
        precision = np.array([prevalence, prevalence], dtype=float)
        recall = np.array([0.0, 1.0], dtype=float)
        ap = prevalence
        return precision, recall, ap

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = float(average_precision_score(y_true, y_proba))
    return precision, recall, ap


def _compute_dca(y_true, y_proba, thresholds):
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length for DCA.")
    if y_true.size == 0:
        return pd.DataFrame(
            columns=["threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none"]
        )

    y_true = y_true.astype(int)
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_proba = np.clip(y_proba, 0.0, 1.0)

    n = float(y_true.size)
    prevalence = float(np.mean(y_true == 1))
    rows = []

    for pt in np.asarray(thresholds).ravel():
        pt = float(pt)
        if pt <= 0.0 or pt >= 1.0:
            continue
        y_pred = (y_proba >= pt).astype(int)
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        odds = pt / (1.0 - pt)
        nb_model = (tp / n) - (fp / n) * odds
        nb_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": pt,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": 0.0,
            }
        )

    return pd.DataFrame(rows)


def _compute_calibration_points(y_true, y_proba, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length for calibration.")

    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_proba = np.clip(y_proba, 0.0, 1.0)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame(
        {
            "mean_predicted_prob": prob_pred,
            "observed_fraction_positive": prob_true,
        }
    )

# -------------------- 1) Performance line plots (based on results_df) --------------------
fig = plt.figure(figsize=(15, 10), dpi=300)

metrics_to_plot = ['Sensitivity', 'Specificity', 'Accuracy', 'PPV', 'NPV', 'F1', "Youden"]

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 3, i)
    sorted_results = results_df.sort_values(metric, ascending=False)
    plt.plot(range(len(sorted_results)), sorted_results[metric], 'o-', linewidth=2, markersize=6)
    plt.title(f'{metric} Performance', fontweight='bold')
    plt.xlabel('Model Rank')
    plt.ylabel(metric)
    plt.xticks(range(len(sorted_results)), sorted_results.index, rotation=45)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
_save_fig(fig, os.path.join(ML_FIG_DIR, 'performance_line_plots.tiff'),
            format='tiff', dpi=300, bbox_inches='tight')

# -------------------- 2) ROC Curves (on Validation Set) --------------------
fig = plt.figure(figsize=(12, 10), dpi=300)

roc_data = {}
for name, model in best_models.items():
    y_proba = _get_proba(model, X_val_scaled)
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = sklearn_auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

for name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], label=f'{name} (AUC = {data["auc"]:.3f})')
    lower = np.maximum(0, data['tpr'] - 0.05)
    upper = np.minimum(1, data['tpr'] + 0.05)
    plt.fill_between(data['fpr'], lower, upper, alpha=0.1)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC Curves (Validation Set)', fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
_save_fig(fig, os.path.join(ML_FIG_DIR, 'roc_curves_VAL.tiff'),
            format='tiff', dpi=300, bbox_inches='tight')

# -------------------- 3) PR Curves (Validation Set; all successful models) --------------------
fig = plt.figure(figsize=(12, 10), dpi=300)

y_val_array = np.asarray(y_val).ravel().astype(int)
val_prevalence = float(np.mean(y_val_array == 1))
n_val_samples = int(y_val_array.shape[0])
pr_rows = []
model_names = list(best_models.keys())
palette = sns.color_palette("husl", n_colors=max(1, len(model_names)))
color_map = {name: palette[i] for i, name in enumerate(model_names)}

for name, model in best_models.items():
    y_proba = _get_proba(model, X_val_scaled)
    precision, recall, ap = _safe_pr(y_val_array, y_proba)
    order = np.argsort(recall)
    plt.plot(
        recall[order],
        precision[order],
        linewidth=plt.rcParams["lines.linewidth"],
        color=color_map[name],
        label=f"{name} (AP = {ap:.3f})",
    )
    pr_rows.append(
        {
            "Model": name,
            "AP": float(ap),
            "positive_prevalence": val_prevalence,
            "n_samples": n_val_samples,
        }
    )

plt.axhline(val_prevalence, color="gray", linestyle="--", linewidth=1.5,
            label=f"Random baseline (Prevalence = {val_prevalence:.3f})")
plt.title("PR Curves (Validation Set)", fontweight="bold")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

pr_fig_path = os.path.join(ML_FIG_DIR, "pr_curves_all_models_VAL.tiff")
_save_fig(fig, pr_fig_path, format='tiff', dpi=300, bbox_inches='tight')

pr_csv_path = os.path.join(ML_METRICS_DIR, "pr_ap_all_models_VAL.csv")
pd.DataFrame(pr_rows).sort_values("AP", ascending=False).to_csv(pr_csv_path, index=False)
print(f"[OK] PR curves saved to: {pr_fig_path}")
print(f"[OK] PR/AP table saved to: {pr_csv_path}")

# -------------------- 4) AUC Forest Plot (based on Validation AUC) --------------------
fig = plt.figure(figsize=(10, 8), dpi=300)

auc_values = results_df['AUC'].sort_values(ascending=True)
y_pos = np.arange(len(auc_values))

ci_lower = auc_values - 1.96 * (auc_values.std(ddof=1) / np.sqrt(len(auc_values)))
ci_upper = auc_values + 1.96 * (auc_values.std(ddof=1) / np.sqrt(len(auc_values)))

plt.barh(y_pos, auc_values, alpha=0.7)
plt.errorbar(auc_values, y_pos,
             xerr=[auc_values - ci_lower, ci_upper - auc_values],
             fmt='o', capsize=5)
plt.yticks(y_pos, auc_values.index)
plt.xlabel('AUC Value')
plt.title('AUC Forest Plot with Approximate Confidence Intervals (Validation Set)', fontweight='bold')
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

for i, v in enumerate(auc_values):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
_save_fig(fig, os.path.join(ML_FIG_DIR, 'auc_forest_plot_VAL.tiff'),
            format='tiff', dpi=300, bbox_inches='tight')

# -------------------- 5) Confusion Matrices (on Validation Set) --------------------
n_models = len(best_models)
n_cols = 5
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), dpi=300)
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (name, model) in enumerate(best_models.items()):
    if i >= len(axes):
        break
    y_pred = model.predict(X_val_scaled)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])
    axes[i].set_title(f'{name}', fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

for j in range(len(best_models), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
_save_fig(fig, os.path.join(ML_FIG_DIR, 'confusion_matrices_VAL.tiff'),
            format='tiff', dpi=300, bbox_inches='tight')

# -------------------- 5b) Best Model Confusion Matrix (Validation Set) --------------------
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
y_val_pred_best = best_model.predict(X_val_scaled)
best_cm = confusion_matrix(y_val, y_val_pred_best)
sns.heatmap(
    best_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    cbar=False,
    xticklabels=["0", "1"],
    yticklabels=["0", "1"],
    linewidths=1.0,
    linecolor="white",
)
ax.set_title(f"Best Model Confusion Matrix (Validation)\n{best_model_name}", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()

best_cm_fig_path = os.path.join(ML_FIG_DIR, "best_model_confusion_matrix_VAL.tiff")
_save_fig(fig, best_cm_fig_path, format='tiff', dpi=300, bbox_inches='tight')
print(f"[OK] Best-model confusion matrix saved to: {best_cm_fig_path}")

# -------------------- 6) Clinical Impact Curves (on Validation Set) --------------------
def plot_clinical_impact_curve(y_true, y_proba, ax=None):
    thresholds = np.linspace(0, 1, 100)
    impacts = []
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        impacts.append(tp - fp)
    if ax is None:
        fig_ci, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(thresholds, impacts, label='Clinical Impact')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Clinical Impact (TP - FP)')
    ax.set_title('Clinical Impact Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), dpi=300)
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (name, model) in enumerate(best_models.items()):
    if i >= len(axes):
        break
    y_proba = _get_proba(model, X_val_scaled)
    plot_clinical_impact_curve(y_val, y_proba, ax=axes[i])
    axes[i].set_title(f'{name}', fontweight='bold')

for j in range(len(best_models), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
_save_fig(fig, os.path.join(ML_FIG_DIR, 'clinical_impact_curves_VAL.tiff'),
            format='tiff', dpi=300, bbox_inches='tight')

# -------------------- 7) Calibration Curves (on Validation Set) --------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), dpi=300)
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (name, model) in enumerate(best_models.items()):
    if i >= len(axes):
        break
    y_proba = _get_proba(model, X_val_scaled)
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10, strategy='uniform')
    axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[i].plot(prob_pred, prob_true, 'o-', label=name)
    axes[i].set_xlabel('Mean Predicted Probability')
    axes[i].set_ylabel('Fraction of Positives')
    axes[i].set_title(f'{name} Calibration', fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

for j in range(len(best_models), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
_save_fig(fig, os.path.join(ML_FIG_DIR, 'calibration_curves_VAL.tiff'),
            format='tiff', dpi=300, bbox_inches='tight')

# -------------------- 8) Best Model Calibration Curve (Validation Set) --------------------
y_val_proba_best = _get_proba(best_model, X_val_scaled)
y_val_proba_best = np.clip(np.nan_to_num(np.asarray(y_val_proba_best).ravel(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

best_calib_df = _compute_calibration_points(y_val_array, y_val_proba_best, n_bins=10)
brier = float(brier_score_loss(y_val_array, y_val_proba_best))

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
ax.plot(
    best_calib_df["mean_predicted_prob"],
    best_calib_df["observed_fraction_positive"],
    "o-",
    linewidth=2,
    markersize=6,
    label=best_model_name,
)
ax.set_title("Best Model Calibration (Validation Set)", fontweight="bold")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Fraction Positive")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
ax.text(
    0.98,
    0.04,
    f"Brier score = {brier:.3f}",
    transform=ax.transAxes,
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
)
plt.tight_layout()

best_calib_fig_path = os.path.join(ML_FIG_DIR, "best_model_calibration_VAL.tiff")
_save_fig(fig, best_calib_fig_path, format='tiff', dpi=300, bbox_inches='tight')

best_calib_csv_path = os.path.join(ML_METRICS_DIR, "best_model_calibration_bins_VAL.csv")
best_calib_df.to_csv(best_calib_csv_path, index=False)
print(f"[OK] Best-model calibration figure saved to: {best_calib_fig_path}")
print(f"[OK] Best-model calibration bins saved to: {best_calib_csv_path}")

# -------------------- 9) Best Model DCA (Validation Set): Curve + Heatmap --------------------
dca_thresholds = np.linspace(0.01, 0.99, 99)
best_dca_df = _compute_dca(y_val_array, y_val_proba_best, dca_thresholds)

# 9a) DCA curve
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(
    best_dca_df["threshold"],
    best_dca_df["net_benefit_model"],
    linewidth=2.8,
    color="#1f77b4",
    label=f"{best_model_name} (Model)",
)
ax.plot(
    best_dca_df["threshold"],
    best_dca_df["net_benefit_all"],
    linestyle="--",
    linewidth=2.4,
    color="black",
    label="Treat All",
)
ax.plot(
    best_dca_df["threshold"],
    best_dca_df["net_benefit_none"],
    linestyle=":",
    linewidth=2.2,
    color="gray",
    label="Treat None",
)
ax.set_title("Decision Curve Analysis", fontweight="bold")
ax.set_xlabel("Threshold Probability")
ax.set_ylabel("Net Benefit")
ax.set_xlim(0.0, 0.8)

window = best_dca_df[best_dca_df["threshold"] <= 0.8]
if window.empty:
    window = best_dca_df
focus_nb = np.concatenate(
    [
        window["net_benefit_model"].to_numpy(dtype=float),
        window["net_benefit_none"].to_numpy(dtype=float),
    ]
)
if focus_nb.size > 0:
    focus_min = float(np.min(focus_nb))
    focus_max = float(np.max(focus_nb))
    if focus_min < 0:
        y_min = focus_min - max(0.01, abs(focus_min) * 0.12)
    else:
        y_min = -0.03
    if focus_max > 0:
        y_max = focus_max + max(0.02, abs(focus_max) * 0.12)
    else:
        y_max = 0.05
    if y_max <= y_min:
        y_max = y_min + 0.1
    ax.set_ylim(y_min, y_max)

ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()

best_dca_curve_fig_path = os.path.join(ML_FIG_DIR, "best_model_dca_curve_VAL.tiff")
_save_fig(fig, best_dca_curve_fig_path, format='tiff', dpi=300, bbox_inches='tight')

# 9b) DCA heatmap
def _nb_at_threshold(df, threshold, col):
    idx = (df["threshold"] - float(threshold)).abs().idxmin()
    return float(df.loc[idx, col])

key_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
heatmap_data = {
    t: [
        _nb_at_threshold(best_dca_df, t, "net_benefit_model"),
        _nb_at_threshold(best_dca_df, t, "net_benefit_all"),
        _nb_at_threshold(best_dca_df, t, "net_benefit_none"),
    ]
    for t in key_thresholds
}
best_dca_heatmap_df = pd.DataFrame(
    heatmap_data,
    index=[best_model_name, "Treat All", "Treat None"],
)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
sns.heatmap(
    best_dca_heatmap_df,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    center=0,
    linewidths=1.0,
    linecolor="black",
    cbar_kws={"label": "Net Benefit"},
    ax=ax,
)
ax.set_title("Best Model DCA Net Benefit Heatmap (Validation Set)", fontweight="bold")
ax.set_xlabel("Threshold Probability")
ax.set_ylabel("Strategy")
plt.tight_layout()

best_dca_heatmap_fig_path = os.path.join(ML_FIG_DIR, "best_model_dca_heatmap_VAL.tiff")
_save_fig(fig, best_dca_heatmap_fig_path, format='tiff', dpi=300, bbox_inches='tight')

best_dca_csv_path = os.path.join(ML_METRICS_DIR, "best_model_dca_VAL.csv")
best_dca_df.to_csv(best_dca_csv_path, index=False)
print(f"[OK] Best-model DCA curve saved to: {best_dca_curve_fig_path}")
print(f"[OK] Best-model DCA heatmap saved to: {best_dca_heatmap_fig_path}")
print(f"[OK] Best-model DCA table saved to: {best_dca_csv_path}")

print(f"\n[OK] All figures saved to: {ML_FIG_DIR}")
print(f"[OK] Added validation PR/AP metrics: {pr_csv_path}")
print(f"[OK] Added validation calibration bins: {best_calib_csv_path}")
print(f"[OK] Added best-model confusion matrix: {best_cm_fig_path}")
print(f"[OK] Added best-model DCA curve: {best_dca_curve_fig_path}")
print(f"[OK] Added best-model DCA heatmap: {best_dca_heatmap_fig_path}")
print(f"[OK] Added validation DCA table: {best_dca_csv_path}")

# ==================== 4.5 Final Test-Set Evaluation ====================
print("\n[SECTION] Final Test-Set Evaluation:")
print("-" * 50)

print(f"Selected model: {best_model_name} (Validation AUC: {results_df.loc[best_model_name, 'AUC']:.4f})")

# Use the validation-selected champion (trained on TRAIN) for TEST evaluation
y_test_pred = best_model.predict(X_test_scaled)
y_test_proba = _get_proba(best_model, X_test_scaled)

test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

print("\nTest performance:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save metrics (model is already saved in 4.3, but keep metrics export here)
pd.DataFrame([{"Model": best_model_name, **{k: round(v, 6) for k, v in test_metrics.items()}}]) \
  .to_csv(os.path.join(ML_METRICS_DIR, "best_model_TEST_metrics.csv"), index=False)

print("\n" + "=" * 60)
print("[INFO] Model training and evaluation completed!")
print("=" * 60)

# ==================== 4.6 Final Model Extraction for SHAP ====================
# Purpose:
#   - Freeze the FINAL model trained in Part 4
#   - Freeze the TEST-set representation used for evaluation
#   - Provide a single, clean source of truth for SHAP analysis
#   - Avoid any dependency on Part 6 (deployment)

print("\n[SECTION] 4.6 Final Model Extraction for SHAP")
print("-" * 70)

# ------------------------------------------------------------------
# 1. Sanity checks: ensure Part 4 executed correctly
# ------------------------------------------------------------------
required_vars = [
    "best_model",
    "best_model_name",
    "best_models",
    "X_test_scaled",
    "y_test"
]

missing = [v for v in required_vars if v not in globals()]
if missing:
    raise RuntimeError(
        f"Missing required variables from Part 4: {missing}. "
        "Ensure Part 4.3-4.5 executed successfully before SHAP."
    )

# ------------------------------------------------------------------
# 2. Freeze model (DO NOT refit, DO NOT clone)
# ------------------------------------------------------------------
shap_model_name = best_model_name
shap_model = best_models[shap_model_name]
model_tag = shap_model_name.replace(" ", "_").lower()

print(f"[INFO] SHAP model fixed as: {shap_model_name}")

# ------------------------------------------------------------------
# 3. Freeze test-set data used for explanation
# ------------------------------------------------------------------
if "X_test_scaled_df" in globals():
    X_test_sample = X_test_scaled_df
else:
    X_test_sample = X_test_scaled
y_test_sample = y_test

print(f"[INFO] SHAP test samples: {X_test_sample.shape[0]}")
print(f"[INFO] SHAP feature dimension: {X_test_sample.shape[1]}")

# ------------------------------------------------------------------
# 4. Freeze feature names (aligned with preprocessing)
# ------------------------------------------------------------------
if "X_train_df" in globals():
    feature_names = list(map(str, X_train_df.columns))
elif "X_test_df" in globals():
    feature_names = list(map(str, X_test_df.columns))
elif "feature_names_ohe" in globals():
    feature_names = list(map(str, feature_names_ohe))
elif "preprocessor" in globals() and "X_raw" in globals() and "get_feature_names_from_preprocessor" in globals():
    feature_names = list(map(str, get_feature_names_from_preprocessor(preprocessor, X_raw.columns)))
else:
    raise RuntimeError(
        "SHAP feature names not found. "
        "Expected X_train_df or X_test_df with column names."
    )

if len(feature_names) != X_test_sample.shape[1]:
    msg = (
        f"Feature name length mismatch: "
        f"{len(feature_names)} names vs {X_test_sample.shape[1]} features."
    )
    print(f"[ERROR] {msg}")
    raise RuntimeError(msg)

# ------------------------------------------------------------------
# 5. Freeze probability helper (for SHAP / plots)
# ------------------------------------------------------------------
try:
    from scipy.special import expit as _expit
except Exception:
    _expit = None

def _prob_from_model(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    if hasattr(m, "decision_function"):
        print("[WARN] Using sigmoid(decision_function) as a pseudo-probability (not calibrated).")
        logits = m.decision_function(X)
        return _expit(logits) if _expit is not None else _sigmoid(logits)
    print("[WARN] Model has no probability/decision_function; using predict() as fallback.")
    return m.predict(X)

print("[INFO] Final model, data, and helpers frozen for SHAP.")

print("\n" + "=" * 70)
print("[INFO] You may now safely proceed to Part 5: SHAP Visualization.")
print("=" * 70)

# ==================== 4.7 Deployment Pipeline Export (Train-only; Val thresholds) ====================
# Purpose:
#   - Build a deployable preprocessing + model pipeline (fit on TRAIN only)
#   - Compute thresholds on VAL only (no test leakage)
#   - Export pipeline + metadata + thresholds for local deployment
print("\n[SECTION] 4.7 Deployment Pipeline Export (Train-only; Val thresholds)")
print("-" * 70)

required_deploy_vars = [
    "X_train_raw",
    "y_train",
    "X_val_raw",
    "y_val",
    "best_model_name",
    "best_model",
    "ml_dir"
]
missing = [v for v in required_deploy_vars if v not in globals()]
if missing:
    raise RuntimeError(
        f"Missing required variables for deployment export: {missing}. "
        "Ensure Parts 4.1-4.5 executed successfully."
    )

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, f1_score

deploy_model_tag = best_model_name.replace(" ", "_").lower()
deploy_dir = DEPLOY_DIR
os.makedirs(deploy_dir, exist_ok=True)

# Preprocessing consistent with training
deploy_preprocess = ColumnTransformer(
    transformers=[
        ("nominal", OneHotEncoder(handle_unknown="ignore"), [c for c in nominal_vars if c in X_train_raw.columns]),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [c for c in ordinal_vars if c in X_train_raw.columns]),
        ("num", StandardScaler(), [c for c in continuous_vars if c in X_train_raw.columns]),
    ],
    remainder="passthrough"
)

deploy_model = clone(best_model)
deploy_pipeline = Pipeline(steps=[
    ("preprocess", deploy_preprocess),
    ("model", deploy_model),
])
deploy_pipeline.fit(X_train_raw, y_train)

pipeline_path = os.path.join(deploy_dir, f"best_model_pipeline_{deploy_model_tag}.pkl")
joblib.dump(deploy_pipeline, pipeline_path)
print(f"[OK] Deployable pipeline saved to: {pipeline_path}")

# Validate probability interface and compute thresholds on VAL
y_val_proba, prob_type = _prob_from_estimator(deploy_pipeline, X_val_raw)
y_val_true = y_val

fpr, tpr, thr = roc_curve(y_val_true, y_val_proba)
if len(thr) > 1:
    J = tpr - fpr
    idx = int(np.argmax(J[1:]) + 1)
    thr_youden = float(thr[idx])
else:
    thr_youden = float(thr[0])

SENS_TARGET = 0.90
cand = np.where(tpr >= SENS_TARGET)[0]
thr_sens90 = float(thr[cand][np.argmin(fpr[cand])]) if len(cand) else thr_youden

SPEC_TARGET = 0.90
spec = 1 - fpr
cand_spec = np.where(spec >= SPEC_TARGET)[0]
thr_spec90 = float(thr[cand_spec][np.argmax(tpr[cand_spec])]) if len(cand_spec) else thr_youden

prec, rec, thr_pr = precision_recall_curve(y_val_true, y_val_proba)
if len(thr_pr):
    f1s = []
    for t in thr_pr:
        y_pred = (y_val_proba >= t).astype(int)
        f1s.append(f1_score(y_val_true, y_pred))
    thr_f1 = float(thr_pr[int(np.argmax(f1s))])
else:
    thr_f1 = thr_youden

def metrics_at(t):
    y_pred = (y_val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = f1_score(y_val_true, y_pred) if (tp + fp) > 0 and (tp + fn) > 0 else 0
    youden = sens + spec - 1
    return dict(
        threshold=round(float(t), 4),
        sensitivity=round(sens, 3),
        specificity=round(spec, 3),
        PPV=round(ppv, 3),
        NPV=round(npv, 3),
        accuracy=round(acc, 3),
        F1=round(f1, 3),
        Youden=round(youden, 3),
    )

summary = pd.DataFrame({
    "Youden_opt": metrics_at(thr_youden),
    "Sens>=0.90": metrics_at(thr_sens90),
    "Spec>=0.90": metrics_at(thr_spec90),
    "MaxF1": metrics_at(thr_f1),
}).T

summary_path = os.path.join(deploy_dir, f"{deploy_model_tag}_thresholds_VAL.csv")
summary.to_csv(summary_path, index=True)

chosen_threshold = thr_youden

# Dual-threshold defaults (low = high sensitivity, high = high specificity)
thr_low = thr_sens90
thr_high = thr_spec90
if thr_low > thr_high:
    thr_low, thr_high = min(thr_low, thr_high), max(thr_low, thr_high)

thr_out_path = os.path.join(deploy_dir, f"{deploy_model_tag}_thresholds_VAL.json")
with open(thr_out_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_name": best_model_name,
        "model_tag": deploy_model_tag,
        "threshold_source": "validation",
        "threshold_Youden": thr_youden,
        "threshold_Sens90": thr_sens90,
        "threshold_Spec90": thr_spec90,
        "threshold_MaxF1": thr_f1,
        "threshold_Chosen": float(chosen_threshold),
        "threshold_low": float(thr_low),
        "threshold_high": float(thr_high),
        "SENS_TARGET": float(SENS_TARGET),
        "SPEC_TARGET": float(SPEC_TARGET),
        "metrics_low": metrics_at(thr_low),
        "metrics_high": metrics_at(thr_high),
    }, f, ensure_ascii=False, indent=2)

meta_path = os.path.join(deploy_dir, f"{deploy_model_tag}_pipeline_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_name": best_model_name,
        "model_tag": deploy_model_tag,
        "has_predict_proba": hasattr(best_model, "predict_proba"),
        "has_decision_function": hasattr(best_model, "decision_function"),
        "probability_type": prob_type,
        "threshold_source": "validation",
        "pipeline_path": pipeline_path,
    }, f, ensure_ascii=False, indent=2)

reloaded = joblib.load(pipeline_path)
print(f"[OK] Saved and reloadable: {pipeline_path}")
print(f"[OK] Thresholds saved to: {thr_out_path}")

# ==================== 4.8 Prospective Prediction (same threshold as deployment) ====================
# Purpose:
#   - Reuse exported deployment pipeline to predict prospective cohort
#   - Apply exactly the same chosen threshold from Section 4.7
#   - Export an Excel prediction report and a confusion matrix figure
print("\n[SECTION] 4.8 Prospective Prediction")
print("-" * 70)

PROSPECTIVE_PATH  = _resolve_data_file("Prospective data.xlsx")
PROSPECTIVE_SHEET = "Train_Pors"
PROSPECTIVE_LABEL = "Reabsorption"
PROSPECTIVE_OUT_DIR = os.path.join(RUN_ROOT, "07_Prospective_Prediction")

if not os.path.isfile(PROSPECTIVE_PATH):
    print(f"[WARN] Prospective file not found. Skip prospective prediction: {PROSPECTIVE_PATH}")
else:
    os.makedirs(PROSPECTIVE_OUT_DIR, exist_ok=True)
    # Extract feature columns from deploy preprocessor to avoid hard-coded drift.
    pros_nominal_cols = []
    pros_ordinal_cols = []
    pros_num_cols = []
    for name, _, cols in deploy_preprocess.transformers:
        col_list = list(cols)
        if name == "nominal":
            pros_nominal_cols.extend(col_list)
        elif name == "ordinal":
            pros_ordinal_cols.extend(col_list)
        elif name == "num":
            pros_num_cols.extend(col_list)
    pros_feature_cols = pros_nominal_cols + pros_ordinal_cols + pros_num_cols

    expected_cols = set(pros_feature_cols + [PROSPECTIVE_LABEL])
    xls = pd.ExcelFile(PROSPECTIVE_PATH)
    available_sheets = list(xls.sheet_names)
    sheet_to_use = PROSPECTIVE_SHEET

    if sheet_to_use not in available_sheets:
        print(
            f"[WARN] Sheet '{PROSPECTIVE_SHEET}' not found. "
            f"Available sheets: {available_sheets}"
        )
        # Fallback: choose the sheet with the best overlap with expected columns.
        best_sheet = None
        best_score = -1
        for s in available_sheets:
            cols = set(map(str, pd.read_excel(PROSPECTIVE_PATH, sheet_name=s, nrows=0).columns))
            if PROSPECTIVE_LABEL not in cols:
                continue
            score = len(cols & expected_cols)
            if score > best_score:
                best_score = score
                best_sheet = s
        if best_sheet is None:
            best_sheet = available_sheets[0]
        sheet_to_use = best_sheet
        print(f"[INFO] Fallback selected sheet: {sheet_to_use}")

    print(f"[INFO] Loading prospective data: {PROSPECTIVE_PATH} | sheet={sheet_to_use}")
    df_pros = pd.read_excel(PROSPECTIVE_PATH, sheet_name=sheet_to_use)

    if PROSPECTIVE_LABEL not in df_pros.columns:
        raise KeyError(
            f"Prospective label column not found: {PROSPECTIVE_LABEL}. "
            f"Current sheet: {sheet_to_use}"
        )

    missing_feature_cols = [c for c in pros_feature_cols if c not in df_pros.columns]
    if missing_feature_cols:
        raise KeyError(f"Prospective data missing required feature columns: {missing_feature_cols}")

    X_pros = df_pros[pros_feature_cols].copy()

    # Match training-time preprocessing cleanup rules.
    for c in pros_num_cols + pros_ordinal_cols:
        if c in X_pros.columns:
            X_pros[c] = pd.to_numeric(X_pros[c], errors="coerce")

    for c in ["Iwabuchi", "Modic", "Spinal_canal_stenosis"]:
        if c in X_pros.columns:
            _tmp = pd.to_numeric(X_pros[c], errors="coerce")
            X_pros[c] = _tmp.apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)

    if "Bull_eye" in X_pros.columns:
        _be = pd.to_numeric(X_pros["Bull_eye"], errors="coerce").round()
        X_pros["Bull_eye"] = _be.astype("Int64").astype(str).replace("<NA>", np.nan)

    if "Gender" in X_pros.columns:
        X_pros["Gender"] = X_pros["Gender"].astype(str).str.strip()
        X_pros["Gender"] = X_pros["Gender"].replace({
            "female": "Female", "FEMALE": "Female", "F": "Female",
            "male": "Male", "MALE": "Male", "M": "Male"
        })
        X_pros["Gender"] = X_pros["Gender"].replace({"nan": np.nan, "NaN": np.nan, "": np.nan})

    y_proba_pros, pros_prob_type = _prob_from_estimator(deploy_pipeline, X_pros)
    y_proba_pros = np.asarray(y_proba_pros, dtype=float)
    y_pred_pros = (y_proba_pros >= float(chosen_threshold)).astype(int)

    y_true_raw = pd.to_numeric(df_pros[PROSPECTIVE_LABEL], errors="coerce")
    valid_label_mask = y_true_raw.isin([0, 1])
    n_invalid = int((~valid_label_mask).sum())
    if n_invalid > 0:
        print(f"[WARN] Prospective labels with non-binary/NA values will be excluded from metrics: {n_invalid}")
    if int(valid_label_mask.sum()) == 0:
        raise ValueError("No valid binary labels (0/1) found in prospective data for confusion matrix.")

    y_true_eval = y_true_raw[valid_label_mask].astype(int).values
    y_pred_eval = y_pred_pros[valid_label_mask.values]
    y_proba_eval = y_proba_pros[valid_label_mask.values]

    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = f1_score(y_true_eval, y_pred_eval) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0
    youden = sens + spec - 1

    if len(np.unique(y_true_eval)) < 2:
        auc = np.nan
        auprc = np.nan
        print("[WARN] Prospective labels contain a single class; AUC/AUPRC set to NA.")
    else:
        auc = roc_auc_score(y_true_eval, y_proba_eval)
        auprc = average_precision_score(y_true_eval, y_proba_eval)

    summary_rows = [
        ("run_root", RUN_ROOT),
        ("model_name", best_model_name),
        ("pipeline_path", pipeline_path),
        ("threshold_json", thr_out_path),
        ("probability_type", pros_prob_type),
        ("threshold_used", float(chosen_threshold)),
        ("n_total_rows", int(len(df_pros))),
        ("n_eval_rows", int(len(y_true_eval))),
        ("TN", int(tn)),
        ("FP", int(fp)),
        ("FN", int(fn)),
        ("TP", int(tp)),
        ("Sensitivity", float(sens)),
        ("Specificity", float(spec)),
        ("PPV", float(ppv)),
        ("NPV", float(npv)),
        ("Accuracy", float(acc)),
        ("F1", float(f1)),
        ("Youden", float(youden)),
        ("AUC", None if pd.isna(auc) else float(auc)),
        ("AUPRC", None if pd.isna(auprc) else float(auprc)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    id_cols = [c for c in ["ID", "Name"] if c in df_pros.columns]
    pred_df = df_pros[id_cols].copy() if id_cols else pd.DataFrame(index=df_pros.index)
    pred_df["TrueLabel"] = y_true_raw
    pred_df["PredProb"] = y_proba_pros
    pred_df["PredLabel"] = y_pred_pros
    true_int = y_true_raw.astype("Int64")
    pred_df["Correct"] = np.where(true_int.isna(), np.nan, (pred_df["PredLabel"] == true_int).astype(int))

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"]
    )
    cm_detail_df = pd.DataFrame([
        ("TN", int(tn)),
        ("FP", int(fp)),
        ("FN", int(fn)),
        ("TP", int(tp)),
    ], columns=["Cell", "Count"])

    report_path = os.path.join(PROSPECTIVE_OUT_DIR, "prospective_prediction_report.xlsx")
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
        cm_detail_df.to_excel(writer, sheet_name="ConfusionMatrix", index=False, startrow=cm_df.shape[0] + 3)

    cm_png_path = os.path.join(PROSPECTIVE_OUT_DIR, "prospective_confusion_matrix.png")
    fig_cm, ax_cm = plt.subplots(figsize=(5.6, 4.6), dpi=200)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title(f"Prospective Confusion Matrix (thr={float(chosen_threshold):.4f})")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig_cm)

    print(f"[OK] Prospective report saved to: {report_path}")
    print(f"[OK] Prospective confusion matrix saved to: {cm_png_path}")


import shap

warnings.filterwarnings("ignore")

# --- SHAP plot style (aligned with 13-model script) ---
SHAP_STYLE = {
    "font_family": "Times New Roman",
    "font_fallback": "Arial",
    "title_size": 16,
    "title_weight": "bold",
    "title_pad": 20,
    "subtitle_size": 14,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
    "grid_alpha": 0.3,
    "beeswarm_max_display": 20,
    "summary_max_display": 8,
    "waterfall_max_display": 10,
    "scatter_alpha": 0.7,
    "scatter_size": 50,
    "line_width": 1.0,
    "vline_width": 2.0,
    "cmap_div": "RdYlBu_r",
    "cmap_heat": "RdBu_r",
    "cmap_target": "coolwarm",
    "small_title_size": 8,
    "small_label_size": 6,
    "small_tick_size": 5,
    "small_legend_size": 5,
    "small_cbar_label": 6,
    "small_figtext_size": 4,
    "figtext_alpha": 0.8,
}

PRO_STYLE = {
    "suptitle_size": 22,
    "ax_label_size": 16,
    "tick_label_size": 16,
    "legend_size": 14,
    "cbar_label_size": 12,
    "grid_wspace": 0.45,
    "grid_hspace": 0.4,
    "summary_cbar_width": 0.015,
    "summary_cbar_height_shrink": 1.0,
    "summary_cbar_pad": 0.01,
    "dep_cbar_width": 0.005,
    "dep_cbar_height_shrink": 1.0,
    "dep_cbar_pad": 0.002,
    "dep_cbar_tick_length": 1,
}

try:
    plt.rcParams["font.family"] = SHAP_STYLE["font_family"]
except Exception:
    plt.rcParams["font.sans-serif"] = [SHAP_STYLE["font_fallback"]]

# ==================== 5 Model explainability & outputs ====================

# ==================== 5.1 SHAP Visualization (best model, model-aware) ====================
# Purpose:
#   - Choose SHAP explainer per model family with appropriate background sampling
#   - Compute SHAP values on preprocessed test data and generate multi-format explanations
#   - Export SHAP figures (summary/bar/beeswarm/waterfall/force/interaction/heatmap etc.) for interpretation

# ==================== 5.1.0 SHAP Setup ====================

tree_models = {"Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM"}
linear_models = {"Logistic Regression", "LDA", "QDA"}

# ---- model binding (aligned with 4.5) ----
shap_model_name = best_model_name
shap_model = best_models[shap_model_name]
model_tag = shap_model_name.replace(" ", "_").lower()

print(f"\n[SECTION] {shap_model_name} SHAP Visualization")
print("=" * 60)

# ---- data binding (aligned with 4.5) ----
if "X_test_scaled_df" in globals():
    X_test_sample = X_test_scaled_df
else:
    X_test_sample = X_test_scaled
y_test_sample = y_test
X_test_values = X_test_sample.values if hasattr(X_test_sample, "values") else np.asarray(X_test_sample)

# ---- feature names ----
if "X_train_df" in globals():
    feature_names = list(map(str, X_train_df.columns))
elif "X_test_df" in globals():
    feature_names = list(map(str, X_test_df.columns))
elif "feature_names_ohe" in globals():
    feature_names = list(map(str, feature_names_ohe))
elif "preprocessor" in globals() and "X_raw" in globals() and "get_feature_names_from_preprocessor" in globals():
    feature_names = list(map(str, get_feature_names_from_preprocessor(preprocessor, X_raw.columns)))
else:
    raise RuntimeError(
        "SHAP feature names not found. "
        "Expected X_train_df or X_test_df with column names."
    )

if len(feature_names) != X_test_sample.shape[1]:
    msg = (
        f"Feature name length mismatch: "
        f"{len(feature_names)} names vs {X_test_sample.shape[1]} features."
    )
    print(f"[ERROR] {msg}")
    raise RuntimeError(msg)

# ---- background data (deterministic source, no new seed) ----
bg_source = X_train_scaled_df if "X_train_scaled_df" in globals() else X_test_sample
bg_n = min(200, len(bg_source))
if hasattr(bg_source, "sample"):
    background = bg_source.sample(n=bg_n, random_state=SEED)
else:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(bg_source), size=bg_n, replace=False)
    background = bg_source[idx]

# ---- explainer selection ----
if shap_model_name in tree_models:
    print("[INFO] Initializing TreeExplainer...")
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_test_sample)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3 and shap_values.shape[-1] == 2:
        shap_values = shap_values[:, :, 1]

elif shap_model_name in linear_models:
    print("[INFO] Initializing LinearExplainer...")
    try:
        explainer = shap.LinearExplainer(shap_model, background)
        shap_values = explainer.shap_values(X_test_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    except Exception:
        print("[WARN] LinearExplainer failed; falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(
            lambda x: _prob_from_model(shap_model, x),
            background
        )
        shap_raw = explainer.shap_values(X_test_sample, nsamples=200)
        if isinstance(shap_raw, list) and len(shap_raw) >= 2:
            shap_values = shap_raw[1]
        elif isinstance(shap_raw, list):
            shap_values = shap_raw[0]
        else:
            shap_values = shap_raw

else:
    print("[INFO] Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(
        lambda x: _prob_from_model(shap_model, x),
        background
    )
    shap_raw = explainer.shap_values(X_test_sample, nsamples=200)
    if isinstance(shap_raw, list) and len(shap_raw) >= 2:
        shap_values = shap_raw[1]
    elif isinstance(shap_raw, list):
        shap_values = shap_raw[0]
    else:
        shap_values = shap_raw

print(f"[INFO] SHAP values computed for {shap_model_name}. Preparing for visualization...")
print(f"[INFO] These SHAP outputs correspond to model: {shap_model_name}")

# ---- output directory ----
shap_dir = SHAP_DIR
os.makedirs(shap_dir, exist_ok=True)
print(f"[INFO] SHAP output directory: {shap_dir}")


# ==================== 5.1.1 SHAP Summary Plot (Supplementary Figure) ====================
print("\n[INFO] 5.1.1 Generating SHAP Summary Plot (Supplementary Figure)...")

plt.figure(figsize=(12, 8), dpi=300)
shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title(
    f"{shap_model_name} SHAP Summary Plot",
    fontsize=SHAP_STYLE["title_size"],
    fontweight=SHAP_STYLE["title_weight"],
    pad=SHAP_STYLE["title_pad"]
)
plt.tight_layout()

fig = plt.gcf()

summary_path = os.path.join(shap_dir, f"{model_tag}_shap_summary.tiff")
_save_fig(fig, summary_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP Summary Plot saved to: {summary_path}")


# ==================== 5.1.2 SHAP Bar Plot (Supplementary Figure) ====================
print("\n[INFO] 5.1.2 Generating SHAP Bar Plot (Supplementary Figure)...")

shap_importance = np.abs(shap_values).mean(axis=0)
features_sorted = np.argsort(shap_importance)

plt.figure(figsize=(10, 8), dpi=300)
plt.barh(
    [feature_names[i] for i in features_sorted],
    shap_importance[features_sorted],
    color="skyblue"
)
plt.title(
    f"{shap_model_name} SHAP Feature Importance",
    fontsize=SHAP_STYLE["title_size"],
    fontweight=SHAP_STYLE["title_weight"]
)
plt.xlabel("Mean |SHAP Value|", fontsize=SHAP_STYLE["label_size"])
plt.tight_layout()

fig = plt.gcf()

bar_path = os.path.join(shap_dir, f"{model_tag}_shap_bar.tiff")
_save_fig(fig, bar_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP Bar Plot saved to: {bar_path}")


# ==================== 5.1.3 SHAP Beeswarm Plot (Supplementary Figure) ====================
print("\n[INFO] 5.1.3 Generating SHAP Beeswarm Plot (Supplementary Figure)...")

plt.figure(figsize=(12, 10), dpi=300)

shap.plots.beeswarm(
    shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),
        data=X_test_sample,
        feature_names=feature_names
    ),
    max_display=SHAP_STYLE["beeswarm_max_display"],
    show=False
)

plt.title(
    f"{shap_model_name} SHAP Beeswarm Plot",
    fontsize=SHAP_STYLE["title_size"],
    fontweight=SHAP_STYLE["title_weight"],
    pad=SHAP_STYLE["title_pad"]
)
plt.tight_layout()

fig = plt.gcf()

beeswarm_path = os.path.join(shap_dir, f"{model_tag}_shap_beeswarm.tiff")
_save_fig(fig, beeswarm_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP Beeswarm Plot saved to: {beeswarm_path}")


# ==================== 5.1.4 SHAP Waterfall Plot (Formal Figure) ====================
print("\n[INFO] 5.1.4 Generating SHAP Waterfall Plot (Formal Figure)...")

sample_indices = [0, 1]

base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = np.atleast_1d(base_val)
    base_val = base_val[1] if base_val.size > 1 else base_val[0]

for idx in sample_indices:
    if idx >= X_test_sample.shape[0]:
        continue

    data_row = X_test_values[idx]
    shap_row = shap_values[idx]
    exp = shap.Explanation(
        values=shap_row,
        base_values=base_val,
        data=data_row,
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.waterfall(exp, max_display=SHAP_STYLE["waterfall_max_display"], show=False)

    prob_arr = _prob_from_model(shap_model, data_row.reshape(1, -1))
    pred_prob = float(prob_arr[0]) if hasattr(prob_arr, "__len__") else float(prob_arr)
    actual_cls = "Reabsorption" if int(y_test_sample[idx]) == 1 else "No Reabsorption"
    pred_cls = "Reabsorption" if pred_prob > 0.5 else "No Reabsorption"

    plt.title(
        f"Actual: {actual_cls} | Predicted P(1): {pred_prob:.4f} | Predicted: {pred_cls}",
        fontsize=SHAP_STYLE["subtitle_size"],
        fontweight=SHAP_STYLE["title_weight"],
        pad=SHAP_STYLE["title_pad"]
    )

    plt.tight_layout()
    plt.show()

    out_path = os.path.join(
        shap_dir, f"{model_tag}_shap_waterfall_sample_{idx + 1}.tiff"
    )
    _save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP Waterfall Plots saved to: {shap_dir}")


# ==================== 5.1.5 SHAP Dependence Plot (Formal Figure) ====================
print("\n[INFO] 5.1.5 Generating SHAP Dependence Plots (Formal Figure)...")

top_features_indices = np.argsort(-np.abs(shap_values).mean(axis=0))[:4]

fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
axes = axes.flatten()

for ax, feat_idx in zip(axes, top_features_indices):
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_test_sample,
        feature_names=feature_names,
        ax=ax,
        show=False
    )
    ax.set_title(
        f"SHAP Dependence: {feature_names[feat_idx]}",
        fontsize=SHAP_STYLE["label_size"],
        fontweight=SHAP_STYLE["title_weight"]
    )

plt.tight_layout()

dependence_path = os.path.join(shap_dir, f"{model_tag}_shap_dependence.tiff")
_save_fig(fig, dependence_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP Dependence Plot saved to: {dependence_path}")


# ==================== 5.1.6 SHAP Force Plot (Formal Figure) ====================
print("\n[INFO] 5.1.6 Generating SHAP Force Plots (Formal Figure)...")

sample_indices = [0, 1]

base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = np.atleast_1d(base_value)
    base_value = base_value[1] if base_value.size > 1 else base_value[0]

for idx in sample_indices:
    if idx >= X_test_sample.shape[0]:
        continue

    rounded_shap = np.round(shap_values[idx], 2)
    feat_row = X_test_values[idx]

    plt.figure(figsize=(20, 3), dpi=300)

    shap.force_plot(
        base_value,
        rounded_shap,
        feat_row,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    actual_label = "Reabsorption" if y_test_sample[idx] == 1 else "No Reabsorption"
    prob_arr = _prob_from_model(shap_model, X_test_values[idx].reshape(1, -1))
    pred_prob = float(prob_arr[0]) if hasattr(prob_arr, "__len__") else float(prob_arr)

    plt.title(
        f"SHAP Force Plot - Sample {idx + 1} | Actual: {actual_label} | Pred Prob: {pred_prob:.3f}",
        fontsize=SHAP_STYLE["subtitle_size"],
        fontweight=SHAP_STYLE["title_weight"],
        pad=SHAP_STYLE["title_pad"]
    )

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()

    out_path = os.path.join(
        shap_dir, f"{model_tag}_shap_force_sample_{idx + 1}.tiff"
    )
    _save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")

print("[OK] SHAP Force Plots completed.")


# ==================== 5.1.7 SHAP Decision Plot (Formal Figure) ====================
print("\n[INFO] 5.1.7 Generating SHAP Decision Plot (Formal Figure)...")

sample_indices = list(range(min(30, len(shap_values))))

if len(sample_indices) == 0:
    print("[WARN] No samples available for SHAP decision plot.")
else:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.atleast_1d(base_value)
        base_value = base_value[1] if base_value.size > 1 else base_value[0]

    fig = plt.figure(figsize=(12, 8), dpi=300)

    shap.decision_plot(
        base_value,
        shap_values[sample_indices],
        X_test_values[sample_indices],
        feature_names=feature_names,
        show=False
    )

    plt.title(
        f"{shap_model_name} SHAP Decision Plot",
        fontsize=SHAP_STYLE["title_size"],
        fontweight=SHAP_STYLE["title_weight"]
    )
    plt.tight_layout()
    out_path = os.path.join(shap_dir, f"{model_tag}_shap_decision.tiff")
    _save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"[OK] SHAP Decision Plot saved to: {out_path}")


# ==================== 5.1.8 SHAP Interaction Values (Supplementary Figure) ====================
print("\n[INFO] 5.1.8 Generating SHAP Interaction Analysis (Supplementary Figure)...")

interaction_sample_size = min(50, len(X_test_sample))
X_interaction_sample = X_test_values[:interaction_sample_size]

try:
    si = explainer.shap_interaction_values(X_interaction_sample)

    if isinstance(si, list):
        si = si[1] if len(si) > 1 else si[0]
    if getattr(si, "ndim", 0) == 4 and si.shape[-1] >= 2:
        si = si[..., 1]

    interaction_matrix = np.abs(si).mean(axis=0)

    plt.figure(figsize=(12, 10), dpi=300)
    mask = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)

    sns.heatmap(
        interaction_matrix,
        mask=mask,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt=".3f",
        cmap=SHAP_STYLE["cmap_div"],
        square=True,
        linewidths=0.5
    )

    plt.title(
        f"{shap_model_name} SHAP Interaction Matrix",
        fontsize=SHAP_STYLE["title_size"],
        fontweight=SHAP_STYLE["title_weight"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig = plt.gcf()
    plt.show()

    out_path = os.path.join(
        shap_dir, f"{model_tag}_shap_interaction_matrix.tiff"
    )
    _save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")

    print(f"[OK] SHAP interaction matrix saved to: {out_path}")

except Exception as e:
    print(f"[WARN] SHAP interaction computation failed: {e}")


# ==================== 5.1.9 SHAP Heatmap (Formal Figure) ====================
print("\n[INFO] 5.1.9 Generating SHAP Heatmap (Formal Figure)...")

shap_sum = shap_values.sum(axis=1)
sorted_idx = np.argsort(shap_sum)
selected_idx = sorted_idx[-min(30, len(sorted_idx)):]

plt.figure(figsize=(15, 8), dpi=300)

im = plt.imshow(
    shap_values[selected_idx].T,
    cmap=SHAP_STYLE["cmap_heat"],
    aspect="auto"
)
plt.colorbar(im, label="SHAP Value")

plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel("Sample Index")
plt.ylabel("Features")
plt.title(
    f"{shap_model_name} SHAP Values Heatmap",
    fontsize=SHAP_STYLE["title_size"],
    fontweight=SHAP_STYLE["title_weight"]
)

actual_labels = np.asarray(y_test_sample)[selected_idx]
for i, lbl in enumerate(actual_labels):
    plt.axvline(
        i,
        color="red" if lbl == 1 else "blue",
        alpha=0.3,
        linewidth=SHAP_STYLE["vline_width"]
    )

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="red", alpha=0.3, label="Reabsorption"),
    Patch(facecolor="blue", alpha=0.3, label="No Reabsorption")
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
fig = plt.gcf()

out_path = os.path.join(shap_dir, f"{model_tag}_shap_heatmap.tiff")
_save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP heatmap saved to: {out_path}")


# ==================== 5.1.10 Top Feature Analysis (Supplementary Figure) ====================
print("\n[INFO] 10. Generating Top Feature Analysis (Supplementary Figure)...")

# Select the most important feature based on mean absolute SHAP values on the test set
top_feature_idx = np.argmax(np.abs(shap_values).mean(0))
top_feature_name = feature_names[top_feature_idx]

fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
axes = axes.reshape(2, 2)

# a) Relationship between the top feature and its SHAP values
plt.sca(axes[0, 0])
sns.regplot(
    x=X_test_values[:, top_feature_idx],
    y=shap_values[:, top_feature_idx],
    scatter_kws={'alpha': 0.5, 's': 50},
    line_kws={'color': 'red'}
)
plt.xlabel(top_feature_name)
plt.ylabel('SHAP Value')
plt.title(f'SHAP Values vs {top_feature_name}', fontweight='bold')

# b) Distribution by actual class (0: No Reabsorption, 1: Reabsorption)
plt.sca(axes[0, 1])
for label, color, label_name in [(0, 'blue', 'No Reabsorption'), (1, 'red', 'Reabsorption')]:
    mask = (y_test_sample == label)
    if np.any(mask):
        sns.kdeplot(
            X_test_values[mask, top_feature_idx],
            color=color,
            label=label_name
        )
plt.xlabel(top_feature_name)
plt.ylabel('Density')
plt.title(f'{top_feature_name} Distribution by Actual Class', fontweight='bold')
plt.legend()

# c) Feature value vs predicted probability (best model on test set)
plt.sca(axes[1, 0])
probs = _prob_from_model(shap_model, X_test_values)
plt.scatter(X_test_values[:, top_feature_idx], probs, alpha=0.7, s=50)
plt.xlabel(top_feature_name)
plt.ylabel('Predicted Probability')
plt.title(f'Predicted Probability vs {top_feature_name}', fontweight='bold')

# d) Boxplot by actual class
plt.sca(axes[1, 1])
feature_data = pd.DataFrame({
    'Value': X_test_values[:, top_feature_idx],
    'Class': np.where(y_test_sample == 1, 'Reabsorption', 'No Reabsorption')
})
sns.boxplot(x='Class', y='Value', data=feature_data)
plt.ylabel(top_feature_name)
plt.title(f'{top_feature_name} by Class', fontweight='bold')

plt.tight_layout()

# Save as TIFF (300 dpi) to the supplementary figures directory
os.makedirs(shap_dir, exist_ok=True)
top_feat_path = os.path.join(shap_dir, f'{model_tag}_top_feature_analysis.tiff')
_save_fig(fig, 
    top_feat_path,
    format='tiff',
    dpi=300,
    bbox_inches='tight'
)
print(f"[OK] Supplementary figure saved to: {top_feat_path}")


# ==================== 5.1.11 Feature Importance Comparison ====================
# Purpose:
#   - Compare model-based (best_model_name) importances with SHAP importances
#   - Normalize/visualize both metrics for side-by-side assessment
#   - Save comparison figure for feature ranking transparency
print("\n[INFO] 11. Generating Feature Importance Comparison...")

model_for_importance = best_models[best_model_name]
model_importance = None
if hasattr(model_for_importance, "feature_importances_"):
    model_importance = model_for_importance.feature_importances_
else:
    print(f"[WARN] {best_model_name} has no feature_importances_; only SHAP importance plotted.")

# Compute SHAP-based feature importance (mean absolute SHAP value)
shap_importance = np.abs(shap_values).mean(0)

# Combine into a comparison DataFrame (model importance may be None)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': shap_importance
})
if model_importance is not None:
    importance_df['Model_Importance'] = model_importance
    importance_df['Model_Normalized'] = importance_df['Model_Importance'] / (importance_df['Model_Importance'].max() or 1.0)
importance_df['SHAP_Normalized'] = importance_df['SHAP_Importance'] / (importance_df['SHAP_Importance'].max() or 1.0)

# Sort by SHAP importance ascending (so the bar chart increases from bottom to top)
importance_df = importance_df.sort_values('SHAP_Normalized', ascending=True)

if model_importance is not None:
    # Create figure: Left = normalized comparison, Right = correlation scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    # Left: normalized importance comparison (best_model_name vs SHAP)
    y_pos = np.arange(len(importance_df))
    ax1.barh(y_pos - 0.2, importance_df['Model_Normalized'], 0.4,
             label=f"{best_model_name} Importance", alpha=0.8, color='skyblue')
    ax1.barh(y_pos + 0.2, importance_df['SHAP_Normalized'], 0.4,
             label='SHAP Importance', alpha=0.8, color='lightcoral')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(importance_df['Feature'])
    ax1.set_xlabel('Normalized Importance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: correlation between model-based and SHAP importance
    ax2.scatter(importance_df['Model_Normalized'], importance_df['SHAP_Normalized'],
                alpha=0.7, s=80, color='green')

    # Annotate feature names
    for _, row in importance_df.iterrows():
        ax2.annotate(row['Feature'],
                     (row['Model_Normalized'], row['SHAP_Normalized']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)

    ax2.set_xlabel(f'{best_model_name} Normalized Importance')
    ax2.set_ylabel('SHAP Normalized Importance')
    ax2.grid(True, alpha=0.3)

    # Add diagonal reference line (perfect correlation)
    max_val = max(importance_df['Model_Normalized'].max(), importance_df['SHAP_Normalized'].max())
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    ax2.legend()
else:
    # Create figure: SHAP-only importance
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=300)

    y_pos = np.arange(len(importance_df))
    ax1.barh(y_pos, importance_df['SHAP_Normalized'], 0.6,
             label='SHAP Importance', alpha=0.8, color='lightcoral')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(importance_df['Feature'])
    ax1.set_xlabel('SHAP Normalized Importance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

plt.tight_layout()

# Save as high-resolution TIFF (300 dpi) in the existing shap_dir
out_path = os.path.join(shap_dir, f'{model_tag}_importance_comparison.tiff')
_save_fig(fig, out_path, format='tiff', dpi=300, bbox_inches='tight')

print("SHAP Visualization - Feature Importance Comparison Completed!")


# ==================== 5.1.12 Advanced SHAP Combined Visualization ====================
print("\n[INFO] 5.1.12 Generating Advanced SHAP Combined Visualization...")

from scipy import signal
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def find_knee_point(x, y, window_length=11, polyorder=3, prominence=0.1):
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    if len(x_sorted) > window_length:
        y_smooth = signal.savgol_filter(y_sorted, window_length, polyorder)
        y_deriv2 = np.gradient(np.gradient(y_smooth))
        peaks, _ = signal.find_peaks(np.abs(y_deriv2), prominence=prominence)
        if len(peaks) > 0:
            idx = peaks[np.argmax(np.abs(y_deriv2[peaks]))]
            return x_sorted[idx], y_sorted[idx]

    mid = len(x_sorted) // 2
    return x_sorted[mid], y_sorted[mid]


fig = plt.figure(figsize=(24, 18), dpi=300)
gs = gridspec.GridSpec(
    3, 3,
    width_ratios=[1.0, 1.0, 0.9],
    height_ratios=[1.0, 1.0, 0.9],
    wspace=0.3,
    hspace=0.3
)

# ---- Summary panel (render SHAP summary into image) ----
ax_summary = plt.subplot(gs[0:2, 0:2])

tmp_fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=feature_names,
    plot_type="dot",
    max_display=8,
    sort=True,
    show=False
)

canvas = FigureCanvas(tmp_fig)
canvas.draw()
w, h = canvas.get_width_height()
img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
plt.close(tmp_fig)

ax_summary.imshow(img)
ax_summary.axis("off")
ax_summary.set_title(
    f"SHAP Feature Importance ({shap_model_name}, Test Set)",
    fontsize=SHAP_STYLE["small_title_size"],
    fontweight=SHAP_STYLE["title_weight"],
    pad=8
)

# ---- Dependence plots ----
axes_dep = [
    plt.subplot(gs[0, 2]),
    plt.subplot(gs[2, 0]),
    plt.subplot(gs[2, 2])
]

top_features_idx = np.argsort(-np.abs(shap_values).mean(axis=0))[:3]

for ax, f_idx in zip(axes_dep, top_features_idx):
    xvals = X_test_values[:, f_idx]
    yvals = shap_values[:, f_idx]

    ax.scatter(
        xvals, yvals,
        c=y_test_sample,
        cmap="coolwarm",
        s=SHAP_STYLE["scatter_size"],
        alpha=0.8
    )

    try:
        sort_idx = np.argsort(xvals)
        xs, ys = xvals[sort_idx], yvals[sort_idx]
        from statsmodels.nonparametric.smoothers_lowess import lowess
        z = lowess(ys, xs, frac=0.3, it=1)
        ax.plot(z[:, 0], z[:, 1], "k-", lw=2)
        knee_x, _ = find_knee_point(z[:, 0], z[:, 1])
        ax.axvline(knee_x, color="red", linestyle="--", alpha=0.55)
    except Exception:
        pass

    ax.axhline(0, color="gray", alpha=0.5)
    ax.set_title(
        f"SHAP Dependence: {feature_names[f_idx]}",
        fontsize=SHAP_STYLE["small_title_size"],
        fontweight=SHAP_STYLE["title_weight"],
        pad=10
    )
    ax.set_xlabel(feature_names[f_idx], fontsize=SHAP_STYLE["small_label_size"], labelpad=5)
    ax.set_ylabel("SHAP Value", fontsize=SHAP_STYLE["small_label_size"], labelpad=10)
    ax.tick_params(labelsize=SHAP_STYLE["small_tick_size"])
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.5)

# ---- Colorbar ----
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap="coolwarm")
sm.set_array([])

cbar_ax = plt.axes([0.90, 0.20, 0.02, 0.60])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label(
    "Target (0: No Reabsorption, 1: Reabsorption)",
    fontsize=SHAP_STYLE["small_cbar_label"]
)
cbar.ax.tick_params(labelsize=SHAP_STYLE["small_tick_size"])

plt.suptitle(
    f"{shap_model_name} SHAP Advanced Visualization (Test Set)",
    fontsize=SHAP_STYLE["small_title_size"],
    fontweight=SHAP_STYLE["title_weight"],
    y=0.98
)

description_text = (
    "SHAP Visualization Description:\n"
    "- Summary: global feature impact (dot summary)\n"
    "- Dependence: feature value vs SHAP value\n"
    "- Color: target value (blue=0, red=1)"
)
plt.figtext(
    0.2, 0.01, description_text,
    ha="left",
    fontsize=SHAP_STYLE["small_figtext_size"],
    fontweight="light",
    bbox=dict(facecolor="white", alpha=SHAP_STYLE["figtext_alpha"], boxstyle="round,pad=0.5")
)


os.makedirs(shap_dir, exist_ok=True)
out_path = os.path.join(shap_dir, f"{model_tag}_shap_advanced_combined.tiff")
_save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] Advanced SHAP combined visualization saved to: {out_path}")


# ==================== 5.1.13 Improved SHAP Quadrant Analysis (Test Set) ====================
print("\n[INFO] 5.1.13 Generating Improved SHAP Quadrant Analysis...")

top_2_idx = np.argsort(-np.abs(shap_values).mean(axis=0))[:2]
f1_idx, f2_idx = top_2_idx

x1 = X_test_values[:, f1_idx]
x2 = X_test_values[:, f2_idx]
s1 = shap_values[:, f1_idx]
s2 = shap_values[:, f2_idx]
feature1_name = feature_names[f1_idx]
feature2_name = feature_names[f2_idx]

probs = _prob_from_model(shap_model, X_test_values)

fig = plt.figure(figsize=(18, 15), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

ax_main = plt.subplot(gs[0, 0])

total_shap = np.abs(s1) + np.abs(s2)
scatter = ax_main.scatter(
    x1, x2,
    c=y_test_sample,
    cmap="coolwarm",
    s=(total_shap / total_shap.max()) * 300 + 30,
    alpha=SHAP_STYLE["scatter_alpha"],
    edgecolor="w",
    linewidth=0.5
)

x1_med, x2_med = np.median(x1), np.median(x2)
ax_main.axvline(x1_med, linestyle="--", color="gray")
ax_main.axhline(x2_med, linestyle="--", color="gray")

label_offset_x = (x1.max() - x1.min()) * 0.05
label_offset_y = (x2.max() - x2.min()) * 0.05
ax_main.text(
    x1.min() + label_offset_x,
    x2_med + label_offset_y,
    f"Q2: Low {feature1_name},\nHigh {feature2_name}",
    fontsize=SHAP_STYLE["label_size"],
    ha="left", va="bottom",
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3")
)
ax_main.text(
    x1_med + label_offset_x,
    x2_med + label_offset_y,
    f"Q1: High {feature1_name},\nHigh {feature2_name}",
    fontsize=SHAP_STYLE["label_size"],
    ha="left", va="bottom",
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3")
)
ax_main.text(
    x1.min() + label_offset_x,
    x2.min() + label_offset_y,
    f"Q3: Low {feature1_name},\nLow {feature2_name}",
    fontsize=SHAP_STYLE["label_size"],
    ha="left", va="bottom",
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3")
)
ax_main.text(
    x1_med + label_offset_x,
    x2.min() + label_offset_y,
    f"Q4: High {feature1_name},\nLow {feature2_name}",
    fontsize=SHAP_STYLE["label_size"],
    ha="left", va="bottom",
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3")
)

ax_main.set_xlabel(feature1_name, fontsize=SHAP_STYLE["label_size"], labelpad=10)
ax_main.set_ylabel(feature2_name, fontsize=SHAP_STYLE["label_size"], labelpad=10)
ax_main.set_title(
    f"Feature Interaction Quadrant Analysis:\n{feature1_name} vs {feature2_name}",
    fontsize=SHAP_STYLE["subtitle_size"],
    fontweight=SHAP_STYLE["title_weight"],
    pad=20
)
ax_main.tick_params(axis="both", which="major", labelsize=SHAP_STYLE["tick_size"])

legend1 = ax_main.legend(
    *scatter.legend_elements(num=5),
    loc="upper right",
    title="Target",
    fontsize=SHAP_STYLE["legend_size"]
)
ax_main.add_artist(legend1)

from matplotlib.lines import Line2D

size_legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           label=f"{i * 25:.0f}% SHAP", markersize=(i * 8 + 4) ** 0.5)
    for i in range(1, 5)
]
ax_main.legend(
    handles=size_legend_elements,
    loc="upper left",
    title="Relative Importance",
    fontsize=SHAP_STYLE["legend_size"]
)

# ---- Right and bottom histograms ----
ax_right = plt.subplot(gs[0, 1], sharey=ax_main)
ax_right.hist(x2, bins=20, orientation="horizontal", alpha=0.7, color="skyblue")
ax_right.axhline(y=x2_med, color="gray", linestyle="--", alpha=0.5)
ax_right.set_xlabel("Count", fontsize=SHAP_STYLE["label_size"], labelpad=10)
ax_right.set_title(f"{feature2_name} Distribution", fontsize=SHAP_STYLE["subtitle_size"], pad=10)
ax_right.tick_params(axis="both", which="major", labelsize=SHAP_STYLE["tick_size"])

ax_bottom = plt.subplot(gs[1, 0], sharex=ax_main)
ax_bottom.hist(x1, bins=20, alpha=0.7, color="skyblue")
ax_bottom.axvline(x=x1_med, color="gray", linestyle="--", alpha=0.5)
ax_bottom.set_ylabel("Count", fontsize=SHAP_STYLE["label_size"], labelpad=10)
ax_bottom.set_title(f"{feature1_name} Distribution", fontsize=SHAP_STYLE["subtitle_size"], pad=10)
ax_bottom.tick_params(axis="both", which="major", labelsize=SHAP_STYLE["tick_size"])

# ---- Quadrant statistics ----
ax_stat = plt.subplot(gs[1, 1])
ax_stat.axis("off")

def quad_stats(x1, x2, y, p, t1, t2):
    q = {
        "Q1": (x1 >= t1) & (x2 >= t2),
        "Q2": (x1 <  t1) & (x2 >= t2),
        "Q3": (x1 <  t1) & (x2 <  t2),
        "Q4": (x1 >= t1) & (x2 <  t2)
    }
    txt = ""
    for k, m in q.items():
        txt += f"{k}: n={m.sum()}, rate={y[m].mean():.2f}, prob={p[m].mean():.3f}\n"
    return txt

ax_stat.text(
    0, 0.5,
    quad_stats(x1, x2, y_test_sample, probs, x1_med, x2_med),
    fontsize=SHAP_STYLE["subtitle_size"],
    va="center",
    bbox=dict(facecolor="white", alpha=SHAP_STYLE["figtext_alpha"], boxstyle="round,pad=0.5")
)

plt.tight_layout()

out_path = os.path.join(shap_dir, f"{model_tag}_shap_quadrant_analysis.tiff")
_save_fig(fig, out_path, format="tiff", dpi=300, bbox_inches="tight")

print(f"[OK] SHAP quadrant analysis saved to: {out_path}")


# ==================== 5.1.14 Advanced SHAP Combined Visualization (Professional) ====================
print("\n[INFO] 5.1.14 Generating professional-grade SHAP combined visualization...")

X_plot_sample = X_test_values
y_plot_sample = y_test_sample

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
}).sort_values("importance", ascending=True)

top_6_features = importance_df["feature"].tail(6).iloc[::-1].tolist()

# ---- Left figure: professional summary panel ----
fig_left = plt.figure(figsize=(12, 15), dpi=300)
ax_main = fig_left.add_subplot(1, 1, 1)

ax_top = ax_main.twiny()
ax_top.barh(
    range(len(importance_df)),
    importance_df["importance"],
    color="lightgray", alpha=0.6
)
ax_top.set_xlabel(
    "Mean Absolute SHAP Value (Global Importance)",
    fontsize=PRO_STYLE["ax_label_size"]
)
ax_top.tick_params(axis="x", labelsize=PRO_STYLE["tick_label_size"])
ax_top.grid(False)

ax_main.set_yticks(range(len(importance_df)))
ax_main.set_yticklabels(importance_df["feature"], fontsize=PRO_STYLE["tick_label_size"])

# Use local random generator for reproducibility (create once before loop)
rng = np.random.default_rng(SEED)
for i, feat in enumerate(importance_df["feature"]):
    idx = feature_names.index(feat)
    jitter = rng.normal(0, 0.08, shap_values.shape[0])
    ax_main.scatter(
        shap_values[:, idx],
        i + jitter,
        c=X_plot_sample[:, idx],
        cmap="viridis",
        s=15, alpha=0.8
    )

ax_main.set_xlabel(
    "SHAP value (impact on model output)",
    fontsize=PRO_STYLE["ax_label_size"]
)
ax_main.tick_params(axis="x", labelsize=PRO_STYLE["tick_label_size"])
ax_main.grid(True, axis="x", linestyle="--", alpha=0.6)

# Summary colorbar
fig_left.canvas.draw()
ax_main_pos = ax_main.get_position()
cax_left = ax_main_pos.x1 + PRO_STYLE["summary_cbar_pad"]
cax_bottom = ax_main_pos.y0 + (
    ax_main_pos.height * (1 - PRO_STYLE["summary_cbar_height_shrink"]) / 2
)
cax_width = PRO_STYLE["summary_cbar_width"]
cax_height = ax_main_pos.height * PRO_STYLE["summary_cbar_height_shrink"]
cax = fig_left.add_axes([cax_left, cax_bottom, cax_width, cax_height])

norm = plt.Normalize(vmin=np.nanmin(X_plot_sample), vmax=np.nanmax(X_plot_sample))
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = fig_left.colorbar(sm, cax=cax)
cbar.set_label("Feature value", rotation=90, labelpad=-15, fontsize=PRO_STYLE["cbar_label_size"])
cbar.outline.set_visible(False)
cbar.set_ticks([])
cbar.ax.text(
    0.6, 1.02, "High", ha="center", va="top",
    transform=cbar.ax.transAxes, fontsize=PRO_STYLE["tick_label_size"]
)
cbar.ax.text(
    0.6, -0.02, "Low", ha="center", va="bottom",
    transform=cbar.ax.transAxes, fontsize=PRO_STYLE["tick_label_size"]
)

fig_left.suptitle(
    f"{shap_model_name} SHAP Professional Summary",
    fontsize=PRO_STYLE["suptitle_size"],
    fontweight=SHAP_STYLE["title_weight"],
    y=0.98
)
fig_left.text(
    0.5, 0.01,
    "SHAP summary panel: global importance bars + sample-level SHAP scatter colored by feature value.",
    ha="center",
    fontsize=8,
    bbox=dict(facecolor="white", alpha=SHAP_STYLE["figtext_alpha"], boxstyle="round,pad=0.5")
)

left_out_path = os.path.join(shap_dir, f"{model_tag}_shap_professional_left.tiff")
_save_fig(fig_left, left_out_path, format="tiff", dpi=300, bbox_inches="tight")
print(f"[OK] Professional SHAP left panel saved to: {left_out_path}")

# ---- Right figure: dependence panels ----
fig_right = plt.figure(figsize=(12, 15), dpi=300)
gs_right = gridspec.GridSpec(
    3, 2,
    wspace=PRO_STYLE["grid_wspace"],
    hspace=PRO_STYLE["grid_hspace"]
)
axes_scatter = [fig_right.add_subplot(gs_right[i, j]) for i in range(3) for j in range(2)]

from matplotlib.lines import Line2D

for i, feature in enumerate(top_6_features):
    ax = axes_scatter[i]
    feature_idx = feature_names.index(feature)
    x_data = X_plot_sample[:, feature_idx]
    y_data = shap_values[:, feature_idx]
    color_data = y_plot_sample

    scatter = ax.scatter(
        x_data, y_data,
        c=color_data,
        cmap=SHAP_STYLE["cmap_target"],
        s=25,
        alpha=0.8
    )

    # colorbar per dependence plot
    fig_right.canvas.draw()
    ax_pos = ax.get_position()
    cax_dep_left = ax_pos.x1 + PRO_STYLE["dep_cbar_pad"]
    cax_dep_bottom = ax_pos.y0 + (
        ax_pos.height * (1 - PRO_STYLE["dep_cbar_height_shrink"]) / 2
    )
    cax_dep_width = PRO_STYLE["dep_cbar_width"]
    cax_dep_height = ax_pos.height * PRO_STYLE["dep_cbar_height_shrink"]
    cax_dep = fig_right.add_axes([cax_dep_left, cax_dep_bottom, cax_dep_width, cax_dep_height])

    cbar = fig_right.colorbar(scatter, cax=cax_dep)
    cbar.ax.set_title("Diagnosis", fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(
        axis="y",
        length=PRO_STYLE["dep_cbar_tick_length"],
        labelsize=PRO_STYLE["tick_label_size"]
    )

    ax.set_xlabel(feature, fontsize=PRO_STYLE["ax_label_size"])
    ax.set_ylabel("SHAP", fontsize=12, labelpad=-8)

    median_val = np.median(x_data)
    try:
        knee_val = find_knee_point(x_data, y_data)
        threshold_val = knee_val[0] if isinstance(knee_val, (tuple, list, np.ndarray)) else knee_val
    except Exception:
        threshold_val = median_val
        print(f"[WARN] Knee point failed for {feature}; using median.")

    ax.axvline(median_val, color="black", linestyle="--", linewidth=1)
    ax.axvline(threshold_val, color="red", linestyle=":", linewidth=1.2)

    line_handles = [
        Line2D([0], [0], color="black", lw=1, linestyle="--", label=f"Median: {median_val:.2f}"),
        Line2D([0], [0], color="red", lw=1, linestyle=":", label=f"Threshold: {threshold_val:.2f}")
    ]
    ax.legend(handles=line_handles, loc="best", fontsize=PRO_STYLE["legend_size"])
    ax.tick_params(axis="both", which="major", labelsize=PRO_STYLE["tick_label_size"])

fig_right.suptitle(
    f"{shap_model_name} SHAP Professional Dependence (Top 6 Features)",
    fontsize=PRO_STYLE["suptitle_size"],
    fontweight=SHAP_STYLE["title_weight"],
    y=0.98
)
fig_right.text(
    0.5, 0.01,
    "Dependence panels: SHAP vs feature value for top 6 features with median and knee-point thresholds.",
    ha="center",
    fontsize=8,
    bbox=dict(facecolor="white", alpha=SHAP_STYLE["figtext_alpha"], boxstyle="round,pad=0.5")
)

right_out_path = os.path.join(shap_dir, f"{model_tag}_shap_professional_right.tiff")
_save_fig(fig_right, right_out_path, format="tiff", dpi=300, bbox_inches="tight")
print(f"[OK] Professional SHAP right panel saved to: {right_out_path}")



