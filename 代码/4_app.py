# app.py - auto-discover best_model_pipeline_*.pkl and *_thresholds.json
import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import json
import numpy as np

st.set_page_config(page_title="Lumbar Disc Herniation Resorption Probability Calculator", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;600&display=swap');
:root {
  --bg: #f4f6fb;
  --panel: #ffffff;
  --panel-2: #f7f9ff;
  --accent: #5292F7;
  --accent-2: #CC247C;
  --accent-3: #4EA660;
  --text: #1b1f2a;
  --muted: #4b5563;
  --border: #c9d7f5;
}
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(160deg, #79CAFB1f 0%, #FBEB6630 100%);
}
[data-testid="stHeader"] {
  background: rgba(0, 0, 0, 0);
}
h1, h2, h3, h4 {
  font-family: 'Playfair Display', Georgia, serif;
  color: var(--text);
  letter-spacing: 0.3px;
}
body, [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
  color: var(--text);
}
.hero {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.3rem 1.5rem;
  box-shadow: 0 8px 28px rgba(31, 36, 48, 0.08);
  margin-bottom: 1rem;
  animation: fadeUp 0.5s ease both;
}
.hero-title {
  font-size: 2.8rem;
  font-weight: 700;
  color: var(--text);
}
.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  margin-top: 0.35rem;
}
.section-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 6px 22px rgba(31, 36, 48, 0.06);
  margin-bottom: 1rem;
  animation: fadeUp 0.6s ease both;
}
.section-title {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.6rem;
}
.helper {
  color: var(--muted);
  font-size: 0.98rem;
  margin-bottom: 0.6rem;
}
.prediction-card {
  background: linear-gradient(135deg, #fef5e9 0%, #fff 65%);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent-2);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  margin-top: 0.6rem;
}
.prediction-title {
  font-weight: 600;
  color: var(--accent-2);
  margin-bottom: 0.2rem;
}
.stButton > button {
  background: var(--accent);
  color: white;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  border: 0;
  font-weight: 600;
}
.stButton > button:hover {
  background: #3c78da;
}
div[data-testid="stMetric"] {
  background: var(--panel-2);
  border: 1px solid var(--border);
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
}
div[data-testid="stWidgetLabel"] > label,
div[data-testid="stWidgetLabel"] > label p,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stSlider label {
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  color: var(--text) !important;
}
input, textarea, .stTextInput input, .stNumberInput input {
  background-color: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid #8fb0ef !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  border: 1px solid #8fb0ef !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] span {
  color: var(--text) !important;
}
input:focus, textarea:focus, .stTextInput input:focus, .stNumberInput input:focus,
div[data-baseweb="select"] > div:focus-within {
  box-shadow: 0 0 0 3px #5292F73a !important;
  border-color: var(--accent) !important;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

# === Model paths: prefer project-level Results, fallback to code-level Results ===
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent

def _pick_run_dirs(root: Path):
    if root.exists():
        runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
        if runs:
            return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return []

def _pick_modeling_dirs_legacy(results_root: Path):
    if results_root.exists():
        candidates = [
            p for p in results_root.iterdir()
            if p.is_dir() and "Machine Learning Modeling" in p.name
        ]
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return [results_root / "2.Machine Learning Modeling"]

def _latest_by_mtime(paths):
    return max(paths, key=lambda p: p.stat().st_mtime)

def _collect_candidates(dirs, pattern):
    paths = []
    for d in dirs:
        if d and d.exists():
            paths.extend(d.glob(pattern))
    return paths

def _uniq_dirs(dirs):
    seen = set()
    out = []
    for d in dirs:
        if d and str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out

def _existing_dirs(dirs):
    return [d.resolve() for d in _uniq_dirs(dirs) if d and d.exists()]

env_results_root = os.getenv("LUMBAR_RESULTS_ROOT")
RESULTS_ROOTS = []
if env_results_root:
    RESULTS_ROOTS.append(Path(env_results_root).expanduser())
RESULTS_ROOTS.extend([
    PROJECT_ROOT / "Results",
    BASE_DIR / "Results",
])
RESULTS_ROOTS = _existing_dirs(RESULTS_ROOTS)

RUN_DIRS = []
for new_root in [root / "Manuscript_v2" for root in RESULTS_ROOTS]:
    RUN_DIRS.extend(_pick_run_dirs(new_root))
RUN_DIRS = sorted(_uniq_dirs(RUN_DIRS), key=lambda p: p.stat().st_mtime, reverse=True)

MODEL_DIRS = [d / "04_ML_ModelDevelopment" for d in RUN_DIRS if (d / "04_ML_ModelDevelopment").exists()]
DEPLOY_DIRS = [d / "06_Calculator_Deployment" / "exported_model" for d in RUN_DIRS if (d / "06_Calculator_Deployment" / "exported_model").exists()]

if not MODEL_DIRS:
    for results_root in RESULTS_ROOTS:
        legacy_dirs = _pick_modeling_dirs_legacy(results_root)
        MODEL_DIRS.extend([p for p in legacy_dirs if p.exists()])
        DEPLOY_DIRS.extend([p / "deployment" for p in legacy_dirs if (p / "deployment").exists()])

RESULTS_DIR = MODEL_DIRS[0] if MODEL_DIRS else (RESULTS_ROOTS[0] if RESULTS_ROOTS else PROJECT_ROOT)
DEPLOY_DIR = DEPLOY_DIRS[0] if DEPLOY_DIRS else RESULTS_DIR
SEARCH_DIRS = _uniq_dirs(DEPLOY_DIRS + MODEL_DIRS + RESULTS_ROOTS + [PROJECT_ROOT, BASE_DIR])

thr_candidates = _collect_candidates(
    SEARCH_DIRS,
    "*_thresholds*.json"
)
thr_json_path = _latest_by_mtime(thr_candidates) if thr_candidates else None
thr_cfg = {}
if thr_json_path:
    try:
        with open(thr_json_path, "r", encoding="utf-8") as f:
            thr_cfg = json.load(f)
    except Exception as e:
        st.warning(f"Thresholds JSON load failed: {e}")
        thr_cfg = {}

pipeline_candidates = _collect_candidates(
    SEARCH_DIRS,
    "best_model_pipeline_*.pkl"
)
pipeline_path = _latest_by_mtime(pipeline_candidates) if pipeline_candidates else None
if pipeline_path is None:
    search_display = ", ".join(str(d) for d in SEARCH_DIRS)
    st.error(
        "Pipeline file not found: expected best_model_pipeline_*.pkl in "
        f"{search_display}"
    )
    st.stop()

model_tag = None
if isinstance(thr_cfg, dict):
    model_tag = thr_cfg.get("model_tag")
if not model_tag and thr_json_path:
    model_tag = thr_json_path.stem.replace("_thresholds", "")
if not model_tag:
    stem = pipeline_path.stem
    model_tag = stem.replace("best_model_pipeline_", "") if stem.startswith("best_model_pipeline_") else stem

model_name = None
if isinstance(thr_cfg, dict):
    model_name = thr_cfg.get("model_name")
if not model_name:
    model_name = model_tag

if model_tag and model_tag not in pipeline_path.name:
    st.warning("Thresholds/pipeline tag mismatch; you may have loaded different model/threshold versions.")

DEFAULT_THR = 0.5
THRESHOLD = DEFAULT_THR
THRESHOLD_LOW = None
THRESHOLD_HIGH = None
thr_source = "Default 0.5 (threshold file missing)"
if thr_json_path and isinstance(thr_cfg, dict) and thr_cfg:
    thr_source = f"Default 0.5 (no threshold keys found in {thr_json_path.name})"
    # Prefer dual-threshold strategy when both low/high are available
    if "threshold_low" in thr_cfg and "threshold_high" in thr_cfg:
        try:
            THRESHOLD_LOW = float(thr_cfg["threshold_low"])
            THRESHOLD_HIGH = float(thr_cfg["threshold_high"])
            if THRESHOLD_LOW > THRESHOLD_HIGH:
                THRESHOLD_LOW, THRESHOLD_HIGH = THRESHOLD_HIGH, THRESHOLD_LOW
            thr_source = f"Dual-threshold (from {thr_json_path.name})"
        except Exception:
            THRESHOLD_LOW = None
            THRESHOLD_HIGH = None
    if THRESHOLD_LOW is None or THRESHOLD_HIGH is None:
        for key, label in (
            ("threshold_Youden", "Youden"),
            ("threshold_MaxF1", "MaxF1"),
            ("threshold_Sens90", "Sens>=0.90"),
            ("threshold_Chosen", "Chosen"),
        ):
            if key in thr_cfg:
                THRESHOLD = float(thr_cfg[key])
                thr_source = f"{label} (from {thr_json_path.name})"
                break

with st.sidebar:
    st.markdown("### Model Info")
    st.write(f"**Model**: {model_name} ({model_tag})")
    st.write(f"**Pipeline**: `{pipeline_path.name}`")
    if thr_json_path:
        st.write(f"**Thresholds**: `{thr_json_path.name}`")
    else:
        st.write("**Thresholds**: None")
    st.markdown("### Threshold Strategy")
    if THRESHOLD_LOW is not None and THRESHOLD_HIGH is not None:
        st.write(f"{thr_source}")
        st.write(f"Low / High: `{THRESHOLD_LOW:.4f}` / `{THRESHOLD_HIGH:.4f}`")
    else:
        st.write(f"{thr_source}")
        st.write(f"Threshold: `{THRESHOLD:.4f}`")

@st.cache_resource
def load_model(pkl_path: Path):
    if not pkl_path.exists():
        st.error(f"Model file not found: {pkl_path}")
        st.stop()
    try:
        return joblib.load(str(pkl_path))
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def _prob_from_estimator(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1], "predict_proba"
    if hasattr(est, "decision_function"):
        st.warning("Using decision_function; pseudo-probability not calibrated.")
        scores = est.decision_function(X)
        return _sigmoid(scores), "decision_function"
    st.warning("No probability interface; output is 0/1 predictions.")
    return est.predict(X), "predict"

def _extract_threshold_metrics(cfg):
    metric_keys = ("sensitivity", "specificity", "PPV", "NPV", "accuracy", "F1", "Youden")
    if not isinstance(cfg, dict):
        return None, None
    top_metrics = {k: cfg[k] for k in metric_keys if k in cfg}
    rows = {}
    for key, val in cfg.items():
        if isinstance(val, dict) and any(m in val for m in metric_keys):
            rows[key] = {m: val.get(m) for m in metric_keys if m in val}
    table = pd.DataFrame(rows).T if rows else None
    return top_metrics if top_metrics else None, table

model = load_model(pipeline_path)

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">Lumbar Disc Herniation Resorption ({model_name})</div>
  <div class="hero-sub">For research and decision support only; not for clinical diagnosis.</div>
</div>
""",
    unsafe_allow_html=True,
)

# === Categorical and numerical variables (consistent with training) ===
CATS = {
    "Gender": [0, 1],  # still use 0/1 in the UI but mapped to Female/Male before submission
    "Herniated_Level": ['L2/3', 'L3/4', 'L4/5', 'L5/S1'],
    "Pfirrmann": [1, 2, 3, 4, 5],
    "Iwabuchi": ["1", "2", "3", "4", "5"],
    "Modic": ["0", "1", "2", "3"],
    "Komori": [1, 2, 3, 4],
    "MSU": [1, 2, 3],
    "Spinal_canal_stenosis": ["0", "1"],
    "Bull_eye": ["(missing)", "1", "2", "3"],
}
NUMS = {
    "Age": (18, 90, 50),
    "SS": (0.0, 60.0, 30.0),
    "Initial_volume": (0.0, 50.0, 4.0),
    "Upper_VB_Posterior_Height_CM": (1.0, 5.0, 2.5),
    "Lower_VB_Posterior_Height_CM": (1.0, 5.0, 2.4),
    "RSI_protrusion_gray": (0.0, 10000.0, 100.0),
    "RSI_csf_gray": (0.01, 10000.0, 100.0),
    "DHI_a": (0.0, 10.0, 1.0),
    "DHI_c": (0.0, 10.0, 1.0),
    "DHI_n": (0.01, 10.0, 1.0),
    "DHI_o": (0.01, 10.0, 1.0),
}
FEATURES = [
    'Age','Gender','Herniated_Level','Pfirrmann','Iwabuchi','Modic','Komori','MSU',
    'Spinal_canal_stenosis','Bull_eye','SS',
    'Upper_VB_Posterior_Height_CM','Lower_VB_Posterior_Height_CM',
    'Initial_volume','RSI','DHI'
]

# === Input section ===
def _section_open(title, subtitle=None):
    st.markdown(f"<div class='section-card'><div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='helper'>{subtitle}</div>", unsafe_allow_html=True)

def _section_close():
    st.markdown("</div>", unsafe_allow_html=True)

_section_open("Demographics", "Patient basics and lesion level")
col1, col2, col3 = st.columns(3)
with col1:
    Gender = st.selectbox(
        "Gender",
        CATS["Gender"],
        index=1,
        format_func=lambda x: "Male (1)" if int(x) == 1 else "Female (0)",
    )
    Age = st.number_input("Age (years)", int(NUMS["Age"][0]), int(NUMS["Age"][1]), int(NUMS["Age"][2]), step=1)
with col2:
    Herniated_Level = st.selectbox("Herniated Level", CATS["Herniated_Level"], index=2)
with col3:
    Spinal_canal_stenosis = st.selectbox(
        "Spinal canal stenosis",
        CATS["Spinal_canal_stenosis"],
        index=0,
        format_func=lambda x: "Yes (1)" if str(x) == "1" else "No (0)",
    )
_section_close()

_section_open("Classification", "Imaging grades and morphology")
col4, col5, col6 = st.columns(3)
with col4:
    Pfirrmann = st.selectbox("Pfirrmann", CATS["Pfirrmann"], index=1)
with col5:
    Iwabuchi = st.selectbox("Iwabuchi", CATS["Iwabuchi"], index=1)
with col6:
    Modic = st.selectbox("Modic", CATS["Modic"], index=0)
col7, col8, col9 = st.columns(3)
with col7:
    Komori = st.selectbox("Komori", CATS["Komori"], index=1)
with col8:
    MSU = st.selectbox("MSU", CATS["MSU"], index=1)
with col9:
    Bull_eye = st.selectbox("Bull-eye (missing allowed)", CATS["Bull_eye"], index=0)
_section_close()

_section_open("Morphology", "Heights, sagittal parameter, and volume")
col10, col11 = st.columns(2)
with col10:
    SS = st.number_input("SS", float(NUMS["SS"][0]), float(NUMS["SS"][1]), float(NUMS["SS"][2]), step=0.5)
with col11:
    Initial_volume = st.number_input(
        "Initial volume (cm^3)",
        float(NUMS["Initial_volume"][0]),
        float(NUMS["Initial_volume"][1]),
        float(NUMS["Initial_volume"][2]),
        step=0.1,
    )
col12, col13 = st.columns(2)
with col12:
    Upper_VB_Posterior_Height_CM = st.number_input(
        "Upper VB posterior height (cm)",
        float(NUMS["Upper_VB_Posterior_Height_CM"][0]),
        float(NUMS["Upper_VB_Posterior_Height_CM"][1]),
        float(NUMS["Upper_VB_Posterior_Height_CM"][2]),
        step=0.1,
    )
with col13:
    Lower_VB_Posterior_Height_CM = st.number_input(
        "Lower VB posterior height (cm)",
        float(NUMS["Lower_VB_Posterior_Height_CM"][0]),
        float(NUMS["Lower_VB_Posterior_Height_CM"][1]),
        float(NUMS["Lower_VB_Posterior_Height_CM"][2]),
        step=0.1,
    )
_section_close()

_section_open("ImageJ & DHI", "Grayscale values and disc heights")
col14, col15 = st.columns(2)
with col14:
    rsi_protrusion_gray = st.number_input(
        "Protrusion grayscale (ImageJ)",
        float(NUMS["RSI_protrusion_gray"][0]),
        float(NUMS["RSI_protrusion_gray"][1]),
        float(NUMS["RSI_protrusion_gray"][2]),
        step=1.0,
    )
with col15:
    rsi_csf_gray = st.number_input(
        "CSF grayscale (ImageJ)",
        float(NUMS["RSI_csf_gray"][0]),
        float(NUMS["RSI_csf_gray"][1]),
        float(NUMS["RSI_csf_gray"][2]),
        step=1.0,
    )
col16, col17 = st.columns(2)
with col16:
    dhi_a = st.number_input(
        "Anterior disc height (cm)",
        float(NUMS["DHI_a"][0]),
        float(NUMS["DHI_a"][1]),
        float(NUMS["DHI_a"][2]),
        step=0.1,
    )
with col17:
    dhi_c = st.number_input(
        "Posterior disc height (cm)",
        float(NUMS["DHI_c"][0]),
        float(NUMS["DHI_c"][1]),
        float(NUMS["DHI_c"][2]),
        step=0.1,
    )
col18, col19 = st.columns(2)
with col18:
    dhi_n = st.number_input(
        "Upper disc depth (cm)",
        float(NUMS["DHI_n"][0]),
        float(NUMS["DHI_n"][1]),
        float(NUMS["DHI_n"][2]),
        step=0.1,
    )
with col19:
    dhi_o = st.number_input(
        "Lower disc depth (cm)",
        float(NUMS["DHI_o"][0]),
        float(NUMS["DHI_o"][1]),
        float(NUMS["DHI_o"][2]),
        step=0.1,
    )
_section_close()

# Map Gender 0/1 to Female/Male (consistent with training phase)
gender_label = "Male" if int(Gender) == 1 else "Female"
bull_eye_value = np.nan if Bull_eye == "(missing)" else float(Bull_eye)
rsi_denom = float(rsi_csf_gray)
RSI = float(rsi_protrusion_gray) / rsi_denom if rsi_denom != 0 else np.nan
dhi_denom = float(dhi_n) + float(dhi_o)
DHI = (float(dhi_a) + float(dhi_c)) / dhi_denom if dhi_denom != 0 else np.nan

row = {
    'Age': float(Age),
    'Gender': gender_label,                       # Key: pass 'Female' / 'Male'
    'Herniated_Level': Herniated_Level,           # String will be handled by OHE inside the pipeline
    'Pfirrmann': int(Pfirrmann),
    'Iwabuchi': str(Iwabuchi),
    'Modic': str(Modic),
    'Komori': int(Komori),
    'MSU': int(MSU),
    'Spinal_canal_stenosis': str(Spinal_canal_stenosis),
    'Bull_eye': bull_eye_value,
    'SS': float(SS),
    'Upper_VB_Posterior_Height_CM': float(Upper_VB_Posterior_Height_CM),
    'Lower_VB_Posterior_Height_CM': float(Lower_VB_Posterior_Height_CM),
    'Initial_volume': float(Initial_volume),
    'RSI': float(RSI),
    'DHI': float(DHI)
}
X_input = pd.DataFrame([row], columns=FEATURES)

_section_open("Input Snapshot")
st.dataframe(X_input, use_container_width=True)
_section_close()

_section_open("Prediction", "Generate prediction and threshold metrics")
predict = st.button("Run Prediction")
if predict:
    try:
        y_proba, prob_type = _prob_from_estimator(model, X_input)
        proba = float(y_proba[0])

        if THRESHOLD_LOW is not None and THRESHOLD_HIGH is not None:
            if proba >= THRESHOLD_HIGH:
                pred_cls = 1
                label = "Reabsorption (1)"
                decision = "High likelihood of resorption (favor conservative management)"
            elif proba <= THRESHOLD_LOW:
                pred_cls = 0
                label = "Non-reabsorption (0)"
                decision = "Low likelihood of resorption (favor surgical evaluation)"
            else:
                pred_cls = None
                label = "Indeterminate"
                decision = "Indeterminate (recommend further assessment/follow-up)"
        else:
            pred_cls = int(proba >= THRESHOLD)
            label = "Reabsorption (1)" if pred_cls == 1 else "Non-reabsorption (0)"
            decision = "Single-threshold decision rule"

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Probability", f"{proba:.2%}")
        if THRESHOLD_LOW is not None and THRESHOLD_HIGH is not None:
            col_b.metric("Thresholds", f"{THRESHOLD_LOW:.4f} / {THRESHOLD_HIGH:.4f}")
        else:
            col_b.metric("Threshold", f"{THRESHOLD:.4f}")
        col_c.metric("Class", label)

        st.markdown(
            f"""
<div class="prediction-card">
  <div class="prediction-title">Decision</div>
  <div>{decision}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        metrics_summary, metrics_table = _extract_threshold_metrics(thr_cfg)
        if metrics_summary:
            st.markdown("<div class='helper'>Threshold metrics summary</div>", unsafe_allow_html=True)
            st.json(metrics_summary)
        if metrics_table is not None:
            st.markdown("<div class='helper'>Threshold metrics table</div>", unsafe_allow_html=True)
            st.dataframe(metrics_table, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Please check that the feature names and values match the training set. The pipeline already includes OHE and scaling.")
_section_close()

