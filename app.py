# app.py —— with best_model_random_forest.pkl in the same folder
import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import json

st.set_page_config(page_title="Lumbar Disc Herniation Resorption Probability Calculator (Random Forest)", layout="centered")

# === Model path: same directory as this script ===
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "best_model_random_forest.pkl"   # ← Change to your saved RF pipeline

# ---- Load saved threshold from training phase (prioritize Youden) ----
THR_PATH = BASE_DIR / "rf_thresholds.json"   # located in the same directory as app.py
DEFAULT_THR = 0.5
try:
    with open(THR_PATH, "r", encoding="utf-8") as f:
        thr_cfg = json.load(f)
    # Priority: Youden > Chosen > default 0.5
    THRESHOLD = float(thr_cfg.get("threshold_Youden",
                       thr_cfg.get("threshold_Chosen", DEFAULT_THR)))
    thr_source = "Youden (from rf_thresholds.json)"
except Exception:
    THRESHOLD = DEFAULT_THR
    thr_source = "Default 0.5 (threshold file missing)"

st.caption(f"Current classification threshold: {THRESHOLD:.4f}  —  {thr_source}")

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

model = load_model(MODEL_PATH)

st.title("Prediction of Lumbar Disc Herniation Resorption (Random Forest)")
st.caption("⚠️ For research and decision support only, not for clinical use")
st.info(f"Model successfully loaded: {MODEL_PATH}")

# === Categorical and numerical variables (consistent with training) ===
CATS = {
    "Gender": [0, 1],  # still use 0/1 in the UI but mapped to Female/Male before submission
    "Herniated_Level": ['L2/3', 'L3/4', 'L4/5', 'L5/S1'],
    "Priffmann": [1, 2, 3, 4, 5],
    "Iwabuchi": [1, 2, 3, 4, 5],
    "Modic": [0, 1, 2],
    "Komori": [1, 2, 3, 4],
    "MSU": [1, 2, 3],
    "Spinal_canal_stenosis": [0, 1],
    "Bull_eye": [1, 2, 3],
}
NUMS = {
    "Age": (18, 90, 50),
    "SS": (0.0, 60.0, 30.0),
    "Months_of_Review": (1, 36, 6),
    "Initial_volume": (0.0, 50.0, 4.0),
    "Upper_VB_Posterior_Height_CM": (1.0, 5.0, 2.5),
    "Lower_VB_Posterior_Height_CM": (1.0, 5.0, 2.4),
}
FEATURES = [
    'Age','Gender','Herniated_Level','Priffmann','Iwabuchi','Modic','Komori','MSU',
    'Spinal_canal_stenosis','Bull_eye','SS',
    'Upper_VB_Posterior_Height_CM','Lower_VB_Posterior_Height_CM',
    'Initial_volume','Months_of_Review'
]

# === Input section ===
st.subheader("Patient Characteristics Input")
col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender (Female=0, Male=1)", CATS["Gender"], index=1)
    Herniated_Level = st.selectbox("Herniated_Level", CATS["Herniated_Level"], index=2)
    Priffmann = st.selectbox("Priffmann", CATS["Priffmann"], index=1)
    Iwabuchi = st.selectbox("Iwabuchi", CATS["Iwabuchi"], index=1)
    Modic = st.selectbox("Modic", CATS["Modic"], index=0)
with col2:
    Komori = st.selectbox("Komori", CATS["Komori"], index=1)
    MSU = st.selectbox("MSU", CATS["MSU"], index=1)
    Spinal_canal_stenosis = st.selectbox("Spinal_canal_stenosis (0=No, 1=Yes)", CATS["Spinal_canal_stenosis"], index=0)
    Bull_eye = st.selectbox("Bull_eye", CATS["Bull_eye"], index=0)

col3, col4 = st.columns(2)
with col3:
    Age = st.number_input("Age", int(NUMS["Age"][0]), int(NUMS["Age"][1]), int(NUMS["Age"][2]), step=1)
    SS = st.number_input("SS", float(NUMS["SS"][0]), float(NUMS["SS"][1]), float(NUMS["SS"][2]), step=0.5)
    Months_of_Review = st.number_input("Months_of_Review",
                                       int(NUMS["Months_of_Review"][0]),
                                       int(NUMS["Months_of_Review"][1]),
                                       int(NUMS["Months_of_Review"][2]), step=1)
with col4:
    Initial_volume = st.number_input("Initial_volume (cm³)", float(NUMS["Initial_volume"][0]),
                                     float(NUMS["Initial_volume"][1]), float(NUMS["Initial_volume"][2]), step=0.1)
    Upper_VB_Posterior_Height_CM = st.number_input("Upper_VB_Posterior_Height_CM",
                                                   float(NUMS["Upper_VB_Posterior_Height_CM"][0]),
                                                   float(NUMS["Upper_VB_Posterior_Height_CM"][1]),
                                                   float(NUMS["Upper_VB_Posterior_Height_CM"][2]), step=0.1)
    Lower_VB_Posterior_Height_CM = st.number_input("Lower_VB_Posterior_Height_CM",
                                                   float(NUMS["Lower_VB_Posterior_Height_CM"][0]),
                                                   float(NUMS["Lower_VB_Posterior_Height_CM"][1]),
                                                   float(NUMS["Lower_VB_Posterior_Height_CM"][2]), step=0.1)

# Map Gender 0/1 to Female/Male — consistent with training phase
gender_label = "Male" if int(Gender) == 1 else "Female"

row = {
    'Age': Age,
    'Gender': gender_label,                       # ← Key: pass 'Female' / 'Male'
    'Herniated_Level': Herniated_Level,           # String will be handled by OHE inside the pipeline
    'Priffmann': int(Priffmann),
    'Iwabuchi': int(Iwabuchi),
    'Modic': int(Modic),
    'Komori': int(Komori),
    'MSU': int(MSU),
    'Spinal_canal_stenosis': int(Spinal_canal_stenosis),
    'Bull_eye': int(Bull_eye),
    'SS': float(SS),
    'Upper_VB_Posterior_Height_CM': float(Upper_VB_Posterior_Height_CM),
    'Lower_VB_Posterior_Height_CM': float(Lower_VB_Posterior_Height_CM),
    'Initial_volume': float(Initial_volume),
    'Months_of_Review': int(Months_of_Review)
}
X_input = pd.DataFrame([row], columns=FEATURES)

st.divider()
st.write("**Current Input Preview:**")
st.dataframe(X_input, use_container_width=True)

st.divider()
if st.button("Predict Resorption Probability"):
    try:
        proba = model.predict_proba(X_input)[:, 1][0]  # The pipeline directly accepts raw features
        pred_cls = int(proba >= THRESHOLD)
        label = "Resorption (1)" if pred_cls == 1 else "Non-resorption (0)"
        st.success(f"Predicted probability: {proba:.2%} ｜ Threshold: {THRESHOLD:.4f} ｜ Classification: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Please check that the feature names and values match the training set. The pipeline already includes OHE and scaling.")
