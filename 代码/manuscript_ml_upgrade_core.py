from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def _save_fig(fig: Any, path: str, **kwargs: Any) -> None:
    save_kwargs = {"dpi": 300, "bbox_inches": "tight"}
    save_kwargs.update(kwargs)
    fig.savefig(path, **save_kwargs)


def _device_requests_gpu(device: str | None) -> bool:
    normalized = str(device or "auto").strip().lower()
    return normalized in {"auto", "gpu", "cuda"}


def _nvidia_gpu_available() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        completed = subprocess.run(
            [exe, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.returncode == 0 and "GPU" in (completed.stdout or "")
    except Exception:
        return False


def _tabpfn_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_tree_backend_use_gpu(device: str | None) -> bool:
    return _device_requests_gpu(device) and _nvidia_gpu_available()


def _resolve_tabpfn_device(device: str | None) -> str:
    normalized = str(device or "auto").strip().lower()
    if normalized == "cpu":
        return "cpu"
    if _tabpfn_cuda_available():
        return "cuda"
    return "cpu"


def _acceleration_summary(device: str | None) -> dict[str, Any]:
    tree_gpu = _resolve_tree_backend_use_gpu(device)
    tabpfn_device = _resolve_tabpfn_device(device)
    return {
        "requested_device": str(device or "auto"),
        "nvidia_gpu_detected": bool(_nvidia_gpu_available()),
        "tree_models_use_gpu": bool(tree_gpu),
        "tabpfn_device": tabpfn_device,
        "tabpfn_cuda_ready": bool(tabpfn_device == "cuda"),
    }


DEFAULT_ENSEMBLE_BASE_MODELS = (
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabPFN",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return expit(np.clip(z, -50, 50))


def _prob_from_estimator(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float), "predict_proba"
        return proba.ravel().astype(float), "predict_proba_ravel"
    if hasattr(estimator, "decision_function"):
        return _sigmoid(estimator.decision_function(X)), "sigmoid(decision_function)"
    return np.asarray(estimator.predict(X), dtype=float).ravel(), "predict_only"


def _predict_from_probability(y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    return (y_proba >= float(threshold)).astype(int)


def _calculate_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    y_pred = _predict_from_probability(y_proba, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc = np.nan
    auprc = np.nan
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_proba))
        auprc = float(average_precision_score(y_true, y_proba))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0
    youden = sens + spec - 1.0

    return {
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Accuracy": float(acc),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1": float(f1),
        "Youden": float(youden),
        "AUC": auc,
        "AUPRC": auprc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def _cohort_probability_table(
    df_source: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    out = df_source.copy().reset_index(drop=True)
    out["TrueLabel"] = np.asarray(y_true).astype(int)
    out["PredProb"] = np.asarray(y_proba, dtype=float)
    out["PredLabel"] = _predict_from_probability(y_proba, threshold)
    return out


def _threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = _calculate_binary_metrics(y_true, y_proba, threshold=threshold)
    return {
        "threshold": float(threshold),
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
        "Accuracy": metrics["Accuracy"],
        "PPV": metrics["PPV"],
        "NPV": metrics["NPV"],
        "F1": metrics["F1"],
        "Youden": metrics["Youden"],
    }


def compute_threshold_bundle(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    fpr, tpr, thr = roc_curve(y_true, y_proba)

    if len(thr) > 1:
        j_stat = tpr - fpr
        idx = int(np.argmax(j_stat[1:]) + 1)
        thr_youden = float(thr[idx])
    else:
        thr_youden = float(thr[0])

    sens_target = 0.90
    cand_sens = np.where(tpr >= sens_target)[0]
    thr_sens90 = float(thr[cand_sens][np.argmin(fpr[cand_sens])]) if len(cand_sens) else thr_youden

    spec = 1 - fpr
    spec_target = 0.90
    cand_spec = np.where(spec >= spec_target)[0]
    thr_spec90 = float(thr[cand_spec][np.argmax(tpr[cand_spec])]) if len(cand_spec) else thr_youden

    prec, rec, thr_pr = precision_recall_curve(y_true, y_proba)
    if len(thr_pr):
        f1_vals = []
        for t in thr_pr:
            y_pred = _predict_from_probability(y_proba, threshold=float(t))
            f1_vals.append(f1_score(y_true, y_pred) if y_pred.sum() else 0.0)
        thr_f1 = float(thr_pr[int(np.argmax(f1_vals))])
    else:
        thr_f1 = thr_youden

    chosen_threshold = thr_youden
    thr_low = min(thr_sens90, thr_spec90)
    thr_high = max(thr_sens90, thr_spec90)

    return {
        "threshold_source": "oof",
        "threshold_Youden": float(thr_youden),
        "threshold_Sens90": float(thr_sens90),
        "threshold_Spec90": float(thr_spec90),
        "threshold_MaxF1": float(thr_f1),
        "threshold_Chosen": float(chosen_threshold),
        "threshold_low": float(thr_low),
        "threshold_high": float(thr_high),
        "SENS_TARGET": float(sens_target),
        "SPEC_TARGET": float(spec_target),
        "metrics_Youden": _threshold_metrics(y_true, y_proba, thr_youden),
        "metrics_Sens90": _threshold_metrics(y_true, y_proba, thr_sens90),
        "metrics_Spec90": _threshold_metrics(y_true, y_proba, thr_spec90),
        "metrics_MaxF1": _threshold_metrics(y_true, y_proba, thr_f1),
        "metrics_low": _threshold_metrics(y_true, y_proba, thr_low),
        "metrics_high": _threshold_metrics(y_true, y_proba, thr_high),
    }


def _resolve_ranking_threshold_strategy(
    threshold_bundle: dict[str, Any],
    strategy: str,
) -> tuple[str, float]:
    normalized = str(strategy or "Youden").strip().lower().replace("_", "")
    mapping = {
        "youden": ("Youden", "threshold_Youden"),
        "maxf1": ("MaxF1", "threshold_MaxF1"),
        "sens90": ("Sens90", "threshold_Sens90"),
        "spec90": ("Spec90", "threshold_Spec90"),
        "low": ("low", "threshold_low"),
        "high": ("high", "threshold_high"),
    }
    label, key = mapping.get(normalized, ("Youden", "threshold_Youden"))
    return label, float(threshold_bundle[key])


def _compute_oof_metric_views(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    config: UpgradeConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], str, float]:
    default_metrics = _calculate_binary_metrics(y_true, y_proba, threshold=0.5)
    threshold_bundle = compute_threshold_bundle(y_true, y_proba)
    threshold_label, threshold_value = _resolve_ranking_threshold_strategy(
        threshold_bundle=threshold_bundle,
        strategy=config.ranking_threshold_strategy,
    )
    ranking_metrics = _calculate_binary_metrics(y_true, y_proba, threshold=threshold_value)
    return (
        default_metrics,
        ranking_metrics,
        threshold_bundle,
        threshold_label,
        float(threshold_value),
    )


def _build_completed_result(
    name: str,
    family: str,
    oof_proba: np.ndarray,
    external_proba: np.ndarray,
    y_retro: np.ndarray,
    y_external: np.ndarray,
    config: UpgradeConfig,
    *,
    final_model: Any,
    training_log: dict[str, Any],
    probability_type: str,
) -> CandidateResult:
    (
        oof_metrics,
        ranking_oof_metrics,
        oof_threshold_bundle,
        ranking_threshold_label,
        ranking_threshold,
    ) = _compute_oof_metric_views(y_retro, oof_proba, config)
    external_metrics = _calculate_binary_metrics(y_external, external_proba, threshold=0.5)
    return CandidateResult(
        name=name,
        family=family,
        status="completed",
        oof_proba=oof_proba,
        external_proba=external_proba,
        oof_metrics=oof_metrics,
        ranking_oof_metrics=ranking_oof_metrics,
        oof_threshold_bundle=oof_threshold_bundle,
        ranking_threshold_label=ranking_threshold_label,
        ranking_threshold=float(ranking_threshold),
        external_metrics=external_metrics,
        final_model=final_model,
        training_log=training_log,
        probability_type=probability_type,
    )


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer, source_cols: list[str]) -> list[str]:
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names.extend(list(transformer.get_feature_names_out(cols)))
            except Exception:
                names.extend(list(transformer.get_feature_names_out()))
        else:
            if isinstance(cols, (list, tuple, np.ndarray)):
                names.extend([str(c) for c in cols])
            else:
                names.append(str(cols))

    used = set()
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, (list, tuple, np.ndarray)):
            used.update(cols)
    passthrough_cols = [c for c in source_cols if c not in used]
    names.extend(list(map(str, passthrough_cols)))
    return names


@dataclass
class UpgradeConfig:
    seed: int = 42
    validation_mode: str = "oof_external"
    oof_folds: int = 5
    inner_cv_folds: int = 3
    bayes_n_iter: int = 12
    enable_tabpfn: bool = True
    tabpfn_search_enabled: bool = True
    enable_voting: bool = True
    enable_stacking: bool = True
    ensemble_base_models: tuple[str, ...] = DEFAULT_ENSEMBLE_BASE_MODELS
    shap_mode: str = "champion_only"
    device: str = "auto"
    model_version: str = "v2.5"
    tabpfn_model_cache_dir: str | None = None
    champion_auc_threshold: float = 0.70
    ranking_threshold_strategy: str = "Youden"
    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Sensitivity": 0.25,
            "AUC": 0.25,
            "Specificity": 0.15,
            "F1": 0.15,
            "PPV": 0.10,
            "AUPRC": 0.10,
        }
    )


@dataclass
class BenchmarkPaths:
    ml_dir: Path
    metrics_dir: Path
    models_dir: Path
    audit_dir: Path
    deploy_dir: Path
    shap_dir: Path
    figures_dir: Path

    def ensure(self) -> None:
        for path in [
            self.ml_dir,
            self.metrics_dir,
            self.models_dir,
            self.audit_dir,
            self.deploy_dir,
            self.shap_dir,
            self.figures_dir,
            self.metrics_dir / "oof_predictions",
            self.metrics_dir / "external_predictions",
            self.audit_dir / "training_logs",
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class CandidateResult:
    name: str
    family: str
    status: str
    oof_proba: np.ndarray | None = None
    external_proba: np.ndarray | None = None
    oof_metrics: dict[str, float] = field(default_factory=dict)
    ranking_oof_metrics: dict[str, float] = field(default_factory=dict)
    oof_threshold_bundle: dict[str, Any] = field(default_factory=dict)
    ranking_threshold_label: str = "Youden"
    ranking_threshold: float = 0.5
    external_metrics: dict[str, float] = field(default_factory=dict)
    final_model: Any = None
    training_log: dict[str, Any] = field(default_factory=dict)
    probability_type: str = "predict_proba"


def _coerce_bull_eye_to_nominal(df: pd.DataFrame) -> pd.DataFrame:
    X_df = pd.DataFrame(df).copy()
    if "Bull_eye" in X_df.columns:
        bull_eye = pd.to_numeric(X_df["Bull_eye"], errors="coerce").round()
        X_df["Bull_eye"] = bull_eye.astype("Int64").astype(str).replace("<NA>", np.nan)
    return X_df


def _normalize_gender(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    values = values.replace(
        {
            "female": "Female",
            "FEMALE": "Female",
            "F": "Female",
            "male": "Male",
            "MALE": "Male",
            "M": "Male",
            "nan": np.nan,
            "NaN": np.nan,
            "": np.nan,
        }
    )
    return values


def clean_modeling_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    continuous_vars: list[str],
    ordinal_vars: list[str],
    nominal_vars: list[str],
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in continuous_vars + ordinal_vars:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in ["Iwabuchi", "Modic", "Spinal_canal_stenosis"]:
        if col in X.columns:
            tmp = pd.to_numeric(X[col], errors="coerce")
            X[col] = tmp.apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)

    if "Gender" in X.columns:
        X["Gender"] = _normalize_gender(X["Gender"])

    # Bull_eye is imputed downstream by the model pipeline; keep missing as NaN
    # so the imputers can learn only from training folds and avoid leakage.
    if "Bull_eye" in X.columns:
        X["Bull_eye"] = pd.to_numeric(X["Bull_eye"], errors="coerce")

    for col in nominal_vars:
        if col in X.columns and col not in {"Bull_eye", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Gender"}:
            X[col] = X[col].astype(str).replace({"nan": np.nan, "NaN": np.nan, "": np.nan})
    return X


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _bull_eye_predictors(
    continuous_vars: list[str],
    ordinal_vars: list[str],
    nominal_vars: list[str],
) -> list[str]:
    """Return the canonical predictor list used for Bull_eye imputation."""
    candidates = list(continuous_vars) + list(ordinal_vars) + [
        "Gender",
        "Herniated_Level",
        "Iwabuchi",
        "Modic",
        "Spinal_canal_stenosis",
    ]
    return _unique_preserve_order(candidates)


class BullEyeImputer:
    def __init__(
        self,
        predictors: list[str],
        seed: int = 42,
        n_estimators: int = 100,
        lasso_C: float = 0.5,
        min_features: int = 3,
        max_features: int = 10,
        output_dir: str | None = None,
    ) -> None:
        self.predictors = predictors
        self.seed = seed
        self.n_estimators = n_estimators
        self.lasso_C = lasso_C
        self.min_features = min_features
        self.max_features = max_features
        self.output_dir = output_dir
        self.lasso = None
        self.clf = None
        self.fallback_value = None
        self.all_cols: list[str] | None = None
        self.selected_cols: list[str] | None = None
        self.selected_cols_original: list[str] | None = None
        self.selected_feature_indices: list[int] | None = None
        self.encoded_feature_names: list[str] | None = None
        self.selected_feature_names: list[str] | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.num_fill_values: dict[str, float] = {}
        self.ohe = None
        self.best_params: dict[str, Any] | None = None
        self.best_cv_score: float | None = None
        self.original_distribution = None
        self.imputed_distribution = None
        self.feature_importance = None
        self.lasso_coefficients = None

    def _prepare_data_for_lasso(self, X: pd.DataFrame, mask: pd.Series) -> tuple[np.ndarray, pd.Series, list[str]]:
        X_be = X.loc[mask, self.all_cols].copy()
        y_be = X.loc[mask, "Bull_eye"].copy().astype(int)

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for col in self.all_cols:
            if pd.api.types.is_numeric_dtype(X_be[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        X_num_df = X_be[numeric_cols].apply(pd.to_numeric, errors="coerce")
        self.num_fill_values = X_num_df.median(numeric_only=True).to_dict()
        X_num_df = X_num_df.fillna(self.num_fill_values)

        if categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_df = X_be[categorical_cols].fillna("__MISSING__").astype(str)
            cat_encoded = ohe.fit_transform(cat_df)
            cat_feature_names = ohe.get_feature_names_out(categorical_cols)
            self.ohe = ohe
            self.categorical_cols = categorical_cols
            self.numeric_cols = numeric_cols
            X_numeric = X_num_df.values
            X_encoded = np.hstack([X_numeric, cat_encoded])
            all_feature_names = numeric_cols + list(cat_feature_names)
        else:
            X_encoded = X_num_df.values
            all_feature_names = list(numeric_cols)
            self.ohe = None
            self.categorical_cols = []
            self.numeric_cols = numeric_cols

        return X_encoded, y_be, all_feature_names

    def _encode_with_fitted_ohe(self, X: pd.DataFrame) -> np.ndarray:
        X_part = X.loc[:, self.all_cols].copy()
        if self.numeric_cols:
            X_num_df = X_part[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            if self.num_fill_values:
                X_num_df = X_num_df.fillna(self.num_fill_values)
            X_numeric = X_num_df.values
        else:
            X_numeric = np.empty((len(X_part), 0))

        if self.ohe is not None and self.categorical_cols:
            X_cat_df = X_part[self.categorical_cols].fillna("__MISSING__").astype(str)
            X_cat = self.ohe.transform(X_cat_df)
            return np.hstack([X_numeric, X_cat])
        return X_numeric

    def _select_features_with_lasso(
        self,
        X_encoded: np.ndarray,
        y: pd.Series,
        feature_names: list[str],
    ) -> list[str]:
        print(f"\n[INFO] ===== STAGE 1: LASSO Feature Selection (Dimensionality Reduction) =====")
        print(f"[INFO] Input: {len(feature_names)} One-Hot encoded features from {len(self.all_cols)} original variables")
        print(f"[INFO] Training samples: {len(y)}")
        print(f"[INFO] Goal: Select most predictive features for Random Forest")
        lasso = LogisticRegression(
            penalty="l1",
            solver="saga",
            multi_class="multinomial",
            C=self.lasso_C,
            max_iter=5000,
            random_state=self.seed,
        )
        lasso.fit(X_encoded, y)
        self.lasso = lasso
        coef_matrix = lasso.coef_
        non_zero_mask = (coef_matrix != 0).any(axis=0)
        selected_indices = np.where(non_zero_mask)[0]

        if len(selected_indices) < self.min_features:
            coef_abs_sum = np.abs(coef_matrix).sum(axis=0)
            selected_indices = np.argsort(coef_abs_sum)[-self.min_features :]

        if len(selected_indices) > self.max_features:
            coef_abs_sum = np.abs(coef_matrix).sum(axis=0)
            selected_coef_abs_sum = coef_abs_sum[selected_indices]
            top_within_selected = np.argsort(selected_coef_abs_sum)[-self.max_features :]
            selected_indices = selected_indices[top_within_selected]
            print(f"[INFO] LASSO selected {len(np.where(non_zero_mask)[0])} features, limiting to top {self.max_features}")

        selected_feature_names = [feature_names[i] for i in selected_indices]
        print(f"[OK] LASSO selected {len(selected_feature_names)} features:")
        for feat in selected_feature_names:
            print(f"       - {feat}")
        return selected_feature_names

    def fit(self, X_train: pd.DataFrame, y_train: Any = None) -> "BullEyeImputer":
        if "Bull_eye" not in X_train.columns:
            return self

        self.original_distribution = X_train["Bull_eye"].value_counts().sort_index()
        self.all_cols = [c for c in self.predictors if c in X_train.columns]
        train_mask = X_train["Bull_eye"].notna()
        if len(self.all_cols) == 0 or int(train_mask.sum()) < 20:
            mode = X_train.loc[train_mask, "Bull_eye"].dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            self.selected_cols_original = self.all_cols
            self.selected_cols = self.all_cols
            print(f"[WARN] Bull_eye imputer: using fallback mode={self.fallback_value} (insufficient data: {train_mask.sum()} samples)")
            return self

        try:
            X_encoded, y_be, feature_names = self._prepare_data_for_lasso(X_train, train_mask)
            self.encoded_feature_names = feature_names
            self.selected_feature_names = self._select_features_with_lasso(X_encoded, y_be, feature_names)
            self.selected_cols_original = self._map_selected_to_original_cols(self.selected_feature_names)
            self.lasso_coefficients = pd.DataFrame({
                "feature": feature_names,
                "coef_abs_sum": np.abs(self.lasso.coef_).sum(axis=0),
                "selected": (self.lasso.coef_ != 0).any(axis=0),
            }).sort_values("coef_abs_sum", ascending=False)
        except Exception:
            X_encoded, _, feature_names = self._prepare_data_for_lasso(X_train, train_mask)
            self.encoded_feature_names = feature_names
            self.selected_feature_names = feature_names
            self.selected_cols_original = self.all_cols
            self.lasso_coefficients = pd.DataFrame({
                "feature": feature_names,
                "coef_abs_sum": np.nan,
                "selected": False,
            })
            print(f"[WARN] LASSO feature selection failed; falling back to all encoded predictors")

        self.selected_feature_indices = [
            self.encoded_feature_names.index(name)
            for name in self.selected_feature_names
            if name in self.encoded_feature_names
        ]
        if not self.selected_feature_indices:
            self.selected_feature_indices = list(range(len(self.encoded_feature_names)))

        self.selected_cols = self.selected_cols_original

        print(f"\n[INFO] ===== STAGE 2: Random Forest Prediction (using LASSO-selected features) =====")
        print(
            f"[INFO] Input: {len(self.selected_feature_names)} OHE features "
            f"({len(self.selected_cols_original)} original vars) selected from "
            f"{len(self.encoded_feature_names)} OHE features ({len(self.all_cols)} original vars)"
        )
        print(f"[INFO] Training Random Forest for final Bull_eye imputation...")

        X_train_be = self._encode_with_fitted_ohe(X_train.loc[train_mask, self.all_cols])
        X_train_be = X_train_be[:, self.selected_feature_indices]
        y_train_be = X_train.loc[train_mask, "Bull_eye"].copy().astype(int)
        if y_train_be.nunique() < 2:
            mode = y_train_be.dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            self.selected_cols = self.selected_cols_original
            print(f"[WARN] Bull_eye imputer: using fallback mode={self.fallback_value} (low variance)")
            return self

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 3],
        }
        print(f"[INFO] Running GridSearchCV to optimize Random Forest hyperparameters...")
        print(f"[INFO] Search space: {len(list(ParameterGrid(param_grid)))} combinations")
        grid = GridSearchCV(
            RandomForestClassifier(
                random_state=self.seed,
                n_jobs=1,
                class_weight="balanced",
            ),
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            scoring="accuracy",
            n_jobs=1,
        )
        grid.fit(X_train_be, y_train_be)
        self.clf = grid.best_estimator_
        self.best_params = dict(grid.best_params_)
        self.best_cv_score = float(grid.best_score_)
        self.feature_importance = pd.DataFrame({
            "feature": self.selected_feature_names,
            "importance": self.clf.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"[INFO] Best hyperparameters found:")
        for param, value in self.best_params.items():
            print(f"       - {param}: {value}")
        print(f"[INFO] Best 5-fold CV accuracy during search: {self.best_cv_score:.4f}")
        cv_scores = cross_val_score(
            self.clf,
            X_train_be,
            y_train_be,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            scoring="accuracy",
        )
        print(f"[INFO] Final imputation model (Random Forest) 5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"[OK] Bull_eye imputer pipeline completed (trained on {train_mask.sum()} samples)")
        print(f"[INFO] Top 5 important features for Bull_eye prediction:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"       - {row['feature']}: {row['importance']:.4f}")
        return self

    def transform(self, X: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
        if "Bull_eye" not in X.columns:
            return X
        X_out = X.copy()
        miss_mask = X_out["Bull_eye"].isna()
        if int(miss_mask.sum()) == 0:
            return X_out

        if self.clf is None:
            fillv = self.fallback_value if self.fallback_value is not None else 1
            X_out.loc[miss_mask, "Bull_eye"] = int(np.clip(fillv, 1, 3))
            pred_counts = pd.Series([int(np.clip(fillv, 1, 3))] * int(miss_mask.sum())).value_counts().sort_index()
            if dataset_name == "Train":
                self.imputed_distribution = pred_counts
            print(f"[INFO] Bull_eye ({dataset_name}): filled {miss_mask.sum()} missing values with fallback={fillv}")
            return X_out

        X_all_miss = self._encode_with_fitted_ohe(X_out.loc[miss_mask, self.all_cols])
        X_miss = X_all_miss[:, self.selected_feature_indices]
        pred = self.clf.predict(X_miss).astype(int)
        pred = np.clip(pred, 1, 3)
        X_out.loc[miss_mask, "Bull_eye"] = pred
        pred_counts = pd.Series(pred).value_counts().sort_index()
        print(f"[INFO] Bull_eye ({dataset_name}): imputed {miss_mask.sum()} missing values")
        print(f"[INFO] Predicted distribution: {dict(pred_counts)}")
        if dataset_name == "Train":
            self.imputed_distribution = pred_counts
        return X_out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train, dataset_name="Train")

    def visualize_diagnostics(self) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        if self.lasso_coefficients is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_features = self.lasso_coefficients.head(20)
            colors = ["green" if s else "gray" for s in top_features["selected"]]
            ax.barh(range(len(top_features)), top_features["coef_abs_sum"], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features["feature"])
            ax.set_xlabel("Absolute Coefficient Sum")
            ax.set_title("LASSO Feature Selection: Top 20 Features")
            ax.legend(
                [plt.Rectangle((0, 0), 1, 1, color="green"), plt.Rectangle((0, 0), 1, 1, color="gray")],
                ["Selected", "Not Selected"],
            )
            plt.tight_layout()
            _save_fig(plt.gcf(), os.path.join(self.output_dir, "bull_eye_lasso_coefficients.tiff"), dpi=300)
            plt.close()

        if self.feature_importance is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(self.feature_importance)), self.feature_importance["importance"])
            ax.set_yticks(range(len(self.feature_importance)))
            ax.set_yticklabels(self.feature_importance["feature"])
            ax.set_xlabel("Feature Importance")
            ax.set_title("Random Forest Feature Importance for Bull_eye Prediction")
            plt.tight_layout()
            _save_fig(plt.gcf(), os.path.join(self.output_dir, "bull_eye_rf_importance.tiff"), dpi=300)
            plt.close()

        if self.original_distribution is not None and self.imputed_distribution is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            self.original_distribution.plot(kind="bar", ax=axes[0], color="steelblue")
            axes[0].set_title("Original Distribution (Observed)")
            axes[0].set_xlabel("Bull_eye")
            axes[0].set_ylabel("Count")

            self.imputed_distribution.plot(kind="bar", ax=axes[1], color="coral")
            axes[1].set_title("Imputed Distribution (Predicted)")
            axes[1].set_xlabel("Bull_eye")
            axes[1].set_ylabel("Count")

            original_pct = self.original_distribution / self.original_distribution.sum() * 100
            imputed_pct = self.imputed_distribution / self.imputed_distribution.sum() * 100
            x = np.arange(len(original_pct))
            width = 0.35
            axes[2].bar(x - width / 2, original_pct, width, label="Original", color="steelblue")
            axes[2].bar(x + width / 2, imputed_pct, width, label="Imputed", color="coral")
            axes[2].set_title("Percentage Comparison")
            axes[2].set_xlabel("Bull_eye")
            axes[2].set_ylabel("Percentage (%)")
            axes[2].set_xticks(x)
            axes[2].set_xticklabels([f"{i}" for i in range(1, 4)])
            axes[2].legend()

            plt.tight_layout()
            _save_fig(plt.gcf(), os.path.join(self.output_dir, "bull_eye_distribution_comparison.tiff"), dpi=300)
            plt.close()

    def diagnostic_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "predictors": list(self.predictors),
            "all_cols": list(self.all_cols or []),
            "selected_cols_original": list(self.selected_cols_original or []),
            "selected_feature_names": list(self.selected_feature_names or []),
            "selected_feature_indices": [int(v) for v in (self.selected_feature_indices or [])],
            "fallback_value": None if self.fallback_value is None else int(self.fallback_value),
            "n_observed_rows": int(self.original_distribution.sum()) if self.original_distribution is not None else None,
            "n_imputed_rows": int(self.imputed_distribution.sum()) if self.imputed_distribution is not None else None,
        }
        if self.original_distribution is not None:
            summary["original_distribution"] = {
                str(k): int(v) for k, v in self.original_distribution.items()
            }
        if self.imputed_distribution is not None:
            summary["imputed_distribution"] = {
                str(k): int(v) for k, v in self.imputed_distribution.items()
            }
        if self.feature_importance is not None:
            summary["top_feature_importance"] = [
                {"feature": str(row["feature"]), "importance": float(row["importance"])}
                for _, row in self.feature_importance.head(10).iterrows()
            ]
        if self.lasso_coefficients is not None:
            summary["top_lasso_features"] = [
                {
                    "feature": str(row["feature"]),
                    "coef_abs_sum": float(row["coef_abs_sum"]),
                    "selected": bool(row["selected"]),
                }
                for _, row in self.lasso_coefficients.head(10).iterrows()
            ]
        return summary


class BullEyeImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        predictors: list[str],
        seed: int = 42,
        n_estimators: int = 100,
        lasso_C: float = 0.5,
        min_features: int = 3,
        max_features: int = 10,
        output_dir: str | None = None,
    ) -> None:
        self.predictors = predictors
        self.seed = seed
        self.n_estimators = n_estimators
        self.lasso_C = lasso_C
        self.min_features = min_features
        self.max_features = max_features
        self.output_dir = output_dir

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BullEyeImputerTransformer":
        X_df = pd.DataFrame(X).copy()
        self.imputer_ = BullEyeImputer(
            predictors=self.predictors,
            seed=self.seed,
            n_estimators=self.n_estimators,
            lasso_C=self.lasso_C,
            min_features=self.min_features,
            max_features=self.max_features,
            output_dir=self.output_dir,
        )
        if self.output_dir is None:
            with redirect_stdout(io.StringIO()):
                self.imputer_.fit(X_df, y_train=y)
        else:
            self.imputer_.fit(X_df, y_train=y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        if self.output_dir is None:
            with redirect_stdout(io.StringIO()):
                X_df = self.imputer_.transform(X_df)
        else:
            X_df = self.imputer_.transform(X_df)
        return _coerce_bull_eye_to_nominal(X_df)


@dataclass
class SharedBullEyeFoldData:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_train: pd.DataFrame
    X_val: pd.DataFrame


@dataclass
class SharedBullEyeImputationBundle:
    full_transformer: BullEyeImputerTransformer | None
    X_retro_full: pd.DataFrame
    X_external_full: pd.DataFrame
    folds: list[SharedBullEyeFoldData] = field(default_factory=list)


class SharedBullEyeModelWrapper(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model: Any,
        bull_eye_transformer: BullEyeImputerTransformer | None = None,
    ) -> None:
        self.model = model
        self.bull_eye_transformer = bull_eye_transformer

    def _resolved_model(self) -> Any:
        return getattr(self, "model_", self.model)

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        if self.bull_eye_transformer is not None and "Bull_eye" in X_df.columns:
            X_df = self.bull_eye_transformer.transform(X_df)
        else:
            X_df = _coerce_bull_eye_to_nominal(X_df)
        return X_df

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SharedBullEyeModelWrapper":
        model = clone(self.model)
        model.fit(self.prepare_features(X), y)
        self.model_ = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._resolved_model().predict_proba(self.prepare_features(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._resolved_model().predict(self.prepare_features(X))

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        model = self._resolved_model()
        if not hasattr(model, "decision_function"):
            raise AttributeError("Wrapped model does not support decision_function.")
        return model.decision_function(self.prepare_features(X))


def _fit_bull_eye_transformer(
    X_train: pd.DataFrame,
    predictors: list[str],
    seed: int,
    output_dir: Path | str | None = None,
) -> BullEyeImputerTransformer:
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    transformer = BullEyeImputerTransformer(
        predictors=predictors,
        seed=seed,
        output_dir=output_dir,
    )
    transformer.fit(X_train)
    return transformer


def _build_shared_bull_eye_imputation_bundle(
    X_retro: pd.DataFrame,
    y_retro: np.ndarray,
    X_external: pd.DataFrame,
    continuous_vars: list[str],
    ordinal_vars: list[str],
    nominal_vars: list[str],
    paths: BenchmarkPaths,
    config: UpgradeConfig,
) -> SharedBullEyeImputationBundle:
    cv_splitter = StratifiedKFold(
        n_splits=int(config.oof_folds),
        shuffle=True,
        random_state=config.seed,
    )
    predictors = _bull_eye_predictors(continuous_vars, ordinal_vars, nominal_vars)

    if "Bull_eye" not in X_retro.columns:
        folds = []
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_retro, y_retro), start=1):
            folds.append(
                SharedBullEyeFoldData(
                    fold=fold,
                    train_idx=np.asarray(train_idx, dtype=int),
                    val_idx=np.asarray(val_idx, dtype=int),
                    X_train=_coerce_bull_eye_to_nominal(X_retro.iloc[train_idx].copy()),
                    X_val=_coerce_bull_eye_to_nominal(X_retro.iloc[val_idx].copy()),
                )
            )
        return SharedBullEyeImputationBundle(
            full_transformer=None,
            X_retro_full=_coerce_bull_eye_to_nominal(X_retro.copy()),
            X_external_full=_coerce_bull_eye_to_nominal(X_external.copy()),
            folds=folds,
        )

    shared_dir = paths.audit_dir / "bull_eye_imputation_diagnostics" / "shared_preimpute"
    full_transformer = _fit_bull_eye_transformer(
        X_train=X_retro,
        predictors=predictors,
        seed=config.seed,
        output_dir=shared_dir,
    )
    X_retro_full = full_transformer.transform(X_retro.copy())
    X_external_full = full_transformer.transform(X_external.copy())

    folds: list[SharedBullEyeFoldData] = []
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_retro, y_retro), start=1):
        X_train_fold = X_retro.iloc[train_idx].copy()
        X_val_fold = X_retro.iloc[val_idx].copy()
        fold_transformer = _fit_bull_eye_transformer(
            X_train=X_train_fold,
            predictors=predictors,
            seed=config.seed,
            output_dir=None,
        )
        folds.append(
            SharedBullEyeFoldData(
                fold=fold,
                train_idx=np.asarray(train_idx, dtype=int),
                val_idx=np.asarray(val_idx, dtype=int),
                X_train=fold_transformer.transform(X_train_fold),
                X_val=fold_transformer.transform(X_val_fold),
            )
        )

    return SharedBullEyeImputationBundle(
        full_transformer=full_transformer,
        X_retro_full=X_retro_full,
        X_external_full=X_external_full,
        folds=folds,
    )


class TabPFNRawClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        feature_cols: list[str],
        continuous_vars: list[str],
        ordinal_vars: list[str],
        nominal_vars: list[str],
        seed: int = 42,
        device: str = "auto",
        model_version: str = "v2.5",
        balance_probabilities: bool = True,
        n_estimators: int = 32,
        softmax_temperature: float = 0.7,
        average_before_softmax: bool = True,
        model_cache_dir: str | None = None,
        bull_eye_mode: str = "internal",
    ) -> None:
        self.feature_cols = feature_cols
        self.continuous_vars = continuous_vars
        self.ordinal_vars = ordinal_vars
        self.nominal_vars = nominal_vars
        self.seed = seed
        self.device = device
        self.model_version = model_version
        self.balance_probabilities = balance_probabilities
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.model_cache_dir = model_cache_dir
        self.bull_eye_mode = bull_eye_mode
        self.bull_eye_imputer_ = None
        self.bull_eye_predictors_ = _bull_eye_predictors(
            continuous_vars=self.continuous_vars,
            ordinal_vars=self.ordinal_vars,
            nominal_vars=self.nominal_vars,
        )

    def _prepare_X(self, X: pd.DataFrame, *, fit_mode: bool = False) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.loc[:, self.feature_cols].copy()

        if "Bull_eye" in X_df.columns:
            if self.bull_eye_mode == "preimputed":
                if X_df["Bull_eye"].isna().any():
                    raise RuntimeError("Bull_eye contains missing values in pre-imputed mode.")
                X_df = _coerce_bull_eye_to_nominal(X_df)
            elif fit_mode:
                self.bull_eye_imputer_ = BullEyeImputerTransformer(
                    predictors=self.bull_eye_predictors_,
                    seed=self.seed,
                    output_dir=None,
                )
                X_df = self.bull_eye_imputer_.fit_transform(X_df)
            elif self.bull_eye_imputer_ is not None:
                X_df = self.bull_eye_imputer_.transform(X_df)
            else:
                raise RuntimeError("Bull_eye imputer is not initialized before prediction.")

        for col in self.continuous_vars + self.ordinal_vars:
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        for col in self.nominal_vars:
            if col not in X_df.columns:
                continue
            X_df[col] = X_df[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)

        if "Gender" in X_df.columns:
            X_df["Gender"] = _normalize_gender(X_df["Gender"])
        return X_df

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabPFNRawClassifier":
        from tabpfn import TabPFNClassifier

        os.environ["TABPFN_MODEL_VERSION"] = self.model_version
        cache_dir = _discover_tabpfn_cache_dir(self.model_cache_dir, self.model_version)
        resolved_device = _resolve_tabpfn_device(self.device)

        if cache_dir is not None:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = cache_dir

        X_prepared = self._prepare_X(X, fit_mode=True)
        y_arr = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = int(X_prepared.shape[1])
        self.feature_names_in_ = np.asarray(list(X_prepared.columns), dtype=object)

        # Try AutoTabPFNClassifier (Post-Hoc Ensemble) for best performance
        try:
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
            from sklearn.preprocessing import OrdinalEncoder as _OE
            print("[INFO] TabPFN: running AutoTabPFN Post-Hoc Ensemble (max_time=300s)...")

            X_hpo = X_prepared.copy()
            str_cols = X_hpo.select_dtypes(include=["object", "category"]).columns.tolist()
            if str_cols:
                oe = _OE(handle_unknown="use_encoded_value", unknown_value=-1)
                X_hpo[str_cols] = oe.fit_transform(X_hpo[str_cols].fillna("__NA__"))

            auto_clf = AutoTabPFNClassifier(
                device=resolved_device,
                max_time=300,
            )
            auto_clf.fit(X_hpo, y_arr)
            self.model_ = auto_clf
            self.hpo_encoder_ = oe if str_cols else None
            self.hpo_str_cols_ = str_cols
            self.resolved_device_ = resolved_device
            print("[OK] AutoTabPFN Post-Hoc Ensemble completed.")
            return self
        except Exception as auto_exc:
            print(f"[WARN] AutoTabPFNClassifier failed ({auto_exc}), falling back to manual config.")

        # Fallback: manual configuration
        model_kwargs: dict[str, Any] = {
            "device": resolved_device,
            "random_state": self.seed,
            "balance_probabilities": self.balance_probabilities,
            "n_estimators": self.n_estimators,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
        }
        if cache_dir is not None:
            checkpoint_path = Path(cache_dir) / _checkpoint_name_for_version(self.model_version)
            if checkpoint_path.exists():
                model_kwargs["model_path"] = str(checkpoint_path)
        try:
            self.model_ = TabPFNClassifier(**model_kwargs)
            self.model_.fit(X_prepared, y_arr)
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "accept its terms" in message.lower():
                raise RuntimeError(
                    "TabPFN model access is blocked by Hugging Face gated-model permissions. "
                    "Accept the terms at https://huggingface.co/Prior-Labs/tabpfn_2_5 and "
                    "configure a local HF token before rerunning."
                ) from exc
            raise
        self.resolved_device_ = resolved_device
        return self

    def _encode_for_hpo(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same ordinal encoding used during HPO fit."""
        encoder = getattr(self, "hpo_encoder_", None)
        str_cols = getattr(self, "hpo_str_cols_", [])
        if encoder is not None and str_cols:
            X = X.copy()
            X[str_cols] = encoder.transform(X[str_cols].fillna("__NA__"))
        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare_X(X, fit_mode=False)
        if getattr(self, "hpo_encoder_", None) is not None:
            X_prepared = self._encode_for_hpo(X_prepared)
        proba = self.model_.predict_proba(X_prepared)
        return np.asarray(proba, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare_X(X, fit_mode=False)
        if getattr(self, "hpo_encoder_", None) is not None:
            X_prepared = self._encode_for_hpo(X_prepared)
        return self.model_.predict(X_prepared)


def _find_bull_eye_imputer(model: Any) -> BullEyeImputer | None:
    if model is None:
        return None
    if isinstance(model, BullEyeImputer):
        return model
    if isinstance(model, BullEyeImputerTransformer):
        return getattr(model, "imputer_", None)
    if isinstance(model, SharedBullEyeModelWrapper):
        transformer = getattr(model, "bull_eye_transformer", None)
        if transformer is not None and hasattr(transformer, "imputer_"):
            return transformer.imputer_
        return _find_bull_eye_imputer(getattr(model, "model_", model.model))
    if isinstance(model, TabPFNRawClassifier):
        transformer = getattr(model, "bull_eye_imputer_", None)
        if transformer is not None and hasattr(transformer, "imputer_"):
            return transformer.imputer_
        return None
    if hasattr(model, "bull_eye_transformer"):
        transformer = getattr(model, "bull_eye_transformer", None)
        if transformer is not None and hasattr(transformer, "imputer_"):
            return transformer.imputer_
    if hasattr(model, "bull_eye_imputer_"):
        transformer = getattr(model, "bull_eye_imputer_", None)
        if isinstance(transformer, BullEyeImputer):
            return transformer
        if transformer is not None and hasattr(transformer, "imputer_"):
            return transformer.imputer_
    if isinstance(model, Pipeline):
        for _, step in model.named_steps.items():
            found = _find_bull_eye_imputer(step)
            if found is not None:
                return found
    if hasattr(model, "named_steps"):
        try:
            for step in model.named_steps.values():
                found = _find_bull_eye_imputer(step)
                if found is not None:
                    return found
        except Exception:
            pass
    if hasattr(model, "base_models"):
        try:
            base_models = getattr(model, "base_models")
            if isinstance(base_models, dict):
                for step in base_models.values():
                    found = _find_bull_eye_imputer(step)
                    if found is not None:
                        return found
        except Exception:
            pass
    if hasattr(model, "meta_model"):
        try:
            found = _find_bull_eye_imputer(getattr(model, "meta_model"))
            if found is not None:
                return found
        except Exception:
            pass
    return None


def _save_bull_eye_diagnostics(model: Any, out_dir: Path | str, model_name: str) -> Path | None:
    imputer = _find_bull_eye_imputer(model)
    if imputer is None:
        return None

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    imputer.output_dir = str(out_path)
    try:
        imputer.visualize_diagnostics()
    except Exception as exc:
        print(f"[WARN] Failed to render Bull_eye diagnostics for {model_name}: {exc}")

    if getattr(imputer, "lasso_coefficients", None) is not None:
        try:
            imputer.lasso_coefficients.to_csv(
                out_path / "bull_eye_lasso_coefficients.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception as exc:
            print(f"[WARN] Failed to save Bull_eye LASSO table for {model_name}: {exc}")
    if getattr(imputer, "feature_importance", None) is not None:
        try:
            imputer.feature_importance.to_csv(
                out_path / "bull_eye_rf_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception as exc:
            print(f"[WARN] Failed to save Bull_eye RF importance table for {model_name}: {exc}")

    payload = {
        "model_name": model_name,
        "predictors": list(imputer.predictors),
        "summary": imputer.diagnostic_summary(),
        "best_params": getattr(imputer, "best_params", None),
        "best_cv_score": getattr(imputer, "best_cv_score", None),
    }
    with open(out_path / "bull_eye_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(payload), f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved Bull_eye diagnostics for {model_name}: {out_path}")
    return out_path


class WeightedSoftVotingEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, base_models: dict[str, Any], weights: dict[str, float]) -> None:
        self.base_models = base_models
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WeightedSoftVotingEnsemble":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ordered_names = list(self.weights.keys())
        weight_arr = np.asarray([self.weights[name] for name in ordered_names], dtype=float)
        probs = []
        for name in ordered_names:
            prob, _ = _prob_from_estimator(self.base_models[name], X)
            probs.append(prob)
        matrix = np.column_stack(probs)
        positive = matrix.dot(weight_arr)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_from_probability(self.predict_proba(X)[:, 1])


class OOFStackingEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_models: dict[str, Any],
        base_model_order: list[str],
        meta_model: Any,
    ) -> None:
        self.base_models = base_models
        self.base_model_order = base_model_order
        self.meta_model = meta_model

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OOFStackingEnsemble":
        return self

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for name in self.base_model_order:
            prob, _ = _prob_from_estimator(self.base_models[name], X)
            cols.append(prob)
        return np.column_stack(cols)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = self._meta_features(X)
        proba, _ = _prob_from_estimator(self.meta_model, meta_X)
        proba = np.asarray(proba, dtype=float).ravel()
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_from_probability(self.predict_proba(X)[:, 1])


def build_traditional_pipeline(
    estimator: BaseEstimator,
    continuous_vars: list[str],
    ordinal_vars: list[str],
    nominal_vars: list[str],
    seed: int,
    include_bull_eye: bool = True,
) -> Pipeline:
    nominal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    try:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
    except TypeError:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal", nominal_encoder, nominal_vars),
            ("ordinal", ordinal_encoder, ordinal_vars),
            ("num", StandardScaler(), continuous_vars),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps: list[tuple[str, Any]] = []
    if include_bull_eye:
        bull_eye_imputer = BullEyeImputerTransformer(
            predictors=_bull_eye_predictors(continuous_vars, ordinal_vars, nominal_vars),
            seed=seed,
        )
        steps.append(("bull_eye", bull_eye_imputer))
    steps.extend(
        [
            ("preprocess", preprocessor),
            ("model", clone(estimator)),
        ]
    )
    return Pipeline(steps=steps)


def _traditional_registry(seed: int, device: str = "auto") -> dict[str, dict[str, Any]]:
    use_gpu = _resolve_tree_backend_use_gpu(device)
    return {
        "Logistic Regression": {
            "family": "traditional",
            "estimator": LogisticRegression(random_state=seed),
            "search_spaces": {
                "model__C": Real(0.01, 100, prior="log-uniform"),
                "model__max_iter": Integer(100, 1000),
            },
        },
        "Random Forest": {
            "family": "traditional",
            "estimator": RandomForestClassifier(random_state=seed, n_jobs=2),
            "search_spaces": {
                "model__n_estimators": Integer(50, 200),
                "model__max_depth": Integer(3, 20),
                "model__min_samples_split": Integer(2, 20),
                "model__min_samples_leaf": Integer(1, 10),
            },
        },
        "Gradient Boosting": {
            "family": "traditional",
            "estimator": GradientBoostingClassifier(random_state=seed),
            "search_spaces": {
                "model__n_estimators": Integer(50, 200),
                "model__max_depth": Integer(3, 10),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            },
        },
        "XGBoost": {
            "family": "traditional",
            "estimator": XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                n_jobs=2,
                tree_method="hist",
                device="cuda" if use_gpu else "cpu",
            ),
            "search_spaces": {
                "model__n_estimators": Integer(50, 200),
                "model__max_depth": Integer(3, 10),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "model__subsample": Real(0.6, 1.0),
            },
        },
        "LightGBM": {
            "family": "traditional",
            "estimator": LGBMClassifier(
                random_state=seed,
                verbosity=-1,
                n_jobs=2,
                device_type="gpu" if use_gpu else "cpu",
            ),
            "search_spaces": {
                "model__n_estimators": Integer(50, 200),
                "model__max_depth": Integer(3, 10),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "model__num_leaves": Integer(10, 100),
            },
        },
        "CatBoost": {
            "family": "traditional",
            "estimator": CatBoostClassifier(
                random_state=seed,
                verbose=0,
                thread_count=2,
                allow_writing_files=False,
                task_type="GPU" if use_gpu else "CPU",
            ),
            "search_spaces": {
                "model__iterations": Integer(50, 300),
                "model__depth": Integer(3, 10),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "model__l2_leaf_reg": Real(1.0, 10.0, prior="log-uniform"),
            },
        },
        "SVM": {
            "family": "traditional",
            "estimator": SVC(random_state=seed, probability=True),
            "search_spaces": {
                "model__C": Real(0.1, 100, prior="log-uniform"),
                "model__gamma": Real(0.001, 1.0, prior="log-uniform"),
                "model__kernel": Categorical(["rbf", "linear"]),
            },
        },
        "KNN": {
            "family": "traditional",
            "estimator": KNeighborsClassifier(n_jobs=1),
            "search_spaces": {
                "model__n_neighbors": Integer(3, 20),
                "model__weights": Categorical(["uniform", "distance"]),
                "model__metric": Categorical(["euclidean", "manhattan"]),
            },
        },
        "Extra Trees": {
            "family": "traditional",
            "estimator": ExtraTreesClassifier(random_state=seed, n_jobs=2),
            "search_spaces": {
                "model__n_estimators": Integer(50, 200),
                "model__max_depth": Integer(3, 20),
                "model__min_samples_split": Integer(2, 20),
            },
        },
        "AdaBoost": {
            "family": "traditional",
            "estimator": AdaBoostClassifier(random_state=seed),
            "search_spaces": {
                "model__n_estimators": Integer(50, 200),
                "model__learning_rate": Real(0.01, 2.0, prior="log-uniform"),
            },
        },
        "LDA": {
            "family": "traditional",
            "estimator": LinearDiscriminantAnalysis(),
            "search_spaces": {
                "model__solver": Categorical(["lsqr"]),
                "model__shrinkage": Real(0.0, 1.0),
            },
        },
    }


def _fit_search_or_estimator(
    estimator: BaseEstimator,
    search_spaces: dict[str, Any] | None,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: UpgradeConfig,
    cv_folds: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    if search_spaces:
        n_splits = int(cv_folds) if cv_folds is not None else int(config.inner_cv_folds)
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=config.bayes_n_iter,
            cv=StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=config.seed,
            ),
            scoring="roc_auc",
            random_state=config.seed,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, {
            "search_type": "BayesSearchCV",
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_folds": n_splits,
        }

    estimator.fit(X_train, y_train)
    return estimator, {"search_type": "none"}


def _fit_single_candidate_oof(
    name: str,
    family: str,
    estimator_builder: Any,
    search_spaces: dict[str, Any] | None,
    X_retro: pd.DataFrame,
    y_retro: np.ndarray,
    X_external: pd.DataFrame,
    y_external: np.ndarray,
    paths: BenchmarkPaths,
    config: UpgradeConfig,
    shared_bull_eye_bundle: SharedBullEyeImputationBundle | None = None,
) -> CandidateResult:
    outer_cv = StratifiedKFold(
        n_splits=config.oof_folds,
        shuffle=True,
        random_state=config.seed,
    )
    oof_proba = np.full(len(X_retro), np.nan, dtype=float)
    fold_logs: list[dict[str, Any]] = []
    probability_type = "predict_proba"

    try:
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_retro, y_retro), start=1):
            X_train_fold = X_retro.iloc[train_idx].copy()
            X_val_fold = X_retro.iloc[val_idx].copy()
            y_train_fold = y_retro[train_idx]

            estimator = estimator_builder()
            fitted, fit_log = _fit_search_or_estimator(
                estimator=estimator,
                search_spaces=search_spaces,
                X_train=X_train_fold,
                y_train=y_train_fold,
                config=config,
            )
            fold_prob, probability_type = _prob_from_estimator(fitted, X_val_fold)
            oof_proba[val_idx] = fold_prob
            fold_logs.append(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    **fit_log,
                }
            )

        if np.isnan(oof_proba).any():
            raise RuntimeError(f"{name} produced incomplete OOF probabilities.")

        final_estimator = estimator_builder()
        final_model, final_log = _fit_search_or_estimator(
            estimator=final_estimator,
            search_spaces=search_spaces,
            X_train=X_retro,
            y_train=y_retro,
            config=config,
        )
        _save_bull_eye_diagnostics(
            final_model,
            paths.audit_dir / "bull_eye_imputation_diagnostics" / _model_tag(name),
            name,
        )
        external_proba, probability_type = _prob_from_estimator(final_model, X_external)
        return _build_completed_result(
            name=name,
            family=family,
            oof_proba=oof_proba,
            external_proba=external_proba,
            y_retro=y_retro,
            y_external=y_external,
            config=config,
            final_model=final_model,
            training_log={
                "fold_logs": fold_logs,
                "final_fit": final_log,
            },
            probability_type=probability_type,
        )
    except Exception as exc:
        return CandidateResult(
            name=name,
            family=family,
            status="failed",
            training_log={"error": f"{type(exc).__name__}: {exc}"},
        )


def _fit_single_candidate_single_cv(
    name: str,
    family: str,
    estimator_builder: Any,
    search_spaces: dict[str, Any] | None,
    X_retro: pd.DataFrame,
    y_retro: np.ndarray,
    X_external: pd.DataFrame,
    y_external: np.ndarray,
    paths: BenchmarkPaths,
    config: UpgradeConfig,
    shared_bull_eye_bundle: SharedBullEyeImputationBundle | None = None,
) -> CandidateResult:
    cv_folds = int(config.oof_folds)
    cv_splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=config.seed,
    )
    oof_proba = np.full(len(X_retro), np.nan, dtype=float)
    fold_logs: list[dict[str, Any]] = []
    probability_type = "predict_proba"

    try:
        search_X = shared_bull_eye_bundle.X_retro_full if shared_bull_eye_bundle is not None else X_retro
        external_X = shared_bull_eye_bundle.X_external_full if shared_bull_eye_bundle is not None else X_external
        search_estimator = estimator_builder()
        final_model, final_log = _fit_search_or_estimator(
            estimator=search_estimator,
            search_spaces=search_spaces,
            X_train=search_X,
            y_train=y_retro,
            config=config,
            cv_folds=cv_folds,
        )

        if shared_bull_eye_bundle is not None:
            fold_iter = [
                (
                    fold_data.fold,
                    fold_data.train_idx,
                    fold_data.val_idx,
                    fold_data.X_train,
                    fold_data.X_val,
                )
                for fold_data in shared_bull_eye_bundle.folds
            ]
        else:
            fold_iter = [
                (
                    fold,
                    np.asarray(train_idx, dtype=int),
                    np.asarray(val_idx, dtype=int),
                    X_retro.iloc[train_idx].copy(),
                    X_retro.iloc[val_idx].copy(),
                )
                for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_retro, y_retro), start=1)
            ]

        for fold, train_idx, val_idx, X_train_fold, X_val_fold in fold_iter:
            y_train_fold = y_retro[train_idx]

            fold_model = clone(final_model)
            fold_model.fit(X_train_fold, y_train_fold)
            fold_prob, probability_type = _prob_from_estimator(fold_model, X_val_fold)
            oof_proba[val_idx] = fold_prob
            fold_logs.append(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "search_type": "fixed_best_estimator_single_cv",
                }
            )

        if np.isnan(oof_proba).any():
            raise RuntimeError(f"{name} produced incomplete single-CV OOF probabilities.")

        wrapped_final_model: Any = final_model
        if shared_bull_eye_bundle is not None and shared_bull_eye_bundle.full_transformer is not None:
            wrapped_final_model = SharedBullEyeModelWrapper(
                model=final_model,
                bull_eye_transformer=shared_bull_eye_bundle.full_transformer,
            )

        _save_bull_eye_diagnostics(
            wrapped_final_model,
            paths.audit_dir / "bull_eye_imputation_diagnostics" / _model_tag(name),
            name,
        )
        external_proba, probability_type = _prob_from_estimator(final_model, external_X)
        return _build_completed_result(
            name=name,
            family=family,
            oof_proba=oof_proba,
            external_proba=external_proba,
            y_retro=y_retro,
            y_external=y_external,
            config=config,
            final_model=wrapped_final_model,
            training_log={
                "fold_logs": fold_logs,
                "selection_fit": final_log,
                "validation_mode": "single_cv_external",
                "bull_eye_strategy": "shared_fold_preimpute" if shared_bull_eye_bundle is not None else "per_model_pipeline",
            },
            probability_type=probability_type,
        )
    except Exception as exc:
        return CandidateResult(
            name=name,
            family=family,
            status="failed",
            training_log={"error": f"{type(exc).__name__}: {exc}"},
        )


def _model_tag(name: str) -> str:
    tag = str(name).strip().lower().replace(" ", "_").replace("/", "_")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag


def _checkpoint_name_for_version(model_version: str) -> str:
    version = str(model_version).strip().lower().replace("_", ".")
    if version in {"v2.5", "2.5"}:
        return "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    return "tabpfn-v2-classifier.ckpt"


@lru_cache(maxsize=8)
def _discover_tabpfn_cache_dir(explicit_dir: str | None, model_version: str) -> str | None:
    checkpoint_name = _checkpoint_name_for_version(model_version)
    candidates: list[Path] = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))

    env_dir = os.getenv("TABPFN_MODEL_CACHE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    cwd = Path.cwd()
    drive_root = Path(cwd.anchor) if cwd.anchor else cwd
    search_roots = [cwd, cwd.parent, cwd.parent.parent, drive_root, Path.home()]
    for base in search_roots:
        candidates.append(base / ".tabpfn_models")
        try:
            candidates.extend([p for p in base.glob("*/.tabpfn_models")])
        except Exception:
            continue

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except Exception:
            resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir() and (candidate / checkpoint_name).exists():
            return str(candidate)
    return None


def _build_candidate_registry(
    feature_cols: list[str],
    continuous_vars: list[str],
    ordinal_vars: list[str],
    nominal_vars: list[str],
    config: UpgradeConfig,
    share_bull_eye_preimpute: bool = False,
) -> dict[str, dict[str, Any]]:
    continuous_model_vars = [c for c in continuous_vars if c in feature_cols]
    ordinal_model_vars = [c for c in ordinal_vars if c in feature_cols]
    nominal_model_vars = [c for c in nominal_vars if c in feature_cols]

    registry: dict[str, dict[str, Any]] = {}
    for name, spec in _traditional_registry(config.seed, device=config.device).items():
        estimator = spec["estimator"]
        search_spaces = spec.get("search_spaces")

        def _builder(
            estimator: BaseEstimator = estimator,
            continuous_vars: list[str] = continuous_model_vars,
            ordinal_vars: list[str] = ordinal_model_vars,
            nominal_vars: list[str] = nominal_model_vars,
            seed: int = config.seed,
        ) -> Pipeline:
            return build_traditional_pipeline(
                estimator=estimator,
                continuous_vars=continuous_vars,
                ordinal_vars=ordinal_vars,
                nominal_vars=nominal_vars,
                seed=seed,
                include_bull_eye=not share_bull_eye_preimpute,
            )

        registry[name] = {
            "family": spec["family"],
            "builder": _builder,
            "search_spaces": search_spaces,
        }

    if config.enable_tabpfn:
        tabpfn_search_spaces = None
        if config.tabpfn_search_enabled:
            tabpfn_search_spaces = {
                "n_estimators": Integer(16, 64),
                "softmax_temperature": Real(0.55, 1.10),
                "average_before_softmax": Categorical([True, False]),
                "balance_probabilities": Categorical([True, False]),
            }
        registry["TabPFN"] = {
            "family": "tabpfn",
            "builder": lambda: TabPFNRawClassifier(
                feature_cols=feature_cols,
                continuous_vars=continuous_vars,
                ordinal_vars=ordinal_vars,
                nominal_vars=nominal_vars,
                seed=config.seed,
                device=config.device,
                model_version=config.model_version,
                balance_probabilities=True,
                model_cache_dir=config.tabpfn_model_cache_dir,
                bull_eye_mode="preimputed" if share_bull_eye_preimpute else "internal",
            ),
            "search_spaces": tabpfn_search_spaces,
        }

    return registry


def _save_candidate_outputs(
    result: CandidateResult,
    paths: BenchmarkPaths,
    retro_source: pd.DataFrame,
    y_retro: np.ndarray,
    external_source: pd.DataFrame,
    y_external: np.ndarray,
) -> None:
    tag = _model_tag(result.name)

    if result.oof_proba is not None:
        oof_table = _cohort_probability_table(
            df_source=retro_source,
            y_true=y_retro,
            y_proba=result.oof_proba,
            threshold=0.5,
        )
        oof_table["PredLabel_RankingThreshold"] = _predict_from_probability(
            result.oof_proba,
            threshold=result.ranking_threshold,
        )
        oof_table.to_csv(
            paths.metrics_dir / "oof_predictions" / f"{tag}_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if result.external_proba is not None:
        external_table = _cohort_probability_table(
            df_source=external_source,
            y_true=y_external,
            y_proba=result.external_proba,
            threshold=0.5,
        )
        external_table.to_csv(
            paths.metrics_dir / "external_predictions" / f"{tag}_external_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    payload = {
        "name": result.name,
        "family": result.family,
        "status": result.status,
        "oof_metrics": result.oof_metrics,
        "ranking_oof_metrics": result.ranking_oof_metrics,
        "oof_threshold_bundle": result.oof_threshold_bundle,
        "ranking_threshold_label": result.ranking_threshold_label,
        "ranking_threshold": result.ranking_threshold,
        "external_metrics": result.external_metrics,
        "probability_type": result.probability_type,
        "training_log": result.training_log,
    }
    with open(paths.audit_dir / "training_logs" / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(payload), f, ensure_ascii=False, indent=2)


def _build_voting_result(
    base_results: dict[str, CandidateResult],
    X_external: pd.DataFrame,
    y_retro: np.ndarray,
    y_external: np.ndarray,
    config: UpgradeConfig,
) -> CandidateResult:
    base_names = [name for name in config.ensemble_base_models if name in base_results]
    missing = [name for name in config.ensemble_base_models if name not in base_results]
    if missing:
        return CandidateResult(
            name="Soft Voting",
            family="ensemble_voting",
            status="failed",
            training_log={"error": f"Missing base models for voting: {missing}"},
        )

    auc_weights = np.asarray(
        [max(float(base_results[name].oof_metrics.get("AUC", 0.0) or 0.0), 1e-6) for name in base_names],
        dtype=float,
    )
    auc_weights = auc_weights / auc_weights.sum()
    weight_map = {name: float(weight) for name, weight in zip(base_names, auc_weights)}

    oof_matrix = np.column_stack([base_results[name].oof_proba for name in base_names])
    external_matrix = np.column_stack([base_results[name].external_proba for name in base_names])
    oof_proba = oof_matrix.dot(auc_weights)
    external_proba = external_matrix.dot(auc_weights)

    final_model = WeightedSoftVotingEnsemble(
        base_models={name: base_results[name].final_model for name in base_names},
        weights=weight_map,
    )
    return _build_completed_result(
        name="Soft Voting",
        family="ensemble_voting",
        oof_proba=oof_proba,
        external_proba=external_proba,
        y_retro=y_retro,
        y_external=y_external,
        config=config,
        final_model=final_model,
        training_log={
            "base_models": base_names,
            "weights": weight_map,
        },
        probability_type="predict_proba",
    )


def _composite_score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(float(metrics.get(m, 0.0) or 0.0) * w for m, w in weights.items())


def _build_lda_tabpfn_voting_result(
    base_results: dict[str, CandidateResult],
    X_external: pd.DataFrame,
    y_retro: np.ndarray,
    y_external: np.ndarray,
    config: UpgradeConfig,
) -> CandidateResult | None:
    """Build a dedicated LDA + TabPFN soft voting ensemble with optimised weights."""
    lda_key = "LDA"
    tabpfn_key = "TabPFN"
    if lda_key not in base_results or tabpfn_key not in base_results:
        return None

    oof_lda = base_results[lda_key].oof_proba
    oof_pfn = base_results[tabpfn_key].oof_proba
    ext_lda = base_results[lda_key].external_proba
    ext_pfn = base_results[tabpfn_key].external_proba

    # Grid search: find LDA weight that maximises OOF composite score
    best_w, best_score = 0.5, -1.0
    grid = np.arange(0.50, 1.00, 0.01)  # LDA weight from 50% to 99%, 1% steps
    search_log: list[dict[str, Any]] = []
    for w_lda in grid:
        w_lda = round(float(w_lda), 2)
        oof_blend = oof_lda * w_lda + oof_pfn * (1.0 - w_lda)
        _, rank_metrics, _, threshold_label, threshold_value = _compute_oof_metric_views(
            y_true=y_retro,
            y_proba=oof_blend,
            config=config,
        )
        cs = _composite_score(rank_metrics, config.ranking_weights)
        search_log.append(
            {
                "w_lda": w_lda,
                "composite": round(cs, 6),
                "sens": round(float(rank_metrics.get("Sensitivity", 0)), 4),
                "threshold_label": threshold_label,
                "threshold": round(float(threshold_value), 6),
            }
        )
        if cs > best_score:
            best_score = cs
            best_w = w_lda

    print(f"[INFO] LDA+TabPFN weight search: best w_LDA={best_w:.2f} (composite={best_score:.4f})")
    for entry in search_log:
        print(
            f"       w_LDA={entry['w_lda']:.2f}  composite={entry['composite']:.6f}  "
            f"sens={entry['sens']:.4f}  {entry['threshold_label']}={entry['threshold']:.6f}"
        )

    w_pfn = round(1.0 - best_w, 2)
    weight_map = {lda_key: best_w, tabpfn_key: w_pfn}

    oof_proba = oof_lda * best_w + oof_pfn * w_pfn
    external_proba = ext_lda * best_w + ext_pfn * w_pfn

    final_model = WeightedSoftVotingEnsemble(
        base_models={name: base_results[name].final_model for name in [lda_key, tabpfn_key]},
        weights=weight_map,
    )
    return _build_completed_result(
        name="LDA+TabPFN Voting",
        family="ensemble_voting",
        oof_proba=oof_proba,
        external_proba=external_proba,
        y_retro=y_retro,
        y_external=y_external,
        config=config,
        final_model=final_model,
        training_log={
            "base_models": [lda_key, tabpfn_key],
            "weights": weight_map,
            "weight_search": search_log,
        },
        probability_type="predict_proba",
    )


def _build_stacking_result(
    base_results: dict[str, CandidateResult],
    X_external: pd.DataFrame,
    y_retro: np.ndarray,
    y_external: np.ndarray,
    config: UpgradeConfig,
    paths: BenchmarkPaths,
) -> CandidateResult:
    base_names = [name for name in config.ensemble_base_models if name in base_results]
    missing = [name for name in config.ensemble_base_models if name not in base_results]
    if missing:
        return CandidateResult(
            name="Stacking",
            family="ensemble_stacking",
            status="failed",
            training_log={"error": f"Missing base models for stacking: {missing}"},
        )

    base_oof_matrix = np.column_stack([base_results[name].oof_proba for name in base_names])
    base_external_matrix = np.column_stack([base_results[name].external_proba for name in base_names])

    base_oof_df = pd.DataFrame(base_oof_matrix, columns=[_model_tag(name) for name in base_names])
    base_oof_df.insert(0, "TrueLabel", np.asarray(y_retro).astype(int))
    base_oof_df.to_csv(
        paths.metrics_dir / "ensemble_base_oof_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    meta_oof = np.full(len(y_retro), np.nan, dtype=float)
    meta_cv = StratifiedKFold(
        n_splits=config.oof_folds,
        shuffle=True,
        random_state=config.seed,
    )
    meta_logs: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(meta_cv.split(base_oof_matrix, y_retro), start=1):
        meta_model = LogisticRegression(random_state=config.seed, max_iter=1000)
        meta_model.fit(base_oof_matrix[train_idx], y_retro[train_idx])
        fold_prob, _ = _prob_from_estimator(meta_model, base_oof_matrix[val_idx])
        meta_oof[val_idx] = fold_prob
        meta_logs.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
        )

    if np.isnan(meta_oof).any():
        return CandidateResult(
            name="Stacking",
            family="ensemble_stacking",
            status="failed",
            training_log={"error": "Meta-model produced incomplete OOF predictions."},
        )

    meta_model_final = LogisticRegression(random_state=config.seed, max_iter=1000)
    meta_model_final.fit(base_oof_matrix, y_retro)
    external_proba, _ = _prob_from_estimator(meta_model_final, base_external_matrix)

    final_model = OOFStackingEnsemble(
        base_models={name: base_results[name].final_model for name in base_names},
        base_model_order=base_names,
        meta_model=meta_model_final,
    )
    return _build_completed_result(
        name="Stacking",
        family="ensemble_stacking",
        oof_proba=meta_oof,
        external_proba=external_proba,
        y_retro=y_retro,
        y_external=y_external,
        config=config,
        final_model=final_model,
        training_log={
            "base_models": base_names,
            "meta_model": "LogisticRegression",
            "meta_fold_logs": meta_logs,
            "base_external_shape": list(base_external_matrix.shape),
            "external_n": int(len(X_external)),
        },
        probability_type="predict_proba",
    )


def _rank_candidates(
    results: dict[str, CandidateResult],
    config: UpgradeConfig,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        row = {
            "Model": name,
            "Family": result.family,
            "Status": result.status,
            "OOF_RankingThresholdLabel": result.ranking_threshold_label,
            "OOF_RankingThreshold": result.ranking_threshold,
        }
        for prefix, metrics in (("OOF", result.oof_metrics), ("External", result.external_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        for key, value in result.ranking_oof_metrics.items():
            row[f"OOF_Rank_{key}"] = value
        rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise RuntimeError("No candidate results were generated.")

    completed = ranking[ranking["Status"] == "completed"].copy()
    if completed.empty:
        failed = ranking[["Model", "Status"]].to_dict("records")
        raise RuntimeError(f"All candidates failed: {failed}")

    eligible = completed[completed["OOF_AUC"] >= float(config.champion_auc_threshold)].copy()
    if eligible.empty:
        eligible = completed.sort_values(["OOF_AUC", "External_AUC"], ascending=False).head(1).copy()

    for metric, weight in config.ranking_weights.items():
        rank_col = f"OOF_Rank_{metric}"
        source_col = rank_col if rank_col in eligible.columns else f"OOF_{metric}"
        eligible[f"weight_{metric}"] = eligible[source_col].astype(float) * float(weight)
    eligible["CompositeScore"] = eligible[[f"weight_{metric}" for metric in config.ranking_weights]].sum(axis=1)

    ranking = ranking.merge(
        eligible[["Model", "CompositeScore"]],
        on="Model",
        how="left",
    )
    ranking["CompositeScore"] = ranking["CompositeScore"].fillna(np.nan)
    ranking = ranking.sort_values(
        by=["CompositeScore", "OOF_AUC", "External_AUC"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    champion_name = str(eligible.sort_values(
        by=["CompositeScore", "OOF_AUC", "External_AUC"],
        ascending=[False, False, False],
    ).iloc[0]["Model"])
    ranking["Champion"] = ranking["Model"].eq(champion_name)
    return ranking, champion_name


def _save_ranking_outputs(
    ranking_df: pd.DataFrame,
    paths: BenchmarkPaths,
) -> None:
    ranking_df.to_csv(
        paths.metrics_dir / "model_ranking_OOF.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking_df.to_csv(
        paths.ml_dir / "model_ranking_OOF.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _save_threshold_outputs(
    champion_name: str,
    threshold_bundle: dict[str, Any],
    paths: BenchmarkPaths,
) -> tuple[Path, Path]:
    tag = _model_tag(champion_name)
    payload = {
        "model_name": champion_name,
        "model_tag": tag,
        **threshold_bundle,
    }

    for key in [
        "metrics_Youden",
        "metrics_Sens90",
        "metrics_Spec90",
        "metrics_MaxF1",
        "metrics_low",
        "metrics_high",
    ]:
        if key in payload and isinstance(payload[key], dict):
            metric_block = payload[key]
            payload[key] = {
                **metric_block,
                "sensitivity": metric_block.get("Sensitivity"),
                "specificity": metric_block.get("Specificity"),
                "accuracy": metric_block.get("Accuracy"),
                "PPV": metric_block.get("PPV"),
                "NPV": metric_block.get("NPV"),
                "F1": metric_block.get("F1"),
                "Youden": metric_block.get("Youden"),
            }

    metrics_path = paths.metrics_dir / f"{tag}_thresholds_OOF.json"
    deploy_path = paths.deploy_dir / f"{tag}_thresholds_OOF.json"
    for target in [metrics_path, deploy_path]:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(_serializable(payload), f, ensure_ascii=False, indent=2)
    return metrics_path, deploy_path


def _save_champion_pipeline(
    champion_name: str,
    champion_model: Any,
    paths: BenchmarkPaths,
) -> tuple[Path, Path]:
    tag = _model_tag(champion_name)
    model_archive_path = paths.models_dir / f"champion_model_{tag}.pkl"
    deploy_path = paths.deploy_dir / f"best_model_pipeline_{tag}.pkl"
    joblib.dump(champion_model, model_archive_path)
    joblib.dump(champion_model, deploy_path)

    meta_payload = {
        "model_name": champion_name,
        "model_tag": tag,
        "model_archive_path": str(model_archive_path),
        "deploy_path": str(deploy_path),
    }
    with open(paths.models_dir / f"champion_model_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)
    return model_archive_path, deploy_path


def _save_external_summary(
    champion_name: str,
    champion_result: CandidateResult,
    threshold_bundle: dict[str, Any],
    y_external: np.ndarray,
    paths: BenchmarkPaths,
) -> Path:
    rows = []
    for label, threshold_key in [
        ("default_0.5", None),
        ("Youden", "threshold_Youden"),
        ("Sens90", "threshold_Sens90"),
        ("Spec90", "threshold_Spec90"),
        ("MaxF1", "threshold_MaxF1"),
        ("Chosen", "threshold_Chosen"),
        ("Low", "threshold_low"),
        ("High", "threshold_high"),
    ]:
        threshold = 0.5 if threshold_key is None else float(threshold_bundle[threshold_key])
        metrics = _calculate_binary_metrics(y_external, champion_result.external_proba, threshold=threshold)
        rows.append(
            {
                "Model": champion_name,
                "ThresholdLabel": label,
                "Threshold": threshold,
                **metrics,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = paths.metrics_dir / "champion_external_metrics.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def _save_prospective_prediction_report(
    champion_name: str,
    champion_result: CandidateResult,
    threshold_bundle: dict[str, Any],
    external_source: pd.DataFrame,
    y_external: np.ndarray,
    paths: BenchmarkPaths,
) -> Path:
    out_dir = paths.ml_dir.parent / "07_Prospective_Prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold_map = {
        "PredLabel_0_5": 0.5,
        "PredLabel_Youden": float(threshold_bundle["threshold_Youden"]),
        "PredLabel_Sens90": float(threshold_bundle["threshold_Sens90"]),
        "PredLabel_Spec90": float(threshold_bundle["threshold_Spec90"]),
        "PredLabel_MaxF1": float(threshold_bundle["threshold_MaxF1"]),
        "PredLabel_Chosen": float(threshold_bundle["threshold_Chosen"]),
        "PredLabel_Low": float(threshold_bundle["threshold_low"]),
        "PredLabel_High": float(threshold_bundle["threshold_high"]),
    }

    prediction_df = external_source.copy().reset_index(drop=True)
    prediction_df["TrueLabel"] = np.asarray(y_external).astype(int)
    prediction_df["PredProb"] = np.asarray(champion_result.external_proba, dtype=float)
    for col, thr in threshold_map.items():
        prediction_df[col] = _predict_from_probability(champion_result.external_proba, thr)

    summary_rows = []
    for label, thr in threshold_map.items():
        metrics = _calculate_binary_metrics(y_external, champion_result.external_proba, threshold=thr)
        summary_rows.append(
            {
                "Model": champion_name,
                "ThresholdLabel": label.replace("PredLabel_", ""),
                "Threshold": float(thr),
                **metrics,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    report_path = out_dir / "prospective_prediction_report.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        prediction_df.to_excel(writer, sheet_name="predictions", index=False)
    return report_path


def run_oof_external_benchmark(
    data_retro: pd.DataFrame,
    data_pros: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    continuous_vars: list[str],
    ordinal_vars: list[str],
    nominal_vars: list[str],
    paths: BenchmarkPaths,
    config: UpgradeConfig,
) -> dict[str, Any]:
    paths.ensure()
    accel = _acceleration_summary(config.device)
    print(f"[INFO] Acceleration summary: {accel}")

    candidate_runner = _fit_single_candidate_oof
    share_bull_eye_preimpute = False
    if config.validation_mode == "single_cv_external":
        candidate_runner = _fit_single_candidate_single_cv
        share_bull_eye_preimpute = True
        print(f"[INFO] Validation mode: single-layer CV + external validation (cv_folds={config.oof_folds})")
        print(f"[INFO] Ranking threshold strategy: per-model OOF {config.ranking_threshold_strategy}")
    elif config.validation_mode != "oof_external":
        raise ValueError(
            f"Unsupported validation_mode='{config.validation_mode}'. "
            "Use 'oof_external' or 'single_cv_external'."
        )

    retro_source = data_retro.reset_index(drop=True).copy()
    external_source = data_pros.reset_index(drop=True).copy()

    if target_col not in retro_source.columns:
        raise KeyError(f"Retrospective target column '{target_col}' not found.")
    if target_col not in external_source.columns:
        raise KeyError(f"Prospective target column '{target_col}' not found.")

    y_retro = pd.to_numeric(retro_source[target_col], errors="coerce").astype(int).to_numpy()
    y_external = pd.to_numeric(external_source[target_col], errors="coerce").astype(int).to_numpy()

    X_retro = clean_modeling_features(
        retro_source,
        feature_cols=feature_cols,
        continuous_vars=continuous_vars,
        ordinal_vars=ordinal_vars,
        nominal_vars=nominal_vars,
    )
    X_external = clean_modeling_features(
        external_source,
        feature_cols=feature_cols,
        continuous_vars=continuous_vars,
        ordinal_vars=ordinal_vars,
        nominal_vars=nominal_vars,
    )

    shared_bull_eye_bundle = None
    if share_bull_eye_preimpute:
        shared_bull_eye_bundle = _build_shared_bull_eye_imputation_bundle(
            X_retro=X_retro,
            y_retro=y_retro,
            X_external=X_external,
            continuous_vars=continuous_vars,
            ordinal_vars=ordinal_vars,
            nominal_vars=nominal_vars,
            paths=paths,
            config=config,
        )
        print(
            "[INFO] Bull_eye strategy: shared pre-imputation per CV fold "
            f"(folds={len(shared_bull_eye_bundle.folds)}) + one full-data fit for final models"
        )

    registry = _build_candidate_registry(
        feature_cols=feature_cols,
        continuous_vars=continuous_vars,
        ordinal_vars=ordinal_vars,
        nominal_vars=nominal_vars,
        config=config,
        share_bull_eye_preimpute=share_bull_eye_preimpute,
    )

    results: dict[str, CandidateResult] = {}
    for name, spec in registry.items():
        result = candidate_runner(
            name=name,
            family=spec["family"],
            estimator_builder=spec["builder"],
            search_spaces=spec.get("search_spaces"),
            X_retro=X_retro,
            y_retro=y_retro,
            X_external=X_external,
            y_external=y_external,
            paths=paths,
            config=config,
            shared_bull_eye_bundle=shared_bull_eye_bundle,
        )
        results[name] = result
        _save_candidate_outputs(
            result=result,
            paths=paths,
            retro_source=retro_source,
            y_retro=y_retro,
            external_source=external_source,
            y_external=y_external,
        )

    completed_base = {name: result for name, result in results.items() if result.status == "completed"}

    if config.enable_voting:
        voting_result = _build_voting_result(
            base_results=completed_base,
            X_external=X_external,
            y_retro=y_retro,
            y_external=y_external,
            config=config,
        )
        results[voting_result.name] = voting_result
        _save_candidate_outputs(
            result=voting_result,
            paths=paths,
            retro_source=retro_source,
            y_retro=y_retro,
            external_source=external_source,
            y_external=y_external,
        )

    lda_tabpfn_result = _build_lda_tabpfn_voting_result(
        base_results=completed_base,
        X_external=X_external,
        y_retro=y_retro,
        y_external=y_external,
        config=config,
    )
    if lda_tabpfn_result is not None:
        results[lda_tabpfn_result.name] = lda_tabpfn_result
        _save_candidate_outputs(
            result=lda_tabpfn_result,
            paths=paths,
            retro_source=retro_source,
            y_retro=y_retro,
            external_source=external_source,
            y_external=y_external,
        )

    if config.enable_stacking:
        stacking_result = _build_stacking_result(
            base_results=completed_base,
            X_external=X_external,
            y_retro=y_retro,
            y_external=y_external,
            config=config,
            paths=paths,
        )
        results[stacking_result.name] = stacking_result
        _save_candidate_outputs(
            result=stacking_result,
            paths=paths,
            retro_source=retro_source,
            y_retro=y_retro,
            external_source=external_source,
            y_external=y_external,
        )

    ranking_df, champion_name = _rank_candidates(results, config=config)
    _save_ranking_outputs(ranking_df, paths=paths)

    champion_result = results[champion_name]
    threshold_bundle = compute_threshold_bundle(y_retro, champion_result.oof_proba)
    threshold_metrics_path, threshold_deploy_path = _save_threshold_outputs(
        champion_name=champion_name,
        threshold_bundle=threshold_bundle,
        paths=paths,
    )
    model_archive_path, deploy_path = _save_champion_pipeline(
        champion_name=champion_name,
        champion_model=champion_result.final_model,
        paths=paths,
    )
    champion_external_metrics_path = _save_external_summary(
        champion_name=champion_name,
        champion_result=champion_result,
        threshold_bundle=threshold_bundle,
        y_external=y_external,
        paths=paths,
    )
    prospective_report_path = _save_prospective_prediction_report(
        champion_name=champion_name,
        champion_result=champion_result,
        threshold_bundle=threshold_bundle,
        external_source=external_source,
        y_external=y_external,
        paths=paths,
    )

    summary = {
        "validation_mode": config.validation_mode,
        "device_request": config.device,
        "acceleration": accel,
        "oof_folds": config.oof_folds,
        "search_cv_folds": config.oof_folds if config.validation_mode == "single_cv_external" else config.inner_cv_folds,
        "bull_eye_strategy": "shared_fold_preimpute" if share_bull_eye_preimpute else "per_model_pipeline",
        "enable_tabpfn": config.enable_tabpfn,
        "tabpfn_search_enabled": config.tabpfn_search_enabled,
        "enable_voting": config.enable_voting,
        "enable_stacking": config.enable_stacking,
        "ensemble_base_models": list(config.ensemble_base_models),
        "ranking_threshold_strategy": config.ranking_threshold_strategy,
        "champion_model": champion_name,
        "champion_family": champion_result.family,
        "champion_oof_metrics": champion_result.ranking_oof_metrics,
        "champion_oof_metrics_default_0_5": champion_result.oof_metrics,
        "champion_oof_metrics_ranking_threshold": champion_result.ranking_oof_metrics,
        "champion_ranking_threshold_label": champion_result.ranking_threshold_label,
        "champion_ranking_threshold": champion_result.ranking_threshold,
        "champion_oof_threshold_bundle": champion_result.oof_threshold_bundle,
        "champion_external_metrics_default_0_5": champion_result.external_metrics,
        "thresholds": threshold_bundle,
        "n_retro": int(len(retro_source)),
        "n_external": int(len(external_source)),
        "retro_target_distribution": pd.Series(y_retro).value_counts().sort_index().to_dict(),
        "external_target_distribution": pd.Series(y_external).value_counts().sort_index().to_dict(),
        "feature_cols": list(feature_cols),
        "continuous_vars": list(continuous_vars),
        "ordinal_vars": list(ordinal_vars),
        "nominal_vars": list(nominal_vars),
        "artifacts": {
            "model_ranking": str(paths.metrics_dir / "model_ranking_OOF.csv"),
            "threshold_metrics_json": str(threshold_metrics_path),
            "threshold_deploy_json": str(threshold_deploy_path),
            "champion_model_archive": str(model_archive_path),
            "champion_deploy_pipeline": str(deploy_path),
            "champion_external_metrics": str(champion_external_metrics_path),
            "prospective_prediction_report": str(prospective_report_path),
        },
    }

    summary_path = paths.ml_dir / "oof_external_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(summary), f, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "results": results,
        "ranking_df": ranking_df,
        "champion_name": champion_name,
        "champion_result": champion_result,
        "threshold_bundle": threshold_bundle,
        "X_retro": X_retro,
        "y_retro": y_retro,
        "X_external": X_external,
        "y_external": y_external,
        "retro_source": retro_source,
        "external_source": external_source,
    }
