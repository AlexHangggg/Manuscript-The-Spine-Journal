from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
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
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


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
    enable_voting: bool = True
    enable_stacking: bool = True
    ensemble_base_models: tuple[str, ...] = DEFAULT_ENSEMBLE_BASE_MODELS
    shap_mode: str = "champion_only"
    device: str = "cpu"
    model_version: str = "v2.5"
    tabpfn_model_cache_dir: str | None = None
    champion_auc_threshold: float = 0.70
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
    external_metrics: dict[str, float] = field(default_factory=dict)
    final_model: Any = None
    training_log: dict[str, Any] = field(default_factory=dict)
    probability_type: str = "predict_proba"


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

    # Bull_eye: encode missing (contrast-enhanced MRI not performed) as a
    # distinct category rather than imputing unobserved values.
    if "Bull_eye" in X.columns:
        be = pd.to_numeric(X["Bull_eye"], errors="coerce")
        X["Bull_eye"] = be.apply(
            lambda v: str(int(v)) if pd.notna(v) else "Not_assessed"
        )

    for col in nominal_vars:
        if col in X.columns and col not in {"Bull_eye", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Gender"}:
            X[col] = X[col].astype(str).replace({"nan": np.nan, "NaN": np.nan, "": np.nan})
    return X


class BullEyeImputer:
    def __init__(
        self,
        predictors: list[str],
        seed: int = 42,
        n_estimators: int = 100,
        lasso_C: float = 0.5,
        min_features: int = 3,
        max_features: int = 10,
    ) -> None:
        self.predictors = predictors
        self.seed = seed
        self.n_estimators = n_estimators
        self.lasso_C = lasso_C
        self.min_features = min_features
        self.max_features = max_features
        self.lasso = None
        self.clf = None
        self.fallback_value = None
        self.all_cols: list[str] | None = None
        self.selected_feature_indices: list[int] | None = None
        self.encoded_feature_names: list[str] | None = None
        self.selected_feature_names: list[str] | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.num_fill_values: dict[str, float] = {}
        self.ohe = None

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

        return [feature_names[i] for i in selected_indices]

    def fit(self, X_train: pd.DataFrame, y_train: Any = None) -> "BullEyeImputer":
        if "Bull_eye" not in X_train.columns:
            return self

        self.all_cols = [c for c in self.predictors if c in X_train.columns]
        train_mask = X_train["Bull_eye"].notna()
        if len(self.all_cols) == 0 or int(train_mask.sum()) < 20:
            mode = X_train.loc[train_mask, "Bull_eye"].dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            return self

        try:
            X_encoded, y_be, feature_names = self._prepare_data_for_lasso(X_train, train_mask)
            self.encoded_feature_names = feature_names
            self.selected_feature_names = self._select_features_with_lasso(X_encoded, y_be, feature_names)
        except Exception:
            X_encoded, _, feature_names = self._prepare_data_for_lasso(X_train, train_mask)
            self.encoded_feature_names = feature_names
            self.selected_feature_names = feature_names

        self.selected_feature_indices = [
            self.encoded_feature_names.index(name)
            for name in self.selected_feature_names
            if name in self.encoded_feature_names
        ]
        if not self.selected_feature_indices:
            self.selected_feature_indices = list(range(len(self.encoded_feature_names)))

        X_train_be = self._encode_with_fitted_ohe(X_train.loc[train_mask, self.all_cols])
        X_train_be = X_train_be[:, self.selected_feature_indices]
        y_train_be = X_train.loc[train_mask, "Bull_eye"].copy().astype(int)
        if y_train_be.nunique() < 2:
            mode = y_train_be.dropna().mode()
            self.fallback_value = int(mode.iloc[0]) if len(mode) > 0 else 1
            return self

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 3],
        }
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
            return X_out

        X_all_miss = self._encode_with_fitted_ohe(X_out.loc[miss_mask, self.all_cols])
        X_miss = X_all_miss[:, self.selected_feature_indices]
        pred = self.clf.predict(X_miss).astype(int)
        pred = np.clip(pred, 1, 3)
        X_out.loc[miss_mask, "Bull_eye"] = pred
        return X_out


class BullEyeImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        predictors: list[str],
        seed: int = 42,
        n_estimators: int = 100,
        lasso_C: float = 0.5,
        min_features: int = 3,
        max_features: int = 10,
    ) -> None:
        self.predictors = predictors
        self.seed = seed
        self.n_estimators = n_estimators
        self.lasso_C = lasso_C
        self.min_features = min_features
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BullEyeImputerTransformer":
        X_df = pd.DataFrame(X).copy()
        self.imputer_ = BullEyeImputer(
            predictors=self.predictors,
            seed=self.seed,
            n_estimators=self.n_estimators,
            lasso_C=self.lasso_C,
            min_features=self.min_features,
            max_features=self.max_features,
        )
        self.imputer_.fit(X_df, y_train=y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        X_df = self.imputer_.transform(X_df)
        if "Bull_eye" in X_df.columns:
            bull_eye = pd.to_numeric(X_df["Bull_eye"], errors="coerce").round()
            X_df["Bull_eye"] = bull_eye.astype("Int64").astype(str).replace("<NA>", np.nan)
        return X_df


class TabPFNRawClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        feature_cols: list[str],
        continuous_vars: list[str],
        ordinal_vars: list[str],
        nominal_vars: list[str],
        seed: int = 42,
        device: str = "cpu",
        model_version: str = "v2.5",
        balance_probabilities: bool = False,
        model_cache_dir: str | None = None,
    ) -> None:
        self.feature_cols = feature_cols
        self.continuous_vars = continuous_vars
        self.ordinal_vars = ordinal_vars
        self.nominal_vars = nominal_vars
        self.seed = seed
        self.device = device
        self.model_version = model_version
        self.balance_probabilities = balance_probabilities
        self.model_cache_dir = model_cache_dir

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.loc[:, self.feature_cols].copy()
        for col in self.continuous_vars + self.ordinal_vars:
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        for col in self.nominal_vars:
            if col not in X_df.columns:
                continue
            if col == "Bull_eye":
                # Bull_eye already encoded as "1"/"2"/"3"/"Not_assessed" by
                # clean_modeling_features; just ensure string type.
                X_df[col] = X_df[col].astype(str).replace({"nan": "Not_assessed", "NaN": "Not_assessed"})
            else:
                X_df[col] = X_df[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)

        if "Gender" in X_df.columns:
            X_df["Gender"] = _normalize_gender(X_df["Gender"])
        return X_df

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabPFNRawClassifier":
        from tabpfn import TabPFNClassifier

        os.environ["TABPFN_MODEL_VERSION"] = self.model_version
        cache_dir = _discover_tabpfn_cache_dir(self.model_cache_dir, self.model_version)
        model_kwargs: dict[str, Any] = {
            "device": self.device,
            "random_state": self.seed,
            "balance_probabilities": self.balance_probabilities,
        }
        if cache_dir is not None:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = cache_dir
            checkpoint_path = Path(cache_dir) / _checkpoint_name_for_version(self.model_version)
            if checkpoint_path.exists():
                model_kwargs["model_path"] = str(checkpoint_path)
        self.model_ = TabPFNClassifier(**model_kwargs)
        self.model_.fit(self._prepare_X(X), np.asarray(y).astype(int).ravel())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model_.predict_proba(self._prepare_X(X))
        return np.asarray(proba, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare_X(X))


class WeightedSoftVotingEnsemble(BaseEstimator, ClassifierMixin):
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


class OOFStackingEnsemble(BaseEstimator, ClassifierMixin):
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
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clone(estimator)),
        ]
    )


def _traditional_registry(seed: int) -> dict[str, dict[str, Any]]:
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
            "estimator": RandomForestClassifier(random_state=seed, n_jobs=-1),
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
                n_jobs=-1,
                tree_method="hist",
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
                n_jobs=-1,
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
                thread_count=-1,
                allow_writing_files=False,
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
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "search_spaces": {
                "model__n_neighbors": Integer(3, 20),
                "model__weights": Categorical(["uniform", "distance"]),
                "model__metric": Categorical(["euclidean", "manhattan"]),
            },
        },
        "Extra Trees": {
            "family": "traditional",
            "estimator": ExtraTreesClassifier(random_state=seed, n_jobs=-1),
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
) -> tuple[Any, dict[str, Any]]:
    if search_spaces:
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=config.bayes_n_iter,
            cv=StratifiedKFold(
                n_splits=config.inner_cv_folds,
                shuffle=True,
                random_state=config.seed,
            ),
            scoring="roc_auc",
            random_state=config.seed,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, {
            "search_type": "BayesSearchCV",
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
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
    config: UpgradeConfig,
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
        external_proba, probability_type = _prob_from_estimator(final_model, X_external)
        return CandidateResult(
            name=name,
            family=family,
            status="completed",
            oof_proba=oof_proba,
            external_proba=external_proba,
            oof_metrics=_calculate_binary_metrics(y_retro, oof_proba),
            external_metrics=_calculate_binary_metrics(y_external, external_proba),
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
) -> dict[str, dict[str, Any]]:
    continuous_model_vars = [c for c in continuous_vars if c in feature_cols]
    ordinal_model_vars = [c for c in ordinal_vars if c in feature_cols]
    nominal_model_vars = [c for c in nominal_vars if c in feature_cols]

    registry: dict[str, dict[str, Any]] = {}
    for name, spec in _traditional_registry(config.seed).items():
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
            )

        registry[name] = {
            "family": spec["family"],
            "builder": _builder,
            "search_spaces": search_spaces,
        }

    if config.enable_tabpfn:
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
                balance_probabilities=False,
                model_cache_dir=config.tabpfn_model_cache_dir,
            ),
            "search_spaces": None,
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
    return CandidateResult(
        name="Soft Voting",
        family="ensemble_voting",
        status="completed",
        oof_proba=oof_proba,
        external_proba=external_proba,
        oof_metrics=_calculate_binary_metrics(y_retro, oof_proba),
        external_metrics=_calculate_binary_metrics(y_external, external_proba),
        final_model=final_model,
        training_log={
            "base_models": base_names,
            "weights": weight_map,
        },
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
    return CandidateResult(
        name="Stacking",
        family="ensemble_stacking",
        status="completed",
        oof_proba=meta_oof,
        external_proba=external_proba,
        oof_metrics=_calculate_binary_metrics(y_retro, meta_oof),
        external_metrics=_calculate_binary_metrics(y_external, external_proba),
        final_model=final_model,
        training_log={
            "base_models": base_names,
            "meta_model": "LogisticRegression",
            "meta_fold_logs": meta_logs,
            "base_external_shape": list(base_external_matrix.shape),
            "external_n": int(len(X_external)),
        },
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
        }
        for prefix, metrics in (("OOF", result.oof_metrics), ("External", result.external_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
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
        eligible[f"weight_{metric}"] = eligible[f"OOF_{metric}"].astype(float) * float(weight)
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

    registry = _build_candidate_registry(
        feature_cols=feature_cols,
        continuous_vars=continuous_vars,
        ordinal_vars=ordinal_vars,
        nominal_vars=nominal_vars,
        config=config,
    )

    results: dict[str, CandidateResult] = {}
    for name, spec in registry.items():
        result = _fit_single_candidate_oof(
            name=name,
            family=spec["family"],
            estimator_builder=spec["builder"],
            search_spaces=spec.get("search_spaces"),
            X_retro=X_retro,
            y_retro=y_retro,
            X_external=X_external,
            y_external=y_external,
            config=config,
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
        "oof_folds": config.oof_folds,
        "enable_tabpfn": config.enable_tabpfn,
        "enable_voting": config.enable_voting,
        "enable_stacking": config.enable_stacking,
        "ensemble_base_models": list(config.ensemble_base_models),
        "champion_model": champion_name,
        "champion_family": champion_result.family,
        "champion_oof_metrics": champion_result.oof_metrics,
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
