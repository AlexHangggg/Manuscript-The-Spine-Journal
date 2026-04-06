from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from manuscript_ml_upgrade_core import (
    BenchmarkPaths,
    CandidateResult,
    SharedBullEyeModelWrapper,
    get_feature_names_from_preprocessor,
)

warnings.filterwarnings("ignore")


TREE_MODEL_TYPES = (
    RandomForestClassifier,
    GradientBoostingClassifier,
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    XGBClassifier,
    LGBMClassifier,
)
LINEAR_MODEL_TYPES = (
    LogisticRegression,
    LinearDiscriminantAnalysis,
)


def _model_tag(name: str) -> str:
    tag = str(name).strip().lower().replace(" ", "_").replace("/", "_")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag


def _sample_df(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=max_n, random_state=seed).reset_index(drop=True)


def _normalize_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        if len(values) >= 2:
            return np.asarray(values[1], dtype=float)
        return np.asarray(values[0], dtype=float)
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return np.asarray(arr[:, :, 1], dtype=float)
    return np.asarray(arr, dtype=float)


def _unwrap_model(model: Any) -> Any:
    if isinstance(model, SharedBullEyeModelWrapper):
        return _unwrap_model(getattr(model, "model_", model.model))
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _resolve_pipeline(model: Any) -> Pipeline | None:
    if isinstance(model, SharedBullEyeModelWrapper):
        return _resolve_pipeline(getattr(model, "model_", model.model))
    if isinstance(model, Pipeline):
        return model
    return None


def _canonicalize_model_input(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    X_df = pd.DataFrame(X).copy()
    if isinstance(model, SharedBullEyeModelWrapper):
        return model.prepare_features(X_df)
    if isinstance(model, Pipeline):
        if "bull_eye" in model.named_steps:
            return model.named_steps["bull_eye"].transform(X_df)
        return X_df

    base_models = getattr(model, "base_models", None)
    if isinstance(base_models, dict) and base_models:
        first_model = next(iter(base_models.values()))
        return _canonicalize_model_input(first_model, X_df)
    return X_df


def _pipeline_transform(
    pipeline: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    X_work = pd.DataFrame(X).copy()
    source_cols = list(X_work.columns)
    pipeline_obj = pipeline
    if isinstance(pipeline_obj, SharedBullEyeModelWrapper):
        X_work = pipeline_obj.prepare_features(X_work)
        source_cols = list(X_work.columns)
        pipeline_obj = getattr(pipeline_obj, "model_", pipeline_obj.model)
    if isinstance(pipeline_obj, Pipeline) and "bull_eye" in pipeline_obj.named_steps:
        X_work = pipeline_obj.named_steps["bull_eye"].transform(X_work)
    if not isinstance(pipeline_obj, Pipeline) or "preprocess" not in pipeline_obj.named_steps:
        return np.asarray(X_work), source_cols, X_work

    preprocessor = pipeline_obj.named_steps["preprocess"]
    transformed = preprocessor.transform(X_work)
    feature_names = get_feature_names_from_preprocessor(preprocessor, source_cols)
    return np.asarray(transformed), feature_names, X_work


def _fit_raw_codec(df_reference: pd.DataFrame) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for col in df_reference.columns:
        series = df_reference[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            metadata.append(
                {
                    "name": col,
                    "kind": "numeric",
                    "fill": fill,
                }
            )
        else:
            categories = [str(v) for v in pd.Series(series).dropna().astype(str).unique().tolist()]
            if not categories:
                categories = ["__MISSING__"]
            metadata.append(
                {
                    "name": col,
                    "kind": "categorical",
                    "categories": categories,
                }
            )
    return metadata


def _encode_raw_df(df: pd.DataFrame, metadata: list[dict[str, Any]]) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for meta in metadata:
        col = meta["name"]
        series = df[col]
        if meta["kind"] == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").fillna(meta["fill"]).astype(float)
            encoded[col] = numeric
        else:
            mapping = {category: idx for idx, category in enumerate(meta["categories"])}
            encoded[col] = series.map(
                lambda v: mapping.get(str(v), -1) if pd.notna(v) else -1
            ).astype(float)
    return encoded


def _decode_raw_array(
    encoded_array: np.ndarray,
    metadata: list[dict[str, Any]],
) -> pd.DataFrame:
    array = np.asarray(encoded_array, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    decoded = pd.DataFrame(index=np.arange(array.shape[0]))
    for idx, meta in enumerate(metadata):
        values = array[:, idx]
        col = meta["name"]
        if meta["kind"] == "numeric":
            decoded[col] = pd.Series(values.astype(float), index=decoded.index, dtype=float)
        else:
            categories = meta["categories"]
            indices = np.rint(values).astype(int)
            restored = []
            for code in indices:
                if code < 0 or code >= len(categories):
                    restored.append(np.nan)
                else:
                    restored.append(categories[code])
            # Keep categorical columns as object even when a perturbed SHAP sample
            # decodes to all-missing values; otherwise pandas promotes the column
            # to float and downstream OneHotEncoder will error on string categories.
            decoded[col] = pd.Series(restored, index=decoded.index, dtype=object)
    return decoded


def _save_shap_artifacts(
    shap_values: np.ndarray,
    plot_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    paths: BenchmarkPaths,
    model_tag: str,
) -> dict[str, str]:
    shap_values = np.asarray(shap_values, dtype=float)
    feature_names = list(plot_features.columns)

    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_values_path = paths.shap_dir / "shap_values.csv"
    shap_values_df.to_csv(shap_values_path, index=False, encoding="utf-8-sig")

    raw_features_path = paths.shap_dir / "shap_source_features.csv"
    raw_features.to_csv(raw_features_path, index=False, encoding="utf-8-sig")

    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_path = paths.shap_dir / "shap_importance.csv"
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    max_display = min(20, len(feature_names))

    plt.figure(figsize=(8.5, 6.0))
    shap.summary_plot(
        shap_values,
        features=plot_features,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    bar_png = paths.figures_dir / f"{model_tag}_shap_bar.png"
    bar_pdf = paths.figures_dir / f"{model_tag}_shap_bar.pdf"
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.savefig(bar_pdf, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.5, 6.0))
    shap.summary_plot(
        shap_values,
        features=plot_features,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    beeswarm_png = paths.figures_dir / f"{model_tag}_shap_beeswarm.png"
    beeswarm_pdf = paths.figures_dir / f"{model_tag}_shap_beeswarm.pdf"
    plt.savefig(beeswarm_png, dpi=300, bbox_inches="tight")
    plt.savefig(beeswarm_pdf, bbox_inches="tight")
    plt.close()

    return {
        "shap_values": str(shap_values_path),
        "shap_source_features": str(raw_features_path),
        "shap_importance": str(importance_path),
        "shap_bar_png": str(bar_png),
        "shap_bar_pdf": str(bar_pdf),
        "shap_beeswarm_png": str(beeswarm_png),
        "shap_beeswarm_pdf": str(beeswarm_pdf),
    }


def _tree_or_linear_shap(
    champion_model: Any,
    model_type: str,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, str]:
    inner_pipeline = _resolve_pipeline(champion_model)
    if inner_pipeline is None or "model" not in inner_pipeline.named_steps:
        raise TypeError("Tree/linear SHAP currently expects a Pipeline champion model.")

    model_core = inner_pipeline.named_steps["model"]
    background_t, feature_names, _ = _pipeline_transform(champion_model, X_background)
    explain_t, _, explain_raw = _pipeline_transform(champion_model, X_explain)

    plot_features = pd.DataFrame(explain_t, columns=feature_names)
    raw_features = explain_raw.reset_index(drop=True)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model_core)
        shap_values = _normalize_shap_values(explainer.shap_values(explain_t))
        return shap_values, plot_features, raw_features, "tree"

    try:
        explainer = shap.LinearExplainer(model_core, background_t)
        shap_values = _normalize_shap_values(explainer.shap_values(explain_t))
        return shap_values, plot_features, raw_features, "linear"
    except Exception:
        predict_fn = lambda x: champion_model.predict_proba(
            _canonicalize_model_input(
                champion_model,
                pd.DataFrame(x, columns=X_explain.columns),
            ).reset_index(drop=True)
        )[:, 1]
        background_small = _canonicalize_model_input(champion_model, X_background).reset_index(drop=True)
        explain_small = _canonicalize_model_input(champion_model, X_explain).reset_index(drop=True)
        explainer = shap.KernelExplainer(predict_fn, background_small)
        shap_values = _normalize_shap_values(explainer.shap_values(explain_small, nsamples=200))
        return shap_values, explain_small, explain_small, "kernel_fallback_from_linear"


def _model_agnostic_shap(
    champion_model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, str]:
    X_background_ready = _canonicalize_model_input(champion_model, X_background).reset_index(drop=True)
    X_explain_ready = _canonicalize_model_input(champion_model, X_explain).reset_index(drop=True)

    metadata = _fit_raw_codec(X_background_ready)
    background_encoded = _encode_raw_df(X_background_ready, metadata)
    explain_encoded = _encode_raw_df(X_explain_ready, metadata)

    def predict_fn(encoded_matrix: np.ndarray) -> np.ndarray:
        decoded_df = _decode_raw_array(encoded_matrix, metadata)
        decoded_df = _canonicalize_model_input(champion_model, decoded_df).reset_index(drop=True)
        proba = champion_model.predict_proba(decoded_df)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    explainer = shap.KernelExplainer(predict_fn, background_encoded)
    shap_values = _normalize_shap_values(explainer.shap_values(explain_encoded, nsamples=200))
    return shap_values, explain_encoded, X_explain_ready, "kernel"


def run_champion_shap(
    champion_name: str,
    champion_result: CandidateResult,
    X_retro: pd.DataFrame,
    X_external: pd.DataFrame,
    paths: BenchmarkPaths,
    seed: int = 42,
    background_size: int = 40,
    explain_size: int = 24,
) -> dict[str, Any]:
    paths.ensure()
    paths.shap_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    model_tag = _model_tag(champion_name)
    champion_model = champion_result.final_model
    model_core = _unwrap_model(champion_model)

    X_background = _sample_df(X_retro, max_n=background_size, seed=seed)
    base_explain_source = X_external if len(X_external) else X_retro
    X_explain = _sample_df(base_explain_source, max_n=explain_size, seed=seed)

    if isinstance(model_core, TREE_MODEL_TYPES):
        shap_values, plot_features, raw_features, shap_method = _tree_or_linear_shap(
            champion_model=champion_model,
            model_type="tree",
            X_background=X_background,
            X_explain=X_explain,
        )
    elif isinstance(model_core, LINEAR_MODEL_TYPES):
        shap_values, plot_features, raw_features, shap_method = _tree_or_linear_shap(
            champion_model=champion_model,
            model_type="linear",
            X_background=X_background,
            X_explain=X_explain,
        )
    else:
        shap_values, plot_features, raw_features, shap_method = _model_agnostic_shap(
            champion_model=champion_model,
            X_background=X_background,
            X_explain=X_explain,
        )

    artifact_paths = _save_shap_artifacts(
        shap_values=shap_values,
        plot_features=plot_features,
        raw_features=raw_features,
        paths=paths,
        model_tag=model_tag,
    )

    summary = {
        "champion_model": champion_name,
        "model_tag": model_tag,
        "shap_method": shap_method,
        "background_size": int(len(X_background)),
        "explain_size": int(len(X_explain)),
        "artifacts": artifact_paths,
    }
    summary_path = paths.shap_dir / "shap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary
