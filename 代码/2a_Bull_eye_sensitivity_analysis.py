"""
2a_Bull_eye_sensitivity_analysis.py
====================================
敏感性分析：Bull_eye 数据完整性对 Champion 模型外部验证性能的影响

实验设计
--------
1. 基线：外部验证集 Bull_eye 保持真值（前瞻性队列采集的真实数据）
2. 遮盲组：将外部验证集 Bull_eye 全部设为 NaN，强制模型使用 BullEyeImputer 插补
3. 对比：量化 Bull_eye 真值 vs 插补值对 AUC 等指标的影响
4. 扩展：内部数据中，Bull_eye 有真值 vs 无真值子群的 OOF 性能差异

临床意义：如果遮盲后 AUC 显著下降，说明增强 MRI（获取 Bull_eye）对预测精度有重要贡献
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
from manuscript_ml_upgrade_core import (
    _calculate_binary_metrics,
    clean_modeling_features,
)

# ── data files ─────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "文件"
RETRO_PATH = DATA_DIR / "Retrospective data.xlsx"
PROS_PATH = DATA_DIR / "Prospective data.xlsx"
TARGET_COL = "Reabsorption"

# ── auto-discover latest champion model ────────────────────────────────────
RESULTS_ROOT = PROJECT_ROOT / "Results" / "Manuscript_v2"
run_dirs = sorted(RESULTS_ROOT.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
MODEL_PATH = None
for run_dir in run_dirs:
    candidates = list((run_dir / "06_Calculator_Deployment" / "exported_model").glob("best_model_pipeline_*.pkl"))
    if candidates:
        MODEL_PATH = candidates[0]
        RUN_DIR = run_dir
        break

if MODEL_PATH is None:
    print("[ERROR] 找不到 Champion 模型文件。请先运行 2_Data_analysis...py")
    sys.exit(1)

print(f"[INFO] 使用模型: {MODEL_PATH}")
print(f"[INFO] 来自运行: {RUN_DIR.name}")

# ── output directory ───────────────────────────────────────────────────────
OUT_DIR = RUN_DIR / "07_Sensitivity_Analysis_BullEye"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── feature definitions ───────────────────────────────────────────────────
continuous_vars = [
    "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
    "Initial_volume", "RSI", "DHI",
]
ordinal_vars = ["Pfirrmann", "Komori", "MSU"]
nominal_vars = ["Gender", "Herniated_Level", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Bull_eye"]
feature_cols = continuous_vars + ordinal_vars + nominal_vars


# ── load data ──────────────────────────────────────────────────────────────
def load_cohort(path, sheet):
    df = pd.read_excel(str(path), sheet_name=sheet)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[pd.notna(df[TARGET_COL])].copy()
    return df


print("\n[1/5] 加载数据...")
data_retro = load_cohort(RETRO_PATH, "Train")
data_pros = load_cohort(PROS_PATH, "Train_Pors")

retro_bull = data_retro["Bull_eye"].notna().sum()
retro_miss = data_retro["Bull_eye"].isna().sum()
pros_bull = data_pros["Bull_eye"].notna().sum()
pros_miss = data_pros["Bull_eye"].isna().sum()

print(f"    回顾性队列: n={len(data_retro)}, Bull_eye 有值={retro_bull} ({retro_bull/len(data_retro):.1%}), 缺失={retro_miss} ({retro_miss/len(data_retro):.1%})")
print(f"    前瞻性队列: n={len(data_pros)}, Bull_eye 有值={pros_bull} ({pros_bull/len(data_pros):.1%}), 缺失={pros_miss} ({pros_miss/len(data_pros):.1%})")

# ── load champion model ───────────────────────────────────────────────────
print("\n[2/5] 加载 Champion 模型...")
champion_model = joblib.load(str(MODEL_PATH))
print(f"    模型类型: {type(champion_model).__name__}")
if hasattr(champion_model, "weights"):
    print(f"    权重: {champion_model.weights}")

# ── prepare features ──────────────────────────────────────────────────────
print("\n[3/5] 准备特征数据...")

X_pros_real = clean_modeling_features(data_pros, feature_cols, continuous_vars, ordinal_vars, nominal_vars)
y_pros = data_pros[TARGET_COL].astype(int).to_numpy()

# 遮盲：Bull_eye 全部设为 NaN
data_pros_masked = data_pros.copy()
data_pros_masked["Bull_eye"] = np.nan
X_pros_masked = clean_modeling_features(data_pros_masked, feature_cols, continuous_vars, ordinal_vars, nominal_vars)

print(f"    前瞻性特征列: {list(X_pros_real.columns)}")
print(f"    基线 Bull_eye NaN 数: {X_pros_real['Bull_eye'].isna().sum()}/{len(X_pros_real)}")
print(f"    遮盲 Bull_eye NaN 数: {X_pros_masked['Bull_eye'].isna().sum()}/{len(X_pros_masked)}")

# ── predict ────────────────────────────────────────────────────────────────
print("\n[4/5] 生成预测概率...")

proba_real = champion_model.predict_proba(X_pros_real)[:, 1]
proba_masked = champion_model.predict_proba(X_pros_masked)[:, 1]

metrics_real = _calculate_binary_metrics(y_pros, proba_real)
metrics_masked = _calculate_binary_metrics(y_pros, proba_masked)

# ── 扩展分析：内部数据子群 ─────────────────────────────────────────────
print("\n[4.5/5] 内部数据子群分析...")

X_retro = clean_modeling_features(data_retro, feature_cols, continuous_vars, ordinal_vars, nominal_vars)
y_retro = data_retro[TARGET_COL].astype(int).to_numpy()

# 加载 OOF 预测结果（如果有）
oof_ranking_path = RUN_DIR / "04_ML_ModelDevelopment" / "model_ranking_OOF.csv"
ranking_df = pd.read_csv(oof_ranking_path) if oof_ranking_path.exists() else None

# 找到 champion 的 OOF predictions（从 metrics/oof_predictions 加载）
champion_tag = MODEL_PATH.stem.replace("best_model_pipeline_", "")
oof_pred_path = RUN_DIR / "04_ML_ModelDevelopment" / "metrics" / "oof_predictions" / f"{champion_tag}_oof_predictions.csv"
retro_subgroup_metrics = None

if oof_pred_path.exists():
    oof_df = pd.read_csv(oof_pred_path)
    # OOF 文件自带 Bull_eye 列，用它判断有值/无值
    has_bull = oof_df["Bull_eye"].notna().to_numpy()

    # 列名是 PredProb（不是 PredictedProba）
    prob_col = "PredProb" if "PredProb" in oof_df.columns else "PredictedProba"
    if prob_col in oof_df.columns and "TrueLabel" in oof_df.columns:
        oof_proba = oof_df[prob_col].to_numpy()
        oof_true = oof_df["TrueLabel"].to_numpy()

        metrics_has = _calculate_binary_metrics(oof_true[has_bull], oof_proba[has_bull])
        metrics_no = _calculate_binary_metrics(oof_true[~has_bull], oof_proba[~has_bull])
        retro_subgroup_metrics = {
            "Bull_eye_present": {"n": int(has_bull.sum()), "metrics": metrics_has},
            "Bull_eye_missing": {"n": int((~has_bull).sum()), "metrics": metrics_no},
        }
        print(f"    Bull_eye 有值子群 (n={has_bull.sum()}): AUC={metrics_has.get('AUC', 'N/A'):.4f}")
        print(f"    Bull_eye 缺失子群 (n={(~has_bull).sum()}): AUC={metrics_no.get('AUC', 'N/A'):.4f}")
    else:
        print(f"    [WARN] OOF 预测文件缺少 {prob_col}/TrueLabel 列")
else:
    print(f"    [WARN] 未找到 OOF 预测文件: {oof_pred_path}")

# ── 子群基线特征对比（混杂因素排查）─────────────────────────────────────
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

print("\n[4.6/5] 子群基线特征对比（混杂因素排查）...")

has_bull_retro = data_retro["Bull_eye"].notna()
df_has = data_retro[has_bull_retro]
df_no = data_retro[~has_bull_retro]

# 阳性率对比
pos_rate_has = df_has[TARGET_COL].mean()
pos_rate_no = df_no[TARGET_COL].mean()
print(f"    阳性率 (Reabsorption=1): 有值={pos_rate_has:.1%} ({int(df_has[TARGET_COL].sum())}/{len(df_has)}), "
      f"缺失={pos_rate_no:.1%} ({int(df_no[TARGET_COL].sum())}/{len(df_no)})")

# 连续变量对比 (Mann-Whitney U)
baseline_rows = []
print()
print("    连续变量 (Mann-Whitney U):")
print(f"    {'变量':<35} {'有值 median(IQR)':<25} {'缺失 median(IQR)':<25} {'p值':<10}")
print("    " + "-" * 95)
for var in continuous_vars:
    if var not in data_retro.columns:
        continue
    v_has = df_has[var].dropna()
    v_no = df_no[var].dropna()
    if len(v_has) < 2 or len(v_no) < 2:
        continue
    stat, pval = mannwhitneyu(v_has, v_no, alternative="two-sided")
    med_has, q1_has, q3_has = v_has.median(), v_has.quantile(0.25), v_has.quantile(0.75)
    med_no, q1_no, q3_no = v_no.median(), v_no.quantile(0.25), v_no.quantile(0.75)
    sig = " *" if pval < 0.05 else ""
    print(f"    {var:<35} {med_has:.2f} ({q1_has:.2f}-{q3_has:.2f}){'':>5} {med_no:.2f} ({q1_no:.2f}-{q3_no:.2f}){'':>5} {pval:.4f}{sig}")
    baseline_rows.append({
        "Variable": var, "Type": "continuous",
        "BullEye_Present_Median": round(med_has, 3), "BullEye_Present_IQR": f"{q1_has:.3f}-{q3_has:.3f}",
        "BullEye_Missing_Median": round(med_no, 3), "BullEye_Missing_IQR": f"{q1_no:.3f}-{q3_no:.3f}",
        "p_value": round(pval, 4), "Significant": pval < 0.05,
    })

# 分类变量对比 (Chi-squared / Fisher)
cat_vars_check = ["Gender", "Herniated_Level", "Pfirrmann", "Iwabuchi", "Modic",
                   "Komori", "MSU", "Spinal_canal_stenosis"]
print()
print("    分类变量 (Chi-squared / Fisher):")
print(f"    {'变量':<35} {'有值 分布':<30} {'缺失 分布':<30} {'p值':<10}")
print("    " + "-" * 105)
for var in cat_vars_check:
    if var not in data_retro.columns:
        continue
    ct = pd.crosstab(data_retro[var].fillna("NA"), has_bull_retro)
    try:
        if ct.shape[0] == 2 and ct.shape[1] == 2:
            _, pval = fisher_exact(ct)
        else:
            _, pval, _, _ = chi2_contingency(ct)
    except Exception:
        pval = float("nan")
    dist_has = df_has[var].value_counts(dropna=False).to_dict()
    dist_no = df_no[var].value_counts(dropna=False).to_dict()
    # 简化显示
    dist_has_str = str({k: v for k, v in sorted(dist_has.items(), key=lambda x: -x[1])[:4]})
    dist_no_str = str({k: v for k, v in sorted(dist_no.items(), key=lambda x: -x[1])[:4]})
    sig = " *" if pval < 0.05 else ""
    print(f"    {var:<35} {dist_has_str:<30} {dist_no_str:<30} {pval:.4f}{sig}")
    baseline_rows.append({
        "Variable": var, "Type": "categorical",
        "BullEye_Present_Distribution": str(dist_has),
        "BullEye_Missing_Distribution": str(dist_no),
        "p_value": round(pval, 4), "Significant": pval < 0.05,
    })

# 阳性率检验
ct_target = pd.crosstab(data_retro[TARGET_COL], has_bull_retro)
try:
    _, pval_target = fisher_exact(ct_target)
except Exception:
    _, pval_target, _, _ = chi2_contingency(ct_target)
sig_t = " *" if pval_target < 0.05 else ""
print()
print(f"    Reabsorption 阳性率检验 (Fisher): p={pval_target:.4f}{sig_t}")
baseline_rows.append({
    "Variable": TARGET_COL, "Type": "target",
    "BullEye_Present_Rate": round(pos_rate_has, 4),
    "BullEye_Missing_Rate": round(pos_rate_no, 4),
    "p_value": round(pval_target, 4), "Significant": pval_target < 0.05,
})

# 保存基线对比表
baseline_df = pd.DataFrame(baseline_rows)
baseline_path = OUT_DIR / "internal_subgroup_baseline_comparison.csv"
baseline_df.to_csv(baseline_path, index=False, encoding="utf-8-sig")
print(f"\n    [OK] 子群基线对比表: {baseline_path}")


# ── 输出结果 ───────────────────────────────────────────────────────────────
print("\n[5/5] 汇总结果...")
print()
print("=" * 80)
print("  Bull_eye 敏感性分析结果 — 外部验证集 (前瞻性队列, n={})".format(len(y_pros)))
print("=" * 80)

key_metrics = ["AUC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "Youden", "Accuracy"]

header = f"{'指标':<16} {'基线(真值)':<14} {'遮盲(插补)':<14} {'差值':<12} {'变化率':<10}"
print(header)
print("-" * 66)

comparison_rows = []
for m in key_metrics:
    v_real = float(metrics_real.get(m, 0))
    v_mask = float(metrics_masked.get(m, 0))
    diff = v_mask - v_real
    pct = (diff / v_real * 100) if v_real != 0 else float("nan")
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"{m:<16} {v_real:<14.4f} {v_mask:<14.4f} {diff:+.4f} {arrow:<2} {pct:+.1f}%")
    comparison_rows.append({
        "Metric": m,
        "Baseline_RealBullEye": round(v_real, 4),
        "Masked_ImputedBullEye": round(v_mask, 4),
        "Difference": round(diff, 4),
        "Change_Percent": round(pct, 2),
    })

print()

# 内部子群结果
if retro_subgroup_metrics:
    print("=" * 80)
    print("  内部数据 (回顾性队列) — Bull_eye 有值 vs 缺失子群 OOF 性能")
    print("=" * 80)
    has_m = retro_subgroup_metrics["Bull_eye_present"]["metrics"]
    no_m = retro_subgroup_metrics["Bull_eye_missing"]["metrics"]
    n_has = retro_subgroup_metrics["Bull_eye_present"]["n"]
    n_no = retro_subgroup_metrics["Bull_eye_missing"]["n"]

    header2 = f"{'指标':<16} {'有值(n={})'.format(n_has):<14} {'缺失(n={})'.format(n_no):<14} {'差值':<12}"
    print(header2)
    print("-" * 56)
    subgroup_rows = []
    for m in key_metrics:
        v_has = float(has_m.get(m, 0))
        v_no = float(no_m.get(m, 0))
        diff = v_has - v_no
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"{m:<16} {v_has:<14.4f} {v_no:<14.4f} {diff:+.4f} {arrow}")
        subgroup_rows.append({
            "Metric": m,
            f"BullEye_Present_n{n_has}": round(v_has, 4),
            f"BullEye_Missing_n{n_no}": round(v_no, 4),
            "Difference": round(diff, 4),
        })
    print()

# ── 保存结果 ───────────────────────────────────────────────────────────────
# 1. 外部验证对比表
comp_df = pd.DataFrame(comparison_rows)
comp_path = OUT_DIR / "external_bull_eye_sensitivity.csv"
comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
print(f"[OK] 外部验证对比表: {comp_path}")

# 2. 预测概率对比（逐样本）
sample_df = pd.DataFrame({
    "SampleIndex": range(len(y_pros)),
    "TrueLabel": y_pros,
    "BullEye_Original": data_pros["Bull_eye"].to_numpy(),
    "Proba_Baseline": np.round(proba_real, 6),
    "Proba_Masked": np.round(proba_masked, 6),
    "Proba_Diff": np.round(proba_masked - proba_real, 6),
})
sample_path = OUT_DIR / "external_sample_probabilities.csv"
sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
print(f"[OK] 逐样本概率对比: {sample_path}")

# 3. 内部子群分析
if retro_subgroup_metrics:
    sub_df = pd.DataFrame(subgroup_rows)
    sub_path = OUT_DIR / "internal_subgroup_bull_eye.csv"
    sub_df.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 内部子群分析: {sub_path}")

# 4. 汇总 JSON
import json

summary = {
    "analysis": "Bull_eye sensitivity analysis",
    "timestamp": datetime.now().isoformat(),
    "champion_model": MODEL_PATH.name,
    "run_dir": RUN_DIR.name,
    "data_completeness": {
        "retrospective": {"n": len(data_retro), "bull_eye_present": int(retro_bull), "bull_eye_missing": int(retro_miss), "completion_rate": round(retro_bull / len(data_retro), 4)},
        "prospective": {"n": len(data_pros), "bull_eye_present": int(pros_bull), "bull_eye_missing": int(pros_miss), "completion_rate": round(pros_bull / len(data_pros), 4)},
    },
    "external_validation": {
        "baseline_real_bull_eye": {m: round(float(metrics_real.get(m, 0)), 4) for m in key_metrics},
        "masked_imputed_bull_eye": {m: round(float(metrics_masked.get(m, 0)), 4) for m in key_metrics},
    },
}
if retro_subgroup_metrics:
    summary["internal_subgroup"] = {
        "bull_eye_present": {"n": retro_subgroup_metrics["Bull_eye_present"]["n"], "metrics": {m: round(float(has_m.get(m, 0)), 4) for m in key_metrics}},
        "bull_eye_missing": {"n": retro_subgroup_metrics["Bull_eye_missing"]["n"], "metrics": {m: round(float(no_m.get(m, 0)), 4) for m in key_metrics}},
        "positive_rate": {"present": round(pos_rate_has, 4), "missing": round(pos_rate_no, 4), "fisher_p": round(pval_target, 4)},
        "confounders_detected": [r["Variable"] for r in baseline_rows if r.get("Significant", False)],
    }

summary_path = OUT_DIR / "bull_eye_sensitivity_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[OK] 汇总报告: {summary_path}")

print()
print("=" * 80)
print("  分析完成")
print("=" * 80)
