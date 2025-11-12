#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标空气质量回归（兼容旧版 scikit-learn）
改动要点：
- 不再使用 mean_squared_error(..., squared=False)，统一改为 RMSE = sqrt(MSE)。
- GridSearchCV 的 scoring 使用 "neg_mean_squared_error"，外部再开根号得到 RMSE。
- 去掉 Ridge/Lasso 中对 random_state 的显式传参，以提高版本兼容性。
"""

from __future__ import annotations
import os
import warnings
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = str(BASE_DIR / "AirQualityUCI.xlsx")
OUTPUT_ROOT = str(BASE_DIR / "air_quality_outputs_multi_compat")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

TARGETS = ["C6H6(GT)", "NO2(GT)", "CO(GT)"]
MISSING_COL_THRESHOLD = 0.30
USE_TIME_AWARE_SPLIT = False

def read_air_quality(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据文件：{filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(filepath, engine="openpyxl")
    else:
        try:
            df_raw = pd.read_csv(filepath, sep=";", decimal=",")
        except Exception:
            df_raw = pd.read_csv(filepath)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    date_col, time_col = None, None
    for c in df_raw.columns:
        lc = c.lower()
        if lc in ("date", "data"):
            date_col = c
        if lc == "time":
            time_col = c
    if date_col is not None and time_col is not None:
        dt = pd.to_datetime(
            df_raw[date_col].astype(str).str.strip() + " " + df_raw[time_col].astype(str).str.strip(),
            errors="coerce", dayfirst=True
        )
        df_raw.insert(0, "Datetime", dt)
    elif date_col is not None:
        df_raw.insert(0, "Datetime", pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True))
    else:
        df_raw.insert(0, "Datetime", pd.RangeIndex(start=0, stop=len(df_raw), step=1))
    df = df_raw.replace(-200, np.nan)
    for c in df.columns:
        if c == "Datetime":
            continue
        if df[c].dtype == object:
            df[c] = (
                df[c].astype(str).str.replace(",", ".", regex=False).str.replace(";", "", regex=False)
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return df

def drop_high_missing_cols(df: pd.DataFrame, missing_threshold: float) -> pd.DataFrame:
    miss_ratio = df.isna().mean().sort_values(ascending=False)
    to_drop = miss_ratio[miss_ratio > missing_threshold].index.tolist()
    kept = df.drop(columns=to_drop)
    if len(to_drop) > 0:
        print(f"[Info] 剔除缺失占比 > {missing_threshold:.0%} 的列：{to_drop}")
    return kept

def build_feature_sets(df: pd.DataFrame, target: str) -> Tuple[str, List[str], pd.Series]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if c != target]
    corr = df[candidates + [target]].corr(numeric_only=True)[target].drop(labels=[target])
    corr_abs = corr.abs().sort_values(ascending=False)
    if corr_abs.empty:
        raise RuntimeError("相关性分析失败：无可用数值列或目标缺失！")
    best_single = corr_abs.index[0]
    multi_feats = corr_abs[corr_abs >= 0.20].index.tolist()
    if best_single not in multi_feats:
        multi_feats = [best_single] + multi_feats
    print(f"[Info] 目标 {target}: 一元基线特征 = {best_single} (r = {corr[best_single]:.3f})，多元特征数 = {len(multi_feats)}")
    return best_single, multi_feats, corr

def make_design_matrices(df: pd.DataFrame, features: List[str], target: str):
    return df[features].copy(), df[target].copy()

def rmse_manual(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse_manual(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred))
    }

def save_coefficients(model, feature_names: List[str], out_path: str):
    if hasattr(model, "coef_"):
        coefs = np.array(model.coef_).ravel()
        df_coef = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        df_coef["abs_coef"] = df_coef["coefficient"].abs()
        df_coef.sort_values("abs_coef", ascending=False, inplace=True)
        df_coef.drop(columns=["abs_coef"], inplace=True)
        df_coef.to_csv(out_path, index=False, encoding="utf-8-sig")

def plot_correlation_heatmap(num_df: pd.DataFrame, out_path: str):
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(cax)
    ax.set_title("Pearson Correlation (Numeric Columns)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def slugify(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe.strip("._-") or "target"

def run_for_target(df_all: pd.DataFrame, target: str) -> List[Dict[str, object]]:
    out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, slugify(target)))
    print("\n" + "="*70)
    print(f"[Target] {target}  —— 输出目录：{out_dir}")
    if target not in df_all.columns:
        raise KeyError(f"目标列 {target!r} 不存在，实际列名：{df_all.columns.tolist()}")

    df = df_all.copy()
    df = df[df[target].notna()].copy()

    cols_keep = ["Datetime", target] + [c for c in df.columns if c not in ["Datetime", target]]
    df = df[cols_keep]
    num_df = df.select_dtypes(include=[np.number])
    num_df.isna().mean().sort_values(ascending=False).to_csv(
        os.path.join(out_dir, "missing_ratio_before.csv"), encoding="utf-8-sig"
    )
    df = drop_high_missing_cols(df, MISSING_COL_THRESHOLD)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_all = [c for c in numeric_cols if c != target]

    df_imp = df.copy()
    try:
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        df_imp[feature_cols_all + [target]] = imputer.fit_transform(df_imp[feature_cols_all + [target]])
        print("[Info] 使用 KNNImputer 完成缺失填补")
    except Exception as e:
        print("[Warn] KNNImputer 失败，改用中位数填补：", e)
        simp = SimpleImputer(strategy="median")
        df_imp[feature_cols_all + [target]] = simp.fit_transform(df_imp[feature_cols_all + [target]])

    best_single, multi_feats, _ = build_feature_sets(df_imp, target)

    try:
        plot_correlation_heatmap(df_imp.select_dtypes(include=[np.number]),
                                 os.path.join(out_dir, "correlation_heatmap.png"))
    except Exception as e:
        print("[Warn] 相关性热图绘制失败：", e)

    X_uni, y = make_design_matrices(df_imp, [best_single], target)
    X_mul, _ = make_design_matrices(df_imp, multi_feats, target)

    if USE_TIME_AWARE_SPLIT and pd.api.types.is_datetime64_any_dtype(df_imp.get("Datetime", pd.Series([], dtype="datetime64[ns]")).dtype):
        n = len(df_imp)
        split_idx = int(n * 0.8)
        train_idx = np.arange(0, split_idx)
        test_idx = np.arange(split_idx, n)
        X_uni_train, X_uni_test = X_uni.iloc[train_idx], X_uni.iloc[test_idx]
        X_mul_train, X_mul_test = X_mul.iloc[train_idx], X_mul.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_uni_train, X_uni_test, y_train, y_test = train_test_split(
            X_uni, y, test_size=0.2, random_state=RANDOM_STATE
        )
        X_mul_train, X_mul_test, _, _ = train_test_split(
            X_mul, y, test_size=0.2, random_state=RANDOM_STATE
        )

    records: List[Dict[str, object]] = []

    def plot_pred_vs_true(y_true, y_pred, title, fname):
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(y_true, y_pred, s=12, alpha=0.6)
        lo = min(float(np.min(y_true)), float(np.min(y_pred)))
        hi = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)

    # 模型 1：一元线性回归
    pipe_lr_uni = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe_lr_uni.fit(X_uni_train, y_train)
    y_pred_lr_uni = pipe_lr_uni.predict(X_uni_test)
    m1 = evaluate_regression(y_test, y_pred_lr_uni)
    records.append({"Target": target, "Model": "Univariate Linear (baseline)", **m1})
    pd.DataFrame([m1]).to_csv(os.path.join(out_dir, f"metrics_{slugify(target)}_lr_uni.csv"),
                              index=False, encoding="utf-8-sig")

    # 模型 2：多元线性回归
    pipe_lr_mul = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe_lr_mul.fit(X_mul_train, y_train)
    y_pred_lr_mul = pipe_lr_mul.predict(X_mul_test)
    m2 = evaluate_regression(y_test, y_pred_lr_mul)
    records.append({"Target": target, "Model": "Multivariate Linear", **m2})
    pd.DataFrame([m2]).to_csv(os.path.join(out_dir, f"metrics_{slugify(target)}_lr_mul.csv"),
                              index=False, encoding="utf-8-sig")

    # 模型 3：Ridge（CV, scoring=neg_mean_squared_error）
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    ridge_param_grid = {"ridge__alpha": np.logspace(-3, 3, 13)}
    ridge_grid = GridSearchCV(
        estimator=ridge_pipe,
        param_grid=ridge_param_grid,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    ridge_grid.fit(X_mul_train, y_train)
    y_pred_ridge = ridge_grid.predict(X_mul_test)
    m3 = evaluate_regression(y_test, y_pred_ridge)
    records.append({"Target": target, "Model": f"Ridge (alpha={ridge_grid.best_params_['ridge__alpha']:.4g})", **m3})

    # 模型 4：Lasso（CV, scoring=neg_mean_squared_error）
    lasso_pipe = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))])
    lasso_param_grid = {"lasso__alpha": np.logspace(-3, 1, 15)}
    lasso_grid = GridSearchCV(
        estimator=lasso_pipe,
        param_grid=lasso_param_grid,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    lasso_grid.fit(X_mul_train, y_train)
    y_pred_lasso = lasso_grid.predict(X_mul_test)
    m4 = evaluate_regression(y_test, y_pred_lasso)
    records.append({"Target": target, "Model": f"Lasso (alpha={lasso_grid.best_params_['lasso__alpha']:.4g})", **m4})

    # 系数与图
    save_coefficients(pipe_lr_mul.named_steps["lr"], multi_feats, os.path.join(out_dir, "coefficients_linear.csv"))
    save_coefficients(ridge_grid.best_estimator_.named_steps["ridge"], multi_feats,
                      os.path.join(out_dir, "coefficients_ridge.csv"))
    save_coefficients(lasso_grid.best_estimator_.named_steps["lasso"], multi_feats,
                      os.path.join(out_dir, "coefficients_lasso.csv"))

    plot_pred_vs_true(y_test, y_pred_lr_mul, f"Pred vs True - Linear ({target})", f"pred_vs_true_linear_{slugify(target)}.png")
    plot_pred_vs_true(y_test, y_pred_lasso, f"Pred vs True - Lasso ({target})", f"pred_vs_true_lasso_{slugify(target)}.png")

    residuals = y_test - y_pred_lasso
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(y_pred_lasso, residuals, s=10, alpha=0.6)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (True - Pred)")
    ax.set_title(f"Residuals - Lasso ({target})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"residuals_lasso_{slugify(target)}.png"), dpi=160)
    plt.close(fig)

    pd.DataFrame(records).to_csv(os.path.join(out_dir, f"metrics_{slugify(target)}.csv"),
                                 index=False, encoding="utf-8-sig")
    print(f"[Done] {target} 完成。指标：")
    print(pd.DataFrame(records)[["Target", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    return records

def main():
    warnings.filterwarnings("ignore")
    print("========== 多目标：空气质量回归建模（兼容版） ==========")
    print(f"[Info] 读取数据：{FILE_PATH}")
    df_all = read_air_quality(FILE_PATH)

    all_records: List[Dict[str, object]] = []
    for tgt in TARGETS:
        try:
            recs = run_for_target(df_all, tgt)
            all_records.extend(recs)
        except Exception as e:
            print(f"[Error] 处理目标 {tgt} 时出错：{e}")

    if all_records:
        df_all_metrics = pd.DataFrame(all_records)[["Target", "Model", "MAE", "RMSE", "R2"]]
        out_path = os.path.join(OUTPUT_ROOT, "metrics_all_targets.csv")
        df_all_metrics.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("\n===== 所有目标汇总指标 =====")
        print(df_all_metrics.to_string(index=False))
        print(f"\n[Done] 汇总表已保存：{out_path}")
    else:
        print("[Warn] 无可用记录，请检查目标列名是否正确。")

if __name__ == "__main__":
    main()
