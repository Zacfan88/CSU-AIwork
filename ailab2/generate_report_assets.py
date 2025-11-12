#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate additional report assets:
- RMSE/MAE bar charts per target (from metrics_summary.csv)
- Residual histograms, QQ plots, and true-vs-pred overlay lines
  using prediction CSVs in air_quality_outputs_multi/<TARGET>/preds_*.csv
- Ridge/Lasso coefficient bar charts (Top-N by absolute value)
  using air_quality_outputs_multi_compat/<TARGET>/coefficients_*.csv

All images are saved to the project root with clear file names.
"""

from pathlib import Path
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

# Inputs
METRICS_SUMMARY = BASE_DIR / "metrics_summary.csv"
PRED_ROOT = BASE_DIR / "air_quality_outputs_multi"
COEF_ROOT = BASE_DIR / "air_quality_outputs_multi_compat"
DATA_FILE = BASE_DIR / "AirQualityUCI.xlsx"

# Targets and model mapping
TARGETS = ["CO_GT", "NO2_GT", "C6H6_GT"]
METRIC_TARGETS = ["CO(GT)", "NO2(GT)", "C6H6(GT)"]
MODEL_FILE_MAP = {
    "LinearRegression_Univariate": "preds_linear_univariate.csv",
    "LinearRegression_Multivariate": "preds_linear_multivariate.csv",
    "RidgeCV": "preds_ridge.csv",
    "LassoCV": "preds_lasso.csv",
}

# Mapping between metrics target labels and directory codes
TARGET_LABEL_TO_CODE = {"CO(GT)": "CO_GT", "NO2(GT)": "NO2_GT", "C6H6(GT)": "C6H6_GT"}
TARGET_CODE_TO_LABEL = {v: k for k, v in TARGET_LABEL_TO_CODE.items()}

def safe_save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_metrics_bars():
    if not METRICS_SUMMARY.exists():
        print(f"[Warn] metrics summary not found: {METRICS_SUMMARY}")
        return
    df = pd.read_csv(METRICS_SUMMARY)
    # Normalize column names to expected schema
    cols = {c.lower(): c for c in df.columns}
    # map lowercase keys to actual column names
    col_target = cols.get("target", "Target" if "Target" in df.columns else None)
    col_model = cols.get("model", "Model" if "Model" in df.columns else None)
    col_rmse = cols.get("rmse", "RMSE" if "RMSE" in df.columns else None)
    col_mae = cols.get("mae", "MAE" if "MAE" in df.columns else None)
    if not all([col_target, col_model, col_rmse, col_mae]):
        print("[Warn] metrics_summary.csv missing required columns; skip bar charts.")
        return
    # 统一模型顺序
    order = ["LinearRegression_Univariate", "LinearRegression_Multivariate", "RidgeCV", "LassoCV"]
    for tgt in METRIC_TARGETS:
        dft = df[df[col_target] == tgt].copy()
        if dft.empty:
            print(f"[Warn] empty metrics for target: {tgt}")
            continue
        dft[col_model] = pd.Categorical(dft[col_model], categories=order, ordered=True)
        dft.sort_values(col_model, inplace=True)
        # RMSE 柱状图
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(dft[col_model], dft[col_rmse], color=["#6baed6", "#3182bd", "#31a354", "#e6550d"], alpha=0.8)
        ax.set_title(f"{tgt} - RMSE Comparison")
        ax.set_ylabel("RMSE")
        ax.set_xticklabels(dft[col_model], rotation=20, ha="right")
        for i, v in enumerate(dft[col_rmse].values):
            ax.text(i, v + 0.02 * (dft[col_rmse].max()), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        safe_save(fig, BASE_DIR / f"bar_rmse_{tgt}.png")
        # MAE 柱状图
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(dft[col_model], dft[col_mae], color=["#9ecae1", "#6baed6", "#74c476", "#fd8d3c"], alpha=0.8)
        ax.set_title(f"{tgt} - MAE Comparison")
        ax.set_ylabel("MAE")
        ax.set_xticklabels(dft[col_model], rotation=20, ha="right")
        for i, v in enumerate(dft[col_mae].values):
            ax.text(i, v + 0.02 * (dft[col_mae].max()), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        safe_save(fig, BASE_DIR / f"bar_mae_{tgt}.png")

def load_preds_csv(path: Path):
    if not path.exists():
        return None
    try:
        dfp = pd.read_csv(path)
    except Exception:
        return None
    if set(["y_true", "y_pred"]).issubset(dfp.columns):
        return dfp[["y_true", "y_pred"]].copy()
    return None

def plot_preds_derived():
    for tgt in TARGETS:
        tgt_dir = PRED_ROOT / tgt
        if not tgt_dir.exists():
            print(f"[Warn] prediction directory not found: {tgt_dir}")
            continue
        for model, fname in MODEL_FILE_MAP.items():
            dfp = load_preds_csv(tgt_dir / fname)
            if dfp is None or dfp.empty:
                print(f"[Warn] {tgt} - {model} predictions unavailable: {fname}")
                continue
            y_true = dfp["y_true"].values
            y_pred = dfp["y_pred"].values
            resid = y_true - y_pred
            # 残差直方图
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.hist(resid, bins=30, color="#6baed6", alpha=0.85, edgecolor="white")
            ax.set_title(f"Residual Histogram - {tgt} - {model}")
            ax.set_xlabel("Residual (True - Pred)")
            ax.set_ylabel("Count")
            safe_save(fig, BASE_DIR / f"residual_hist_{tgt}_{model}.png")
            # QQ图
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            stats.probplot(resid, dist="norm", plot=ax)
            ax.set_title(f"QQ Plot - Residuals - {tgt} - {model}")
            safe_save(fig, BASE_DIR / f"qq_{tgt}_{model}.png")
            # 真值-预测折线叠加（按样本顺序）
            fig, ax = plt.subplots(figsize=(8.0, 4.0))
            ax.plot(y_true, label="True", color="#2b8cbe", linewidth=1.3)
            ax.plot(y_pred, label="Pred", color="#e6550d", linewidth=1.1, alpha=0.9)
            ax.set_title(f"True vs Pred (Sequence) - {tgt} - {model}")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value")
            ax.legend()
            safe_save(fig, BASE_DIR / f"overlay_line_{tgt}_{model}.png")

def plot_coeff_bars():
    for tgt in TARGETS:
        coef_dir = COEF_ROOT / tgt
        if not coef_dir.exists():
            print(f"[Warn] coefficients directory not found: {coef_dir}")
            continue
        for mdl in ["ridge", "lasso"]:
            path = coef_dir / f"coefficients_{mdl}.csv"
            if not path.exists():
                print(f"[Warn] missing coefficients file: {path}")
                continue
            try:
                dfc = pd.read_csv(path)
            except Exception:
                print(f"[Warn] cannot read coefficients file: {path}")
                continue
            if not set(["feature", "coefficient"]).issubset(dfc.columns):
                print(f"[Warn] unexpected columns in: {path}")
                continue
            dfc["abs_coef"] = dfc["coefficient"].abs()
            dfc = dfc.sort_values("abs_coef", ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.barh(dfc["feature"], dfc["coefficient"], color="#74c476", alpha=0.85)
            ax.axvline(0, color="#555", linewidth=1)
            ax.set_title(f"Top Coefficients ({mdl.title()}) - {tgt}")
            ax.set_xlabel("Coefficient")
            ax.invert_yaxis()
    safe_save(fig, BASE_DIR / f"coef_bar_{tgt}_{mdl.title()}.png")

# ===== Composite figures ===== #
def _read_metrics_for_target(target_label: str):
    if not METRICS_SUMMARY.exists():
        return None
    df = pd.read_csv(METRICS_SUMMARY)
    cols = {c.lower(): c for c in df.columns}
    ct = cols.get("target", "Target" if "Target" in df.columns else None)
    cm = cols.get("model", "Model" if "Model" in df.columns else None)
    cr = cols.get("rmse", "RMSE" if "RMSE" in df.columns else None)
    ca = cols.get("mae", "MAE" if "MAE" in df.columns else None)
    if not all([ct, cm, cr, ca]):
        return None
    dft = df[df[ct] == target_label].copy()
    dft.rename(columns={ct: "Target", cm: "Model", cr: "RMSE", ca: "MAE"}, inplace=True)
    return dft

def _best_model_from_metrics(dft: pd.DataFrame):
    if dft is None or dft.empty:
        return None
    # prefer lowest RMSE
    idx = dft["RMSE"].astype(float).idxmin()
    return dft.loc[idx, "Model"]

def _plot_bar(ax, dft: pd.DataFrame, metric: str, title: str):
    order = ["LinearRegression_Univariate", "LinearRegression_Multivariate", "RidgeCV", "LassoCV"]
    dft = dft.copy()
    dft["Model"] = pd.Categorical(dft["Model"], categories=order, ordered=True)
    dft.sort_values("Model", inplace=True)
    colors_map = {
        "RMSE": ["#6baed6", "#3182bd", "#31a354", "#e6550d"],
        "MAE": ["#9ecae1", "#6baed6", "#74c476", "#fd8d3c"],
    }
    ax.bar(dft["Model"], dft[metric], color=colors_map.get(metric, ["#6baed6"]*len(dft)), alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticklabels(dft["Model"], rotation=18, ha="right")
    vmax = float(dft[metric].max())
    for i, v in enumerate(dft[metric].astype(float).values):
        ax.text(i, v + 0.02 * vmax, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

def _load_best_preds(target_code: str, best_model: str):
    fname = MODEL_FILE_MAP.get(best_model)
    if not fname:
        return None
    dfp = load_preds_csv(PRED_ROOT / target_code / fname)
    return dfp

def _plot_coef_on_ax(ax, target_code: str, best_model: str):
    coef_dir = COEF_ROOT / target_code
    mdl = None
    if best_model == "RidgeCV":
        mdl = "ridge"
    elif best_model == "LassoCV":
        mdl = "lasso"
    else:
        mdl = "linear"
    path = coef_dir / f"coefficients_{mdl}.csv"
    if not path.exists():
        ax.set_title("Coefficients: file missing")
        ax.axis("off")
        return
    try:
        dfc = pd.read_csv(path)
    except Exception:
        ax.set_title("Coefficients: read error")
        ax.axis("off")
        return
    if not set(["feature", "coefficient"]).issubset(dfc.columns):
        ax.set_title("Coefficients: unexpected columns")
        ax.axis("off")
        return
    dfc["abs_coef"] = dfc["coefficient"].abs()
    dfc = dfc.sort_values("abs_coef", ascending=False).head(10)
    ax.barh(dfc["feature"], dfc["coefficient"], color="#74c476", alpha=0.85)
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_title(f"Top Coefficients ({best_model})")
    ax.set_xlabel("Coefficient")
    ax.invert_yaxis()

def compose_target_summary(target_code: str):
    target_label = TARGET_CODE_TO_LABEL.get(target_code, target_code)
    dft = _read_metrics_for_target(target_label)
    if dft is None or dft.empty:
        print(f"[Warn] metrics unavailable for {target_label}")
        return
    best_model = _best_model_from_metrics(dft)
    # figure layout 3x2
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])
    ax_overlay = fig.add_subplot(gs[1, 0])
    ax_resid = fig.add_subplot(gs[1, 1])
    ax_qq = fig.add_subplot(gs[2, 0])
    ax_coef = fig.add_subplot(gs[2, 1])
    _plot_bar(ax_rmse, dft, "RMSE", f"{target_label} - RMSE Comparison")
    _plot_bar(ax_mae, dft, "MAE", f"{target_label} - MAE Comparison")
    dfp = _load_best_preds(target_code, best_model)
    if dfp is not None and not dfp.empty:
        y_true = dfp["y_true"].values
        y_pred = dfp["y_pred"].values
        resid = y_true - y_pred
        # overlay
        ax_overlay.plot(y_true, label="True", color="#2b8cbe", linewidth=1.2)
        ax_overlay.plot(y_pred, label="Pred", color="#e6550d", linewidth=1.0, alpha=0.9)
        ax_overlay.set_title(f"True vs Pred - Best Model ({best_model})")
        ax_overlay.set_xlabel("Sample Index")
        ax_overlay.set_ylabel("Value")
        ax_overlay.legend(fontsize=8)
        # residual hist
        ax_resid.hist(resid, bins=30, color="#6baed6", alpha=0.85, edgecolor="white")
        ax_resid.set_title("Residual Histogram (Best Model)")
        ax_resid.set_xlabel("Residual")
        ax_resid.set_ylabel("Count")
        # qq plot
        stats.probplot(resid, dist="norm", plot=ax_qq)
        ax_qq.set_title("QQ Plot (Residuals)")
    else:
        ax_overlay.axis("off"); ax_overlay.set_title("Predictions not available")
        ax_resid.axis("off"); ax_resid.set_title("Residuals not available")
        ax_qq.axis("off")
    _plot_coef_on_ax(ax_coef, target_code, best_model)
    fig.suptitle(f"Target Summary - {target_label}", fontsize=13, y=0.98)
    safe_save(fig, BASE_DIR / f"summary_{target_code}.png")

def compose_overview():
    # Top row: RMSE bars for three targets, Bottom row: selected correlation heatmap
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], hspace=0.32, wspace=0.25)
    axes_rmse = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # draw RMSE bars
    for i, tgt_label in enumerate(METRIC_TARGETS):
        dft = _read_metrics_for_target(tgt_label)
        if dft is None or dft.empty:
            axes_rmse[i].axis("off"); axes_rmse[i].set_title(f"{tgt_label} - metrics missing")
            continue
        _plot_bar(axes_rmse[i], dft, "RMSE", f"{tgt_label} - RMSE")
    # correlation heatmap selected
    ax_heat = fig.add_subplot(gs[1, :])
    try:
        if DATA_FILE.exists():
            df_raw = pd.read_excel(DATA_FILE, engine="openpyxl")
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            df = df_raw.replace(-200, np.nan).copy()
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(df[c], errors="ignore")
            subset = [
                "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
                "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
                "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
            ]
            cols_present = [c for c in subset if c in df.columns]
            if len(cols_present) >= 2:
                corr_sub = df[cols_present].corr()
                sns.heatmap(corr_sub, cmap="Reds", vmin=-1, vmax=1, center=0,
                            annot=True, fmt=".2f", linewidths=0.4, square=True, ax=ax_heat)
                ax_heat.set_title("Correlation Heatmap (Selected)")
            else:
                ax_heat.axis("off"); ax_heat.set_title("Heatmap: insufficient columns")
        else:
            ax_heat.axis("off"); ax_heat.set_title("Heatmap: data file missing")
    except Exception as e:
        ax_heat.axis("off"); ax_heat.set_title(f"Heatmap error: {e}")
    fig.suptitle("Overview: RMSE per Target & Correlation", fontsize=13, y=0.98)
    safe_save(fig, BASE_DIR / "overview_summary.png")

def compose_overlay_best_models():
    # three subplots, each best model overlay line
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, tgt_code in zip(axes, TARGETS):
        tgt_label = TARGET_CODE_TO_LABEL.get(tgt_code, tgt_code)
        dft = _read_metrics_for_target(tgt_label)
        if dft is None or dft.empty:
            ax.axis("off"); ax.set_title(f"{tgt_label} - metrics missing")
            continue
        best_model = _best_model_from_metrics(dft)
        dfp = _load_best_preds(tgt_code, best_model)
        if dfp is None or dfp.empty:
            ax.axis("off"); ax.set_title(f"{tgt_label} - predictions missing")
            continue
        y_true = dfp["y_true"].values
        y_pred = dfp["y_pred"].values
        ax.plot(y_true, label="True", color="#2b8cbe", linewidth=1.2)
        ax.plot(y_pred, label="Pred", color="#e6550d", linewidth=1.0, alpha=0.9)
        ax.set_title(f"{tgt_label} - Best {best_model}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Best Models Overlay (All Targets)", fontsize=13, y=0.98)
    safe_save(fig, BASE_DIR / "overlay_best_models.png")

def main():
    os.environ["PYTHONWARNINGS"] = "ignore"
    print("[Info] Generating bar charts...")
    plot_metrics_bars()
    print("[Info] Generating prediction-derived charts...")
    plot_preds_derived()
    print("[Info] Generating coefficient bar charts...")
    plot_coeff_bars()
    print("[Info] Generating composite summary figures...")
    for tgt in TARGETS:
        compose_target_summary(tgt)
    compose_overview()
    compose_overlay_best_models()
    print("[Info] Generating correlation heatmaps...")
    try:
        # simple loader
        if DATA_FILE.exists():
            df_raw = pd.read_excel(DATA_FILE, engine="openpyxl")
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            # replace -200 with NaN and coerce to numeric
            df = df_raw.replace(-200, np.nan).copy()
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(df[c], errors="ignore")
            # selected subset columns if present
            subset = [
                "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
                "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
                "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
            ]
            cols_present = [c for c in subset if c in df.columns]
            # full numeric
            num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
            # seaborn heatmap for subset
            if len(cols_present) >= 2:
                corr_sub = df[cols_present].corr()
                fig, ax = plt.subplots(figsize=(10, 9))
                sns.heatmap(corr_sub, cmap="Reds", vmin=-1, vmax=1, center=0,
                            annot=True, fmt=".2f", linewidths=0.5, square=True, ax=ax)
                ax.set_title("Correlation Heatmap (Selected Sensors & Weather)")
                safe_save(fig, BASE_DIR / "correlation_heatmap_selected.png")
            # seaborn heatmap for all numeric
            if num_df.shape[1] >= 2:
                corr_all = num_df.corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_all, cmap="Reds", vmin=-1, vmax=1, center=0,
                            annot=False, linewidths=0.2, square=False, ax=ax)
                ax.set_title("Correlation Heatmap (All Numeric Columns)")
                safe_save(fig, BASE_DIR / "correlation_heatmap_all.png")
        else:
            print(f"[Warn] data file not found: {DATA_FILE}")
    except Exception as e:
        print(f"[Warn] failed to generate correlation heatmaps: {e}")
    print("[Done] Images saved to project root.")

if __name__ == "__main__":
    main()