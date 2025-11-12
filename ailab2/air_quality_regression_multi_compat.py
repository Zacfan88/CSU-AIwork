import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "AirQualityUCI.xlsx"
OUTPUT_BASE = BASE_DIR / "air_quality_outputs_multi"
TARGETS = ["CO(GT)", "NO2(GT)", "C6H6(GT)"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

ALPHAS = np.logspace(-3, 3, 20)
TOP_K_FEATURES = 10  # for SelectKBest

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_missing_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    missing_ratio = df.isna().mean()
    out = pd.DataFrame({"column": missing_ratio.index, "missing_ratio": missing_ratio.values})
    out.to_csv(out_dir / "missing_ratio_before.csv", index=False)


def plot_corr_heatmap(df: pd.DataFrame, out_dir: Path, title: str = "Correlation Heatmap") -> None:
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()


def sanitize_name(s: str) -> str:
    return s.replace("(", "_").replace(")", "").replace("/", "_").replace(" ", "_")


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, s=14, alpha=0.7)
    # 45-degree reference line
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.text(0.05, 0.95, f'RMSE={rmse:.3f}\nMAE={mae:.3f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Data Loading & Preprocessing
# -----------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    # AirQualityUCI.xlsx commonly stores Date, Time and sensor columns
    # Use openpyxl engine for reliability with xlsx
    df = pd.read_excel(data_path, engine="openpyxl")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Replace dataset sentinel -200 with NaN for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(-200, np.nan)

    # Build datetime if Date and Time exist; drop raw Date/Time afterwards
    if "Date" in df.columns and "Time" in df.columns:
        try:
            dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce", dayfirst=True)
            df["datetime"] = dt
            # Optional time-derived features (can help models)
            df["hour"] = dt.dt.hour
            df["month"] = dt.dt.month
        except Exception:
            pass
    
    # Drop original Date and Time string columns to avoid leakage and non-numeric issues
    for col in ["Date", "Time"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def split_and_scale(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


# -----------------------------
# Feature Selection
# -----------------------------

def select_top_k_features(X: pd.DataFrame, y: pd.Series, k: int = TOP_K_FEATURES) -> Tuple[np.ndarray, List[str]]:
    # Use f_regression to select top-K predictive features
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_sel = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_cols = X.columns[mask].tolist()
    return X_sel, selected_cols


def best_univariate_feature(df: pd.DataFrame, target: str) -> str:
    corr = df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    corr = corr.dropna()
    if corr.empty:
        # Fallback to the first numeric feature available
        candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        return candidates[0] if candidates else None
    best_feat = corr.abs().sort_values(ascending=False).index[0]
    return best_feat


# -----------------------------
# Modeling
# -----------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae}


def run_models_for_target(df: pd.DataFrame, target: str, out_dir: Path, common_out_dir: Path) -> List[Dict]:
    ensure_dir(out_dir)

    # Save missing ratio and correlation heatmap of the raw cleaned data
    save_missing_ratio(df, out_dir)
    plot_corr_heatmap(df, out_dir, title=f"Correlation Heatmap - {target}")

    # Prepare features/target
    if target not in df.columns:
        print(f"[WARN] Target {target} not in columns; skipping.")
        return []

    # Drop rows with missing target
    df_t = df.dropna(subset=[target]).copy()

    # Features: all numeric columns except target
    feature_cols = [c for c in df_t.select_dtypes(include=[np.number]).columns if c != target]
    if not feature_cols:
        print(f"[WARN] No numeric features for {target}; skipping.")
        return []

    X = df_t[feature_cols]
    y = df_t[target]

    # Impute feature missing values with median
    X = X.fillna(X.median(numeric_only=True))

    # Univariate Linear Regression (one best feature)
    uni_feat = best_univariate_feature(df_t[feature_cols + [target]], target)
    results = []
    t_name = sanitize_name(target)

    if uni_feat is not None:
        X_uni = df_t[[uni_feat]].fillna(df_t[[uni_feat]].median(numeric_only=True))
        X_train_u, X_test_u, y_train_u, y_test_u, scaler_u = split_and_scale(X_uni, y)
        lr_uni = LinearRegression()
        lr_uni.fit(X_train_u, y_train_u)
        y_pred_u = lr_uni.predict(X_test_u)
        metrics_u = evaluate(y_test_u, y_pred_u)
        results.append({"target": target, "model": "LinearRegression_Univariate", "feature": uni_feat, **metrics_u})
        
        # Save predictions
        pd.DataFrame({"y_true": y_test_u, "y_pred": y_pred_u}).to_csv(out_dir / "preds_linear_univariate.csv", index=False)
        # Save prediction plot to common directory
        plot_pred_vs_true(y_test_u, y_pred_u, common_out_dir / f"pred_vs_true_{t_name}_LinearRegression_Univariate.png",
                          title=f"{target} - Linear (univariate: {uni_feat})")
    else:
        print(f"[WARN] Univariate feature selection failed for {target}.")

    # Feature selection for multivariate models
    X_sel, sel_cols = select_top_k_features(X, y, k=TOP_K_FEATURES)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(pd.DataFrame(X_sel, columns=sel_cols), y)

    # Multivariate Linear Regression
    lr_multi = LinearRegression()
    lr_multi.fit(X_train, y_train)
    y_pred_lr = lr_multi.predict(X_test)
    metrics_lr = evaluate(y_test, y_pred_lr)
    results.append({"target": target, "model": "LinearRegression_Multivariate", "features": ",".join(sel_cols), **metrics_lr})
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_lr}).to_csv(out_dir / "preds_linear_multivariate.csv", index=False)
    plot_pred_vs_true(y_test, y_pred_lr, common_out_dir / f"pred_vs_true_{t_name}_LinearRegression_Multivariate.png",
                      title=f"{target} - Linear (multivariate)")

    # Ridge Regression with CV
    ridge = RidgeCV(alphas=ALPHAS, cv=5)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    metrics_ridge = evaluate(y_test, y_pred_ridge)
    results.append({"target": target, "model": "RidgeCV", "alpha": float(ridge.alpha_), **metrics_ridge})
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_ridge}).to_csv(out_dir / "preds_ridge.csv", index=False)
    plot_pred_vs_true(y_test, y_pred_ridge, common_out_dir / f"pred_vs_true_{t_name}_RidgeCV.png",
                      title=f"{target} - RidgeCV (alpha={ridge.alpha_:.4f})")

    # Lasso Regression with CV
    # Use max_iter to be safe on convergence
    lasso = LassoCV(alphas=ALPHAS, cv=5, random_state=RANDOM_STATE, max_iter=5000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    metrics_lasso = evaluate(y_test, y_pred_lasso)
    results.append({"target": target, "model": "LassoCV", "alpha": float(lasso.alpha_), **metrics_lasso})
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_lasso}).to_csv(out_dir / "preds_lasso.csv", index=False)
    plot_pred_vs_true(y_test, y_pred_lasso, common_out_dir / f"pred_vs_true_{t_name}_LassoCV.png",
                      title=f"{target} - LassoCV (alpha={lasso.alpha_:.4f})")

    # Save metrics summary per-target
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    return results


# -----------------------------
# Main
# -----------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    ensure_dir(OUTPUT_BASE)

    # Load and clean
    df = load_data(DATA_PATH)
    df = clean_dataframe(df)

    # Iterate targets and collect summary
    summary_rows: List[Dict] = []
    for target in TARGETS:
        out_dir = OUTPUT_BASE / sanitize_name(target)
        results = run_models_for_target(df, target, out_dir, BASE_DIR)
        summary_rows.extend(results)
        print(f"[INFO] Finished target: {target}")

    # Save combined metrics summary in the same directory as script
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(BASE_DIR / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()