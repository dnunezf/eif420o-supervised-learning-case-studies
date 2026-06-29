#!/usr/bin/env python3
"""
Lab03 - Part (d): Cross-Family Regression Model Comparison
EIF420O Inteligencia Artificial - UNA
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Output dirs ──
OUT = Path("outputs_cross_family")
FIG = OUT / "figures"
RES = OUT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# ── Load data ──
df = pd.read_csv("diabetes_regression_lab03.csv")
TARGET = "target"
print(f"Dataset: {df.shape[0]} obs, {df.shape[1]} cols")
print(f"Target range: [{df[TARGET].min():.1f}, {df[TARGET].max():.1f}], mean={df[TARGET].mean():.2f}, std={df[TARGET].std():.2f}")
print(f"Missing: {df.isna().sum().sum()}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# ── Define cross-family models ──
# Each entry: (name, family_label, model_object)
MODELS = [
    # Family 1: Linear
    ("LinearRegression",        "Lineal",       LinearRegression()),
    ("Ridge (α=1)",             "Lineal",       Ridge(alpha=1.0)),
    ("RidgeCV",                 "Lineal",       RidgeCV(alphas=np.logspace(-4, 4, 30))),
    ("Lasso (α=0.01)",         "Lineal",       Lasso(alpha=0.01, max_iter=20000)),
    ("LassoCV",                 "Lineal",       LassoCV(max_iter=20000, cv=5)),
    # Family 2: SVM / Kernel
    ("SVR (rbf)",               "Kernel/SVM",   SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")),
    ("SVR (linear)",            "Kernel/SVM",   SVR(kernel="linear", C=100, epsilon=0.1)),
    ("SVR (poly)",              "Kernel/SVM",   SVR(kernel="poly", C=100, degree=3, epsilon=0.1)),
    # Family 3: Tree
    ("DecisionTree (std)",      "Árbol",        DecisionTreeRegressor(random_state=42)),
    ("DecisionTree (podado)",   "Árbol",        DecisionTreeRegressor(max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42)),
    # Family 4: Ensemble
    ("RandomForest (100)",      "Ensamble",     RandomForestRegressor(n_estimators=100, random_state=42)),
    ("RandomForest (reg)",      "Ensamble",     RandomForestRegressor(n_estimators=120, max_depth=5, min_samples_leaf=4, max_features="sqrt", random_state=42)),
    ("GradientBoosting",        "Ensamble",     GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=2, subsample=0.9, random_state=42)),
]

if HAS_XGB:
    MODELS.append(
        ("XGBoost",             "Ensamble",     XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=2, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=1))
    )

# ── Train, CV and evaluate ──
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"neg_rmse": "neg_root_mean_squared_error",
           "neg_mae": "neg_mean_absolute_error",
           "r2": "r2"}

rows = []
best_models = {}

for name, family, model in MODELS:
    print(f"  Fitting {name}...", end=" ")
    try:
        cv_res = cross_validate(model, X_train_sc, y_train, cv=cv, scoring=scoring, return_train_score=False)
        cv_rmse = -cv_res["test_neg_rmse"].mean()
        cv_mae  = -cv_res["test_neg_mae"].mean()
        cv_r2   = cv_res["test_r2"].mean()

        from sklearn.base import clone
        fitted = clone(model).fit(X_train_sc, y_train)
        y_pred = fitted.predict(X_test_sc)

        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mae  = float(mean_absolute_error(y_test, y_pred))
        test_r2   = float(r2_score(y_test, y_pred))
        test_mse  = float(mean_squared_error(y_test, y_pred))

        rows.append({
            "Modelo": name, "Familia": family,
            "CV_RMSE": round(cv_rmse, 4), "CV_MAE": round(cv_mae, 4), "CV_R2": round(cv_r2, 4),
            "Test_MSE": round(test_mse, 4), "Test_RMSE": round(test_rmse, 4),
            "Test_MAE": round(test_mae, 4), "Test_R2": round(test_r2, 4)
        })
        best_models[name] = {"model": fitted, "y_pred": y_pred, "family": family}
        print(f"OK  RMSE_cv={cv_rmse:.2f}  RMSE_test={test_rmse:.2f}  R2_test={test_r2:.4f}")
    except Exception as e:
        print(f"FAIL: {e}")

benchmark = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
benchmark.to_csv(RES / "benchmark_cross_family.csv", index=False)
print("\n=== BENCHMARK CROSS-FAMILY ===")
print(benchmark.to_string(index=False))

# ── Best per family ──
best_per_family = benchmark.sort_values("CV_RMSE").groupby("Familia", as_index=False).first().sort_values("CV_RMSE")
best_per_family.to_csv(RES / "best_per_family.csv", index=False)
print("\n=== BEST PER FAMILY ===")
print(best_per_family.to_string(index=False))

# ── Overall best model ──
overall_best_name = benchmark.iloc[0]["Modelo"]
overall_best_family = benchmark.iloc[0]["Familia"]
print(f"\n*** Overall best model: {overall_best_name} (Family: {overall_best_family}) ***")

# ── FIGURES ──
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# 1. Full benchmark bar chart - RMSE
fig, ax = plt.subplots(figsize=(10, 6))
data = benchmark.sort_values("Test_RMSE", ascending=True)
colors = {"Lineal": "#2196F3", "Kernel/SVM": "#FF9800", "Árbol": "#4CAF50", "Ensamble": "#9C27B0"}
bar_colors = [colors.get(f, "#999") for f in data["Familia"]]
ax.barh(data["Modelo"], data["Test_RMSE"], color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Test RMSE")
ax.set_title("Comparación entre familias: RMSE en conjunto de prueba")
ax.grid(axis="x", alpha=0.3)
# legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, label=f) for f, c in colors.items()]
ax.legend(handles=handles, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "cross_family_rmse.png", bbox_inches="tight")
plt.close()

# 2. Full benchmark - R2
fig, ax = plt.subplots(figsize=(10, 6))
data2 = benchmark.sort_values("Test_R2", ascending=True)
bar_colors2 = [colors.get(f, "#999") for f in data2["Familia"]]
ax.barh(data2["Modelo"], data2["Test_R2"], color=bar_colors2, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Test R²")
ax.set_title("Comparación entre familias: R² en conjunto de prueba")
ax.grid(axis="x", alpha=0.3)
ax.legend(handles=handles, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "cross_family_r2.png", bbox_inches="tight")
plt.close()

# 3. Best per family
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bpf = best_per_family.sort_values("Test_RMSE", ascending=True)
fam_colors = [colors.get(f, "#999") for f in bpf["Familia"]]

axes[0].barh(bpf["Familia"] + "\n(" + bpf["Modelo"] + ")", bpf["Test_RMSE"], color=fam_colors, edgecolor="white")
axes[0].set_xlabel("Test RMSE")
axes[0].set_title("Mejor modelo por familia — RMSE")
axes[0].grid(axis="x", alpha=0.3)

bpf2 = best_per_family.sort_values("Test_R2", ascending=True)
fam_colors2 = [colors.get(f, "#999") for f in bpf2["Familia"]]
axes[1].barh(bpf2["Familia"] + "\n(" + bpf2["Modelo"] + ")", bpf2["Test_R2"], color=fam_colors2, edgecolor="white")
axes[1].set_xlabel("Test R²")
axes[1].set_title("Mejor modelo por familia — R²")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "best_per_family.png", bbox_inches="tight")
plt.close()

# 4. Predicted vs Real for best model
best_pred = best_models[overall_best_name]["y_pred"]
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(y_test, best_pred, alpha=0.6, edgecolors="k", linewidth=0.3)
mn, mx = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
ax.plot([mn, mx], [mn, mx], "--", color="red", linewidth=1.5, label="Ideal")
ax.set_xlabel("Valor real")
ax.set_ylabel("Valor predicho")
ax.set_title(f"Predicho vs. Real — {overall_best_name}")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "pred_vs_real_best.png", bbox_inches="tight")
plt.close()

# 5. Residuals for best model
residuals = y_test.values - best_pred
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(best_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.3)
ax.axhline(0, linestyle="--", color="red", linewidth=1.5)
ax.set_xlabel("Valor predicho")
ax.set_ylabel("Residuo")
ax.set_title(f"Residuos — {overall_best_name}")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "residuos_best.png", bbox_inches="tight")
plt.close()

# 6. CV vs Test RMSE grouped scatter
fig, ax = plt.subplots(figsize=(8, 6))
for fam, grp in benchmark.groupby("Familia"):
    ax.scatter(grp["CV_RMSE"], grp["Test_RMSE"], s=80, label=fam, color=colors.get(fam, "#999"), edgecolors="k", linewidth=0.5)
    for _, row in grp.iterrows():
        ax.annotate(row["Modelo"], (row["CV_RMSE"], row["Test_RMSE"]), fontsize=6.5, alpha=0.8,
                     xytext=(4, 4), textcoords="offset points")
mn = min(benchmark["CV_RMSE"].min(), benchmark["Test_RMSE"].min()) - 2
mx = max(benchmark["CV_RMSE"].max(), benchmark["Test_RMSE"].max()) + 2
ax.plot([mn, mx], [mn, mx], "--", color="gray", alpha=0.5)
ax.set_xlabel("CV RMSE")
ax.set_ylabel("Test RMSE")
ax.set_title("CV RMSE vs Test RMSE — Comparación entre familias")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "cv_vs_test_rmse.png", bbox_inches="tight")
plt.close()

# 7. Heatmap of metrics per family
metrics_fam = best_per_family.set_index("Familia")[["Test_RMSE", "Test_MAE", "Test_R2"]].copy()
metrics_fam.columns = ["RMSE", "MAE", "R²"]
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(metrics_fam, annot=True, fmt=".2f", cmap="YlOrRd_r", ax=ax, linewidths=0.5)
ax.set_title("Métricas del mejor modelo por familia")
plt.tight_layout()
plt.savefig(FIG / "heatmap_families.png", bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to {FIG}")
print(f"All results saved to {RES}")
print("Done.")
