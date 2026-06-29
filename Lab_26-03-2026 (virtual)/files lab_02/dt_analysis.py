#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree Analysis - Lab 02 - EIF420O Artificial Intelligence
Universidad Nacional de Costa Rica
Based on the Supervisado class pattern built in class.

Dataset: BankChurners (Bank Customer Churn Prediction)
Target: Attrition_Flag (Existing Customer / Attrited Customer)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Supervisado Class (extended for DT)
class Supervisado:
    def __init__(self, df):
        self.__df = df

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    def preparar_datos(self, target_col='target', test_size=0.25, random_state=42):
        """
        Prepare data for DT: NO scaling (tree-based methods are scale-invariant).
        """
        X = self.__df.drop(columns=[target_col])
        y = self.__df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, y

    # ---- DT Methods ----
    def modeloDT(self, X_train, y_train, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        model = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        return model

    def predecir(self, model, X_test):
        return model.predict(X_test)

    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        return self.indices_general(MC, list(np.unique(y)))

    def indices_general(self, MC, nombres=None):
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres is not None:
            precision_categoria.columns = nombres
        return {
            "Matriz de Confusión": MC,
            "Precisión Global": precision_global,
            "Error Global": error_global,
            "Precisión por categoría": precision_categoria
        }

# Data Preparation
def preparar_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['ID'])
    le_target = LabelEncoder()
    df['target'] = le_target.fit_transform(df['Attrition_Flag'])
    df = df.drop(columns=['Attrition_Flag'])
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df, le_target

# Visualization Functions
def plot_confusion_matrix(y_test, y_pred, class_names, output_path, title=''):
    MC = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ConfusionMatrixDisplay(confusion_matrix=MC, display_labels=class_names).plot(ax=ax, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_depth_vs_precision(df_results, output_path):
    """Precision vs max_depth for each criterion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for i, crit in enumerate(['gini', 'entropy']):
        ax = axes[i]
        subset = df_results[(df_results['criterion'] == crit) &
                            (df_results['min_samples_leaf'] == 1)]
        for mss in [2, 5, 10]:
            data = subset[subset['min_samples_split'] == mss]
            ax.plot(data['max_depth'], data['precision_global'],
                    marker='o', markersize=4, label=f'min_split={mss}', linewidth=1.5)
        ax.set_xlabel('max_depth', fontsize=11)
        ax.set_ylabel('Global Precision', fontsize=11)
        ax.set_title(f'Criterion: {crit}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 21))
    fig.suptitle('Decision Tree: Precision vs max_depth', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_heatmap_dt(df_results, output_path):
    """Heatmap: max_depth vs min_samples_split (best criterion, min_samples_leaf=1)."""
    best_crit = df_results.groupby('criterion')['precision_global'].mean().idxmax()
    subset = df_results[(df_results['criterion'] == best_crit) &
                        (df_results['min_samples_leaf'] == 1)]
    pivot = subset.pivot_table(index='min_samples_split', columns='max_depth',
                               values='precision_global')
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Global Precision'})
    ax.set_title(f'DT Precision Heatmap (criterion={best_crit}, min_samples_leaf=1)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('max_depth', fontsize=11)
    ax.set_ylabel('min_samples_split', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_leaf_effect(df_results, output_path):
    """Effect of min_samples_leaf on precision for different depths."""
    best_crit = df_results.groupby('criterion')['precision_global'].mean().idxmax()
    subset = df_results[(df_results['criterion'] == best_crit) &
                        (df_results['min_samples_split'] == 2)]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for leaf in sorted(subset['min_samples_leaf'].unique()):
        data = subset[subset['min_samples_leaf'] == leaf]
        ax.plot(data['max_depth'], data['precision_global'],
                marker='o', markersize=4, label=f'min_leaf={leaf}', linewidth=1.5)
    ax.set_xlabel('max_depth', fontsize=11)
    ax.set_ylabel('Global Precision', fontsize=11)
    ax.set_title(f'Effect of min_samples_leaf (criterion={best_crit}, min_split=2)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_top10_dt(df_results, output_path):
    top10 = df_results.nlargest(10, 'precision_global').reset_index(drop=True)
    top10['label'] = top10.apply(
        lambda r: f"d={r['max_depth']}, {r['criterion']}, split={r['min_samples_split']}, leaf={r['min_samples_leaf']}",
        axis=1
    )
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    bars = ax.barh(range(len(top10)), top10['precision_global'],
                   color=sns.color_palette('magma', len(top10)), edgecolor='black')
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['label'], fontsize=9)
    ax.set_xlabel('Global Precision', fontsize=11)
    ax.set_title('Top 10 Decision Tree Configurations', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, top10['precision_global']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return top10


def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(range(len(idx)), importances[idx],
            color=sns.color_palette('crest', len(idx)), edgecolor='black')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(np.array(feature_names)[idx], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Feature Importance - Best Decision Tree', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_tree_visualization(model, feature_names, class_names, output_path):
    fig, ax = plt.subplots(figsize=(28, 12), dpi=120)
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, fontsize=8,
              proportion=True, impurity=True, max_depth=4)
    ax.set_title('Decision Tree Structure (top 4 levels)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_summary_by_depth(df_results, output_path):
    summary = df_results.groupby('max_depth').agg(
        best=('precision_global', 'max'),
        worst=('precision_global', 'min'),
        mean=('precision_global', 'mean'),
        std=('precision_global', 'std')
    ).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(summary['max_depth'], summary['best'], 'g-o', label='Best', markersize=5)
    ax.plot(summary['max_depth'], summary['mean'], 'b--s', label='Mean', markersize=4)
    ax.plot(summary['max_depth'], summary['worst'], 'r-^', label='Worst', markersize=4)
    ax.fill_between(summary['max_depth'],
                    summary['mean'] - summary['std'],
                    summary['mean'] + summary['std'],
                    alpha=0.15, color='blue')
    ax.set_xlabel('max_depth', fontsize=11)
    ax.set_ylabel('Global Precision', fontsize=11)
    ax.set_title('DT Performance Summary by max_depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return summary

# Main Execution
if __name__ == '__main__':
    import os
    output_dir = '/home/claude/output_dt'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("DECISION TREE ANALYSIS - LAB 02 - BANK CHURNERS")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading and preparing data...")
    df, le_target = preparar_dataset('/mnt/user-data/uploads/BankChurners.csv')
    print(f"    Shape: {df.shape}")
    print(f"    Target: {le_target.classes_}")
    print(f"    Distribution:\n{df['target'].value_counts()}")

    sup = Supervisado(df)
    X_train, X_test, y_train, y_test, y = sup.preparar_datos(
        target_col='target', random_state=42
    )
    feature_names = list(X_train.columns)

    # 2. Standard DT
    print("\n[2] Standard DT (gini, no max_depth, min_split=2, min_leaf=1)...")
    model_std = sup.modeloDT(X_train, y_train, criterion='gini', max_depth=None,
                             min_samples_split=2, min_samples_leaf=1)
    y_pred_std = sup.predecir(model_std, X_test)
    idx_std = sup.evaluar(y_test, y_pred_std, y)
    print(f"    Precision: {idx_std['Precisión Global']:.4f}")
    print(f"    Error:     {idx_std['Error Global']:.4f}")
    print(f"    By category:\n{idx_std['Precisión por categoría']}")
    print(f"    Tree depth: {model_std.get_depth()}, leaves: {model_std.get_n_leaves()}")
    print(f"    CM:\n{idx_std['Matriz de Confusión']}")

    plot_confusion_matrix(y_test, y_pred_std, le_target.classes_,
                          os.path.join(output_dir, 'dt_cm_standard.png'),
                          'Confusion Matrix - Standard DT (gini, unrestricted)')

    # 3. Systematic variations
    print("\n[3] Running DT variations...")
    resultados = []
    depths = list(range(1, 21))
    criteria = ['gini', 'entropy']
    splits = [2, 5, 10, 15, 20]
    leaves = [1, 2, 5, 10]

    for crit in criteria:
        for d in depths:
            for mss in splits:
                for msl in leaves:
                    model = sup.modeloDT(X_train, y_train, criterion=crit,
                                         max_depth=d, min_samples_split=mss,
                                         min_samples_leaf=msl)
                    y_pred = sup.predecir(model, X_test)
                    idx = sup.evaluar(y_test, y_pred, y)
                    resultados.append({
                        'criterion': crit, 'max_depth': d,
                        'min_samples_split': mss, 'min_samples_leaf': msl,
                        'precision_global': idx['Precisión Global'],
                        'error_global': idx['Error Global'],
                        'tree_depth': model.get_depth(),
                        'n_leaves': model.get_n_leaves(),
                        'y_pred': y_pred, 'model': model
                    })

    df_results = pd.DataFrame([{k: v for k, v in r.items()
                                 if k not in ['y_pred', 'model']} for r in resultados])
    print(f"    Configs evaluated: {len(df_results)}")
    print(f"    Best: {df_results['precision_global'].max():.4f}")
    print(f"    Worst: {df_results['precision_global'].min():.4f}")

    # 4. Select best
    print("\n[4] Best model...")
    idx_best = df_results['precision_global'].idxmax()
    best = resultados[idx_best]
    best_indices = sup.evaluar(y_test, best['y_pred'], y)
    best_cfg = {k: best[k] for k in ['criterion', 'max_depth',
                'min_samples_split', 'min_samples_leaf']}
    print(f"    Config: {best_cfg}")
    print(f"    Precision: {best_indices['Precisión Global']:.4f}")
    print(f"    Error:     {best_indices['Error Global']:.4f}")
    print(f"    By category:\n{best_indices['Precisión por categoría']}")
    print(f"    CM:\n{best_indices['Matriz de Confusión']}")
    print(f"    Depth: {best['model'].get_depth()}, Leaves: {best['model'].get_n_leaves()}")

    # 5. Classification report
    print("\n[5] Classification Report (Best):")
    report = classification_report(y_test, best['y_pred'], target_names=le_target.classes_)
    print(report)

    # 6. Generate plots
    print("\n[6] Generating visualizations...")
    plot_depth_vs_precision(df_results, os.path.join(output_dir, 'dt_precision_vs_depth.png'))
    plot_heatmap_dt(df_results, os.path.join(output_dir, 'dt_heatmap.png'))
    plot_leaf_effect(df_results, os.path.join(output_dir, 'dt_leaf_effect.png'))
    top10 = plot_top10_dt(df_results, os.path.join(output_dir, 'dt_top10.png'))
    plot_feature_importance(best['model'], feature_names,
                            os.path.join(output_dir, 'dt_feature_importance.png'))
    plot_tree_visualization(best['model'], feature_names, list(le_target.classes_),
                            os.path.join(output_dir, 'dt_tree_structure.png'))
    plot_confusion_matrix(y_test, best['y_pred'], le_target.classes_,
                          os.path.join(output_dir, 'dt_cm_best.png'),
                          f"CM - Best DT ({best_cfg['criterion']}, depth={best_cfg['max_depth']})")
    summary = plot_summary_by_depth(df_results, os.path.join(output_dir, 'dt_summary_by_depth.png'))

    # 7. Print top 10
    print("\n[7] Top 10:")
    print(top10[['criterion', 'max_depth', 'min_samples_split',
                 'min_samples_leaf', 'precision_global']].to_string(index=False))

    # 8. Summary by depth
    print("\n[8] Summary by depth:")
    print(summary.to_string(index=False))

    # Save report
    with open(os.path.join(output_dir, 'dt_report.txt'), 'w') as f:
        f.write(f"Standard DT (gini, unrestricted)\n")
        f.write(f"Precision: {idx_std['Precisión Global']:.4f}\n")
        f.write(f"Error: {idx_std['Error Global']:.4f}\n")
        f.write(f"Depth: {model_std.get_depth()}, Leaves: {model_std.get_n_leaves()}\n\n")
        f.write(f"Best DT: {best_cfg}\n")
        f.write(f"Precision: {best_indices['Precisión Global']:.4f}\n")
        f.write(f"Error: {best_indices['Error Global']:.4f}\n")
        f.write(f"Depth: {best['model'].get_depth()}, Leaves: {best['model'].get_n_leaves()}\n\n")
        f.write("Classification Report (Best):\n")
        f.write(report)

    print("\n[DONE] Outputs:", output_dir)
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
