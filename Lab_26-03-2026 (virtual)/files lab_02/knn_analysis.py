#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Analysis - Lab 02 - EIF420O Artificial Intelligence
Universidad Nacional de Costa Rica

Dataset: BankChurners (Bank Customer Churn Prediction)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Supervisado Class (adapted from M_Caso.py course package)
class Supervisado:
    """
    Supervised learning class following the pythonic model
    developed during the EIF420O course.
    """

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
        Prepare data: separate features/target, scale, split.
        """
        X = self.__df.drop(columns=[target_col])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self.__df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, y

    # ---- KNN Methods ----
    def modeloKNN(self, X_train, y_train, n_neighbors=5, algorithm='auto',
                  weights='uniform', metric='minkowski', p=2):
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=algorithm,
            weights=weights, metric=metric, p=p
        )
        model.fit(X_train, y_train)
        return model

    def predecir(self, model, X_test):
        return model.predict(X_test)

    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MC, list(np.unique(y)))
        return indices

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

    # ---- KNN Standard Analysis ----
    def KNN_standard(self, target_col='target', random_state=42):
        """Standard KNN with default config (k=5, auto, uniform, euclidean)."""
        X_train, X_test, y_train, y_test, y = self.preparar_datos(
            target_col=target_col, random_state=random_state
        )
        model = self.modeloKNN(X_train, y_train, n_neighbors=5,
                               algorithm='auto', weights='uniform',
                               metric='minkowski', p=2)
        y_pred = self.predecir(model, X_test)
        indices = self.evaluar(y_test, y_pred, y)
        return {
            'model': model, 'indices': indices,
            'y_test': y_test, 'y_pred': y_pred,
            'config': {'n_neighbors': 5, 'algorithm': 'auto',
                       'weights': 'uniform', 'metric': 'euclidean'}
        }

    # ---- KNN Variations ----
    def KNN_variaciones(self, target_col='target', random_state=42):
        """
        Systematic exploration of KNN hyperparameters:
        - n_neighbors: 1-20
        - algorithm: auto, ball_tree, kd_tree, brute
        - weights: uniform, distance
        - metric/p: euclidean(p=2), manhattan(p=1), minkowski(p=3)
        """
        X_train, X_test, y_train, y_test, y = self.preparar_datos(
            target_col=target_col, random_state=random_state
        )

        resultados = []
        k_values = list(range(1, 21))
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        weight_opts = ['uniform', 'distance']
        metrics = [
            ('minkowski', 2, 'Euclidean'),
            ('minkowski', 1, 'Manhattan'),
            ('minkowski', 3, 'Minkowski(p=3)')
        ]

        for k in k_values:
            for algo in algorithms:
                for w in weight_opts:
                    for metric, p, metric_name in metrics:
                        model = self.modeloKNN(
                            X_train, y_train,
                            n_neighbors=k, algorithm=algo,
                            weights=w, metric=metric, p=p
                        )
                        y_pred = self.predecir(model, X_test)
                        indices = self.evaluar(y_test, y_pred, y)

                        resultados.append({
                            'k': k,
                            'algorithm': algo,
                            'weights': w,
                            'metric': metric_name,
                            'precision_global': indices['Precisión Global'],
                            'error_global': indices['Error Global'],
                            'y_pred': y_pred,
                            'model': model
                        })

        df_results = pd.DataFrame([{
            k: v for k, v in r.items() if k not in ['y_pred', 'model']
        } for r in resultados])

        return resultados, df_results, X_train, X_test, y_train, y_test, y

    # ---- Best Model Selection ----
    def seleccionar_mejor_modelo(self, resultados, df_results, y_test, y):
        """Select the best model by global precision."""
        idx_best = df_results['precision_global'].idxmax()
        best = resultados[idx_best]
        best_indices = self.evaluar(y_test, best['y_pred'], y)
        return best, best_indices

# Data Preparation
def preparar_dataset(path):
    """Load and prepare BankChurners dataset."""
    df = pd.read_csv(path)

    # Drop ID column
    df = df.drop(columns=['ID'])

    # Encode target: Attrited=1, Existing=0
    le_target = LabelEncoder()
    df['target'] = le_target.fit_transform(df['Attrition_Flag'])
    df = df.drop(columns=['Attrition_Flag'])

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df, le_target

# Visualization Functions
def plot_precision_vs_k(df_results, output_path):
    """Plot global precision vs k for different weight/metric combos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for i, w in enumerate(['uniform', 'distance']):
        ax = axes[i]
        subset = df_results[(df_results['weights'] == w) & (df_results['algorithm'] == 'auto')]
        for metric in subset['metric'].unique():
            data = subset[subset['metric'] == metric]
            ax.plot(data['k'], data['precision_global'], marker='o', markersize=3,
                    label=metric, linewidth=1.5)
        ax.set_xlabel('Number of Neighbors (k)', fontsize=11)
        ax.set_ylabel('Global Precision', fontsize=11)
        ax.set_title(f'Weights: {w}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 21))

    fig.suptitle('KNN: Global Precision vs Number of Neighbors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(df_results, output_path):
    """Compare algorithms for best k."""
    best_k = df_results.groupby('k')['precision_global'].mean().idxmax()
    subset = df_results[(df_results['k'] == best_k) & (df_results['metric'] == 'Euclidean')]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    pivot = subset.pivot_table(index='algorithm', columns='weights',
                               values='precision_global')
    pivot.plot(kind='bar', ax=ax, edgecolor='black', width=0.6)
    ax.set_title(f'Algorithm Comparison (k={best_k}, Euclidean)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Global Precision', fontsize=11)
    ax.set_xlabel('Algorithm', fontsize=11)
    ax.legend(title='Weights', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return best_k


def plot_confusion_matrix(y_test, y_pred, class_names, output_path, title=''):
    """Plot confusion matrix."""
    MC = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=MC, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_top10_models(df_results, output_path):
    """Plot top 10 models by global precision."""
    top10 = df_results.nlargest(10, 'precision_global').reset_index(drop=True)
    top10['label'] = top10.apply(
        lambda r: f"k={int(r['k'])}, {r['weights']}, {r['metric']}", axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.barh(range(len(top10)), top10['precision_global'],
                   color=sns.color_palette('viridis', len(top10)), edgecolor='black')
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['label'], fontsize=9)
    ax.set_xlabel('Global Precision', fontsize=11)
    ax.set_title('Top 10 KNN Configurations', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, top10['precision_global']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return top10


def plot_metrics_heatmap(df_results, output_path):
    """Heatmap of precision by k and metric (weights=distance, algo=auto)."""
    subset = df_results[(df_results['weights'] == 'distance') &
                        (df_results['algorithm'] == 'auto')]
    pivot = subset.pivot_table(index='metric', columns='k', values='precision_global')

    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Global Precision'})
    ax.set_title('KNN Precision Heatmap (weights=distance, algorithm=auto)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('k (Number of Neighbors)', fontsize=11)
    ax.set_ylabel('Distance Metric', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def generate_results_table(df_results, output_path):
    """Generate summary table for LaTeX."""
    # Group by k and get the best precision for each k (across all combos)
    summary = df_results.groupby('k').agg(
        best_precision=('precision_global', 'max'),
        worst_precision=('precision_global', 'min'),
        mean_precision=('precision_global', 'mean'),
        std_precision=('precision_global', 'std')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(summary['k'], summary['best_precision'], 'g-o', label='Best', markersize=5)
    ax.plot(summary['k'], summary['mean_precision'], 'b--s', label='Mean', markersize=4)
    ax.plot(summary['k'], summary['worst_precision'], 'r-^', label='Worst', markersize=4)
    ax.fill_between(summary['k'],
                    summary['mean_precision'] - summary['std_precision'],
                    summary['mean_precision'] + summary['std_precision'],
                    alpha=0.15, color='blue')
    ax.set_xlabel('Number of Neighbors (k)', fontsize=11)
    ax.set_ylabel('Global Precision', fontsize=11)
    ax.set_title('KNN Performance Summary by k', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return summary


def export_standard_vs_best_table(std_indices, best_config, best_indices, output_path):
    """Create comparison table image."""
    data = {
        'Metric': ['Global Precision', 'Global Error',
                    'Precision (Class 0)', 'Precision (Class 1)'],
        'Standard (k=5, uniform, euclidean)': [
            f"{std_indices['Precisión Global']:.4f}",
            f"{std_indices['Error Global']:.4f}",
            f"{std_indices['Precisión por categoría'].iloc[0, 0]:.4f}",
            f"{std_indices['Precisión por categoría'].iloc[0, 1]:.4f}"
        ],
        f"Best (k={best_config['k']}, {best_config['weights']}, {best_config['metric']})": [
            f"{best_indices['Precisión Global']:.4f}",
            f"{best_indices['Error Global']:.4f}",
            f"{best_indices['Precisión por categoría'].iloc[0, 0]:.4f}",
            f"{best_indices['Precisión por categoría'].iloc[0, 1]:.4f}"
        ]
    }
    df_table = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=150)
    ax.axis('off')
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#D9E2F3')
        cell.set_edgecolor('#B4C7E7')
    ax.set_title('Standard vs Best KNN Configuration', fontsize=13,
                 fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return df_table

# Main Execution
if __name__ == '__main__':
    import os
    output_dir = '/home/claude/output_knn'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("KNN ANALYSIS - LAB 02 - BANK CHURNERS DATASET")
    print("=" * 60)

    # 1. Load and prepare data
    print("\n[1] Loading and preparing data...")
    df, le_target = preparar_dataset('/mnt/user-data/uploads/BankChurners.csv')
    print(f"    Dataset shape: {df.shape}")
    print(f"    Target classes: {le_target.classes_}")
    print(f"    Target distribution:\n{df['target'].value_counts()}")

    # 2. Initialize Supervisado
    sup = Supervisado(df)

    # 3. Standard KNN Analysis
    print("\n[2] Running Standard KNN (k=5, uniform, euclidean)...")
    std_result = sup.KNN_standard(target_col='target', random_state=42)
    std_indices = std_result['indices']
    print(f"    Global Precision: {std_indices['Precisión Global']:.4f}")
    print(f"    Global Error:     {std_indices['Error Global']:.4f}")
    print(f"    Precision by category:\n{std_indices['Precisión por categoría']}")
    print(f"    Confusion Matrix:\n{std_indices['Matriz de Confusión']}")

    # Plot standard confusion matrix
    plot_confusion_matrix(
        std_result['y_test'], std_result['y_pred'],
        le_target.classes_,
        os.path.join(output_dir, 'knn_cm_standard.png'),
        title='Confusion Matrix - Standard KNN (k=5, uniform, euclidean)'
    )

    # 4. KNN Variations
    print("\n[3] Running KNN Variations (480 configurations)...")
    resultados, df_results, X_train, X_test, y_train, y_test, y = \
        sup.KNN_variaciones(target_col='target', random_state=42)
    print(f"    Total configurations evaluated: {len(df_results)}")
    print(f"    Best precision: {df_results['precision_global'].max():.4f}")
    print(f"    Worst precision: {df_results['precision_global'].min():.4f}")

    # 5. Select best model
    print("\n[4] Selecting best model...")
    best, best_indices = sup.seleccionar_mejor_modelo(resultados, df_results, y_test, y)
    best_config = {
        'k': best['k'], 'algorithm': best['algorithm'],
        'weights': best['weights'], 'metric': best['metric']
    }
    print(f"    Best config: {best_config}")
    print(f"    Global Precision: {best_indices['Precisión Global']:.4f}")
    print(f"    Global Error:     {best_indices['Error Global']:.4f}")
    print(f"    Precision by category:\n{best_indices['Precisión por categoría']}")
    print(f"    Confusion Matrix:\n{best_indices['Matriz de Confusión']}")

    # 6. Generate all plots
    print("\n[5] Generating visualizations...")

    plot_precision_vs_k(df_results, os.path.join(output_dir, 'knn_precision_vs_k.png'))
    best_k_algo = plot_algorithm_comparison(df_results, os.path.join(output_dir, 'knn_algorithm_comparison.png'))

    plot_confusion_matrix(
        y_test, best['y_pred'], le_target.classes_,
        os.path.join(output_dir, 'knn_cm_best.png'),
        title=f"Confusion Matrix - Best KNN (k={best_config['k']}, {best_config['weights']}, {best_config['metric']})"
    )

    top10 = plot_top10_models(df_results, os.path.join(output_dir, 'knn_top10.png'))
    plot_metrics_heatmap(df_results, os.path.join(output_dir, 'knn_heatmap.png'))
    summary = generate_results_table(df_results, os.path.join(output_dir, 'knn_summary_by_k.png'))
    df_table = export_standard_vs_best_table(
        std_indices, best_config, best_indices,
        os.path.join(output_dir, 'knn_comparison_table.png')
    )

    # 7. Print top 10
    print("\n[6] Top 10 configurations:")
    print(top10[['k', 'algorithm', 'weights', 'metric', 'precision_global']].to_string(index=False))

    # 8. Summary by k
    print("\n[7] Summary by k:")
    print(summary.to_string(index=False))

    # 9. Classification report for best model
    print("\n[8] Classification Report (Best Model):")
    report = classification_report(y_test, best['y_pred'],
                                   target_names=le_target.classes_)
    print(report)

    # Save classification report
    with open(os.path.join(output_dir, 'knn_classification_report.txt'), 'w') as f:
        f.write(f"Standard KNN (k=5, uniform, euclidean)\n")
        f.write(f"Global Precision: {std_indices['Precisión Global']:.4f}\n")
        f.write(f"Global Error: {std_indices['Error Global']:.4f}\n\n")
        f.write(f"Best KNN: {best_config}\n")
        f.write(f"Global Precision: {best_indices['Precisión Global']:.4f}\n")
        f.write(f"Global Error: {best_indices['Error Global']:.4f}\n\n")
        f.write("Classification Report (Best Model):\n")
        f.write(report)

    print("\n[DONE] All outputs saved to:", output_dir)
    print("Files generated:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
