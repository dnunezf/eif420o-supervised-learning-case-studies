# -*- coding: utf-8 -*-
"""
ModuloACPLab01.py
=================
Modulo de Analisis de Componentes Principales para el Laboratorio 01.
Curso: EIF420O Inteligencia Artificial, I Ciclo 2026 - UNA
Profesor: Dr. Juan Murillo-Morera

Extiende el paquete pythonico del curso (PaqNOSup.py) con:
  - Clase ACPAnalisis: ACP completo usando sklearn (sin dependencia de prince).
  - Clase ACPExperimento: ejecuta configuracion estandar + variaciones,
    compara modelos y selecciona el mejor automaticamente.

Uso rapido:
    from ModuloACPLab01 import ACPAnalisis, ACPExperimento
    exp = ACPExperimento(df_numerico)
    exp.ejecutar()
    exp.mejor_modelo.plot_circulo()
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# CLASE 1: ACPAnalisis
class ACPAnalisis:
    """
    Analisis de Componentes Principales completo.
    Calcula: eigenvalues, varianza explicada, correlaciones variable-componente,
    coordenadas de individuos, contribuciones, cos2 de individuos y variables.
    """

    def __init__(self, datos, n_componentes=None, nombre='PCA'):
        self.__nombre = nombre
        self.__variables = datos.columns.tolist()
        self.__n_obs, self.__n_vars = datos.shape

        if n_componentes is None:
            n_componentes = min(self.__n_obs, self.__n_vars)
        self.__n_comp = n_componentes

        # Ajuste del modelo
        self.__modelo = PCA(n_components=n_componentes)
        self.__scores = self.__modelo.fit_transform(datos)
        self.__loadings = self.__modelo.components_.T
        self.__eigenvalues = self.__modelo.explained_variance_
        self.__var_explicada = self.__modelo.explained_variance_ratio_ * 100
        self.__var_acumulada = np.cumsum(self.__var_explicada)

        # Correlacion variables-componentes: cor(Xj, CPk) = loading_jk * std(CPk) / std(Xj)
        std_scores = np.std(self.__scores, axis=0, ddof=1)
        std_vars = np.std(datos.values, axis=0, ddof=1)
        self.__correlacion_var = pd.DataFrame(
            self.__loadings * std_scores / std_vars[:, np.newaxis],
            index=self.__variables,
            columns=[f'CP{i}' for i in range(n_componentes)]
        )

        # Coordenadas de individuos
        self.__coordenadas_ind = pd.DataFrame(
            self.__scores,
            index=datos.index,
            columns=[f'CP{i}' for i in range(n_componentes)]
        )

        # Contribucion de individuos (%)
        scores2 = self.__scores ** 2
        self.__contribucion_ind = pd.DataFrame(
            scores2 / scores2.sum(axis=0) * 100,
            index=datos.index,
            columns=[f'CP{i}' for i in range(n_componentes)]
        )

        # Cos2 de individuos
        dist2 = np.sum(scores2, axis=1, keepdims=True)
        self.__cos2_ind = pd.DataFrame(
            scores2 / dist2,
            index=datos.index,
            columns=[f'CP{i}' for i in range(n_componentes)]
        )

        # Cos2 de variables
        self.__cos2_var = self.__correlacion_var ** 2

    # PROPIEDADES
    @property
    def nombre(self):
        return self.__nombre

    @property
    def variables(self):
        return self.__variables

    @property
    def n_obs(self):
        return self.__n_obs

    @property
    def n_vars(self):
        return self.__n_vars

    @property
    def n_comp(self):
        return self.__n_comp

    @property
    def modelo(self):
        return self.__modelo

    @property
    def scores(self):
        return self.__scores

    @property
    def loadings(self):
        return self.__loadings

    @property
    def eigenvalues(self):
        return self.__eigenvalues

    @property
    def var_explicada(self):
        return self.__var_explicada

    @property
    def var_acumulada(self):
        return self.__var_acumulada

    @property
    def correlacion_var(self):
        return self.__correlacion_var

    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind

    @property
    def contribucion_ind(self):
        return self.__contribucion_ind

    @property
    def cos2_ind(self):
        return self.__cos2_ind

    @property
    def cos2_var(self):
        return self.__cos2_var

    # METODOS DE CONSULTA
    def resumen(self):
        """Retorna DataFrame con eigenvalues, %varianza y %acumulado."""
        return pd.DataFrame({
            'Valor Propio': self.__eigenvalues[:self.__n_comp],
            '% Varianza': self.__var_explicada[:self.__n_comp],
            '% Acumulado': self.__var_acumulada[:self.__n_comp]
        }, index=[f'CP{i}' for i in range(self.__n_comp)])

    def kaiser(self):
        """Retorna el numero de componentes con eigenvalue > 1."""
        return int((self.__eigenvalues > 1).sum())

    def interpretacion(self, top_n=5):
        """Imprime las variables mas correlacionadas con cada componente."""
        for cp in range(self.__n_comp):
            print(f"\nCP{cp} ({self.__var_explicada[cp]:.1f}% varianza):")
            cors = self.__correlacion_var.iloc[:, cp]
            top = cors.abs().sort_values(ascending=False).head(top_n)
            for var in top.index:
                val = cors[var]
                signo = '+' if val > 0 else '-'
                print(f"  {signo} {var}: {abs(val):.3f}")

    # METODOS DE GRAFICACION
    def plot_varianza(self, max_comp=None, guardar=None):
        """Scree plot con regla de Kaiser."""
        if max_comp is None:
            max_comp = min(self.__n_comp, 15)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        x = range(max_comp)
        ax.bar(x, self.__var_explicada[:max_comp],
               color='steelblue', alpha=0.7, label='Individual')
        ax.plot(x, self.__var_acumulada[:max_comp],
                'ro-', label='Acumulada')
        ax.axhline(y=100 / self.__n_vars, color='red',
                   linestyle='--', alpha=0.5, label='Kaiser (promedio)')
        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('% Varianza Explicada')
        ax.set_title(f'Scree Plot - {self.__nombre}')
        ax.set_xticks(list(x))
        ax.set_xticklabels([f'CP{i}' for i in x], rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_plano_principal(self, ejes=[0, 1], colores=None,
                             ind_labels=False, titulo=None, guardar=None):
        """Plano principal: proyeccion de individuos sobre dos componentes."""
        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
        x = self.__scores[:, ejes[0]]
        y = self.__scores[:, ejes[1]]
        if colores is not None:
            scatter = ax.scatter(x, y, c=colores, cmap='tab10',
                                alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(x, y, color='steelblue', alpha=0.5,
                       s=30, edgecolors='k', linewidths=0.3)
        if ind_labels:
            for i, txt in enumerate(self.__coordenadas_ind.index):
                ax.annotate(txt, (x[i], y[i]), fontsize=6, alpha=0.7)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        ix = round(self.__var_explicada[ejes[0]], 1)
        iy = round(self.__var_explicada[ejes[1]], 1)
        ax.set_xlabel(f'CP{ejes[0]} ({ix}%)')
        ax.set_ylabel(f'CP{ejes[1]} ({iy}%)')
        ax.set_title(titulo or f'Plano Principal - {self.__nombre}')
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_circulo(self, ejes=[0, 1], var_labels=True,
                     titulo=None, guardar=None):
        """Circulo de correlaciones."""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        cor = self.__correlacion_var.iloc[:, ejes].values
        circle = plt.Circle((0, 0), 1, color='steelblue',
                             fill=False, linewidth=1.5)
        ax.add_patch(circle)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        for i in range(cor.shape[0]):
            ax.arrow(0, 0, cor[i, 0] * 0.92, cor[i, 1] * 0.92,
                     color='steelblue', alpha=0.7,
                     head_width=0.04, head_length=0.03)
            if var_labels:
                ax.text(cor[i, 0] * 1.08, cor[i, 1] * 1.08,
                        self.__variables[i], color='darkblue',
                        ha='center', va='center', fontsize=7)
        ix = round(self.__var_explicada[ejes[0]], 1)
        iy = round(self.__var_explicada[ejes[1]], 1)
        ax.set_xlabel(f'CP{ejes[0]} ({ix}%)')
        ax.set_ylabel(f'CP{ejes[1]} ({iy}%)')
        ax.set_title(titulo or f'Circulo de Correlacion - {self.__nombre}')
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_sobreposicion(self, ejes=[0, 1], ind_labels=False,
                           var_labels=True, titulo=None, guardar=None):
        """Biplot: individuos + vectores de variables superpuestos."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        x = self.__scores[:, ejes[0]]
        y = self.__scores[:, ejes[1]]
        ax.scatter(x, y, color='gray', alpha=0.3, s=20)
        if ind_labels:
            for i, txt in enumerate(self.__coordenadas_ind.index):
                ax.annotate(txt, (x[i], y[i]), fontsize=6, alpha=0.6)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        cor = self.__correlacion_var.iloc[:, ejes].values
        scale = min(
            (x.max() - x.min()) / (cor[:, 0].max() - cor[:, 0].min() + 1e-10),
            (y.max() - y.min()) / (cor[:, 1].max() - cor[:, 1].min() + 1e-10)
        ) * 0.6
        for i in range(cor.shape[0]):
            ax.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale,
                     color='firebrick', alpha=0.7,
                     head_width=scale * 0.04, head_length=scale * 0.03)
            if var_labels:
                ax.text(cor[i, 0] * scale * 1.12, cor[i, 1] * scale * 1.12,
                        self.__variables[i], color='firebrick',
                        ha='center', va='center', fontsize=7, weight='bold')
        ix = round(self.__var_explicada[ejes[0]], 1)
        iy = round(self.__var_explicada[ejes[1]], 1)
        ax.set_xlabel(f'CP{ejes[0]} ({ix}%)')
        ax.set_ylabel(f'CP{ejes[1]} ({iy}%)')
        ax.set_title(titulo or f'Biplot - {self.__nombre}')
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_cos2_variables(self, n_comp=None, guardar=None):
        """Mapa de calor del cos2 de las variables."""
        if n_comp is None:
            n_comp = self.__n_comp
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        data = self.__cos2_var.iloc[:, :n_comp]
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, vmin=0, vmax=1,
                    cbar_kws={'label': 'cos2'})
        ax.set_title(f'Calidad de Representacion (cos2) - {self.__nombre}')
        ax.set_ylabel('Variable')
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_contribuciones(self, comp=0, guardar=None):
        """Barras horizontales de contribucion de variables a una componente."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        contrib = self.__cos2_var.iloc[:, comp].sort_values(ascending=True)
        umbral = 1 / self.__n_vars
        colors = ['steelblue' if v >= umbral else 'lightgray'
                  for v in contrib.values]
        ax.barh(range(len(contrib)), contrib.values, color=colors,
                edgecolor='black', linewidth=0.3)
        ax.axvline(x=umbral, color='red', linestyle='--', alpha=0.5,
                   label=f'Umbral ({umbral:.2f})')
        ax.set_yticks(range(len(contrib)))
        ax.set_yticklabels(contrib.index, fontsize=8)
        ax.set_xlabel('cos2 (contribucion relativa)')
        ax.set_title(f'Contribucion Variables a CP{comp} - {self.__nombre}')
        ax.legend(fontsize=7)
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def __str__(self):
        return (f'ACPAnalisis("{self.__nombre}", n_comp={self.__n_comp}, '
                f'var_acum={self.__var_acumulada[self.__n_comp-1]:.1f}%)')

    def __repr__(self):
        return self.__str__()


# CLASE 2: ACPExperimento
class ACPExperimento:
    """
    Ejecuta multiples configuraciones de PCA, compara y selecciona el mejor.

    Parametros
    ----------
    df_numerico : pd.DataFrame
        DataFrame con variables numericas sin escalar.
    df_completo : pd.DataFrame o None
        DataFrame completo (con categoricas) para la variacion con dummies.
    carpeta : str o None
        Ruta para guardar figuras automaticamente. Si None, no guarda.

    Uso
    ---
        exp = ACPExperimento(df_numerico, df_completo=df_raw)
        exp.ejecutar()
        print(exp.benchmark())
        exp.mejor_modelo.plot_circulo()
        exp.mejor_modelo.interpretacion()
    """

    def __init__(self, df_numerico, df_completo=None, carpeta=None):
        self.__df_num = df_numerico
        self.__df_completo = df_completo
        self.__carpeta = carpeta
        self.__modelos = {}
        self.__mejor_nombre = None

    @property
    def modelos(self):
        return self.__modelos

    @property
    def mejor_modelo(self):
        if self.__mejor_nombre:
            return self.__modelos[self.__mejor_nombre]
        return None

    @property
    def mejor_nombre(self):
        return self.__mejor_nombre

    def ejecutar(self):
        """Ejecuta las 6 configuraciones de PCA y selecciona el mejor."""
        print("=" * 60)
        print("EXPERIMENTO ACP - INICIO")
        print("=" * 60)

        # --- 1. Estandar (StandardScaler + Kaiser) ---
        print("\n[1/6] StandardScaler + Kaiser...")
        scaler_std = StandardScaler()
        datos_std = pd.DataFrame(
            scaler_std.fit_transform(self.__df_num),
            columns=self.__df_num.columns,
            index=self.__df_num.index
        )
        acp_full = ACPAnalisis(datos_std, nombre='Estandar_full')
        n_kaiser = acp_full.kaiser()
        acp_std = ACPAnalisis(datos_std, n_componentes=n_kaiser,
                              nombre=f'StandardScaler ({n_kaiser} comp.)')
        self.__modelos['StandardScaler'] = acp_std
        print(f"   Kaiser: {n_kaiser} comp., Var. acum: {acp_std.var_acumulada[n_kaiser-1]:.2f}%")

        # --- 2. MinMaxScaler ---
        print("\n[2/6] MinMaxScaler...")
        datos_mm = pd.DataFrame(
            MinMaxScaler().fit_transform(self.__df_num),
            columns=self.__df_num.columns,
            index=self.__df_num.index
        )
        acp_mm = ACPAnalisis(datos_mm, n_componentes=n_kaiser,
                             nombre=f'MinMaxScaler ({n_kaiser} comp.)')
        self.__modelos['MinMaxScaler'] = acp_mm
        print(f"   Var. acum: {acp_mm.var_acumulada[n_kaiser-1]:.2f}%")

        # --- 3. RobustScaler ---
        print("\n[3/6] RobustScaler...")
        datos_rob = pd.DataFrame(
            RobustScaler().fit_transform(self.__df_num),
            columns=self.__df_num.columns,
            index=self.__df_num.index
        )
        acp_rob = ACPAnalisis(datos_rob, n_componentes=n_kaiser,
                              nombre=f'RobustScaler ({n_kaiser} comp.)')
        self.__modelos['RobustScaler'] = acp_rob
        print(f"   Var. acum: {acp_rob.var_acumulada[n_kaiser-1]:.2f}%")

        # --- 4. Criterio de codo ---
        print("\n[4/6] Criterio de codo...")
        diffs = np.abs(np.diff(acp_full.var_explicada))
        n_codo = max(np.argmax(diffs < 3) + 1, 2)
        acp_codo = ACPAnalisis(datos_std, n_componentes=n_codo,
                               nombre=f'Estandar Codo ({n_codo} comp.)')
        self.__modelos['Codo'] = acp_codo
        print(f"   Codo: {n_codo} comp., Var. acum: {acp_codo.var_acumulada[n_codo-1]:.2f}%")

        # --- 5. Sin estandarizar ---
        print("\n[5/6] Sin estandarizar...")
        acp_crudo = ACPAnalisis(self.__df_num, n_componentes=n_kaiser,
                                nombre=f'Sin estandarizar ({n_kaiser} comp.)')
        self.__modelos['Sin_estandarizar'] = acp_crudo
        print(f"   Var. acum: {acp_crudo.var_acumulada[n_kaiser-1]:.2f}%")

        # --- 6. Con dummies ---
        if self.__df_completo is not None:
            print("\n[6/6] Con dummies categoricas...")
            df_dum = pd.get_dummies(self.__df_completo, drop_first=True).astype(float)
            datos_dum = pd.DataFrame(
                StandardScaler().fit_transform(df_dum),
                columns=df_dum.columns, index=df_dum.index
            )
            acp_dum_full = ACPAnalisis(datos_dum, nombre='Dummies_full')
            k_dum = acp_dum_full.kaiser()
            acp_dum = ACPAnalisis(datos_dum, n_componentes=k_dum,
                                  nombre=f'Con Dummies ({k_dum} comp.)')
            self.__modelos['Con_dummies'] = acp_dum
            print(f"   Kaiser: {k_dum} comp., Var. acum: {acp_dum.var_acumulada[k_dum-1]:.2f}%")
        else:
            print("\n[6/6] Sin df_completo, se omite variacion con dummies.")

        # --- Seleccion del mejor ---
        self.__seleccionar_mejor()

        print("\n" + "=" * 60)
        print(f"MEJOR MODELO: {self.__mejor_nombre}")
        print(f"  {self.__modelos[self.__mejor_nombre]}")
        print("=" * 60)

    def __seleccionar_mejor(self):
        """Selecciona el mejor modelo: mayor varianza acumulada entre modelos
        validos (excluye sin estandarizar y con dummies por ininterpretabilidad)."""
        candidatos = {k: v for k, v in self.__modelos.items()
                      if k not in ('Sin_estandarizar', 'Con_dummies')}
        mejor = max(candidatos.items(),
                    key=lambda x: x[1].var_acumulada[x[1].n_comp - 1])
        self.__mejor_nombre = mejor[0]

    def benchmark(self):
        """Retorna tabla comparativa de todos los modelos."""
        filas = []
        for nombre, m in self.__modelos.items():
            var_total = m.var_acumulada[m.n_comp - 1]
            avg_cos2 = m.cos2_var.iloc[:, :m.n_comp].sum(axis=1).mean()
            filas.append({
                'Modelo': nombre,
                'N Comp.': m.n_comp,
                'N Vars': m.n_vars,
                '% Var. Acum.': round(var_total, 2),
                'cos2 prom.': round(avg_cos2, 4),
                'Eigenval CP0': round(m.eigenvalues[0], 4)
            })
        return pd.DataFrame(filas)

    def plot_comparacion_varianza(self, guardar=None):
        """Grafico comparativo de varianza acumulada."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for nombre, m in self.__modelos.items():
            nc = min(m.n_comp, 10)
            ax.plot(range(nc), m.var_acumulada[:nc], 'o-',
                    label=nombre, markersize=5)
        ax.axhline(y=70, color='red', linestyle=':', alpha=0.5, label='70%')
        ax.axhline(y=80, color='darkred', linestyle=':', alpha=0.5, label='80%')
        ax.set_xlabel('Componentes')
        ax.set_ylabel('% Varianza Acumulada')
        ax.set_title('Comparacion de Varianza Acumulada')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_comparacion_circulos(self, guardar=None):
        """Circulos de correlacion lado a lado de todos los modelos."""
        n = len(self.__modelos)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), dpi=150)
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for idx, (nombre, m) in enumerate(self.__modelos.items()):
            ax = axes[idx]
            if m.n_comp >= 2:
                cor = m.correlacion_var.iloc[:, [0, 1]].values
                c = plt.Circle((0, 0), 1, color='steelblue',
                               fill=False, linewidth=1)
                ax.add_patch(c)
                ax.set_xlim(-1.3, 1.3)
                ax.set_ylim(-1.3, 1.3)
                ax.set_aspect('equal')
                ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
                ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
                for i in range(cor.shape[0]):
                    ax.arrow(0, 0, cor[i, 0] * 0.9, cor[i, 1] * 0.9,
                             color='steelblue', alpha=0.6,
                             head_width=0.04, head_length=0.03)
                    ax.text(cor[i, 0] * 1.08, cor[i, 1] * 1.08,
                            m.variables[i], color='darkblue',
                            ha='center', va='center', fontsize=6)
                ax.set_title(nombre, fontsize=9)
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle('Comparacion de Circulos de Correlacion', fontsize=13)
        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def analisis_mejor(self):
        """Ejecuta el analisis completo del mejor modelo seleccionado."""
        m = self.mejor_modelo
        if m is None:
            print("Ejecute .ejecutar() primero.")
            return
        print(f"\n{'='*60}")
        print(f"ANALISIS DEL MEJOR MODELO: {self.__mejor_nombre}")
        print(f"{'='*60}")
        print(f"\n--- Resumen ---")
        print(m.resumen().round(4))
        print(f"\n--- Correlaciones variable-componente ---")
        print(m.correlacion_var.round(4))
        print(f"\n--- cos2 de variables ---")
        print(m.cos2_var.round(4))
        print(f"\n--- Interpretacion ---")
        m.interpretacion()
        g = self.__carpeta
        m.plot_varianza(guardar=f'{g}/screeplot.png' if g else None)
        m.plot_plano_principal(guardar=f'{g}/plano_principal.png' if g else None)
        m.plot_circulo(guardar=f'{g}/circulo.png' if g else None)
        m.plot_sobreposicion(guardar=f'{g}/biplot.png' if g else None)
        m.plot_cos2_variables(guardar=f'{g}/cos2.png' if g else None)
        for cp in range(min(m.n_comp, 4)):
            m.plot_contribuciones(comp=cp,
                guardar=f'{g}/contrib_CP{cp}.png' if g else None)
