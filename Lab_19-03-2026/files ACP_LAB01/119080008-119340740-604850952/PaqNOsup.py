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
				
class TSNE_UMAP_Experimento:
    def __init__(self, df, modelo_acp=None):
        import pandas as pd
        self.df = df
        self.modelo_acp = modelo_acp
        self.resultados = {}

    # -------------------------------
    # Escalado
    # -------------------------------
    def _escalar(self, metodo):
        from sklearn.preprocessing import StandardScaler, RobustScaler
        import pandas as pd

        if metodo == "standard":
            scaler = StandardScaler()
        elif metodo == "robust":
            scaler = RobustScaler()
        else:
            return self.df

        return pd.DataFrame(
            scaler.fit_transform(self.df),
            columns=self.df.columns,
            index=self.df.index
        )

    # -------------------------------
    # T-SNE
    # -------------------------------
    def _tsne(self, data, nombre):
        from sklearn.manifold import TSNE

        modelo = TSNE(n_components=2, random_state=42, perplexity=30)
        res = modelo.fit_transform(data)

        self.resultados[nombre] = res
        return res

    # -------------------------------
    # UMAP
    # -------------------------------
    def _umap(self, data, nombre):
        import umap.umap_ as umap

        modelo = umap.UMAP(n_components=2, random_state=42)
        res = modelo.fit_transform(data)

        self.resultados[nombre] = res
        return res

    # -------------------------------
    # Plot
    # -------------------------------
    def _plot(self, data, titulo):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7,5))
        plt.scatter(data[:,0], data[:,1], s=15)
        plt.title(titulo)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid()
        plt.show()

    # -------------------------------
    # Ejecutar experimentos
    # -------------------------------
    def ejecutar(self):
        print("\n" + "=" * 60)
        print("EXPERIMENTO T-SNE / UMAP")
        print("=" * 60)

        # 1. Standard
        df_std = self._escalar("standard")
        self._plot(self._tsne(df_std, "tsne_std"), "T-SNE (Standard)")
        self._plot(self._umap(df_std, "umap_std"), "UMAP (Standard)")

        # 2. Robust
        df_rob = self._escalar("robust")
        self._plot(self._tsne(df_rob, "tsne_robust"), "T-SNE (Robust)")
        self._plot(self._umap(df_rob, "umap_robust"), "UMAP (Robust)")

        # 3. ACP
        if self.modelo_acp is not None:
            df_pca = self.modelo_acp.coordenadas_ind
            self._plot(self._tsne(df_pca, "tsne_pca"), "T-SNE (ACP)")
            self._plot(self._umap(df_pca, "umap_pca"), "UMAP (ACP)")

    # -------------------------------
    # Comparación simple
    # -------------------------------
    def comparar(self):
        print("\n" + "=" * 60)
        print("COMPARACION DE METODOS")
        print("=" * 60)

        print("Analiza visualmente:")
        print("- Separación de clusters")
        print("- Compactación de grupos")
        print("- Claridad de estructuras\n")

        print("Usualmente:")
        print("UMAP > T-SNE en estabilidad")
        print("ACP + UMAP/T-SNE = mejor resultado")
        
        
        
        
        
#HAC

# SECCION HAC (Hierarchical Agglomerative Clustering)
class HAC_Class:
    def __init__(self, df):
        self.__df = df
        self.__resultados = {}
        self.__etiquetas = None
        self.__n_clusters = None

    @property
    def df(self):
        return self.__df

    def plot_dendrogramas(self):
        """Genera dendrogramas para los 4 métodos principales de vinculación."""
        metodos = {
            'Ward': ward(self.__df),
            'Average': average(self.__df),
            'Single': single(self.__df),
            'Complete': complete(self.__df)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        axes = axes.flatten()
        
        for i, (nombre, res) in enumerate(metodos.items()):
            dendrogram(res, labels=self.__df.index.tolist(), ax=axes[i])
            axes[i].set_title(f'Dendrograma - Método: {nombre}')
            axes[i].grid(False)
            
        plt.tight_layout()
        plt.show()

    def analizar_silhouette(self, metodos=['ward', 'average', 'complete', 'single'], k_max=10):
        """Calcula y grafica el Silhouette Score para diferentes valores de K y métodos."""
        resultados_silhouette = []
        
        plt.figure(figsize=(12, 6))
        for metodo in metodos:
            scores = []
            ks = range(2, k_max + 1)
            for k in ks:
                modelo = AgglomerativeClustering(n_clusters=k, linkage=metodo)
                etiquetas = modelo.fit_predict(self.__df)
                score = silhouette_score(self.__df, etiquetas)
                scores.append(score)
            
            mejor_k = ks[np.argmax(scores)]
            mejor_score = max(scores)
            resultados_silhouette.append({'Metodo': metodo, 'K_Optimo': mejor_k, 'Score': mejor_score})
            
            plt.plot(ks, scores, marker='o', label=f'{metodo} (Max: {mejor_score:.2f})')

        plt.title('Análisis de Silhouette por Método de Vinculación')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
        return pd.DataFrame(resultados_silhouette)

    def ajustar_modelo(self, n_clusters, metodo='ward'):
        """Ajusta el modelo final con un K específico y devuelve las etiquetas."""
        self.__n_clusters = n_clusters
        modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=metodo)
        self.__etiquetas = modelo.fit_predict(self.__df)
        return self.__etiquetas

    def plot_perfil_clusters(self, tipo='bar'):
        """
        Genera gráficos de perfil (Bar o Radar) para los clusters creados.
        Requiere haber ejecutado ajustar_modelo primero.
        """
        if self.__etiquetas is None:
            print("Error: Primero debes ejecutar ajustar_modelo(n_clusters).")
            return

        # Calcular centroides manualmente para HAC
        centroides = []
        for i in range(self.__n_clusters):
            # Usamos el método estático de la clase base analisisEDA
            c = analisisEDA.centroide(i, self.__df.values, self.__etiquetas)
            centroides.append(c.values[0])
        
        centroides = np.array(centroides)

        if tipo == 'bar':
            analisisEDA.bar_plot(centroides, self.__df.columns, scale=True)
        elif tipo == 'radar':
            analisisEDA.radar_plot(centroides, self.__df.columns)
        
        plt.show()

    def visualizar_2d(self):
        """Visualiza los clusters en un plano 2D usando PCA."""
        if self.__etiquetas is None:
            print("Error: Primero debes ajustar el modelo.")
            return

        pca = PCA(n_components=2)
        comp = pca.fit_transform(self.__df)
        
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(comp[:, 0], comp[:, 1], c=self.__etiquetas, cmap='viridis', s=50)
        plt.title(f'Visualización HAC (k={self.__n_clusters})')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(alpha=0.3)
        plt.show()
#K-medias

# CLASE 3: KMeansAnalisis
class KMeansAnalisis:
    """
    Ejecuta un unico modelo K-Means sobre un DataFrame numerico.

    Parametros
    ----------
    datos : pd.DataFrame
        Datos numericos ya preparados para este experimento.
    nombre : str
        Nombre descriptivo del modelo.
    k_range : iterable
        Valores de K a evaluar.
    scaler : objeto scaler o None
        Escalador a aplicar antes de K-Means.
    quitar_outliers : bool
        Si True, elimina observaciones con |z| > 3 antes de ajustar.
    random_state : int
        Semilla para reproducibilidad.
    n_init : int
        Numero de inicializaciones de K-Means.
    max_iter : int
        Maximo de iteraciones del algoritmo.
    """

    def __init__(self,
                 datos,
                 nombre='KMeans',
                 k_range=range(2, 9),
                 scaler=None,
                 quitar_outliers=False,
                 random_state=42,
                 n_init=20,
                 max_iter=300):

        # ----------- Validaciones basicas -----------
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("datos debe ser un DataFrame de pandas.")

        if datos.shape[1] < 2:
            raise ValueError("K-Means requiere al menos 2 variables numericas.")

        self.__nombre = nombre
        self.__datos_originales = datos.copy()
        self.__columnas = datos.columns.tolist()
        self.__k_range = list(k_range)
        self.__scaler = scaler
        self.__quitar_outliers = quitar_outliers
        self.__random_state = random_state
        self.__n_init = n_init
        self.__max_iter = max_iter

        # Atributos que se completan durante el ajuste
        self.__datos_filtrados = None
        self.__datos_transformados = None
        self.__modelo = None
        self.__labels = None
        self.__k_optimo = None
        self.__silhouette = None
        self.__inercia = None
        self.__centroides_scaled = None
        self.__centroides_originales = None
        self.__tabla_metricas = None
        self.__resumen_clusters = None

        # Ajuste automatico del modelo al instanciar
        self.__ajustar()

    # ---------------- PROPIEDADES ----------------
    @property
    def nombre(self):
        return self.__nombre

    @property
    def columnas(self):
        return self.__columnas

    @property
    def datos_filtrados(self):
        return self.__datos_filtrados

    @property
    def datos_transformados(self):
        return self.__datos_transformados

    @property
    def modelo(self):
        return self.__modelo

    @property
    def labels(self):
        return self.__labels

    @property
    def k_optimo(self):
        return self.__k_optimo

    @property
    def silhouette(self):
        return self.__silhouette

    @property
    def inercia(self):
        return self.__inercia

    @property
    def centroides_scaled(self):
        return self.__centroides_scaled

    @property
    def centroides_originales(self):
        return self.__centroides_originales

    @property
    def tabla_metricas(self):
        return self.__tabla_metricas

    @property
    def resumen_clusters(self):
        return self.__resumen_clusters

    # ---------------- METODOS INTERNOS ----------------
    def __quitar_outliers_zscore(self, df, umbral=3.0):
        """
        Elimina filas con al menos una variable fuera de |z| > umbral.
        Se calcula sobre el propio DataFrame recibido.
        """
        z = (df - df.mean()) / df.std(ddof=0)
        mask = (z.abs() <= umbral).all(axis=1)
        return df.loc[mask].copy()

    def __preparar_datos(self):
        """
        Aplica limpieza minima:
        - elimina NaN
        - opcionalmente quita outliers
        - opcionalmente escala datos
        """
        df = self.__datos_originales.select_dtypes(include=[np.number]).copy()
        df = df.dropna()

        if self.__quitar_outliers:
            df = self.__quitar_outliers_zscore(df)

        self.__datos_filtrados = df.copy()

        # Si hay escalador, transformar; si no, usar datos crudos
        if self.__scaler is not None:
            X = self.__scaler.fit_transform(df)
        else:
            X = df.values

        self.__datos_transformados = X

    def __buscar_mejor_k(self):
        """
        Evalua distintos valores de K y selecciona el mejor por silhouette.
        Retorna:
            - tabla de metricas por K
            - mejor K
        """
        filas = []

        for k in self.__k_range:
            # Seguridad: no intentar K mayor o igual al numero de filas
            if k >= len(self.__datos_filtrados):
                continue

            modelo = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=self.__n_init,
                max_iter=self.__max_iter,
                random_state=self.__random_state
            )

            labels = modelo.fit_predict(self.__datos_transformados)

            # Silhouette requiere al menos 2 clusters distintos
            if len(np.unique(labels)) < 2:
                sil = np.nan
            else:
                sil = silhouette_score(self.__datos_transformados, labels)

            filas.append({
                'K': k,
                'Inercia': modelo.inertia_,
                'Silhouette': sil
            })

        tabla = pd.DataFrame(filas)

        if tabla.empty:
            raise ValueError("No fue posible evaluar ningun valor de K.")

        # Seleccion principal: mayor silhouette
        # Desempate: menor inercia
        tabla = tabla.sort_values(
            by=['Silhouette', 'Inercia'],
            ascending=[False, True]
        ).reset_index(drop=True)

        mejor_k = int(tabla.iloc[0]['K'])
        return tabla, mejor_k

    def __ajustar_modelo_final(self):
        """
        Ajusta el modelo final con el K seleccionado.
        """
        modelo = KMeans(
            n_clusters=self.__k_optimo,
            init='k-means++',
            n_init=self.__n_init,
            max_iter=self.__max_iter,
            random_state=self.__random_state
        )

        labels = modelo.fit_predict(self.__datos_transformados)

        self.__modelo = modelo
        self.__labels = labels
        self.__silhouette = silhouette_score(self.__datos_transformados, labels)
        self.__inercia = modelo.inertia_
        self.__centroides_scaled = pd.DataFrame(
            modelo.cluster_centers_,
            columns=self.__columnas,
            index=[f'Cluster {i}' for i in range(self.__k_optimo)]
        )

        # Si hubo escalado, regresar centroides a escala original
        if self.__scaler is not None:
            centros_originales = self.__scaler.inverse_transform(
                modelo.cluster_centers_
            )
        else:
            centros_originales = modelo.cluster_centers_

        self.__centroides_originales = pd.DataFrame(
            centros_originales,
            columns=self.__columnas,
            index=[f'Cluster {i}' for i in range(self.__k_optimo)]
        )

        # Armar tabla con datos + cluster para resumen descriptivo
        df_aux = self.__datos_filtrados.copy()
        df_aux['Cluster'] = labels

        # Tamaños de cluster
        tamanos = df_aux['Cluster'].value_counts().sort_index()

        # Promedios por cluster en escala original
        promedios = df_aux.groupby('Cluster')[self.__columnas].mean()

        # Tabla resumen compacta
        resumen = promedios.copy()
        resumen.insert(0, 'n', tamanos.values)
        resumen.index = [f'Cluster {i}' for i in resumen.index]

        self.__resumen_clusters = resumen

    def __ajustar(self):
        """
        Flujo completo:
        1) preparar datos
        2) buscar mejor K
        3) ajustar modelo final
        """
        self.__preparar_datos()
        self.__tabla_metricas, self.__k_optimo = self.__buscar_mejor_k()
        self.__ajustar_modelo_final()

    # ---------------- METODOS DE CONSULTA ----------------
    def benchmark_k(self):
        """
        Retorna la tabla de metricas por K evaluado.
        """
        return self.__tabla_metricas.copy()

    def centroides(self, escala='original'):
        """
        Retorna centroides del modelo:
        - escala='original' -> escala interpretable
        - escala='scaled'   -> escala interna usada por K-Means
        """
        if escala == 'original':
            return self.__centroides_originales.copy()
        elif escala == 'scaled':
            return self.__centroides_scaled.copy()
        else:
            raise ValueError("escala debe ser 'original' o 'scaled'.")

    def variables_mas_discriminatorias(self, top_n=10):
        """
        Identifica variables que mas separan clusters.
        Se usa el rango (max - min) entre centroides originales.
        """
        centros = self.__centroides_originales
        rango = centros.max(axis=0) - centros.min(axis=0)
        out = pd.DataFrame({
            'Rango_entre_centroides': rango
        }).sort_values('Rango_entre_centroides', ascending=False)
        return out.head(top_n)

    def interpretar_clusters(self, top_n=5):
        """
        Imprime una interpretacion simple:
        - tamano del cluster
        - variables mas altas y mas bajas del centroide
        """
        print(f"\n{'='*60}")
        print(f"INTERPRETACION DE CLUSTERS - {self.__nombre}")
        print(f"{'='*60}")
        print(f"K optimo: {self.__k_optimo}")
        print(f"Silhouette: {self.__silhouette:.4f}")
        print(f"Inercia: {self.__inercia:.4f}")

        centros = self.__centroides_originales

        for idx in centros.index:
            serie = centros.loc[idx].sort_values(ascending=False)
            altas = serie.head(top_n)
            bajas = serie.tail(top_n).sort_values()

            n_cluster = int(self.__resumen_clusters.loc[idx, 'n'])

            print(f"\n{idx} -> n = {n_cluster}")
            print("  Variables mas altas del centroide:")
            for var, val in altas.items():
                print(f"    + {var}: {val:.4f}")

            print("  Variables mas bajas del centroide:")
            for var, val in bajas.items():
                print(f"    - {var}: {val:.4f}")

    def resumen(self):
        """
        Retorna un resumen general del modelo.
        """
        return pd.DataFrame([{
            'Modelo': self.__nombre,
            'Observaciones': len(self.__datos_filtrados),
            'Variables': len(self.__columnas),
            'K optimo': self.__k_optimo,
            'Silhouette': round(self.__silhouette, 4),
            'Inercia': round(self.__inercia, 4),
            'Outliers removidos': self.__quitar_outliers,
            'Scaler': self.__scaler.__class__.__name__ if self.__scaler is not None else 'None'
        }])

    # ---------------- METODOS DE GRAFICACION ----------------
    def plot_elbow_silhouette(self, guardar=None):
        """
        Grafico combinado:
        - izquierda: inercia (elbow)
        - derecha: silhouette
        """
        tabla = self.__tabla_metricas.sort_values('K')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

        # Grafico de inercia
        axes[0].plot(tabla['K'], tabla['Inercia'], 'o-', linewidth=1.8)
        axes[0].axvline(self.__k_optimo, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title(f'Elbow - {self.__nombre}')
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Inercia')
        axes[0].grid(alpha=0.3)

        # Grafico de silhouette
        axes[1].plot(tabla['K'], tabla['Silhouette'], 'o-', linewidth=1.8)
        axes[1].axvline(self.__k_optimo, color='red', linestyle='--', alpha=0.7)
        axes[1].set_title(f'Silhouette - {self.__nombre}')
        axes[1].set_xlabel('K')
        axes[1].set_ylabel('Silhouette')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_clusters_pca(self, guardar=None):
        """
        Proyecta los datos a 2 componentes con PCA solo para visualizacion.
        No cambia el modelo K-Means; es un apoyo grafico.
        """
        pca = PCA(n_components=2, random_state=self.__random_state)
        coords = pca.fit_transform(self.__datos_transformados)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=self.__labels,
            cmap='tab10',
            alpha=0.70,
            s=35,
            edgecolors='k',
            linewidths=0.25
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # Centroides proyectados
        centros_2d = pca.transform(self.__modelo.cluster_centers_)
        ax.scatter(
            centros_2d[:, 0],
            centros_2d[:, 1],
            marker='X',
            s=250,
            color='black',
            label='Centroides'
        )

        ax.set_title(f'Clusters proyectados en PCA - {self.__nombre}')
        ax.set_xlabel('CP1')
        ax.set_ylabel('CP2')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def plot_perfiles_centroides(self, top_n=12, guardar=None):
        """
        Grafico de barras de las variables con mayor rango entre centroides.
        Se usan centroides en escala original para facilitar interpretacion.
        """
        vars_top = self.variables_mas_discriminatorias(top_n=top_n).index.tolist()
        centros = self.__centroides_originales[vars_top].T

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        centros.plot(kind='bar', ax=ax)

        ax.set_title(f'Perfiles de centroides - {self.__nombre}')
        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor promedio del centroide')
        ax.legend(title='Cluster')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def __str__(self):
        return (f'KMeansAnalisis("{self.__nombre}", '
                f'k_optimo={self.__k_optimo}, '
                f'silhouette={self.__silhouette:.4f}, '
                f'inercia={self.__inercia:.4f})')

    def __repr__(self):
        return self.__str__()


# CLASE 4: KMeansExperimento
class KMeansExperimento:
    """
    Ejecuta configuracion estandar y variaciones de K-Means,
    compara resultados y selecciona automaticamente el mejor modelo.

    Variaciones incluidas:
    ----------------------
    1) Estandar_StandardScaler     -> baseline recomendado
    2) Sin_Escalar                -> para mostrar el efecto del escalado
    3) MinMaxScaler               -> variacion de normalizacion
    4) RobustScaler               -> mas robusto ante outliers
    5) StandardScaler_sin_outliers -> version con limpieza adicional

    Uso
    ---
        exp = KMeansExperimento(df_numerico)
        exp.ejecutar()
        print(exp.benchmark())
        exp.analisis_mejor()
    """

    def __init__(self, df_numerico, carpeta=None, k_range=range(2, 9)):
        self.__df_num = df_numerico.copy()
        self.__carpeta = carpeta
        self.__k_range = list(k_range)
        self.__modelos = {}
        self.__mejor_nombre = None

    @property
    def modelos(self):
        return self.__modelos

    @property
    def mejor_nombre(self):
        return self.__mejor_nombre

    @property
    def mejor_modelo(self):
        if self.__mejor_nombre is None:
            return None
        return self.__modelos[self.__mejor_nombre]

    def ejecutar(self):
        """
        Ejecuta todos los modelos y selecciona el mejor.
        """
        print("=" * 60)
        print("EXPERIMENTO K-MEANS - INICIO")
        print("=" * 60)

        # ----------------------------------------------------
        # 1) Configuracion estandar:
        #    StandardScaler + K-Means++ + seleccion de K por silhouette
        # ----------------------------------------------------
        print("\n[1/5] Estandar: StandardScaler...")
        self.__modelos['Estandar_StandardScaler'] = KMeansAnalisis(
            datos=self.__df_num,
            nombre='Estandar StandardScaler',
            k_range=self.__k_range,
            scaler=StandardScaler(),
            quitar_outliers=False,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        print(self.__modelos['Estandar_StandardScaler'])

        # ----------------------------------------------------
        # 2) Sin escalar
        # ----------------------------------------------------
        print("\n[2/5] Variacion: sin escalar...")
        self.__modelos['Sin_Escalar'] = KMeansAnalisis(
            datos=self.__df_num,
            nombre='Sin Escalar',
            k_range=self.__k_range,
            scaler=None,
            quitar_outliers=False,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        print(self.__modelos['Sin_Escalar'])

        # ----------------------------------------------------
        # 3) MinMaxScaler
        # ----------------------------------------------------
        print("\n[3/5] Variacion: MinMaxScaler...")
        self.__modelos['MinMaxScaler'] = KMeansAnalisis(
            datos=self.__df_num,
            nombre='MinMaxScaler',
            k_range=self.__k_range,
            scaler=MinMaxScaler(),
            quitar_outliers=False,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        print(self.__modelos['MinMaxScaler'])

        # ----------------------------------------------------
        # 4) RobustScaler
        # ----------------------------------------------------
        print("\n[4/5] Variacion: RobustScaler...")
        self.__modelos['RobustScaler'] = KMeansAnalisis(
            datos=self.__df_num,
            nombre='RobustScaler',
            k_range=self.__k_range,
            scaler=RobustScaler(),
            quitar_outliers=False,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        print(self.__modelos['RobustScaler'])

        # ----------------------------------------------------
        # 5) StandardScaler + quitar outliers
        # ----------------------------------------------------
        print("\n[5/5] Variacion: StandardScaler sin outliers...")
        self.__modelos['StandardScaler_sin_outliers'] = KMeansAnalisis(
            datos=self.__df_num,
            nombre='StandardScaler sin outliers',
            k_range=self.__k_range,
            scaler=StandardScaler(),
            quitar_outliers=True,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        print(self.__modelos['StandardScaler_sin_outliers'])

        # Seleccion del mejor modelo
        self.__seleccionar_mejor()

        print("\n" + "=" * 60)
        print(f"MEJOR MODELO: {self.__mejor_nombre}")
        print(f"  {self.mejor_modelo}")
        print("=" * 60)

    def __seleccionar_mejor(self):
        """
        Selecciona el mejor modelo.
        Regla:
        1) mayor silhouette
        2) si silhouettes son muy cercanos, menor inercia
        """
        filas = []
        for nombre, modelo in self.__modelos.items():
            filas.append({
                'Modelo': nombre,
                'Silhouette': modelo.silhouette,
                'Inercia': modelo.inercia
            })

        tabla = pd.DataFrame(filas).sort_values(
            by=['Silhouette', 'Inercia'],
            ascending=[False, True]
        ).reset_index(drop=True)

        self.__mejor_nombre = tabla.iloc[0]['Modelo']

    def benchmark(self):
        """
        Retorna tabla comparativa entre todos los modelos probados.
        """
        filas = []
        for nombre, m in self.__modelos.items():
            filas.append({
                'Modelo': nombre,
                'Obs. usadas': len(m.datos_filtrados),
                'Variables': len(m.columnas),
                'K optimo': m.k_optimo,
                'Silhouette': round(m.silhouette, 4),
                'Inercia': round(m.inercia, 4),
                'Scaler': m.resumen().iloc[0]['Scaler'],
                'Outliers removidos': m.resumen().iloc[0]['Outliers removidos']
            })
        return pd.DataFrame(filas).sort_values(
            by=['Silhouette', 'Inercia'],
            ascending=[False, True]
        ).reset_index(drop=True)

    def plot_benchmark(self, guardar=None):
        """
        Grafico comparativo de silhouette por modelo.
        """
        tabla = self.benchmark()

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        ax.bar(tabla['Modelo'], tabla['Silhouette'])
        ax.set_title('Comparacion de modelos K-Means')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('Silhouette')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()

        if guardar:
            plt.savefig(guardar, bbox_inches='tight')
        plt.show()

    def analisis_mejor(self):
        """
        Ejecuta el analisis completo del mejor modelo seleccionado.
        """
        m = self.mejor_modelo
        if m is None:
            print("Ejecute .ejecutar() primero.")
            return

        print(f"\n{'='*60}")
        print(f"ANALISIS DEL MEJOR MODELO K-MEANS: {self.__mejor_nombre}")
        print(f"{'='*60}")

        print("\n--- Resumen general ---")
        print(m.resumen().round(4))

        print("\n--- Benchmark por K del mejor modelo ---")
        print(m.benchmark_k().sort_values('K').round(4))

        print("\n--- Centroide por cluster (escala original) ---")
        print(m.centroides(escala='original').round(4))

        print("\n--- Resumen descriptivo por cluster ---")
        print(m.resumen_clusters.round(4))

        print("\n--- Variables mas discriminatorias entre centroides ---")
        print(m.variables_mas_discriminatorias(top_n=10).round(4))

        print("\n--- Interpretacion textual de clusters ---")
        m.interpretar_clusters(top_n=5)

        g = self.__carpeta
        m.plot_elbow_silhouette(
            guardar=f'{g}/kmeans_elbow_silhouette.png' if g else None
        )
        m.plot_clusters_pca(
            guardar=f'{g}/kmeans_clusters_pca.png' if g else None
        )
        m.plot_perfiles_centroides(
            guardar=f'{g}/kmeans_perfiles_centroides.png' if g else None
        )
        self.plot_benchmark(
            guardar=f'{g}/kmeans_benchmark_modelos.png' if g else None
        )
        
        

