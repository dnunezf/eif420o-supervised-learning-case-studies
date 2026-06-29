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

        print("💡 Usualmente:")
        print("✔ UMAP > T-SNE en estabilidad")
        print("✔ ACP + UMAP/T-SNE = mejor resultado")