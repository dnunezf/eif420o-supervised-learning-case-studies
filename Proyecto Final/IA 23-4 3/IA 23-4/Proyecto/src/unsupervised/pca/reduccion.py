from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..no_supervisado_base import NoSupervisadoBase


class ReduccionDimensional(NoSupervisadoBase):
    def pca(self, n_componentes: int = 2):
        X = self.get_matrix()
        model = PCA(n_components=n_componentes, random_state=42)
        componentes = model.fit_transform(X)
        return {
            "model": model,
            "componentes": componentes,
            "varianza_explicada": model.explained_variance_ratio_,
        }

    def tsne(self, n_componentes: int = 2, perplexity: float = 30.0):
        X = self.get_matrix()
        model = TSNE(n_components=n_componentes, perplexity=perplexity, random_state=42, init="pca")
        componentes = model.fit_transform(X)
        return {"model": model, "componentes": componentes}

    def umap(self, n_componentes: int = 2, n_neighbors: int = 15):
        try:
            import umap as um
        except Exception as exc:
            raise ImportError("Install 'umap-learn' to use UMAP.") from exc

        X = self.get_matrix()
        model = um.UMAP(n_components=n_componentes, n_neighbors=n_neighbors, random_state=42)
        componentes = model.fit_transform(X)
        return {"model": model, "componentes": componentes}

