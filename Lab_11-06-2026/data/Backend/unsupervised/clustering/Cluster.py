from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans

from ..no_supervisado_base import NoSupervisadoBase


class ClusterModelos(NoSupervisadoBase):
    def kmeans(self, n_clusters: int = 3, n_init: str | int = "auto", max_iter: int = 300):
        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42,
        )
        return self.fit_predict(model, "kmeans")

    def mini_batch_kmeans(self, n_clusters: int = 3, batch_size: int = 512):
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init="auto",
            max_iter=120,
            random_state=42,
        )
        return self.fit_predict(model, "mini_batch_kmeans")

    def kmedoids(self, n_clusters: int = 3, metric: str = "euclidean", max_iter: int = 300):
        try:
            from sklearn_extra.cluster import KMedoids
        except Exception as exc:
            raise ImportError("Install 'scikit-learn-extra' to use KMedoids.") from exc

        model = KMedoids(n_clusters=n_clusters, metric=metric, max_iter=max_iter, random_state=42)
        return self.fit_predict(model, "kmedoids")

    def hac(self, n_clusters: int = 3, linkage: str = "ward"):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric="euclidean")
        return self.fit_predict(model, f"hac_{linkage}")

    def dbscan(self, eps: float = 0.5, min_samples: int = 5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        return self.fit_predict(model, "dbscan")
