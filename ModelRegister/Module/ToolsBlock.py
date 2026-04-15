from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class ToolsBlock:
    def __init__(self, data):
        self.data = data

    def pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        pca.fit(self.data)
        return pca.transform(self.data), pca

    def kmeans(self, n_clusters=2, is_show_Cluster=False, save_path=None):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.data)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        if is_show_Cluster or save_path:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='rainbow')
            plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='input_map')
            plt.title('K-means Clustering')
            plt.xlabel('X')
            plt.ylabel('Y')
            if save_path:
                plt.savefig(save_path)
            if is_show_Cluster:
                plt.show()
        return labels, kmeans
