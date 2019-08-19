if __name__ == '__main__':
    import chaindetection
    import plot
    import dataloader
    from scipy.cluster.hierarchy import single, fcluster, dendrogram
    from scipy.spatial.distance import pdist
    from sklearn.cluster import DBSCAN, KMeans, MeanShift

    # Load data
    points, labels = dataloader.random2cluster(.3, .05, .05)

    # Set params
    eps = .3
    allowedVar = .5

    # Cluster points with DBSCAN
    def DBSCANclustering(points):
        return DBSCAN(eps=eps, min_samples=2).fit(points).labels_

    def hierarchyCluster(points):
        y = pdist([p for p in points])
        Z = single(y)
        return fcluster(Z, eps, criterion='distance')

    def KMeansClustering(points):
        return KMeans(n_clusters=6, random_state=0).fit(points).labels_

    def meanShiftClustering(points):
        return MeanShift(bandwidth=20).fit(points).labels_

    clusteringAlgo = hierarchyCluster
    labels = clusteringAlgo(points)
    plot.plot(points, labels, randomcolor=True, title="ClusteringAlgo results")
    #a,b = (zip(*[(i, x) for i, x in enumerate(points) if labels[i] == 0]))

    labelsCD = chaindetection.chainDetection(
        points, [l for l in labels], eps, allowedVar, clusteringAlgo, chainDim=1)

    plot.plot(points, labelsCD, title="Final output",
              randomcolor=True, dotSize=5)
