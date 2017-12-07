import numpy
import plot
import itertools
import math
import DBSCAN
from sklearn.decomposition import PCA

def detectChainpoints(points, eps, minPts, labels, currentClusterID, nBCountsOfClusterPoints, debug):
    pca = PCA(n_components=1)
    clusterIndizes = nBCountsOfClusterPoints.keys()
    clusterPoints = [points[p] for p in clusterIndizes]
    pcaIndicator = list()
    for i,p in enumerate(clusterPoints):
        nbh = [clusterPoints[j] for j in  DBSCAN.regionQuery(clusterPoints, i, eps)]
        pca.fit(nbh)
        pcaIndicator.append(pca.explained_variance_ratio_[0])

    plot.plot(clusterPoints, [math.floor(x*100) for x in pcaIndicator])
    print()

