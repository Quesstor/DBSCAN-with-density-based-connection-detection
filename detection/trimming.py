import numpy
import plot
import itertools
import math
import DBSCAN

def detectChainpoints(points, eps, minPts, labels, currentClusterID, nBCountsOfClusterPoints, debug):
    clusterIndizes = [i for i in nBCountsOfClusterPoints.keys()]
    trimCount = math.floor(len(nBCountsOfClusterPoints) / 10)
    trim = sorted(nBCountsOfClusterPoints.values())[trimCount]
    while len(clusterIndizes) > trimCount:
        trimmedPoints = [
            i for i in clusterIndizes if nBCountsOfClusterPoints[i] <= trim]
        plot.plot([points[i] for i in clusterIndizes],
                  [-2 if i in trimmedPoints else -1 for i in clusterIndizes],
                  title="Trimmed 10% points with the lowest Neighbours count")

        clusterIndizes = [i for i in clusterIndizes if not i in trimmedPoints]
        nBCountsOfClusterPoints = dict()
        for i in clusterIndizes:
            nBCountsOfClusterPoints[i] = len(DBSCAN.regionQuery(points, i, eps))
        trim = sorted(nBCountsOfClusterPoints.values())[trimCount]


