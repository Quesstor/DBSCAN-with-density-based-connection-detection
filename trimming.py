import numpy
import plot
import itertools
import math


def DBSCAN(points, eps, minPts, clusterFoundFunction):
    clusterID = 1
    labels = [0] * len(points)
    #if(debug): plot.plot(points, [-1]*len(points), eps=eps, title="Data for DBSCAN with eps="+str(eps)+" and minPts="+str(minPts))

    for pIndex in range(len(points)):
        if labels[pIndex] != 0:
            continue
        NeighborPts = regionQuery(points, pIndex, eps)
        if len(NeighborPts) < minPts:
            labels[pIndex] = -1  # Outlier
        else:
            labels[pIndex] = clusterID

            # This saves the # of Neighbors for each point in cluster
            nBCountsOfClusterPoints = {}
            nBCountsOfClusterPoints[pIndex] = len(NeighborPts)

            # Expand Cluster
            i = 0
            while i < len(NeighborPts):
                neighborIndex = NeighborPts[i]
                if labels[neighborIndex] == 0 or labels[neighborIndex] == -1:
                    labels[neighborIndex] = clusterID
                    PnNeighborPts = regionQuery(points, neighborIndex, eps)
                    nBCountsOfClusterPoints[neighborIndex] = len(PnNeighborPts)
                    if len(PnNeighborPts) >= minPts:
                        NeighborPts = NeighborPts + PnNeighborPts
                i += 1

            if debug:
                plot.plot(points, [-1 if i == 0 else i for i in labels],
                          title="Basic DBScan detected a cluster")
            if enableChainpointDetection:
                clusterID = clusterFoundFunction(
                    points, eps, minPts, labels, clusterID, nBCountsOfClusterPoints, debug)
            clusterID += 1
    return labels


def Algorithm(points, eps, minPts, enableChainpointDetection=True, debug=False):
    return DBSCAN(points, eps, minPts, detectChainpoints)


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
            nBCountsOfClusterPoints[i] = len(regionQuery(points, i, eps))
        trim = sorted(nBCountsOfClusterPoints.values())[trimCount]


def regionQuery(points, index, eps):
    neighbors = []
    for p in range(len(points)):
        if p != index and numpy.linalg.norm(points[index] - points[p]) < eps:
            neighbors.append(p)
    return neighbors
