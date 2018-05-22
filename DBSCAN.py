import plot
import numpy

def Algorithm(points, eps, minPts, clusterFoundFunction=False, debug=False):
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

            if debug and False:
                plot.plot(points, [-1 if i == 0 else i for i in labels],
                          title="Basic DBScan detected a cluster")
            if clusterFoundFunction != False:
                clusterID = clusterFoundFunction(points, eps, minPts, labels, clusterID, nBCountsOfClusterPoints, debug)
            clusterID += 1
    return labels

def regionQuery(points, index, eps):
    return [p for p in range(len(points)) if p!=index and numpy.linalg.norm(points[index] - points[p]) < eps] 
