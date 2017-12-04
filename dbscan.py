import numpy
import plot
import itertools

def MyDBSCAN(points, eps, minPts, connectionDensityFactor=0.5, enableChainpointDetection=True, debug = False):
    clusterID = 1
    labels = [0] * len(points)
    if(debug): plot.plot(points, [-1]*len(points), eps=eps, title="Data for DBSCAN with eps="+str(eps)+" and minPts="+str(minPts))

    for pIndex in range(len(points)):
        if labels[pIndex] != 0: continue
        NeighborPts = regionQuery(points, pIndex, eps, labels)
        if len(NeighborPts) < minPts: labels[pIndex] = -1 #Outlier
        else:
            labels[pIndex] = clusterID

            nBCountsOfClusterPoints = {} #This saves the # of Neighbors for each point in cluster
            nBCountsOfClusterPoints[pIndex] = len(NeighborPts)

            # Expand Cluster
            i = 0
            while i < len(NeighborPts):
                neighborIndex = NeighborPts[i]
                if labels[neighborIndex] == 0 or labels[neighborIndex] == -1:
                    labels[neighborIndex] = clusterID
                    PnNeighborPts = regionQuery(points, neighborIndex, eps, labels)
                    nBCountsOfClusterPoints[neighborIndex] = len(PnNeighborPts)
                    if len(PnNeighborPts) >= minPts:
                        NeighborPts = NeighborPts + PnNeighborPts
                i += 1

            if debug: plot.plot(points, [-1 if i == 0 else i for i in labels], title="Basic DBScan detected a cluster")
            if enableChainpointDetection:
                clusterID = detectChainpoints(points, eps, minPts, connectionDensityFactor, labels, clusterID, nBCountsOfClusterPoints, debug)
            clusterID += 1
    return labels

def detectChainpoints(points, eps, minPts, connectionDensityFactor, labels, currentClusterID, nBCountsOfClusterPoints, debug):
    clusterIndizes = [i for i in nBCountsOfClusterPoints.keys()]

    # Get chainpointCandidates
    mean = sum(nBCountsOfClusterPoints.values()) / len(nBCountsOfClusterPoints)
    chainpointCandidateIndizes = [i for i in clusterIndizes if nBCountsOfClusterPoints[i] < mean * connectionDensityFactor]
    if len(chainpointCandidateIndizes) == 0: return currentClusterID
    if debug: plot.plot([points[i] for i in clusterIndizes],
                        [-2 if i in chainpointCandidateIndizes else currentClusterID for i in clusterIndizes],
                        title="Chainpoint candidates = Points with less than "+str(connectionDensityFactor)+"*(mean Neighbors) Neighbors",
                        eps=eps,
                        text=[nBCountsOfClusterPoints[i] for i in clusterIndizes])

    # Add new outlier to chainpointCandidates
    indizesWithoutCandidates = [i for i in clusterIndizes if i not in chainpointCandidateIndizes]
    newLabes = MyDBSCAN([points[i] for i in indizesWithoutCandidates], eps, minPts, enableChainpointDetection=False)
    if debug:
        if -1 in newLabes:
            plot.plot([points[i] for i in indizesWithoutCandidates],newLabes,title="Clustering cluster without candidates. New outliers are added to chainpoint candidates")
    for i in range(len(newLabes)):
        if newLabes[i] == -1:
            chainpointCandidateIndizes.append(indizesWithoutCandidates[i])

    # Cluster chainpointCandidates
    chainpointLabels = MyDBSCAN([points[i] for i in chainpointCandidateIndizes], eps, minPts, enableChainpointDetection=False)
    chainpointClusters = {}
    for i in range(len(chainpointCandidateIndizes)):
        #if chainpointLabels[i] == -1: continue
        if not chainpointLabels[i] in chainpointClusters: chainpointClusters[chainpointLabels[i]] = []
        chainpointClusters[chainpointLabels[i]].append(chainpointCandidateIndizes[i])
    if len(chainpointClusters)==1: return currentClusterID
    if debug: plot.plot([points[i] for i in chainpointCandidateIndizes],
                        chainpointLabels,
                        title="Chainpoint candidates clusters. Outliers are removed from chainpoint candidates")

    if -1 in chainpointClusters: del chainpointClusters[-1] #chainpoint outliers can not be part of chains
    chainpointCandidateIndizes = []
    for c in chainpointClusters.values():
        chainpointCandidateIndizes.extend(c)

    # Check chainpointClusters by if two clusters merge in the union of chainpointCluster and pointsWithoutChainClusterCandidates
    chainpoints = []
    pointsWithoutChainClusterCandidates = [points[p] for p in clusterIndizes if not p in chainpointCandidateIndizes]
    labelsWithoutChainClusterCandidates = MyDBSCAN(pointsWithoutChainClusterCandidates, eps, minPts, enableChainpointDetection=False)
    newClusterCount = max(labelsWithoutChainClusterCandidates)
    for chainpointCluster in chainpointClusters.values():
        scanningPoints = pointsWithoutChainClusterCandidates + [points[p] for p in chainpointCluster]
        clustersCount = max(MyDBSCAN(scanningPoints, eps, minPts, enableChainpointDetection=False))
        if clustersCount < newClusterCount: #chainPointCluster is indeed a connecting chain
            chainpoints.extend(chainpointCluster)
            for p in chainpointCluster: labels[p] =-2

            title ="Confirmed chain"
        else: title ="Is not a chain"
        if debug: plot.plot(scanningPoints,
                  ([currentClusterID] * len(pointsWithoutChainClusterCandidates)) + ([-2] * len(chainpointCluster)),
                  title=title)

    # Label new clusters without chainpoints
    indizesWithoutChainpoints = [p for p in clusterIndizes if not p in chainpoints]
    newLabels = MyDBSCAN([points[p] for p in indizesWithoutChainpoints], eps, minPts, enableChainpointDetection=False)
    if debug: plot.plot([points[p] for p in indizesWithoutChainpoints], newLabels, title="New clustering without chains")
    for i in range(len(indizesWithoutChainpoints)):
        p = indizesWithoutChainpoints[i]
        if newLabels[i] == -1: labels[p] = -2 #new outliers can only occur between chainpoints -> is chanepoint
        else: labels[p] = newLabels[i] + currentClusterID - 1

    return currentClusterID + max(newLabels) - 1


def regionQuery(points, index, eps, labels):
    neighbors = []
    for p in range(len(points)):
        if labels[p] != -2 and numpy.linalg.norm(points[index] - points[p]) < eps:
            neighbors.append(p)
    return neighbors
