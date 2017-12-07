import numpy
import plot
import itertools
import DBSCAN

def detectChainpoints(points, eps, minPts, labels, currentClusterID, nBCountsOfClusterPoints, debug):
    connectionDensityFactor = 0.5

    clusterIndizes = [i for i in nBCountsOfClusterPoints.keys()]

    # Get chainpointCandidates
    mean = sum(nBCountsOfClusterPoints.values()) / len(nBCountsOfClusterPoints)
    chainpointCandidateIndizes = [
        i for i in clusterIndizes if nBCountsOfClusterPoints[i] < mean * connectionDensityFactor]
    if len(chainpointCandidateIndizes) == 0:
        return currentClusterID
    if debug:
        plot.plot([points[i] for i in clusterIndizes],
                  [-2 if i in chainpointCandidateIndizes else currentClusterID for i in clusterIndizes],
                  title="Chainpoint candidates = Points with less than " +
                  str(connectionDensityFactor) + "*(mean Neighbors) Neighbors",
                  eps=eps,
                  text=[nBCountsOfClusterPoints[i] for i in clusterIndizes])

    # Add new outlier to chainpointCandidates
    indizesWithoutCandidates = [
        i for i in clusterIndizes if i not in chainpointCandidateIndizes]
    newLabes = DBSCAN.Algorithm([points[i] for i in indizesWithoutCandidates], eps, minPts)
    if debug:
        if -1 in newLabes:
            plot.plot([points[i] for i in indizesWithoutCandidates], newLabes,
                      title="Clustering cluster without candidates. New outliers are added to chainpoint candidates")
    for i in range(len(newLabes)):
        if newLabes[i] == -1:
            chainpointCandidateIndizes.append(indizesWithoutCandidates[i])

    # Cluster chainpointCandidates
    chainpointLabels = DBSCAN.Algorithm([points[i] for i in chainpointCandidateIndizes], eps, minPts)
    chainpointClusters = {}
    for i in range(len(chainpointCandidateIndizes)):
        # if chainpointLabels[i] == -1: continue
        if not chainpointLabels[i] in chainpointClusters:
            chainpointClusters[chainpointLabels[i]] = []
        chainpointClusters[chainpointLabels[i]].append(
            chainpointCandidateIndizes[i])
    if len(chainpointClusters) == 1:
        return currentClusterID
    if debug:
        plot.plot([points[i] for i in chainpointCandidateIndizes],
                  chainpointLabels,
                  title="Chainpoint candidates clusters. Outliers are removed from chainpoint candidates")

    if -1 in chainpointClusters:
        # chainpoint outliers can not be part of chains
        del chainpointClusters[-1]
    chainpointCandidateIndizes = []
    for c in chainpointClusters.values():
        chainpointCandidateIndizes.extend(c)

    # Check chainpointClusters by if two clusters merge in the union of chainpointCluster and pointsWithoutChainClusterCandidates
    chainpoints = []
    pointsWithoutChainClusterCandidates = [
        points[p] for p in clusterIndizes if not p in chainpointCandidateIndizes]
    labelsWithoutChainClusterCandidates = DBSCAN.Algorithm(pointsWithoutChainClusterCandidates, eps, minPts)
    newClusterCount = max(labelsWithoutChainClusterCandidates)
    for chainpointCluster in chainpointClusters.values():
        scanningPoints = pointsWithoutChainClusterCandidates + \
            [points[p] for p in chainpointCluster]
        clustersCount = max(DBSCAN.Algorithm(scanningPoints, eps, minPts))
        if clustersCount < newClusterCount:  # chainPointCluster is indeed a connecting chain
            chainpoints.extend(chainpointCluster)
            for p in chainpointCluster:
                labels[p] = -2

            title = "Confirmed chain"
        else:
            title = "Is not a chain"
        if debug:
            plot.plot(scanningPoints,
                      ([currentClusterID] * len(pointsWithoutChainClusterCandidates)
                       ) + ([-2] * len(chainpointCluster)),
                      title=title)

    # Label new clusters without chainpoints
    indizesWithoutChainpoints = [
        p for p in clusterIndizes if not p in chainpoints]
    newLabels = DBSCAN.Algorithm([points[p] for p in indizesWithoutChainpoints], eps, minPts)
    if debug:
        plot.plot([points[p] for p in indizesWithoutChainpoints],
                  newLabels, title="New clustering without chains")
    for i in range(len(indizesWithoutChainpoints)):
        p = indizesWithoutChainpoints[i]
        if newLabels[i] == -1:
            # new outliers can only occur between chainpoints -> is chanepoint
            labels[p] = -2
        else:
            labels[p] = newLabels[i] + currentClusterID - 1

    return currentClusterID + max(newLabels) - 1
