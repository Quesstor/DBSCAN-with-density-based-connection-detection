import numpy
import plot

def MyDBSCAN(points, eps, minPts, connectionDensityFactor=0.5, detectWaypoints=True, labels=[], debug = False):
    clusterID = 1
    if(len(labels) == 0): labels = [0] * len(points)

    for pIndex in range(len(points)):
        if labels[pIndex] != 0: continue
        NeighborPts = regionQuery(points, pIndex, eps, labels)
        if len(NeighborPts) < minPts: labels[pIndex] = -1 #Outlier
        else:
            labels[pIndex] = clusterID

            nBCountsOfClusterPoints = {} #This saves the # of Neighbours for each point in cluster
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


            if detectWaypoints:
                if debug: plot.plot(points, [-1 if i == 0 else i for i in labels],
                                    title="Basic DBScan detected a cluster")

                clusterIndizes = [i for i in nBCountsOfClusterPoints.keys()]

                # Get waypointCandidates
                mean = sum(nBCountsOfClusterPoints.values()) / len(nBCountsOfClusterPoints)
                waypointCandidateIndizes = [i for i in clusterIndizes if nBCountsOfClusterPoints[i] < mean * connectionDensityFactor]
                if debug: plot.plot([points[i] for i in clusterIndizes], [-2 if i in waypointCandidateIndizes else -1 for i in clusterIndizes], title="Waypoint candidates")

                # Cluster waypoints
                waypointLabels = MyDBSCAN([points[i] for i in waypointCandidateIndizes], eps, minPts, detectWaypoints=False)
                if debug: plot.plot([points[i] for i in waypointCandidateIndizes] + [points[i] for i in clusterIndizes if not i in waypointCandidateIndizes],
                                    waypointLabels + [-1 for i in clusterIndizes if not i in waypointCandidateIndizes],
                                    title="Waypoint clusters" )

                waypointClusters = {}
                for i in range(len(waypointCandidateIndizes)):
                    if waypointLabels[i] == -1: continue
                    if not waypointLabels[i] in waypointClusters: waypointClusters[waypointLabels[i]] = []
                    waypointClusters[waypointLabels[i]].append(waypointCandidateIndizes[i])
                # Remove waypointClusters
                clusterID += removeWaypointClusters(points, eps, minPts, labels, clusterID, clusterIndizes, waypointClusters, debug)
            clusterID += 1
    return labels

def removeWaypointClusters(points, eps, minPts, labels, clusterID, clusterIndizes, waypointClusters, debug):
    newClustersFound = 0
    remainingWaypointClusters = [i for i in waypointClusters.keys()]
    numberOfClustersToRemove = 1
    while numberOfClustersToRemove <= len(remainingWaypointClusters):
        startIndex = 0
        while startIndex < len(remainingWaypointClusters):
            if len(remainingWaypointClusters) == 0: break
            keys = []
            for i in range(0, numberOfClustersToRemove):
                keys.append(remainingWaypointClusters[(startIndex + i) % len(remainingWaypointClusters)])

            waypoints = []
            for k in keys: waypoints.extend(waypointClusters[k])
            if removeWaypoints(points, eps, minPts, clusterIndizes, waypoints, labels, clusterID, newClustersFound,
                               debug):
                newClustersFound += 1
                numberOfClustersToRemove = 1
                startIndex = 0
                print("Found: " + str(keys))
                for k in keys: remainingWaypointClusters.remove(k)
            else:
                startIndex += 1
        numberOfClustersToRemove += 1
    return newClustersFound

def removeWaypoints(points, eps, minPts, clusterIndizes, waypointIndizes, labels, clusterID, newClustersCount, debug=False):
    if debug: plot.plot([points[i] for i in clusterIndizes],
                        [-2 if i in waypointIndizes else -1 for i in clusterIndizes],
                        title="Waypoints to remove")

    # prelabel Waypoints and check if a new Cluster emerges
    preLabels = []
    for i in range(len(clusterIndizes)):
        if clusterIndizes[i] in waypointIndizes or labels[clusterIndizes[i]] == -2:
            preLabels.append(-2)
        else:
            preLabels.append(0)

    newLabels = MyDBSCAN([points[i] for i in clusterIndizes], eps, minPts, detectWaypoints=False, labels=preLabels)
    if (max(newLabels) > newClustersCount+1):  # a new cluster is found
        for i in range(len(newLabels)):
            if newLabels[i] > 1:  # is point of the new cluster
                labels[clusterIndizes[i]] = clusterID + newLabels[i] - 1
            if newLabels[i] < 0: labels[clusterIndizes[i]] = -2  # is indeed a waypoint or a new outlier -> waypoint
        if debug: plot.plot([points[i] for i in clusterIndizes], newLabels, title="A new cluster was found")
        return True
    return False

def regionQuery(points, index, eps, labels):
    neighbors = []
    for p in range(len(points)):
        if labels[p] != -2 and numpy.linalg.norm(points[index] - points[p]) < eps:
            neighbors.append(p)
    return neighbors
