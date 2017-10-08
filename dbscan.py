import numpy
import plot

def MyDBSCAN(points, eps, minPts, collectWaypoints=True, labels=[]):
    clusterID = 1
    if(len(labels) == 0): labels = [0] * len(points)

    possibleWaypoints = []
    possibleWaypointIndizes = []

    for pIndex in range(len(points)):
        if labels[pIndex] != 0: continue
        NeighborPts = regionQuery(points, pIndex, eps)
        if len(NeighborPts) < minPts:
            labels[pIndex] = -1
        else: #Expand Cluster

            nBCounts = {}
            nBCounts[pIndex] = len(NeighborPts)

            labels[pIndex] = clusterID
            i = 0
            while i < len(NeighborPts):
                neighborIndex = NeighborPts[i]
                if labels[neighborIndex] == -1:
                    labels[neighborIndex] = clusterID

                elif labels[neighborIndex] == 0:
                    labels[neighborIndex] = clusterID
                    PnNeighborPts = regionQuery(points, neighborIndex, eps)
                    if len(PnNeighborPts) >= minPts:

                        nBCounts[neighborIndex] = len(PnNeighborPts)

                        NeighborPts = NeighborPts + PnNeighborPts
                i += 1
            if collectWaypoints:
                #Get possible Waypoints
                values = nBCounts.values()
                mean = sum(nBCounts.values()) / len(values)
                for index, nBCount in nBCounts.items():
                    if nBCount < mean/2:
                        possibleWaypoints.append(points[index])
                        possibleWaypointIndizes.append(index)
            clusterID += 1

    if collectWaypoints:
        # Cluster Possible Waypoints and remove outlier
        waypointLabels = MyDBSCAN(possibleWaypoints, eps, minPts, False)

        waypointClusters = {}
        for i in range(len(possibleWaypoints)):
            if waypointLabels[i] == -1: continue
            if not waypointLabels[i] in waypointClusters: waypointClusters[waypointLabels[i]] = []
            waypointClusters[waypointLabels[i]].append(possibleWaypointIndizes[i])

        # prelabel Cluster of Waypoints and check if new Clusters emerge
        for waypointCluster in waypointClusters.values():
            preLabels = []
            for l in labels:
                if l==-2: preLabels.append(-2)
                else: preLabels.append(0)
            for i in waypointCluster: preLabels[i] = -2

            newLabels = MyDBSCAN(points, eps, minPts, False, preLabels)
            if(max(newLabels) > max(labels)):
                labels = newLabels
    return labels

def regionQuery(points, index, eps):
    neighbors = []
    for p in range(len(points)):
        if numpy.linalg.norm(points[index] - points[p]) < eps:
            neighbors.append(p)
    return neighbors

