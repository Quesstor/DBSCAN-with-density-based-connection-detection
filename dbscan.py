import numpy
import plot

def MyDBSCAN(points, eps, minPts, connectionDensityFactor=0.5, detectWaypoints=True, labels=[]):
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
                if labels[neighborIndex] == -1:
                    labels[neighborIndex] = clusterID

                elif labels[neighborIndex] == 0:
                    labels[neighborIndex] = clusterID
                    PnNeighborPts = regionQuery(points, neighborIndex, eps, labels)
                    if len(PnNeighborPts) >= minPts:
                        nBCountsOfClusterPoints[neighborIndex] = len(PnNeighborPts)
                        NeighborPts = NeighborPts + PnNeighborPts
                i += 1


            if detectWaypoints:
                clusterIndizes = [i for i in nBCountsOfClusterPoints.keys()]

                #Get possible Waypoints
                mean = sum(nBCountsOfClusterPoints.values()) / len(nBCountsOfClusterPoints)
                possibleWaypointIndizes = [i for i in clusterIndizes if nBCountsOfClusterPoints[i] < mean * connectionDensityFactor]

                # Cluster Possible Waypoints and remove outlier
                waypointLabels = MyDBSCAN([points[i] for i in possibleWaypointIndizes], eps, minPts, detectWaypoints=False)

                waypointClusters = {}
                for i in range(len(possibleWaypointIndizes)):
                    if waypointLabels[i] == -1: continue
                    if not waypointLabels[i] in waypointClusters: waypointClusters[waypointLabels[i]] = []
                    waypointClusters[waypointLabels[i]].append(possibleWaypointIndizes[i])

                newClustersCount = 1 #default 1 as the whole cluster is one cluster
                for waypointCluster in waypointClusters.values():
                    # prelabel Waypoints and check if a new Cluster emerges
                    preLabels = []
                    for i in range(len(clusterIndizes)):
                        if clusterIndizes[i] in waypointCluster or labels[clusterIndizes[i]]==-2: preLabels.append(-2)
                        else: preLabels.append(0)

                    newLabels = MyDBSCAN([points[i] for i in clusterIndizes], eps, minPts, detectWaypoints=False, labels=preLabels)
                    newClustersCount = max(newLabels)
                    if (newClustersCount > 1): #a new cluster is found
                        for i in range(len(newLabels)):
                            if newLabels[i] > 1 : #is point of the new cluster
                                labels[clusterIndizes[i]] = clusterID + newLabels[i] - 1
                            if newLabels[i] < 0: labels[clusterIndizes[i]] = -2 #is indeed a waypoint or a new outlier -> waypoint
                clusterID += newClustersCount -1
            clusterID += 1
    return labels

def regionQuery(points, index, eps, labels):
    neighbors = []
    for p in range(len(points)):
        if labels[p] != -2 and numpy.linalg.norm(points[index] - points[p]) < eps:
            neighbors.append(p)
    return neighbors

