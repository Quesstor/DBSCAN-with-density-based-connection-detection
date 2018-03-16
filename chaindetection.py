import numpy
import plot
import itertools
import math
from functools import partial
from sklearn.cluster import DBSCAN as DBSCAN_Algorithm
from sklearn.neighbors import NearestNeighbors
import pool


def calculateError(i, points, dimensions, chainDim, NearestNeighborsCalculator):
    nbh = [points[j] for j in regionQuery(points, i, NearestNeighborsCalculator)]
    if len(nbh)==0: return 0
    nbh.append(points[i])

    nbh = numpy.array(nbh)
    nbh_normalized = nbh - nbh.mean(axis=0)
    cov = numpy.cov(nbh_normalized, rowvar=False, ddof=0)
    eigenValues = numpy.linalg.eigvalsh(cov) #Compute Eigenvalues (EV)

    eigenValuesNormed = eigenValues / eigenValues.sum(0) #Compute ratio of EVs
    eigenValuesNormed = sorted(eigenValuesNormed, reverse=True)

    indicatorValues = eigenValuesNormed[-(dimensions-chainDim):]
    indicator = sum(indicatorValues) 
    indicatorNormed = indicator * (dimensions / (dimensions-chainDim))  #is between 1 and 0: 
                                                                        #1 = perfect distribution of nbs in all dimensions, 
                                                                        #0 = all neighbours in hyperplane
    return indicatorNormed

def regionQuery(points, index, NearestNeighborsCalculator):
    nbh = NearestNeighborsCalculator.radius_neighbors([points[index]])
    nbh = nbh[1][0]
    nbh = [x for x in nbh if not x==index]
    return nbh
    #return [p for p in range(len(points)) if p!=index and numpy.linalg.norm( numpy.subtract(points[index],points[p])) < eps] 

def validatedChainCandidate(remainingPointsNNC, candidatesCluster, remainingPointsLabels, points):
    touchingClusters = set()
    for candidate in candidatesCluster:
        nbh = remainingPointsNNC.radius_neighbors([points[candidate]])
        for i in nbh[1][0]:
            if remainingPointsLabels[i] != -1: touchingClusters.add(remainingPointsLabels[i])
            if len(touchingClusters) >= 2: return touchingClusters
    return False

def detectChainpoints(points, eps, minPts, chainDim=1, allowedVar=0.2, eps2=0, debug=False, makeChainClusters=True):
    if eps2==0: eps2 = eps
    
    if debug: print("Starting Chainpointdetection on "+str(len(points))+" points")

    DBSCAN = DBSCAN_Algorithm(eps=eps, min_samples=minPts)

    dimensions = len(points[0])
    clusterIndizes = range(len(points))
    labels = [0] * len(points)

    normedErrors = list()
    NearestNeighborsCalculator = NearestNeighbors(radius=eps2)
    NearestNeighborsCalculator.fit(points)

    if debug: print("Calculating normed errors ... ", end="", flush=True)
    calculateErrorFnc = partial(calculateError, points=points, dimensions=dimensions, chainDim=chainDim, NearestNeighborsCalculator=NearestNeighborsCalculator)
    #normedErrors = [calculateErrorFnc(x) for x in clusterIndizes]
    normedErrors = pool.getThreadPool().map(calculateErrorFnc, clusterIndizes)
    
    if debug: 
        print("done")
        plot.plot(points, [math.floor(x*100) for x in normedErrors], autumn=True, title="Normed error values")
                        #text=[math.floor(x*1000)/1000 for x in pcaIndicator],                         

    candidates = [index for index,normedError in enumerate(normedErrors) if normedError <= allowedVar]
    if len(candidates) == len(points) or len(candidates) == 0: return labels
    if debug: plot.plot(points, [1 if i in candidates else -1 for i in clusterIndizes], title="Chainpoint candidates")

    #Cluster remaining points
    if debug: print("Refining chain-point candidates: Cluster remaining points")
    remainingPoints = [i for i in clusterIndizes if not i in candidates]
    remainingPointsLabels = DBSCAN.fit([points[i] for i in remainingPoints]).labels_
    if debug: plot.plot([points[i] for i in remainingPoints], [1 if l==-1 else -1 for l in remainingPointsLabels], title="Refining chainpoints: Remaining points clustered, Outliers are added to candidates")
    if(max(remainingPointsLabels)) == 0: return labels 
    for i,index in enumerate(remainingPoints): 
        if remainingPointsLabels[i] == -1: candidates.append(index) #Add outliers to candidates
    remainingPoints = [i for i in clusterIndizes if not i in candidates] #Rebuild remaining points because candidates were added
    if len(remainingPoints) == 0: return labels

    #Cluster Candidates
    if debug: print("Refining chain-point candidates: Cluster Candidates")
    candidateLabels = DBSCAN.fit([points[i] for i in candidates]).labels_
    if debug: plot.plot([points[i] for i in candidates], [1 if l==-1 else -1 for l in candidateLabels], title="Refining chainpoints: Chainpoint candidates clustered, Outliers are removed from candidates")
    for i,label in enumerate(candidateLabels):
        if label==-1: 
            remainingPoints.append(candidates[i]) #Add outlier to remainingPoints

    candidates = [c for i,c in enumerate(candidates) if not candidateLabels[i]==-1]
    candidateLabels = [l for l in candidateLabels if not l==-1]

    remainingPointsLabels = DBSCAN.fit([points[i] for i in remainingPoints]).labels_

    if debug: 
        plot.plot([points[i] for i in (candidates + remainingPoints)], candidateLabels + ([-1]*len(remainingPoints)), title="Chaincandidates")
        plot.plot([points[i] for i in (candidates + remainingPoints)], [-1]*len(candidates) + remainingPointsLabels.tolist(), title="Remaining points clustered")

    #Validate candidate clusters
    validatedChainpoints = list()
    remainingPointsNNC = NearestNeighbors(radius=eps)
    remainingPointsNNC.fit([points[i] for i in remainingPoints])
    if debug: print("Validating "+str(len(set(candidateLabels)))+" chaincandidates ", end="")
    for c in set(candidateLabels):
        candidatesCluster = [index for i,index in enumerate(candidates) if candidateLabels[i]==c]
        isChain = validatedChainCandidate(remainingPointsNNC, candidatesCluster, remainingPointsLabels, points)
        if isChain: validatedChainpoints.extend(candidatesCluster)

        if debug: 
            print(".", end="", flush=True)
            title = "IS chain"
            if not isChain: title = "NO chain"
            union = candidatesCluster + remainingPoints
            debugPoints = [points[p] for p in union]
            if isChain: 
                connectedClusters = []
                connectedLabels = []
                for cci, connectedCluster in enumerate(isChain):
                    clusterpoints = [points[p] for i,p in enumerate(remainingPoints) if remainingPointsLabels[i] == connectedCluster]
                    connectedClusters += clusterpoints
                    connectedLabels += [cci]*len(clusterpoints)
                plot.plot([p for p in points] + connectedClusters + [points[p] for p in candidatesCluster], [-1]*len(points) + connectedLabels + [-2]*len(candidatesCluster), title=title+": Connected Clusters")
            else: plot.plot(debugPoints, [-2]*len(candidatesCluster) + [-1]*len(remainingPoints), title=title)

    if len(validatedChainpoints) == 0: return labels
            
    #Candidates are now validated and we can cluster without chains
    remainingPoints = [i for i in clusterIndizes if not i in validatedChainpoints]
    remainingPointsLabels = DBSCAN.fit([points[i] for i in remainingPoints]).labels_
    for i,index in enumerate(remainingPoints):
        labels[index] = remainingPointsLabels[i]

    if makeChainClusters:
        currentLabel = max(labels) + 1
        chainLabels = DBSCAN.fit([points[i] for i in validatedChainpoints]).labels_
        for i, index in enumerate(validatedChainpoints): labels[index] = currentLabel + chainLabels[i]
    else:
        for index in validatedChainpoints: labels[index] = -2
    
    if debug: plot.plot([points[i] for i in remainingPoints], remainingPointsLabels, title="New Clustering without chains")
    return labels