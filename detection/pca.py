import numpy
import plot
import itertools
import math
import DBSCAN
import numpy
from sklearn.decomposition import PCA

def detectChainpoints(points, eps, minPts, labels, currentClusterID, nBCountsOfClusterPoints, debug):
    pca = PCA()
    dimensions = len(points[0])
    clusterIndizes = nBCountsOfClusterPoints.keys()
    clusterPoints = [points[p] for p in clusterIndizes]
    pcaIndicator = list()
    for i,p in enumerate(clusterPoints):
        nbh = [clusterPoints[j] for j in  DBSCAN.regionQuery(clusterPoints, i, eps)]
        nbh.append(clusterPoints[i])
        nbh = numpy.array(nbh)
        nbh_normalized = nbh - nbh.mean(axis=0)
        cov = numpy.cov(nbh_normalized, rowvar=False)
        eigenValues = numpy.linalg.eigvalsh(cov) #Compute Eigenvalues (EV)

        eigenValues = eigenValues / eigenValues.sum(0) #Compute ratio of EVs
        indicator = min(eigenValues) * dimensions   #is between 1 and 0: 
                                                    #1 = perfect distribution of nbhs in all dimensions, 
                                                    #0 = all neighbours in hyperplane
        pcaIndicator.append(indicator)

    if debug: plot.plot(clusterPoints, [math.floor(x*100) for x in pcaIndicator], text=[math.floor(x*100)/100 for x in pcaIndicator], title="Indicator values")

    candidates = [index for i,index in enumerate(clusterIndizes) if pcaIndicator[i] < 0.5]
    if len(candidates) == len(clusterPoints): return currentClusterID
    if debug: plot.plot([points[i] for i in clusterIndizes], [-2 if i in candidates else -1 for i in clusterIndizes], title="Chainpoint candidates")

    #Cluster remaining points
    remainingPoints = [i for i in clusterIndizes if not i in candidates]
    remainingPointsLabels = DBSCAN.Algorithm([points[i] for i in remainingPoints], eps, minPts)
    if debug: plot.plot([points[i] for i in remainingPoints], remainingPointsLabels, title="Remaining points clustered")
    maxClustersCount = max(remainingPointsLabels)
    if(maxClustersCount) == 1: return currentClusterID #It is not possible to find new Clusters by removing chains
    for i,index in enumerate(remainingPoints): 
        if remainingPointsLabels[i] == -1: candidates.append(index) #Add outliers to candidates
    remainingPoints = [i for i in clusterIndizes if not i in candidates] #Rebuild remaining points because candidates were added
    if len(remainingPoints) == 0: return currentClusterID

    #Cluster Candidates
    candidateLabels = DBSCAN.Algorithm([points[i] for i in candidates], eps, minPts)
    if debug: plot.plot([points[i] for i in candidates], candidateLabels, title="Chainpoint candidate clusters")
    for i,label in enumerate(candidateLabels):
        if label==-1: remainingPoints.append(candidates[i]) #Add outlier to remainingPoints
    candidates = [p for p in candidates if not p in remainingPoints] #Rebuild candidates because outliers have to be removed
    candidateLabels = [i for i in candidateLabels if not i==-1]

    if debug: plot.plot(clusterPoints, [-2 if not i in remainingPoints else -1 for i in clusterIndizes], title="Final candidates set")

    #Validated candidate clusters
    for c in set(candidateLabels):
        candidatesCluster = [index for i,index in enumerate(candidates) if candidateLabels[i]==c]
        union = numpy.array(candidatesCluster + remainingPoints)
        unionLabels = DBSCAN.Algorithm([points[i] for i in union], eps, minPts)
        title = "Is a chain"
        if max(unionLabels) == maxClustersCount: #is not a chain
            title = "Is NOT a chain"
            for p in candidatesCluster: candidates.remove(p)
        if debug: plot.plot([points[p] for p in union], [-1 if i in remainingPoints else -2 for i in union], title=title)
            
    #Candidates are now validated and we can cluster without chains
    remainingPoints = [i for i in clusterIndizes if not i in candidates]
    remainingPointsLabels = DBSCAN.Algorithm([points[i] for i in remainingPoints], eps, minPts)
    for i,index in enumerate(remainingPoints):
        labels[index] = currentClusterID + remainingPointsLabels[i] -1
    for index in candidates:
        labels[index] = -2
    
    if debug: plot.plot([points[i] for i in remainingPoints], remainingPointsLabels, title="New Clustering without chains")
    return currentClusterID + max(remainingPointsLabels) - 1