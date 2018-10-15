if __name__ == '__main__':
    import chaindetection
    import plot
    import dataloader
    from sklearn.cluster import DBSCAN

    #Load data
    points = dataloader.loadcsv()

    #Set params
    eps = .01
    minPts = 15

    #Cluster points with DBSCAN
    labels = DBSCAN(eps=.01, min_samples=15).fit(points).labels_
    currentClusterID = max(labels)+1

    #For each cluster 
    for cluster in range(0, currentClusterID):
        clusterpointindizes, clusterpoints = zip(*[(i,x) for i,x in enumerate(points) if labels[i]==cluster])

        #Apply chaindetection
        #Signature: detectChainpoints(points, eps, minPts, chainDim=1, allowedVar=0.2, eps2=0, debug=False, makeChainClusters=True)
        #   points              - The points on which the chaindetection is applied
        #   eps                 - epsilon value of the overlying DBSCAN
        #   minPts              - minPts parameter of the overlying DBSCAN
        #   chainDim            - The dimensionality of chains the user wants to detect (default = 1)
        #   allowedVar          - in [0,1[ The allowed variation of chains (default = 0.2)
        #   eps2                - The epsilon value to determine the range of range queries when calculating the normed error. If set to 0 eps2 will be set to eps (default = 0)
        #   debug               - Enable this to get a lot of plots and console outputs on how the algorithm works (default = False)
        #   makeChainClusters   - Enable this to assign chains to a new cluster (default = True)
        clusterlabels = chaindetection.detectChainpoints(clusterpoints, eps, minPts, debug=True)

        #Update labels
        for i,pointindex in enumerate(clusterpointindizes):
            if clusterlabels[i] > 0: labels[pointindex] = currentClusterID + clusterlabels[i] - 1
            if clusterlabels[i] == -2: labels[pointindex] = -2
        currentClusterID += max(clusterlabels)

    plot.plot(points, labels, title="Final output", randomcolor=True, dotSize=5)