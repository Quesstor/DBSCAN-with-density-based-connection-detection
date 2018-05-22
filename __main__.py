if __name__ == '__main__':
    import plot
    import detection.density
    import detection.pca
    import json
    import testdata
    from sklearn.cluster import DBSCAN
    import numpy as np
    import winsound
    import time 

    points = testdata.test4()
    eps = 1
    minPts = 3

    plot.plot(points, [-1]*len(points), title="Dataset")
    dbs = DBSCAN(eps=eps, min_samples=minPts).fit(points)
    labels = dbs.labels_

    clusterID = max(labels)+1
    DBSCANClusterID = clusterID

    for cluster in range(0,clusterID):
        clusterpointindizes, clusterpoints = zip(*[(i,x) for i,x in enumerate(points) if labels[i]==cluster])
        clusterlabels = detection.pca.detectChainpoints(clusterpoints, eps, minPts)

        #plot.plot(clusterpoints, clusterlabels, title="Cluster with Chaindetection")

        for i,pointindex in enumerate(clusterpointindizes):
            if clusterlabels[i] > 0: 
                labels[pointindex] = clusterID + clusterlabels[i] - 1
            if clusterlabels[i] == -2: 
                labels[pointindex] = -2
        
        #plot.plot(points, labels, title="Clustering after chaindetection")
        clusterID += max(clusterlabels)

    plot.plot(points, labels, title="Final output")

    labels = [max(0, l - DBSCANClusterID + 1) for l in labels]
    labels = [-1 if l==0 else l for l in labels]
    plot.plot(points, labels, title="New Clusters with Chaindetection")

    #print("done")
    winsound.Beep(500, 1000)

