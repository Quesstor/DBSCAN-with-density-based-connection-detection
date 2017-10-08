import numpy as np
import random
import dbscan
import plot


def generatePoints(x,y,offset=0.5,count=40):
    return np.array([(random.uniform(x-offset, x+offset), random.uniform(y-offset, y+offset)) for i in range(count)])



c1 = generatePoints(1.5,3)
c2 = generatePoints(-1.5,-3)
c3 = generatePoints(1,1)

outlier = np.array([(-5,0),(0,5)])

waypoints1 = [(x,x) for x in np.arange(1, 5, 0.2)]
waypoints2 = [(x,pow(x,3)) for x in np.arange(-1.5, 1.5, 0.041)]

points = np.concatenate([c1, c2,  waypoints2 ])
#points = np.load("data.npy")

labels = dbscan.MyDBSCAN(points, eps=.3, minPts=3)
plot.plot(points, labels)
print("done")


