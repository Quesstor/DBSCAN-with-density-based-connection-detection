import numpy as np
import random
import dbscan
import plot


def generatePoints(x,y,offset=0.5,count=40):
    return np.array([(random.uniform(x-offset, x+offset), random.uniform(y-offset, y+offset)) for i in range(count)])

test0 = np.concatenate([generatePoints(0,0), generatePoints(5,5), [(x,x) for x in np.arange(0,5,0.1)]])

test1 = np.concatenate([generatePoints(0,0), generatePoints(4,0), generatePoints(-4,0), [(x,0) for x in np.arange(-4,4,0.2)],
                        [(-2,x) for x in np.arange(0,2,0.2)], [(2,x) for x in np.arange(0,2,0.2)], [(4,x) for x in np.arange(0,2,0.2)] ])

test2 = np.concatenate([generatePoints(1.5,3), [(x,pow(x,3)) for x in np.arange(-1.5, 1.5, 0.1)], generatePoints(-1.5,-3)])

test3 = np.concatenate([generatePoints(5,0), generatePoints(0,0),generatePoints(5,5), generatePoints(8,8),
                        [(x,x) for x in np.arange(0, 8, 0.2)], [(x,0) for x in np.arange(0, 5, 0.2)], [(5,x) for x in np.arange(0, 5, 0.2)],
                        generatePoints(5,5,5,100)])

test4 = np.concatenate([
    generatePoints(5, 0), generatePoints(5,3), generatePoints(5,6),
    [(0, x) for x in np.arange(0, 6, 0.2)],[(5,x) for x in np.arange(0, 6, 0.2)],
    [(x, 0) for x in np.arange(0, 5, 0.2)],[(x,3) for x in np.arange(0, 5, 0.2)],[(x,6) for x in np.arange(0, 5, 0.2)],
    generatePoints(0, 0), generatePoints(0,3), generatePoints(0,6),
])
outlier = np.array([(0,0),(0,5)])

test = np.concatenate([test1, outlier])

np.save("data", test)
#test = np.load("data.npy")
labels = dbscan.MyDBSCAN(test, eps=.4, minPts=3, debug=False)
plot.plot(test, labels, False, "Final output", legend=True)
print("done")


