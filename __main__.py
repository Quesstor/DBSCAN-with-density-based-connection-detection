import numpy as np
import random
import plot
import DBSCAN
import detection.lowdensitychains
import detection.trimming

def generatePoints(x, y, offset=0.5, count=40):
    return np.array([(random.uniform(x - offset, x + offset), random.uniform(y - offset, y + offset)) for i in range(count)])


def test0():
    return np.concatenate([generatePoints(1, 1), generatePoints(3, 1), [(x, 1) for x in np.arange(1, 3, 0.1)]])

# chains with tentacles
def test1():
    return np.concatenate([generatePoints(0, 0), generatePoints(4, 0), generatePoints(-4, 0), [(x, 0) for x in np.arange(-4, 4, 0.2)],
                        [(-2, x) for x in np.arange(0, 2, 0.2)], [(2, x) for x in np.arange(0, 2, 0.2)], [(4, x) for x in np.arange(0, 2, 0.2)]])

# Non linear chains
def test2():
    return np.concatenate([generatePoints(1.5, 3), [(x, pow(x, 3))
                                                 for x in np.arange(-1.5, 1.5, 0.1)], generatePoints(-1.5, -3)])

# Cyclic cluster chains
def test3():
    return np.concatenate([
        generatePoints(5, 0), generatePoints(5, 3), generatePoints(5, 6),
        [(0, x) for x in np.arange(0, 6, 0.2)], [(5, x)
                                                for x in np.arange(0, 6, 0.2)],
        [(x, 0) for x in np.arange(0, 5, 0.2)], [(x, 3)
                                                for x in np.arange(0, 5, 0.2)], [(x, 6) for x in np.arange(0, 5, 0.2)],
        generatePoints(0, 0), generatePoints(0, 3), generatePoints(0, 6),
])

def problem1():
    return np.concatenate([generatePoints(-2, -.5, offset=1.5, count=400), generatePoints(5, 4, offset=1.5, count=400),
                           [(x, x + .6) for x in np.arange(-1, 4, 0.15)],
                           [(x, x + .4) for x in np.arange(-1, 4, 0.1)],
                           [(x, x + .2) for x in np.arange(-1, 4, 0.15)],
                           [(x, x) for x in np.arange(-1, 4, 0.1)],
                           [(x, x - .2) for x in np.arange(-1, 4, 0.15)], ])

def problem2():
    return np.concatenate([generatePoints(1, 1, count=100), generatePoints(3, 3, count=100), [(x, x) for x in np.arange(1.2, 2.8, 0.03)]])

# All tests
def test4():
    return np.concatenate([generatePoints(5, 0), generatePoints(0, 0), generatePoints(5, 5), generatePoints(8, 8), generatePoints(5, 5, 5, 100),
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(-0.3, 1, 0.05)],
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(1, 1.8, 0.07)],
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(1.7, 3.5, 0.18)],
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(3.5, 5.3, 0.09)],
                        [(x, x) for x in np.arange(5, 8, 0.2)],
                        [(x, 0) for x in np.arange(0, 5, 0.2)],
                        [(5, x) for x in np.arange(0, 5, 0.25)],
                        [(8, x) for x in np.arange(4, 8, 0.25)]
                        ])

test = test1()

#np.save("data", test)
#test = np.load("example.npy")

labels = DBSCAN.Algorithm(test, .6, 3, detection.lowdensitychains.detectChainpoints, True)
plot.plot(test, labels, False, "Final output", legend=True)
print("done")
