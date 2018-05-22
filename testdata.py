import numpy as np
import random


def generatePoints(x, y, offset=0.3, count=100, normal=True):
    if normal: return [x for x in zip(np.random.normal(x,offset,count), np.random.normal(y,offset,count))]
    else: return np.array([(random.uniform(x - offset, x + offset), random.uniform(y - offset, y + offset)) for i in range(count)])


def test0():
    return np.concatenate([generatePoints(0, 0, count=300, offset=1), generatePoints(20, 0, count=300, offset=1), [(x, 0) for x in np.arange(1.5, 19.5, 0.01)]])

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
    return np.concatenate([generatePoints(1, 1, count=50), generatePoints(3, 3, count=50), [(x, x) for x in np.arange(1.5, 2.5, 0.03)]])

# All tests
def test4():
    return np.concatenate([generatePoints(-3, 6), generatePoints(5, 0), generatePoints(0, 0), generatePoints(5, 5, .45, 180), generatePoints(8, 8),
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(-0.3, 1, 0.05)],
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(1, 1.8, 0.07)],
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(1.7, 3.5, 0.18)],
                        [(x, -pow(x - 2.9, 2) + 10)
                         for x in np.arange(3.5, 5.3, 0.08)],
                        [(x, x) for x in np.arange(5, 8, 0.2)],
                        [(x, 0) for x in np.arange(0, 5, 0.2)],
                        [(5, x) for x in np.arange(0, 5, 0.25)],
                        [(8, x) for x in np.arange(4, 8, 0.25)],
                        generatePoints(.8, 5.6, .3, 3),
                        ])
