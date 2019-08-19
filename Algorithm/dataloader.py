import numpy as np
import scipy
import math


def loadcsv():
    import csv
    data = list()
    with open('accidents_2012_to_2014.csv', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        headers = next(reader)
        for row in reader:
            lng = float(row[3])
            lat = float(row[4])

            # All
            # data.append([lng,lat])

            # London
            #if lng>-1.2 and lng<.75 and lat>51 and lat<52: data.append([lng,lat])

            #Liverpool and Manchester
            if lng > -3.1 and lng < -1.9 and lat > 53.2 and lat < 53.7:
                data.append([lng, lat])
    return data


def random2cluster(clusterSD, chainPointDistance, chainSD):
    centerX = np.array([0, 1, 2, 3])
    centerY = np.random.uniform(0, 4, 4)

    xs = np.random.normal(centerX[0], clusterSD, 500)
    trueLabels = [1] * 500
    xs = np.concatenate((xs, np.random.normal(centerX[3], clusterSD, 500)))
    trueLabels = np.concatenate((trueLabels, [2] * 500))

    ys = np.random.normal(centerY[0], clusterSD, 500)
    ys = np.concatenate((ys, np.random.normal(centerY[3], clusterSD, 500)))

    spline = scipy.interpolate.UnivariateSpline(centerX, centerY)
    x = min(centerX)
    xsInter = [x]

    while x < max(centerX):
        best = x
        bestDist = 10
        for i in np.linspace(x, x+chainPointDistance, num=100):
            d = abs(math.sqrt((x-i)**2 + (spline(x) - spline(i))
                              ** 2) - chainPointDistance)
            if d < bestDist:
                best = i
                bestDist = d
        x = best
        xsInter.append(x)

    ysInter = spline(xsInter)
    trueLabels = np.concatenate((trueLabels, [3] * len(xsInter)))

    xsInter += np.random.normal(0, chainSD, len(xsInter))
    ysInter += np.random.normal(0, chainSD, len(ysInter))

    xs = np.concatenate((xs, xsInter))
    ys = np.concatenate((ys, ysInter))
    points = np.array(list(zip(xs, ys)))
    return points, trueLabels
