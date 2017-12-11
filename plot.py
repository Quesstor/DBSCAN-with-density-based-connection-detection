import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.pyplot
import numpy as np


def clusterColor(i, labelCount):
    if i == -3:
        i = labelCount + 3  # Test
    if i == -2:
        return (1, 0, 0, 1)  # Waypoints
    if i == -1:
        return (0, 0, 0, 1)  # Outlier
    norm = Normalize(vmin=0, vmax=labelCount + 3)
    return cm.rainbow(norm(i))


def plot(points, labels, writeIndex=False, title="", legend=False, eps=0, text=[]):
    if len(points) != len(labels):
        print("PLOT failed: len(points) != len(labels)")
        return
    if len(points) == 0:
        print("PLOT failed: len(points) == 0")
    try:
        points = np.array(points)
        labels = np.array(labels)
        labelsCount = max(labels)

        colors = [clusterColor(i, labelsCount) for i in labels]
        matplotlib.pyplot.figure(figsize=(10, 10))
        matplotlib.pyplot.subplots_adjust(
            left=0.035, right=0.99, top=0.95, bottom=0.03)
        xs = np.array([x for x, y in points])
        ys = np.array([y for x, y in points])
        matplotlib.pyplot.scatter(xs, ys, color=colors)
        if eps > 0:
            ax = plt.gca()
            for p in points:
                ax.add_artist(plt.Circle(
                    (p[0], p[1]), eps, color='red', fill=False))
        if writeIndex:
            for i, p in enumerate(points):
                matplotlib.pyplot.annotate(i, (p[0], p[1]))
        if len(text) > 0:
            ax = plt.gca()
            for i, txt in enumerate(text):
                ax.annotate(txt, (points[i][0], points[i][1]))
        if legend:
            red_patch = mpatches.Patch(color='red', label='Chainpoints')
            black_patch = mpatches.Patch(color='black', label='Noise')
            plt.legend(handles=[red_patch, black_patch])

        matplotlib.pyplot.title(title)
        matplotlib.pyplot.show()
    except Exception as e:
        print("PLOT FAILED")
        print(e)
        print(xs)
        print()
        print(ys)
        print()
        print(colors)
        print()
    return


def plotPoints(points):
    matplotlib.pyplot.scatter([x for x, y in points], [y for x, y in points])
    matplotlib.pyplot.show()
