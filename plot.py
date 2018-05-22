import time
import winsound
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import winsound
import pool
from functools import partial

currentid = str(time.time())[-5:]
plotcounter = 0
plt.axis("equal")
plt.subplots_adjust(left=0.035, right=0.99, top=0.95, bottom=0.03)

fig = plt.figure(figsize=(60, 36))


def clusterColor(i, norm, autumn=False):
    if i == -2: return (1, 0, 0, 1)  # Waypoints
    if i == -1: return (0, 0, 0, .1)  # Outlier
    if i == 0: return (0, 0, 0, 1)  
    if autumn: return cm.autumn(norm(i))
    return cm.rainbow(norm(i))


def plot(points, labels, writeIndex=False, title="", legend=False, eps=0, text=[], showAxis=True, autumn=False):
    global currentid, fig, plotcounter
    ax = fig.add_subplot(111)
    dataid = currentid+"."+str(plotcounter)
    plotcounter += 1
    np.save("plots/"+dataid+"_points", points)
    np.save("plots/"+dataid+"_labels", labels)

    if len(points) != len(labels):
        print("PLOT failed: len(points) != len(labels)")
        return
    if len(points) == 0:
        print("PLOT failed: len(points) == 0")
    points = np.array(points)
    labels = np.array(labels)
    norm = Normalize(vmin=0, vmax=round(max(labels)*1.1))

    getColorFnc = partial(clusterColor, norm=norm, autumn=autumn)
    colors = pool.getThreadPool().map(getColorFnc, labels)
    xs, ys = zip(*points)
    ax.scatter(xs, ys, color=colors, s=[2]*len(xs))
    if eps > 0:
        ax = plt.gca()
        for p in points:
            ax.add_artist(plt.Circle((p[0], p[1]), eps, color='red', fill=False))
    if writeIndex:
        for i, p in enumerate(points):
            plt.annotate(i, (p[0], p[1]))
    if len(text) > 0:
        ax = plt.gca()
        for i, txt in enumerate(text):
            ax.annotate(txt, (points[i][0], points[i][1]))
    if legend:
        red_patch = mpatches.Patch(color='red', label='Chainpoints')
        black_patch = mpatches.Patch(color='black', label='Noise')
        plt.legend(handles=[red_patch, black_patch])
    if not showAxis:
        plt.axis('off')

    plt.title(title)
    #plt.show()
    plt.savefig('plots/'+dataid+"_"+title.replace(":","")+'.png')
    fig.clear()
    winsound.Beep(1000, 500)


    return


def plotPoints(points):
    plt.scatter([x for x, y in points], [y for x, y in points])
    plt.show()
