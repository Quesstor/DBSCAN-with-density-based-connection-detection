import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.pyplot

def clusterColor(i, labelCount):
    if i==-3: i = labelCount+3 #Test
    if i==-2: return "red" #Waypoints
    if i==-1: return "black" #Outlier
    norm = Normalize(vmin=1, vmax=labelCount+3)
    return cm.rainbow(norm(i))

def plot(points, labels):
    labelsCount = max(labels)

    colors = [clusterColor(i,labelsCount) for i in labels]
    matplotlib.pyplot.scatter([x for x,y in points],[y for x,y in points], color=colors)

    red_patch = mpatches.Patch(color='red', label='Connecting points')
    black_patch = mpatches.Patch(color='black', label='Noise')
    plt.legend(handles=[red_patch, black_patch])

    matplotlib.pyplot.show()
    return

def plotPoints(points):
    matplotlib.pyplot.scatter([x for x, y in points], [y for x, y in points])
    matplotlib.pyplot.show()