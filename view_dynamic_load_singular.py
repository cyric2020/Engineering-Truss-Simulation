from truss_materials import Truss
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
import time
import numpy as np

# Create a truss object
truss = Truss()

# Load the truss
# FILENAME = 'testing_trusses/warren_flat.yaml'
# FILENAME = 'testing_trusses/pratt_flat_w_pylon.yaml'
FILENAME = 'testing_trusses/pratt_rise.yaml'
truss.loadData(FILENAME)

# Setup the figure
fig = plt.figure()
ax = fig.add_subplot(111)

# Nodes for load
# nfl = range(1, 8)
nfl = range(1, 12)
print(list(nfl))

# Set the load that will be applied
load = [0, -300_000]

# Setup an empty list for the displacements
displacements = []

clusters = [[] for _ in range(len(truss.Nodes))]

# Set all external loads to 0
for i in range(len(truss.ExternalForces)):
    truss.ExternalForces[i] = [0, 0]

# Loop through each nfl node and apply the load
for i in nfl:
    truss.ExternalForces[i] = load

    # Solve the truss
    disp, forces = truss.solveTruss()

    # Convert the displacements to global coordinates
    for j, node in enumerate(truss.Nodes):
        disp[j*2] += node[0]
        disp[j*2+1] += node[1]

        # clusters[i].append([disp[j*2], disp[j*2+1]])
        clusters[j].append([disp[j*2][0], disp[j*2+1][0]])

    # Append the displacements to the list
    displacements.append(disp)

    # Remove the load
    truss.ExternalForces[i] = [0, 0]

# Plot the clusters as light blue polygons
for cluster in clusters:
    cluster = np.array(cluster)
    # print(cluster)

    try:
        hull = ConvexHull(cluster)

        # use ShapelyPolygon to make the cluster round
        shapely_cluster = ShapelyPolygon(cluster[hull.vertices])
        shapely_cluster.buffer(1)
        
        area = shapely_cluster.area

        # Plot the cluster
        color = "blue"
        ax.add_patch(Polygon(shapely_cluster.exterior.coords, closed=True, fill=True, facecolor=color, edgecolor=color, alpha=0.55))
    except Exception as e:
        # print(e)
        pass

# Set the axis equal
plt.axis('equal')

# Show the plot
plt.show()