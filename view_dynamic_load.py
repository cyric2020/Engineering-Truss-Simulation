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

trusses = ['howe', 'pratt', 'warren']
truss_colors = ['green', 'red', 'blue']

# Setup the figure
fig = plt.figure()
ax = fig.add_subplot(111)

legend_elements = []

areas = {}

for t, truss_name in enumerate(trusses):
    # Load the truss
    truss.loadData(f'base_trusses/{truss_name}.yaml')

    # Nodes for load
    nfl = range(1, 6)

    # Set the load that will be applied
    load = [0, -2_000]

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
    color = truss_colors[t]
    for cluster in clusters:
        cluster = np.array(cluster)
        # ax.add_patch(Polygon(cluster, closed=True, fill=True, facecolor=color, edgecolor=color, alpha=0.55))
        # ax.add_patch(Polygon(cluster, closed=True, fill=True, facecolor='None', edgecolor=color, alpha=0.55, hatch='.'))
        
        try:
            hull = ConvexHull(cluster)
            # ax.add_patch(Polygon(cluster[hull.vertices], closed=True, fill=True, facecolor=color, edgecolor=color, alpha=0.55))

            # use ShapelyPolygon to make the cluster round
            shapely_cluster = ShapelyPolygon(cluster[hull.vertices])
            shapely_cluster.buffer(1)

            area = shapely_cluster.area
            if truss_name not in areas.keys():
                areas[truss_name] = area
            else:
                areas[truss_name] += area

            ax.add_patch(Polygon(shapely_cluster.exterior.coords, closed=True, fill=True, facecolor=color, edgecolor=color, alpha=0.55))
        except:
            pass


        # # Calculate the area of the polygon
        # area = 0
        # for i in range(len(cluster)):
        #     j = (i + 1) % len(cluster)
        #     area += cluster[i][0] * cluster[j][1]
        #     area -= cluster[j][0] * cluster[i][1]

        # area = abs(area) / 2

        # # add the area to each other cluster with that same truss type
        # # areas[t] = areas[t] + area
        # if truss_name not in areas.keys():
        #     areas[truss_name] = area
        # else:
        #     areas[truss_name] += area

    legend_elements.append(Line2D([0], [0], color=color, lw=4, label=truss_name))

print(areas)

# Set the legend
ax.legend(handles=legend_elements)

# Set the axis equal
plt.axis('equal')

# Show the plot
plt.show()