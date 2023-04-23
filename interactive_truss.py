from truss_materials import Truss
from matplotlib.backend_bases import MouseButton
from matplotlib import pyplot as plt
import time
import numpy as np

# Create a truss object
truss = Truss()

# Load the base truss
truss.loadData('base_trusses/warren.yaml')

FAC = 1500

last_displacements = None

CLOSEST_NODE_LOCK = 1

def on_move(event):
    global last_displacements, CLOSEST_NODE_LOCK

    start = time.time()

    if event.inaxes:
        coords = (event.xdata, event.ydata)

        # if there is a last displacements, use that to get the nodes
        if last_displacements is None:
            nodes = truss.Nodes
        else:
            nodes = []
            for displacement, node in zip(np.array(last_displacements).reshape(-1, 2), truss.Nodes):
                nodes.append([node[0] + displacement[0], node[1] + displacement[1]])

            nodes = np.array(nodes)

        closest_node = None
        closest_node_distance = None

        LOCK_CLOSEST_NODE = True
        if LOCK_CLOSEST_NODE:
            node = nodes[CLOSEST_NODE_LOCK]
            x, y = node
            x, y = float(x), float(y)

            distance = ((x - coords[0])**2 + (y - coords[1])**2)**0.5

            closest_node = node
            closest_node_distance = distance
        else:
            for node in nodes:
                x, y = node
                x, y = float(x), float(y)

                distance = ((x - coords[0])**2 + (y - coords[1])**2)**0.5

                if closest_node_distance is None or distance < closest_node_distance:
                    closest_node = node
                    closest_node_distance = distance

        # Get the index of the closest node using np
        closest_node = np.where(np.all(nodes == closest_node, axis=1))[0][0]

        # clear all external forces
        for i, force in enumerate(truss.ExternalForces):
            truss.ExternalForces[i] = [0, 0]

        # apply the force based on the distance and direction
        x, y = coords
        x, y = float(x), float(y)
        x, y = x - nodes[closest_node][0], y - nodes[closest_node][1]
        truss.ExternalForces[closest_node] = [x * FAC, y * FAC]

        # solve the truss
        displacements, forces = truss.solveTruss()

        last_displacements = displacements

        # clear the plot
        plt.cla()

        # view the truss deformed
        truss.viewTrussExtras(displacements, forces, NodeNumbers=False, MemberNumbers=True, drawOriginal=False)

        # Draw an arrow from the closest node to the mouse
        plt.arrow(nodes[closest_node][0], nodes[closest_node][1], x, y, width=0.015, color='red')

        # update the plot
        plt.draw()

        end = time.time()

        print(f'FPS: {1 / (end - start)}')

def on_press(event):
    # Key press event
    global CLOSEST_NODE_LOCK

    if event.key == 'right':
        CLOSEST_NODE_LOCK += 1
    elif event.key == 'left':
        CLOSEST_NODE_LOCK -= 1

    print(f'CLOSEST_NODE_LOCK: {CLOSEST_NODE_LOCK}')

binding_id = plt.connect('motion_notify_event', on_move)
plt.connect('key_press_event', on_press)

plt.axis('equal')
plt.show()