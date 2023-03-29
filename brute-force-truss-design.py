from truss_materials import Truss
from matplotlib import pyplot as plt
import random
import time
import copy

# Create a truss object
truss = Truss()

# FILENAME = 'truss-materials.yaml'
FILENAME = 'example_trusses/warren_big.yaml'
# Load the truss
truss.loadData(FILENAME)

BATCH_SIZE = 100
RANDOM_MAX = 0.01

NODE_INDEX = 3
NODE_INITIAL_WEIGHT = -50_000

def randomiseNodes(truss):
    for node in truss.Nodes:
        if node[1] != 0:
            node[0] += random.uniform(-RANDOM_MAX, RANDOM_MAX)
            node[1] += random.uniform(-RANDOM_MAX, RANDOM_MAX)

def EPOCH(best_truss=None):
    # Create 100 trusses based of the original truss with their nodes moved randomly
    trusses = []

    # Add the best truss to the batch
    if best_truss is not None:
        trusses.append(best_truss)

    for i in range(BATCH_SIZE - 1):
        if best_truss is not None:
            # truss_copy = best_truss
            truss_copy = copy.deepcopy(best_truss)
        else:
            truss_copy = Truss()
            truss_copy.loadData(FILENAME)
        
        randomiseNodes(truss_copy)

        # Calculate the weight of the truss
        weight = truss_copy.calculateWeight()

        # Add the weight to the node
        truss_copy.Nodes[NODE_INDEX][1] == NODE_INITIAL_WEIGHT - weight

        trusses.append(truss_copy)

    # Solve the trusses
    displacements = []
    forces = []
    for truss in trusses:
        d, f = truss.solveTruss()
        displacements.append(d)
        forces.append(f)

    # Find the truss with the lowest maximum member stress
    min_stress = 10_000_000_000
    min_stress_truss = None
    for i in range(len(trusses)):
        truss = trusses[i]
        max_stress = abs(max(truss.Stresses, key=abs)[0])
        if max_stress < min_stress:
            min_stress = max_stress
            min_stress_truss = truss

    return min_stress_truss, displacements[trusses.index(min_stress_truss)], forces[trusses.index(min_stress_truss)]

best_truss = None
max_stresses = []
weights = []
try:
    i = 0
    while True:
        start = time.time()
        # best_truss = EPOCH(best_truss)
        best_epoch, displacements, forces = EPOCH(best_truss)

        if best_epoch is not None:
            best_truss = best_epoch
            max_stress = abs(max(best_truss.Stresses, key=abs)[0])

            # Clear the plot
            plt.clf()

            # Save an image of the truss
            best_truss.viewTrussExtras(displacements, forces)

            # Put the epoch number in the top left corner
            plt.text(0.05, 0.95, 'Epoch: {}'.format(i), horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

            # Put the maximum stress in the top right corner
            plt.text(0.95, 0.95, 'Max Stress: {:,}'.format(round(max_stress, 4)), horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

            # Before saving the image, change the figure size to 10x10
            plt.gcf().set_size_inches(10, 5)

            plt.savefig('images/truss-{}.png'.format(i))

            max_stresses.append(max_stress)
            weights.append(best_truss.calculateWeight(False))
        else:
            break
        end = time.time()

        print('Epoch: {}, Epoch Time: {:.2f}s, Max Stress: {:.4f}, Weight: {:.2f}Kg'.format(i, round(end - start, 2), round(max_stress, 4), round(best_truss.calculateWeight(False), 2)))

        i += 1
except KeyboardInterrupt:
    pass

plt.clf()

# Plot max_stresses and weights on the same plot with 2 different scales
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Max Stress', color=color)
ax1.plot(max_stresses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Weight', color=color)  # we already handled the x-label with ax1
ax2.plot(weights, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()