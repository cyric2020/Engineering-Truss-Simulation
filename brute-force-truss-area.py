from truss_materials import Truss
from matplotlib import pyplot as plt
import random
import time
import copy
import numpy as np

# Create a truss object
truss = Truss()

# Define the filename
# FILENAME = 'example_trusses/warren_big.yaml'
FILENAME = 'example_trusses/warren_big_middle_support.yaml'

# Load the truss
truss.loadData(FILENAME)

BATCH_SIZE = 50
RANDOM_MAX = 0.001
INITIAL_AREA = 0.025

NODE_INDEX = 3
NODE_INITIAL_WEIGHT = -50_000

def randomAreas(truss):
    for member in truss.Members:
        member[3] = float(member[3]) + random.uniform(-RANDOM_MAX, RANDOM_MAX)

        if float(member[3]) < 0:
            member[3] = 0.00001

def trussFails(truss):
    for member, stress in zip(truss.Members, truss.Stresses):
        # get the material
        material = truss.Materials[member[2]]
        # get the area
        area = float(member[3])
        # get the yield stress of the material
        max_stress = truss.Materials[member[2]]['MaxStress']
        # check if the stress is greater than the yield stress
        if abs(stress[0]) > max_stress:
            return True
    return False

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

            # Set all member areas to 1
            for member in truss_copy.Members:
                member[3] = INITIAL_AREA
        randomAreas(truss_copy)

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

    # Find the truss with the lowest average area
    min_area = 10_000_000_000
    min_area_truss = None
    for i in range(len(trusses)):
        truss = trusses[i]
        area = sum([float(member[3]) for member in truss.Members]) / len(truss.Members)
        max_area = max([float(member[3]) for member in truss.Members])
        if area < min_area:
            if trussFails(truss) or area < 0:
                continue
            else:
                min_area = area
                min_area_truss = truss
        elif area == min_area and max_area < max([float(member[3]) for member in min_area_truss.Members]):
            if trussFails(truss) or area < 0:
                continue
            else:
                min_area = area
                min_area_truss = truss

    return min_area_truss, displacements[trusses.index(min_area_truss)], forces[trusses.index(min_area_truss)]

best_truss = None
average_areas = []
max_areas = []
min_areas = []

max_stresses = []
try:
    i = 0
    while True:
        start = time.time()
        best_epoch, displacements, forces = EPOCH(best_truss)

        if best_epoch is not None:
            best_truss = best_epoch
            average_area = sum([float(member[3]) for member in best_truss.Members]) / len(best_truss.Members)

            # Clear the plot
            plt.clf()

            average_areas.append(average_area)
            max_stresses.append(max([abs(stress[0]) for stress in best_truss.Stresses]))

            # Append the maximum and minimum areas
            max_areas.append(max([float(member[3]) for member in best_truss.Members]))
            min_areas.append(min([float(member[3]) for member in best_truss.Members]))
        else:
            break
        end = time.time()

        print('Epoch: {}, Epoch Time: {:.2f}s, Average Area: {:.6f}, Max Stress: {:.4f}, Weight: {:.2f}Kg'.format(i, round(end - start, 2), round(average_area, 6), round(max([abs(stress[0]) for stress in best_truss.Stresses]), 4), round(best_truss.calculateWeight(False), 2)))

        i += 1
except KeyboardInterrupt:
    # save a report for the best truss
    with open("smallest_area.txt", "w") as f:
        displacements, forces = best_truss.solveTruss()
        best_truss.Displacements = displacements
        best_truss.Forces = forces
        report = best_truss.generateReport()
        f.write(report)

    # Save the state of the truss
    best_truss.saveState('smallest_area.yaml')
    pass

plt.clf()
plt.plot(average_areas)

# Plot a semi transparent section between the max and min areas
plt.fill_between(range(len(max_areas)), max_areas, min_areas, alpha=0.5)

plt.show()