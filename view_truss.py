from truss_materials import Truss
from matplotlib import pyplot as plt
import time

def plotForces(truss):
    plt.boxplot([abs(f[0]) for f in truss.Forces])
    # plt.hist([abs(f[0]) for f in truss.Forces])
    plt.ylabel('Force (N)')
    plt.title('Forces')
    plt.show()

# Create a truss object
truss = Truss()

# @@@@@@@@@@@@@@@@
# Saving the truss
# @@@@@@@@@@@@@@@@
# truss.loadData('truss-materials.yaml')

# displacements, forces = truss.solveTruss()

# truss.Displacements = displacements
# truss.Forces = forces
# print(truss.R)

# truss.saveState("smallest_area.yaml")

# exit()

# @@@@@@@@@@@@@@@@@
# Loading the truss
# @@@@@@@@@@@@@@@@@
# truss.loadState("example_trusses/warren_big.yaml")

# truss.viewTruss(NodeNumbers=True, MemberNumbers=True)

# displacements, forces, R = truss.Displacements, truss.Forces, truss.R

# truss.viewTrussExtras(displacements, forces, NodeNumbers=True)
# plt.axis('equal')
# plt.show()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Just solve the truss and view displacements
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# truss.loadData('example_trusses/truss_test.yaml')
# truss.loadData('example_trusses/warren_big.yaml')
# truss.loadData('example_trusses/warren_big_middle_support.yaml')
# truss.loadData('truss-materials.yaml')
# truss.loadData('example_trusses/double_intersection_warren_big.yaml')

# Trusses for research
# truss.loadData('base_trusses/pratt.yaml')
# truss.loadData('base_trusses/howe.yaml')
# truss.loadData('base_trusses/warren.yaml')

# Trusses for testing
# truss.loadData('testing_trusses/warren_flat.yaml')
# truss.loadData('testing_trusses/pratt_flat_w_pylon.yaml')
# truss.loadData('testing_trusses/pratt_rise.yaml')
truss.loadData('testing_trusses/warren_rise.yaml')

# truss.loadState('example_trusses/warren_big_middle_support.yaml')

# print(truss.calculateWeight())

# truss.viewTruss(NodeNumbers=True, MemberNumbers=False)
# plt.axis('equal')
# plt.show()
# exit()

displacements, forces = truss.solveTruss()

# plotForces(truss)

# Print out solve time in milliseconds
print(f"Solve time: {truss.solveTime * 1000}ms")

truss.viewTrussExtras(displacements, forces, NodeNumbers=False, MemberNumbers=True, drawOriginal=True)

with open('tmp.txt', 'w') as f:
    truss.Displacements = displacements
    truss.Forces = forces
    report = truss.generateReport()
    f.write(report)

plt.axis('equal')
plt.show()