from truss_materials import Truss
from matplotlib import pyplot as plt
import time

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
truss.loadData('example_trusses/truss_test.yaml')
# truss.loadData('example_trusses/warren_big.yaml')
# truss.loadData('example_trusses/warren_big_middle_support.yaml')
# truss.loadData('truss-materials.yaml')
# truss.loadData('example_trusses/double_intersection_warren_big.yaml')

# truss.loadState('example_trusses/warren_big_middle_support.yaml')

# print(truss.calculateWeight())

# truss.viewTruss(NodeNumbers=True, MemberNumbers=True)
# plt.axis('equal')
# plt.show()
# exit()

displacements, forces = truss.solveTruss()

# Print out solve time in milliseconds
print(f"Solve time: {truss.solveTime * 1000}ms")

truss.viewTrussExtras(displacements, forces, NodeNumbers=True, MemberNumbers=True)

with open('tmp.txt', 'w') as f:
    truss.Displacements = displacements
    truss.Forces = forces
    report = truss.generateReport()
    f.write(report)

plt.axis('equal')
plt.show()