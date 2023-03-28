from truss_materials import Truss
from matplotlib import pyplot as plt

# Create a truss object
truss = Truss()

# %%%%%%%%%%%%%%%%
# Saving the truss
# %%%%%%%%%%%%%%%%
# truss.loadData('truss-materials.yaml')

# displacements, forces = truss.solveTruss()

# truss.Displacements = displacements
# truss.Forces = forces
# print(truss.R)

# truss.saveState("smallest_area.yaml")

# exit()

# %%%%%%%%%%%%%%%%%
# Loading the truss
# %%%%%%%%%%%%%%%%%
truss.loadState("smallest_area.yaml")

# truss.viewTruss(NodeNumbers=True, MemberNumbers=True)

displacements, forces, R = truss.Displacements, truss.Forces, truss.R

truss.viewTrussExtras(displacements, forces, NodeNumbers=True)
plt.axis('equal')
plt.show()