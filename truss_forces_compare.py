from truss_materials import Truss
from matplotlib import pyplot as plt
import time
import numpy as np

# Create a truss object
truss = Truss()

# trusses = ['howe', 'pratt', 'warren']
trusses = ['pratt_flat_w_pylon', 'pratt_rise', 'warren_rise', 'warren_flat']

forces_abs = [[] for _ in trusses]
forces_l = [[] for _ in trusses]

for i, truss_name in enumerate(trusses):
    # truss.loadData(f'base_trusses/{truss_name}.yaml')
    truss.loadData(f'testing_trusses/{truss_name}.yaml')
    displacements, forces = truss.solveTruss()

    # add the forces to the list of forces
    forces_l[i] = [f[0] for f in forces]
    forces_abs[i] = [abs(f[0]) for f in forces]

# boxplot the stresses and forces
fig, ax = plt.subplots(1, 2, sharey=False)
# fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].boxplot(forces_abs, labels=trusses)
ax[1].boxplot(forces_l, labels=trusses)

# set the ylabels
ax[0].set_ylabel('Force (N)')
ax[1].set_ylabel('Force (N)')

# set the title
ax[0].set_title('Forces (Absolute)')
ax[1].set_title('Forces')

plt.show()
