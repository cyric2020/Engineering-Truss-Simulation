from truss_materials import Truss
from matplotlib import pyplot as plt
import time
import numpy as np

# Create a truss object
truss = Truss()

trusses = ['howe', 'pratt', 'warren']

stresses = [[] for _ in trusses]
forces_l = [[] for _ in trusses]

for i, truss_name in enumerate(trusses):
    truss.loadData(f'base_trusses/{truss_name}.yaml')
    displacements, forces = truss.solveTruss()

    # convert the list of arrays to a list of scalars
    truss_stresses = [abs(s[0]) for s in truss.Stresses]
    # truss_stresses = [s[0] for s in truss.Stresses]

    # add the stresses to the list of stresses
    stresses[i] = truss_stresses

    # add the forces to the list of forces
    forces_l[i] = [abs(f[0]) for f in forces]
    # forces_l[i] = [f[0] for f in forces]

# boxplot the stresses and forces
fig, ax = plt.subplots(1, 2)
ax[0].boxplot(stresses, labels=trusses)
ax[1].boxplot(forces_l, labels=trusses)

# set the ylabels
ax[0].set_ylabel('Stress (N/m^2)')
ax[1].set_ylabel('Force (N)')

# set the title
ax[0].set_title('Stresses')
ax[1].set_title('Forces')


# Create a second figure for plotting the distribution of forces
fig2, ax2 = plt.subplots(1, 3, sharey=True, sharex=True)

# Plot the distribution of forces
for i, truss_name in enumerate(trusses):
    ax2[i].hist([abs(number) for number in forces_l[i]])
    # ax2[i].hist(forces_l[i])
    ax2[i].set_title(truss_name)

# # Get the maximum x range for all the axes
# max_x = max([ax2[i].get_xlim()[1] for i in range(len(ax2))])
# min_x = min([ax2[i].get_xlim()[0] for i in range(len(ax2))])

# # Set the x range for all the axes
# for i in range(len(ax2)):
#     ax2[i].set_xlim(min_x, max_x)

# Display the y axis as percentages
for i in range(len(ax2)):
    yticks = ax2[i].get_yticks()
    ax2[i].set_yticklabels([f'{y/len(forces_l[i])*100:.2f}%' for y in yticks])

# Set the ylabels
ax2[0].set_ylabel('Percentage of Forces')

# Set the xlabels
ax2[0].set_xlabel('Force (N)')
ax2[1].set_xlabel('Force (N)')
ax2[2].set_xlabel('Force (N)')

plt.show()
