# area can be solved smartly using the following formula:
# A = F / sigma-1 (sigma = stress)

from truss_materials import Truss
from matplotlib import pyplot as plt
import time
import numpy as np

# Create a truss object
truss = Truss()

# Define the filename
FILENAME = 'example_trusses/warren_big.yaml'

# Define FOS
FOS = 1.5

# Load the truss
truss.loadData(FILENAME)

def trussFails(truss):
    for member, stress in zip(truss.Members, truss.Stresses):
        # get the yield stress of the material
        max_stress = truss.Materials[member[2]]['MaxStress']
        # check if the stress is greater than the yield stress
        if abs(stress[0]) > max_stress:
            return True
    return False

truss.solveTruss()

for i, member in enumerate(truss.Members):
    # get the material
    material = truss.Materials[member[2]]
    # get the yield stress of the material
    max_stress = truss.Materials[member[2]]['MaxStress']
    # Get the stress from the truss
    stress = abs(truss.Stresses[i][0])
    # Get the force
    force = abs(truss.Forces[i][0])

    # Check if the member is a zero force member
    if force == 0:
        # Set the area to zero
        # member[3] = 0.0000000001
        member[3] = 0
        continue
    
    # Calculate the area
    # area = force / (max_stress - 1)
    # Incorperate the FOS
    area = force / (max_stress / FOS - 1)
    # print(force, stress, area)
    # Set the area
    member[3] = area

# Solve the truss
displacements, forces = truss.solveTruss()

# Save the report to the reports folder
with open('reports/smart-area-report.txt', 'w') as f:
    truss.Displacements = displacements
    truss.Forces = forces
    report = truss.generateReport()
    f.write(report)

# Check if the truss fails
if trussFails(truss):
    print('Truss failed')

# Plot the truss
truss.viewTrussExtras(displacements, forces, NodeNumbers=False, MemberNumbers=True)

plt.axis('equal')
# plt.show()