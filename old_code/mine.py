import numpy as np
import matplotlib.pyplot as plt
import time

# Start timer
start = time.time()

# Define global variables
E = 70*10**9 # Placeholder (Pa)
A = 0.0002 # Placeholder (m^2)

# Define nodes
nodes = np.array([
    [0, 0.3],  # Node 0
    [0.5, 0.3], # Node 1
    [0.9, 0]   # Node 2
])

# Define members
members = np.array([
    [0, 1], # Member 1 (Node 0 to Node 1)
    [1, 2]  # Member 2 (Node 1 to Node 2)
])

# Create the global stiffness matrix
n_nodes = nodes.shape[0]
n_dofs = 2 * n_nodes
K = np.zeros((n_dofs, n_dofs))

# Define stiffness matrices for each member
stiffnessMatrices = []
for member in members:
    # Get node IDs
    i, j = member
    
    # Get node coordinates
    node_i = nodes[i]
    node_j = nodes[j]
    
    # Calculate length of member
    L = np.linalg.norm(node_j - node_i)

    # Calculate cosines and sines
    c = (node_j[0] - node_i[0]) / L
    s = (node_j[1] - node_i[1]) / L
    
    # Calculate stiffness matrix
    k_m = np.array([
        [c**2, c*s, -c**2, -c*s],
        [c*s, s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2, c*s],
        [-c*s, -s**2, c*s, s**2]
    ])

    # Calculate stiffness matrix with global variables
    k = (E * A / L) * k_m
    # print(k)

    # Append stiffness matrix to array
    stiffnessMatrices.append(k)

    # Add the local stiffness matrix to the global stiffness matrix at the correct positions
    dofs = [2*i, 2*i+1, 2*j, 2*j+1]
    # print(dofs)
    # print(np.tile(k, (2, 2)))
    K[np.ix_(dofs, dofs)] += k

# print(K * L / (E * A)) # correct so far

# Finite Element Equation
# [K]{u} = {F}

# Define external forces
external_forces = np.array([
    [0, 0], # Node 0
    # [0, -12000], # Node 1 (12,000 N downwards)
    [0, -100000], # Node 1
    [0, 0] # Node 2
])

# Assemble the global external forces vector F
F = np.zeros((n_dofs, 1))
for i, force in enumerate(external_forces):
    node_id = i
    # dofs = [2*node_id+1]  # Apply the force in the y-direction only (bruh)
    # F[dofs, 0] = force[1]

    # Apply the force in both the x and y directions
    dofs = [2*node_id, 2*node_id+1]
    F[dofs, 0] = force

# print(F)

# Add boundary conditions

# Support types
NONE = 0
PIN = 1
ROLLER = 2

supports = np.array([
    [0, PIN], # Node 0 is a pin support
    [1, NONE], # Node 1 is not a support
    [2, PIN] # Node 2 is a roller support
])

# Remove the rows and columns of the global stiffness matrix corresponding to the supports
removedOffset = 0
removedDofs = []

# Create temporary variables for K and F for solving
K_solve = K.copy()
F_solve = F.copy()

for support in supports:
    node_id, support_type = support
    if support_type == PIN:
        dofs = np.array([2*node_id, 2*node_id+1])
    elif support_type == ROLLER:
        dofs = np.arrya([2*node_id+1])
    else:
        continue

    # Remove the rows and columns
    K_solve = np.delete(K_solve, dofs - removedOffset, 0)
    K_solve = np.delete(K_solve, dofs - removedOffset, 1)

    # Remove the external forces
    F_solve = np.delete(F_solve, dofs - removedOffset, 0)

    # Update the offset
    removedOffset += len(dofs)

    # Add the removed dofs to the list
    removedDofs.extend(dofs)

# print(K * L / (E * A))
# print(F)

# print(removedDofs)

# Solve for the nodal displacements u
u = np.linalg.solve(K_solve, F_solve)

# print(u * 1000) # in mm

# Create a new U vector with the removed dofs as 0
U = np.zeros((n_dofs, 1))
U[removedDofs] = 0
U[np.setdiff1d(np.arange(n_dofs), removedDofs)] = u

# print(U)

# Calculate stresses
# loop through every member
stresses = []
forces = []
for i, member in enumerate(members):
    # Get node IDs
    i, j = member

    # Get node coordinates
    node_i = nodes[i]
    node_j = nodes[j]

    # Calculate length of member
    L = np.linalg.norm(node_j - node_i)

    # Calculate cosines and sines
    c = (node_j[0] - node_i[0]) / L
    s = (node_j[1] - node_i[1]) / L

    # Get the q vector
    # q is the vector of the displacements of the two dofs of the member
    q = np.array([U[2*i], U[2*i+1], U[2*j], U[2*j+1]]) # I think its correct?
    # print(q)

    # Compute the stress matrix M
    M = np.array([-c, -s, c, s])

    # Compute the stress
    stress = E / L * np.matmul(M, q)

    # Append the stress to the list
    stresses.append(stress)

    print("Stress in member", i, "is", round(stress[0], 2), "N/m") # WORKS

    # Calculate the force from the stress
    # stress = F / A
    # F = stress * A
    force = stress * A

    # Append the force to the list
    forces.append(force)

    print("Force in member", i, "is", round(force[0], 2), "N") # WORKS

# End the timer
end = time.time()

# Print the time taken in ms
print("Time taken:", round((end - start) * 1000, 2), "ms")


# ************************************
# Functions for plotting/visualisation
# ************************************

# View Truss (Nodes and Members)
def viewTruss(Nodes, Members):
    # Plot the nodes
    plt.scatter(Nodes[:, 0], Nodes[:, 1])

    # Plot the members
    for member in Members:
        # Get node IDs
        i, j = member

        # Get node coordinates
        node_i = Nodes[i]
        node_j = Nodes[j]

        # Plot the member
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k')

    # Show the plot
    plt.show()

# viewTruss(nodes, members)

# Create a function to render the deformed truss with the forces (red = compression, blue = tension)
def viewTrussDeformed(Nodes, Members, Displacements, Forces):
    # Move the nodes by the displacements
    for i, node in enumerate(Nodes):
        Nodes[i] = node + Displacements[2*i:2*i+2].T

    # Plot the nodes
    plt.scatter(Nodes[:, 0], Nodes[:, 1])

    # Plot the members
    for member in Members:
        # Get node IDs
        i, j = member

        # Get node coordinates
        node_i = Nodes[i]
        node_j = Nodes[j]

        # Plot the member
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k')

    # Plot the forces
    for i, member in enumerate(Members):
        # Get node IDs
        i, j = member

        # Get node coordinates
        node_i = Nodes[i]
        node_j = Nodes[j]

        # Get the force
        force = Forces[i]

        # Plot the force
        if force > 0:
            plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'b')
        else:
            plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'r')

# View Truss (Extras)
# Nodes, Members, Supports, Forces and Displacements
def viewTrussExtras(Nodes, Members, Supports, Forces, Displacements):
    # Plot the nodes
    plt.scatter(Nodes[:, 0], Nodes[:, 1])

    # Plot the members
    for member in Members:
        # Get node IDs
        i, j = member

        # Get node coordinates
        node_i = Nodes[i]
        node_j = Nodes[j]

        # Plot the member
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k')

    # Plot the supports
    for support in Supports:
        # Get node ID
        node_id, support_type = support

        # Get node coordinates
        node = Nodes[node_id]

        # Plot the support
        if support_type == PIN:
            plt.plot(node[0], node[1], 'ro')
        elif support_type == ROLLER:
            plt.plot(node[0], node[1], 'bo')

    # Plot the forces
    for i, force in enumerate(Forces):
        # Get node ID
        node_id = i

        # Get node coordinates
        node = Nodes[node_id]

        # Plot the force
        # plt.arrow(node[0], node[1], 0, 0.1, head_width=0.03, head_length=0.03, color='r')

    # Plot the displacements
    for i, displacement in enumerate(Displacements.reshape(-1, 2)):
        # Get node ID
        node_id = i

        # Get node coordinates
        node = Nodes[node_id]

        # if the displacements are 0, skip
        if np.all(displacement == 0):
            continue

        # Plot the displacement
        plt.arrow(node[0], node[1], displacement[0], displacement[1], head_width=0.01, head_length=0.01, color='b')

    # Draw the deformed truss
    viewTrussDeformed(Nodes, Members, Displacements, Forces)

    # Show the plot
    plt.show()

viewTrussExtras(nodes, members, supports, forces, U)