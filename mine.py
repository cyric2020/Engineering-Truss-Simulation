import numpy as np

# Define global variables
E = 70*10**9 # Placeholder (Pa)
A = 20 # Placeholder (m^2)

# Define nodes
nodes = np.array([
    [0, 30],  # Node 0
    [50, 30], # Node 1
    [90, 0]   # Node 2
]);

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

    # print(k_m)

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

print(K * L / (E * A)) # correct so far

# Finite Element Equation
# [K]{u} = {F}

# Define external forces
external_forces = np.array([
    [0, 0], # Node 0
    [0, -12000], # Node 1 (12,000 N downwards)
    [0, 0] # Node 2
])

# Assemble the global external forces vector F
F = np.zeros((n_dofs, 1))
for i, force in enumerate(external_forces):
    node_id = i
    dofs = [2*node_id+1]  # Apply the force in the y-direction only
    F[dofs, 0] = force[1]

print(F)

# Solve for the nodal displacements u
# u = np.linalg.solve(K, F)

# Use lstsq to solve for u (least squares)
u, residuals, rank, s = np.linalg.lstsq(K, F, rcond=None)

# Calculate the member forces f
f = []
for member, stiffness in zip(members, stiffnessMatrices):
    i, j = member
    dofs = [2*i, 2*i+1, 2*j, 2*j+1]
    u_e = u[dofs]
    f_e = np.dot(stiffness, u_e)
    f.append(f_e)

print(f)