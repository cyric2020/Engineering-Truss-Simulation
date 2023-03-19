import numpy as np

# Define the geometry of the truss as a set of nodes and members
nodes = np.array([[0, 0], [0, 5], [0, 10]])
members = np.array([[0, 1], [1, 2]])

# Define the stiffness matrix for each member
E = 200e9  # Young's modulus of steel in Pa
A = 0.01  # Cross-sectional area of each member in m^2
L = np.linalg.norm(nodes[members[:, 1]] - nodes[members[:, 0]], axis=1)
k = (E * A / L)[:, np.newaxis] * np.array([[-1, 1], [1, -1]])

# Assemble the global stiffness matrix K
n_nodes = nodes.shape[0]
n_dofs = 2 * n_nodes  # Number of degrees of freedom (DOFs)
K = np.zeros((n_dofs, n_dofs))
for member, stiffness in zip(members, k):
    i, j = member
    dofs = [2*i, 2*i+1, 2*j, 2*j+1]
    K[np.ix_(dofs, dofs)] += np.tile(stiffness, (2, 2))

# Define the external forces applied to the truss
external_forces = np.array([[0, 25], [0, -50], [0, 25]])

# Assemble the global external forces vector F
F = np.zeros((n_dofs, 1))
for i, force in enumerate(external_forces):
    node_id = i
    dofs = [2*node_id+1]  # Apply the force in the y-direction only
    F[dofs, 0] = force[1]

# Solve for the nodal displacements u
u = np.linalg.solve(K, F)

# Calculate the member forces f
f = []
for member, stiffness in zip(members, k):
    i, j = member
    dofs = [2*i, 2*i+1, 2*j, 2*j+1]
    u_e = u[dofs]
    f_e = np.dot(stiffness, u_e)
    f.append(f_e)

# Print the results
print("Nodal displacements (m):")
print(u.reshape((n_nodes, 2)))
print("\nMember forces (N):")
print(np.array(f))