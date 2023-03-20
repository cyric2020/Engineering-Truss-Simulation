import numpy as np
from scipy.linalg import solve

# Define the geometry and loading of the truss
nodes = np.array([[0, 0], [4, 3], [8, 0]]) # x, y coordinates of nodes
members = np.array([[0, 1], [1, 2], [0, 2]]) # indices of nodes for each member
loads = np.array([[0, -10], [0, -10], [0, -10]]) # loads applied to each node

# Calculate the length and orientation of each member
lengths = np.linalg.norm(nodes[members[:,0]] - nodes[members[:,1]], axis=1)
cosines = (nodes[members[:,1]] - nodes[members[:,0]]) / lengths[:, np.newaxis]
sines = np.array([[-c[1], c[0]] for c in cosines])

# Define the global stiffness matrix
num_nodes = nodes.shape[0]  # Total number of nodes
k = np.zeros((num_nodes*2, num_nodes*2))  # Initialize the global stiffness matrix

for i, (n1, n2) in enumerate(members):
    # Extract nodal coordinates for the truss member
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]

    # Calculate the length, cosine, and sine of the truss member
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    cosine = dx / length
    sine = dy / length

    # Define the local stiffness matrix for the truss member
    local_k = np.outer(np.array([cosine, sine, -cosine, -sine]), np.array([cosine, sine, -cosine, -sine])) / length

    # Add the local stiffness matrix to the global stiffness matrix
    indices = [n1*2, n1*2+1, n2*2, n2*2+1]  # Indices of the degrees of freedom in the global stiffness matrix
    k[np.ix_(indices, indices)] += local_k  # Add the local stiffness matrix to the global stiffness matrix

def print_matrix(m):
    # elegantly print the matrix
    s = [[str(round(e, 4)) for e in row] for row in m]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

print_matrix(k)

loads = np.concatenate([loads[:,0], loads[:,1]])
print(loads)

# check if the global stiffness matrix is singular
if np.linalg.det(k) == 0:
    print("The global stiffness matrix is singular")
    exit()

# Solve for the nodal displacements
u = solve(k, loads)

# Calculate the internal forces in each member
forces = np.zeros(members.shape[0])
for i, (n1, n2) in enumerate(members):
    forces[i] = (u[n2*2]-u[n1*2])*cosines[i,0] + (u[n2*2+1]-u[n1*2+1])*sines[i,0]

print(forces)