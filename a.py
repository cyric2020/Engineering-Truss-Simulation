import numpy as np

# Define the geometry of the truss
coordinates = np.array([[0, 0], [0, 3], [4, 0], [4, 3], [8, 0], [8, 3]])
members = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5]])

# Define external loads and reactions
external_forces = np.array([[0, -10], [0, -10], [0, -10]])
external_moments = np.array([0, 0, 0])

# Initialize the internal force matrix
num_members = len(members)
internal_forces = np.zeros((num_members, 2))

# Loop through each joint and solve for the internal forces
for i in range(num_members):
    member = members[i]
    node1 = member[0]
    node2 = member[1]

    # Calculate the member vector and length
    member_vec = coordinates[node2] - coordinates[node1]
    member_len = np.linalg.norm(member_vec)

    # Calculate the unit vector in the direction of the member
    unit_vec = member_vec / member_len

    # Calculate the forces acting on the joint
    forces = np.zeros((2,))
    moments = 0
    if node1 == 0:
        forces += external_forces[0]
        moments += external_moments[0]
    if node1 == 1:
        forces += external_forces[1]
        moments += external_moments[1]
    if node1 == 2:
        forces += external_forces[2]
        moments += external_moments[2]

    if node2 == 0:
        forces -= external_forces[0]
        moments -= external_moments[0]
    if node2 == 1:
        forces -= external_forces[1]
        moments -= external_moments[1]
    if node2 == 2:
        forces -= external_forces[2]
        moments -= external_moments[2]

    # Solve for the internal forces
    A = np.array([[unit_vec[0], -unit_vec[1]], [unit_vec[1], unit_vec[0]]])
    b = np.array([-forces[0], -forces[1]])
    x = np.linalg.solve(A, b)

    internal_forces[i] = x

    # Update the external loads and moments
    external_forces[node1] += x
    external_forces[node2] -= x
    external_moments[node1] += moments
    external_moments[node2] -= moments

# Print the internal forces
for i in range(num_members):
    member = members[i]
    print(f"Member {i+1} internal force: {internal_forces[i]}")