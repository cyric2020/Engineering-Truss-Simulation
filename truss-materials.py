import numpy as np
import yaml
import matplotlib.pyplot as plt
import time

class Truss:
    def __init__(self):
        pass

    def loadData(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.Nodes = np.array(data['Joints'])
            self.Members = np.array(data['Members'])
            self.Supports = np.array(data['Supports'])
            self.ExternalForces = np.array(data['ExternalForces'])
            self.Materials = data['Materials']

        self.filename = filename

    def generateTable(self, columns, rows, sep="|"):
        columnLengths = [0] * len(columns)
        columnNames = columns
        for i, row in enumerate(rows):
            # the row is an array of values
            for j, value in enumerate(row):
                columnLengths[j] = max(columnLengths[j], len(str(value))) | max(columnLengths[j], len(" " + columnNames[j] + " "))
        table = ""

        # Add a seperator line
        for i, column in enumerate(columns):
            table += "-" * (columnLengths[i]+2) + sep
        table += "\n"

        # Add the column names
        for i, column in enumerate(columns):
            table += " " + column.ljust(columnLengths[i]) + " " + sep
        table += "\n"
        
        # Add the seperator
        for i, column in enumerate(columns):
            table += "-" * (columnLengths[i]+2) + sep
        table += "\n"

        # Add the rows
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table += " " + str(value).ljust(columnLengths[j]) + " " + sep

            table += "\n"

        # Add a seperator line
        for i, column in enumerate(columns):
            table += "-" * (columnLengths[i]+2) + sep
        table += "\n"

        return table
    
    def prettyMatrix(self, matrix, decimals=4):
        # Print out the matrix in a nice format
        # Get the maximum length of each column
        columnLengths = [0] * len(matrix[0])
        for i, row in enumerate(matrix):
            # the row is an array of values
            for j, value in enumerate(row):
                columnLengths[j] = max(columnLengths[j], len(str(round(value, decimals))))
        table = ""

        # Add the rows
        for i, row in enumerate(matrix):
            table += "["
            for j, value in enumerate(row):
                # Line up the numbers to the decimal point
                # get the length of ther remainder of the number after the decimal point
                remainderLength = len(str(round(value, decimals)))-len(str(round(value, decimals)).split('.')[0])-1

                numberLength = len(str(round(value, decimals)))

                # Add the spaces before the number
                table += " " * (columnLengths[j] - numberLength + remainderLength)

                # Add the number
                table += str(round(value, decimals))

                # Add the spaces after the number the inverse of the spaces before the number
                table += " " * (decimals - remainderLength) + " "

            table += "]\n"

        return table

    def generateReport(self):
        # Generate a report of the solved truss
        report = ""

        seperatorCharacter = "-"

        # Add the title
        title = "Truss Report for \"" + self.filename.split('.')[0] + "\" | " + time.strftime("%d/%m/%Y %H:%M:%S")
        report += seperatorCharacter * (len(title)+2) + "\n"
        report += " " + title + " \n"
        report += seperatorCharacter * (len(title)+2) + "\n\n"

        # Overview Debug data
        report += "--------\n"
        report += "Overview\n"
        report += "--------\n"
        report += "Nodes: " + str(len(self.Nodes)) + "\n"
        report += "Members: " + str(len(self.Members)) + "\n"
        report += "Supports: " + str(len(self.Supports)) + "\n"
        report += "External Forces: " + str(len(self.ExternalForces)) + "\n"
        report += "Solve Time: " + str(round(self.solveTime, 6)) + "s\n\n"

        # Node format:
        # Node ID | X | Y | Displacement X | Displacement Y
        report += "Nodes\n"

        # Generate a table for the nodes where each column is equal widths across rows
        # Get the maximum length of each column
        nodesRows = []
        for i, node in enumerate(self.Nodes):
            nodesRows.append([i, node[0], node[1], round(self.Displacements[2*i][0], 4), round(self.Displacements[2*i+1][0], 4), self.Supports[i][1]])
        nodesTable = self.generateTable(["Node ID", "X", "Y", "Displacement X", "Displacement Y", "Support Type"], nodesRows)
        report += nodesTable + "\n\n"


        # Member format:
        # Member ID | Node 1 | Node 2 | Material | Area | Force | Stress
        report += "Members\n"

        # Generate a table for the members where each column is equal widths across rows
        # Get the maximum length of each column
        membersRows = []
        for i, member in enumerate(self.Members):
            membersRows.append([i, member[0], member[1], member[2], member[3], round(self.Forces[i][0], 4), round(self.Stresses[i][0], 4)])
        membersTable = self.generateTable(["Member ID", "Node 1", "Node 2", "Material", "Area", "Force", "Stress"], membersRows)
        report += membersTable + "\n\n"


        # External Forces format:
        # Node ID | Force X | Force Y
        report += "External Forces\n"

        # Generate a table for the external forces where each column is equal widths across rows
        # Get the maximum length of each column
        externalForcesRows = []
        for i, force in enumerate(self.ExternalForces):
            externalForcesRows.append([i, force[0], force[1]])
        externalForcesTable = self.generateTable(["Node ID", "Force X", "Force Y"], externalForcesRows)
        report += externalForcesTable + "\n\n"


        # Reaction Forces format:
        # Node ID | Force X | Force Y
        report += "Reaction Forces\n"

        # Generate a table for the reaction forces where each column is equal widths across rows
        # Get the maximum length of each column
        reactionForcesRows = []
        for i, force in enumerate(self.R.reshape(-1, 2)):
            reactionForcesRows.append([i, round(force[0], 4), round(force[1], 4)])
        reactionForcesTable = self.generateTable(["Node ID", "Force X", "Force Y"], reactionForcesRows)
        report += reactionForcesTable + "\n\n"


        # Material format:
        # Material Name | Young's Modulus | Max Stress
        report += "Materials\n"

        # Generate a table for the materials where each column is equal widths across rows
        # Get the maximum length of each column
        materialsRows = []
        for i, material in enumerate(self.Materials):
            materialsRows.append([material, self.Materials[material]['E'], self.Materials[material]['MaxStress']])
        materialsTable = self.generateTable(["Material Name", "Young's Modulus", "Max Stress"], materialsRows)
        report += materialsTable + "\n\n"


        # Other debug info (K, U, R matrices)
        report += seperatorCharacter * 27 + "\n"
        report += "Debug Information (K, F, R)\n"
        report += seperatorCharacter * 27 + "\n\n"

        report += "K Matrix\n"
        # report += str(self.K) + "\n\n"
        # make the matrix more readable with equal column widths
        # matrixTable = self.generateTable([""]*len(self.K), self.K)
        matrixTable = self.prettyMatrix(self.K, 1)
        report += matrixTable + "\n\n"

        report += "U Matrix\n"
        # report += str(self.U) + "\n\n"
        report += self.prettyMatrix(self.U) + "\n\n"

        report += "R Matrix\n"
        # report += str(self.R) + "\n\n"
        report += self.prettyMatrix(self.R, 2) + "\n\n"

        return report

    def viewTruss(self):
        # Plot the nodes
        plt.scatter(self.Nodes[:, 0], self.Nodes[:, 1])

        # Plot the members
        for member in self.Members:
            # Get node IDs
            i, j = member

            # Get node coordinates
            node_i = self.Nodes[i]
            node_j = self.Nodes[j]

            # Plot the member
            plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k')

        # Make the plot axis equal
        plt.axis('equal')

        # Show the plot
        plt.show()

    def viewTrussDeformed(self, Displacements, Forces):
        """
        This function is used internally to plot the
        deflected truss. It is not intended to be used
        to draw the truss in its undeformed state.
        """

        # Make Nodes a float array
        Nodes = self.Nodes.astype(float)

        # Move the nodes by the displacements
        for i, node in enumerate(Nodes):
            displacement = Displacements[2*i:2*i+2].T

            Nodes[i] = node + displacement

        # Plot the nodes
        plt.scatter(Nodes[:, 0], Nodes[:, 1])

        # Plot the forces
        for o, member in enumerate(self.Members):
            # Get node IDs
            i, j, Material, A = member
            i, j = int(i), int(j)

            # Get node coordinates
            node_i = Nodes[i]
            node_j = Nodes[j]

            # Get the force
            force = Forces[o]

            # Plot the force
            if round(force[0], 2) > 0:
                plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'b')
            elif round(force[0], 2) < 0:
                plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'r')
            else:
                plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'g')

            # Add the force as a label with a white background
            plt.text((node_i[0] + node_j[0]) / 2, (node_i[1] + node_j[1]) / 2, str(round(force[0], 2)) + "N", fontsize=10, bbox=dict(facecolor='white', edgecolor='none', pad=1), horizontalalignment='center', verticalalignment='center')

    def viewTrussExtras(self, Displacements, Forces):
        # Plot the nodes
        plt.scatter(self.Nodes[:, 0], self.Nodes[:, 1])

        # Plot the members
        for member in self.Members:
            # Get node IDs
            i, j, Material, A = member
            i, j = int(i), int(j)

            # Get node coordinates
            node_i = self.Nodes[i]
            node_j = self.Nodes[j]

            # Plot the member
            plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k:')

        # Plot the supports
        for support in self.Supports:
            # Get node ID
            node_id, support_type = support

            # Convert node_id to an integer
            node_id = int(node_id)

            # Get node coordinates
            node = self.Nodes[node_id]

            # Plot the support
            if support_type == "PIN":
                plt.plot(node[0], node[1], 'ro')
            elif support_type == "ROLLER":
                plt.plot(node[0], node[1], 'bo')
        
        # Plot the forces
        # TODO

        # Plot the displacements
        for i, displacement in enumerate(Displacements.reshape(-1, 2)):
            # Get node ID
            node_id = i

            # Get node coordinates
            node = self.Nodes[node_id]

            # Id the displacement is zero, skip it
            if np.all(displacement == 0):
                continue

            # Plot the displacement
            plt.arrow(node[0], node[1], displacement[0], displacement[1], head_width=0.01, head_length=0.01, color="m")

        # Draw the reaction forces (arrows)
        for i, force in enumerate(self.R.reshape(-1, 2)):
            force_x = force[0]
            force_y = force[1]

            arrow_length = 1
            arrow_width = 0.1

            node = self.Nodes[i]

            node_x = node[0]
            node_y = node[1]

            if force_x != 0:
                # draw an arrow in the x direction at the support
                plt.arrow(node_x-arrow_length-arrow_width, node_y, arrow_length, 0, head_width=arrow_width, head_length=arrow_width, color="g")

                # draw the text for the reaction force at the base of the arrow
                plt.text(node_x-arrow_length-arrow_width, node_y, str(round(force_x, 2)) + "N", fontsize=10, bbox=dict(facecolor='white', edgecolor='none', pad=1), horizontalalignment='right', verticalalignment='center')
            if force_y != 0:
                # draw an arrow in the y direction at the support
                plt.arrow(node_x, node_y-arrow_length-arrow_width, 0, arrow_length, head_width=arrow_width, head_length=arrow_width, color="g")

                # draw the text for the reaction force at the base of the arrow
                plt.text(node_x, node_y-arrow_length-arrow_width, str(round(force_y, 2)) + "N", fontsize=10, bbox=dict(facecolor='white', edgecolor='none', pad=1), horizontalalignment='center', verticalalignment='top')


        # Draw the deformed truss
        self.viewTrussDeformed(Displacements, Forces)

        # Make the plot axis equal
        plt.axis('equal')

        # Show the plot
        plt.show()

    def singularityCheck(self, K):
        # Get the determinant of the matrix
        det = np.linalg.det(K)

        # If the determinant is zero, the matrix is singular
        return det == 0

    def solveTruss(self):
        # Start the timer
        start = time.time()

        # Create the global stiffness matrix
        n_nodes = self.Nodes.shape[0]
        n_dofs = 2 * n_nodes
        K = np.zeros((n_dofs, n_dofs))

        # Define stiffness matricies for each member
        stiffnessMatricies = []
        for member in self.Members:
            # Get node IDs
            i, j, Material, A = member
            i, j, Material, A = int(i), int(j), str(Material), float(A)
            E = float(self.Materials[Material]['E'])

            # Get node coordinates
            node_i = self.Nodes[i]
            node_j = self.Nodes[j]

            # Get the length of the member
            L = np.linalg.norm(node_j - node_i)

            # Calculate cosines and sines
            c = (node_j[0] - node_i[0]) / L
            s = (node_j[1] - node_i[1]) / L

            # Calculate the stiffness matrix
            k_m = np.array([
                [c**2, c*s, -c**2, -c*s],
                [c*s, s**2, -c*s, -s**2],
                [-c**2, -c*s, c**2, c*s],
                [-c*s, -s**2, c*s, s**2]
            ])

            # Calculate the stiffness matrix for the member
            k = (E * A / L) * k_m

            # Add the stiffness matrix to the list
            stiffnessMatricies.append(k)

            # Add the local stiffness matrix to the global stiffness matrix at the correct positions
            dofs = [2*i, 2*i+1, 2*j, 2*j+1] # Degrees of freedom

            K[np.ix_(dofs, dofs)] += k

        # Finite Element Equation
        # [K]{u} = {f}

        # Assemble the global external forces vector F
        F = np.zeros((n_dofs, 1))
        for i, force in enumerate(self.ExternalForces):
            # Get the node ID
            node_id = i

            # Apply the force in both the x and y directions
            dofs = [2*node_id, 2*node_id+1]
            F[dofs, 0] = force

        # Add boundary conditions

        # Remove the rows and columns of the global stiffness matrix corresponding to the supports
        removedOffset = 0
        removedDofs = []

        # Create temporary variables for K and F for solving
        K_solve = K.copy()
        F_solve = F.copy()

        for support in self.Supports:
            node_id, support_type = support

            # Convert node_id to int (because it is loaded as a string)
            node_id = int(node_id)

            if support_type == "PIN":
                dofs = np.array([2*node_id, 2*node_id+1])
            elif support_type == "ROLLER":
                dofs = np.array([2*node_id+1])
            else:
                continue

            # Remove the rows and columns
            K_solve = np.delete(K_solve, dofs - removedOffset, axis=0)
            K_solve = np.delete(K_solve, dofs - removedOffset, axis=1)

            # Remove the external forces
            F_solve = np.delete(F_solve, dofs - removedOffset, axis=0)

            # Update the offset
            removedOffset += len(dofs)   

            # Add the removed dofs to the list
            removedDofs.extend(dofs)

        # Check to see if the matrix is singular
        if self.singularityCheck(K_solve):
            raise Exception("The truss is not in equilibrium.")

        # Solve for the nodal displacements u
        u = np.linalg.solve(K_solve, F_solve)

        # Create a new U vector with the removed dofs as 0s
        U = np.zeros((n_dofs, 1))
        U[removedDofs] = 0
        U[np.setdiff1d(np.arange(n_dofs), removedDofs)] = u

        # Save the U vector
        self.U = U

        # Calculate stresses
        stresses = []
        forces = []
        for o, member in enumerate(self.Members):
            # Get node IDs
            i, j, Material, A = member
            i, j, Material, A = int(i), int(j), str(Material), float(A)
            E = float(self.Materials[Material]['E'])

            # Get node coordinates
            node_i = self.Nodes[i]
            node_j = self.Nodes[j]

            # Get the length of the member
            L = np.linalg.norm(node_j - node_i)

            # Calculate cosines and sines
            c = (node_j[0] - node_i[0]) / L
            s = (node_j[1] - node_i[1]) / L

            # Get the q vector
            q = np.array([U[2*i], U[2*i+1], U[2*j], U[2*j+1]])

            # compute the stress matrix M
            M = np.array([-c, -s, c, s])

            # Compute the stress
            stress = E / L * np.matmul(M, q)

            # Append the stress to the list
            stresses.append(stress)

            # Calculate the force from the stress
            force = stress * A

            # Append the force to the list
            forces.append(force)

        # Save the forces to the object
        self.Forces = forces

        # Save the stresses to the object
        self.Stresses = stresses

        # Calculate the reaction forces
        # R = K * U
        R = np.matmul(K, U)

        # Remove every reaction force that ISNT a support
        for support in self.Supports:
            node_id, support_type = support

            # Convert node_id to int (because it is loaded as a string)
            node_id = int(node_id)

            if support_type == "NONE":
                dofs = np.array([2*node_id, 2*node_id+1])
            elif support_type == "ROLLER":
                dofs = np.array([2*node_id])
            else:
                continue

            # Remove the rows
            R[dofs] = 0

        # Save R to the object
        self.R = R

        # Save K to the object
        self.K = K

        # Save stresses to the object
        self.stresses = stresses

        # Stop the timer
        end = time.time()
        self.solveTime = end - start

        return U, forces

myTruss = Truss()
# myTruss.loadData('example-truss-materials.yaml')
myTruss.loadData('truss-materials.yaml')

# myTruss.viewTruss()
# exit()

displacements, forces = myTruss.solveTruss()

# myTruss.viewTrussExtras(displacements, forces)

# exit()

# Print out K in a nice format
# print("K = ")
# for row in myTruss.K:
#     print("[", end="")
#     for col in row:
#         print("{:10.2f}".format(col), end=" ")
#     print("]")
# print()

# Save the report
myTruss.Displacements = displacements
report = myTruss.generateReport()
with open("report.txt", "w") as f:
    f.write(report)

# Print out how long it took to solve in ms
print("Solve time: {:.2f} ms".format(myTruss.solveTime * 1000))


