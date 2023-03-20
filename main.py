import numpy as np

# joints
with open('joints.in','r') as fin:
    joints = np.genfromtxt(fin, comments="#", delimiter="\t")

# print(joints)

# members
with open('members.in', 'r') as fin:
    members = np.genfromtxt(fin, comments="#", delimiter="\t")
for n in range(8):
    members = np.concatenate((members, [[0]]*members.shape[0]), 1)
# members table now has five extra columns (for geometric data) 
# and three more for max load results

# print(members)

# supports 
with open('supports.in', 'r') as fin:
    supports = np.genfromtxt(fin, comments="#")
reactions = ['R'+str(int(supports[0]))+'x',\
             'R'+str(int(supports[1]))+'y',\
             'R'+str(int(supports[2]))+'y']

print(supports)
print(reactions)