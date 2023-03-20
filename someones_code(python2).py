#! /usr/bin/env python

'''
A modification of the truss analyser. This one performs simulated annealing to
attempt to improve the bridge design (purely by moving joint positions, not by
changing any member connections)
Copyright (c) 2013 Bennett Lovelady
pseudocode:
s = s0, e = E(s)
s_best = s, e_best = e
k = 0
while k<k_max && e>e_max:    # i.e. time left && not good enough
    T = Temperature(k/k_max)
    s_new = SomeNeighbour(s)
    e_new = E(s_new)
    if P(e,e_new,T) > rand():
        s = s_new, e = e_new
    if e_new < e_best:
        s_best = s_new, e_best = e_new
    k++
return s_best
E(state) returns the energy of a given state
Temperature(float) returns a float to represent the cooling temperature
SomeNeighbour(state) returns a nearby state (e.g. jiggled a bit)
P(e,e',T) returns probability of moving from s->s' at temp T
    usually defined as 1 if e'<e, or exp(-(e'-e)/T) otherwise
Can define restart(): s=s_best, e=e_best to help with cooling
In a truss context the energy can be replaced by maximum load which
is maximised rather than minimised
new cooling schedule:
iterate(k++)
if k = kmax: k = k0	# heat it back to 'hot' temperature
if no improvements in last "m" tries: reset to the best
if "n" resets in a row: convergence achieved! 
'''

import numpy as np
import copy

bridgenumber = raw_input('input the numbered file in "/anneal" containing the .in files: ')
inpath = '/home/bennett/Documents/ensc1002/anneal/' + bridgenumber + '/'
collating = True
outpath = inpath + 'results/'
outputname = str(bridgenumber)


'''
---input files---
joints: 
[joint #] [x coord (mm)] [y coord (mm)] [degree of freedom]
memers:
[member #] [joint i] [joint j] [dx] [dy] [L] [lij] [mij] [maxL_y] [maxL_b] [maxL]
supports:
[joint # with x-reaction]
[joint # with y-reaction]
[joint # with y-reaction]
applied:
[joint #] [F_x] [F_y]
material:
list of material properties accessed from a dictionary
'''

# ------ a single data structure for the truss ------
class Truss:
    def __init__(self, joints_, members_, supports_, applied_, material_):
        self.joints = joints_ 
        self.members = members_
        self.supports = supports_
        self.applied = applied_
        self.material = material_
        self.C = None
        self.P = None
        self.Q = None
        self.maxL = None
        self.maxL_m = None
        self.singular = None

    def copy(self):
        return copy.deepcopy(self)

    def LoadData(self):
        # set up the data tables and prepare the matrices

        # applied forces
        n = self.joints.shape[0]
        self.P = np.zeros(2*n)
        num = self.applied.shape[0]-1
        for i in range(1,num+1):
            self.P[2*self.applied[i,0]-2] = -self.applied[i,1]
            self.P[2*self.applied[i,0]-1] = -self.applied[i,2]
        # members table
        for k in range(self.members.shape[0]):
            i   = self.members[k,1]                       # Joint i (referenced by number)
            j   = self.members[k,2]                       # Joint j (referenced by number)
            dx  = self.joints[j-1,1] - self.joints[i-1,1] # x distance
            dy  = self.joints[j-1,2] - self.joints[i-1,2] # y distance
            L   = np.sqrt(dx**2 + dy**2)                  # length
            lij = dx/L                                    # cosine
            mij = dy/L                                    # sine
            self.members[k,3], self.members[k,4] = dx,dy
            self.members[k,5], self.members[k,6], self.members[k,7] = L, lij, mij
        # coefficient matrix
        Cm = np.zeros((2*n,2*n))
        for k in range(self.members.shape[0]):
            i   = self.members[k,1] # Joint i (referenced by number)
            j   = self.members[k,2] # Joint j (referenced by number)
            lij = self.members[k,6] # cosine
            mij = self.members[k,7] # sine
            Cm[2*i-2, k] = lij
            Cm[2*i-1, k] = mij
            Cm[2*j-2, k] = -lij
            Cm[2*j-1, k] = -mij
        # now add the coefficients for the reaction forces:
        Cm[2*self.supports[0]-2, 2*n-3] = 1 # First support X Reaction
        Cm[2*self.supports[1]-1, 2*n-2] = 1 # First support Y Reaction
        Cm[2*self.supports[2]-1, 2*n-1] = 1 # Second support Y Reaction
        self.C = Cm
        # degrees of freedom for each joint
        # roadbed can move l/r
        for i in range(self.joints.shape[0]):
            if self.joints[i,2] == 0:
                self.joints[i,3] = 1
            else:
                self.joints[i,3] = 3
        # supports can't move
        self.joints[self.supports[0]-1,3] = 0
        self.joints[self.supports[1]-1,3] = 0
        self.joints[self.supports[2]-1,3] = 0
        # loaded joints can't move
        for i in range(1,num+1):
            self.joints[self.applied[i,0]-1,3] = 0

    def Solve(self):
        self.singular = False
        try:
            # invert the matrix
            self.Q = np.linalg.solve(self.C,self.P)
        except np.linalg.LinAlgError:
            # if inversion doesn't work use numerical approximation
            print "singular matrix. using least squares.."
            self.singular = True
            self.Q = np.linalg.lstsq(self.C,self.P)[0]
        for i in range(len(self.Q)):
            # it sometimes gives tiny values like 1e-13, let's get rid of those:
            if abs(self.Q[i]) < 0.01:
                self.Q[i] = 0.0
        # find the maximum load
        for k in range(self.members.shape[0]):
            if self.Q[k] != 0:
                # yield: F/A <= s.f * s_y
                L_yield = self.material['sf']*self.material['s_y']*self.material['A']/abs(self.Q[k])
                self.members[k,8] = L_yield
                # buckling: F <= s.f * pi^2 * E * I / l^2
                if self.Q[k] < 0:
                    L_buckling = self.material['sf']*np.pi**2*self.material['E']*self.material['I'] \
                                    / (abs(self.Q[k])*self.members[k,5]**2)
                    self.members[k,9] = L_buckling
                    # which is the smaller limit?
                    self.members[k,10] = min(self.members[k,8], self.members[k,9])
                else:
                    self.members[k,9] = 0
                    self.members[k,10] = self.members[k,8]
        self.maxL = 1e6
        for k in range(self.members.shape[0]):
            m = self.members[k,10]
            if m != 0 and m <= self.maxL:
                self.maxL = m
        self.maxL_m = []
        for k in range(self.members.shape[0]):
            # to account for floating point errors,
            # check if the member's maxL is within 1% of maxL
            if np.allclose([self.members[k,10]], [self.maxL], rtol=1e-2):
                self.maxL_m.append(k)

    # maximum load of each member as a string
    def MaxLStrings(self):
        maxL_str = ['']*self.members.shape[0]
        for k in range(self.members.shape[0]):
            m = self.members[k,10]
            maxL_str[k] = '---' if m == 0 else str(round(m,2))
        return maxL_str
    
    # maximum x-coord => bridge length
    def TrussLength(self):
        # Loop through every joint and find the one with the largest x-coord
        maxX = 0
        for i in range(self.joints.shape[0]):
            if self.joints[i,1] > maxX:
                maxX = self.joints[i,1]
        return maxX
    
    # total length of material used
    def TotalMaterial(self):
        # Loop through every member and add up the lengths
        sumL = 0
        for k in range(self.members.shape[0]):
            sumL = sumL + self.members[k,5]
        return sumL
        


# ------ functions for annealing ------


def Temperature(k, kmax):
    #t = 1-k/kmax
    #t = np.exp(-k/kmax)
    #t = 1/(1+np.exp(k/kmax))
    #t = 2/(1+np.exp(10*k/kmax))
    t = 2/(1+np.exp(15*k/kmax))
    return t

def Prob(load1, load2, temp):
    if load2 > load1:
        return 1.0
    else:
        return np.exp(-10*(load1-load2)/temp)        

# ------ functions for output ------
# some things used by both regular and html output sections:
# tension as a string
def TensionState(x):
    if (x<0):
        return 'comp.'
    elif (x>0):
        return 'tens.'
    else:
        return ''

# ------ load initial data ------

# material properties
material = {}
with open(inpath+'material.in', 'r') as fin:
    for line in fin:
        if (line[0] != '#'):
            spl = line.split()
            if len(spl)==2:
                (key, value) = spl 
                material[key] = float(value)

# joints
with open(inpath+'joints.in','r') as fin:
    joints = np.genfromtxt(fin, comments="#", delimiter="\t")

# supports 
with open(inpath+'supports.in', 'r') as fin:
    supports = np.genfromtxt(fin, comments="#")
reactions = ['R'+str(int(supports[0]))+'x',\
             'R'+str(int(supports[1]))+'y',\
             'R'+str(int(supports[2]))+'y']
coord = ['x', 'y', 'y']

# applied forces
with open(inpath+'applied.in','r') as fin:
    applied = np.genfromtxt(fin, comments="#", delimiter="\t")
n = joints.shape[0]
P = np.zeros(2*n)
num = applied.shape[0]-1
for i in range(1,num+1):
    P[2*applied[i,0]-2] = -applied[i,1]
    P[2*applied[i,0]-1] = -applied[i,2]

# members
with open(inpath+'members.in', 'r') as fin:
    members = np.genfromtxt(fin, comments="#", delimiter="\t")
for n in range(8):
    members = np.concatenate((members, [[0]]*members.shape[0]), 1)
# members table now has five extra columns (for geometric data) 
# and three more for max load results

truss = Truss(joints, members, supports, applied, material)
truss.LoadData()
truss.Solve()


# ------ the main annealing loop ------
# always chooses a better bridge, may choose a worse bridge
# if the temp is high enough
# if it reaches "maxResets" resets in a row, the solution has converged
k0 = 0.0
k = k0
kmax = float(raw_input("length of heating cycle? "))
iterations = 0
fails = 0
failsPerReset = 400
resets = 0
maxResets = 5
truss_best = truss
hitsSinceLastDraw = 0
hitsPerDraw = 20
f = open(outpath + outputname + '.maxL.out','w')
# draw each improvement?
drawingSteps = True if raw_input("draw each step? [y/n] ") == 'y' else False

while resets < maxResets:
    joints_new = Jiggle(truss.joints)
    truss_new = Truss(joints_new, members, supports, applied, material)
    truss_new.LoadData()
    truss_new.Solve()
    if truss_new.maxL > truss_best.maxL:
        truss_best = truss_new.copy()
        resets = 0
        # draw it?
        if drawingSteps:
            hitsSinceLastDraw = hitsSinceLastDraw + 1
            if hitsSinceLastDraw == hitsPerDraw:
                hitsSinceLastDraw = 0
    T = Temperature(k, kmax)
    Pr = Prob(truss.maxL, truss_new.maxL, T)
    if Pr > np.random.random()**2:
        # good stuff, continue
        f.write('# ' + str(int(iterations)) + ':\t' + str(truss_new.maxL) + '\t' \
            + str(T) + '\t' + str(Pr) + '\n')
        truss = truss_new.copy()
    else:
        fails = fails + 1

    iterations = iterations + 1
    if (iterations % 500) == 0:
        print "iteration #" + str(int(iterations)) + " - T: " + str(T)
    
    k = k + 1
    if k >= kmax:
        #k0 = (kmax + k0)/2
        k = k0
        if abs(k-kmax) < 100:
            # don't bother heating by less than 50 units
            print "can't heat anymore.. ending simulation"
            break
        print "heating! k: " + str(k) + ", T: " + str(Temperature(k, kmax))

    if fails == failsPerReset:
        fails = 0
        resets = resets + 1
        truss = truss_best.copy()
        print "..reset! " + str(maxResets-resets) + " resets remain"

f.close()