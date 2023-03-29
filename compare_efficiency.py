import time
import numpy as np

c = 0
s = 0
def func1():
    k_m = np.array([
        [c**2, c*s, -c**2, -c*s],
        [c*s, s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2, c*s],
        [-c*s, -s**2, c*s, s**2]
    ])

def func2():
    c2 = c**2
    cs = c*s
    s2 = s**2
    k_m = np.array([
        [c2, cs, -c2, -cs],
        [cs, s2, -cs, -s2],
        [-c2, -cs, c2, cs],
        [-cs, -s2, cs, s2]
    ])

# Time the functions
tests = 100_000
start = time.time()
for i in range(tests):
    func1()
end = time.time()
func1_time = end - start
print(f"func1: {func1_time}")

start = time.time()
for i in range(tests):
    func2()
end = time.time()
func2_time = end - start
print(f"func2: {func2_time}")

print(f"func2 is {func1_time / func2_time} times faster than func1")