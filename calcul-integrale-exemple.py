from scipy.integrate import simps
from numpy import trapz

import numpy as np

def function(x):
    return x**2

x = np.arange(1,10,0.1)
y = function(x)

print(x)
print(y)

# primitive :

print("area: ", 1.0 / 3.0 * ( x[len(x)-1]**3 - x[0]**3 ))

# using Trapezoidal rule:

area = trapz(y,x)
print('area: ',area)

# using Simpson's rule:

area = simps(y,x)
print('area: ',area)