import math
from math import factorial as fact
import matplotlib.pyplot as plt
import numpy as np

def deg1(x):
    p = 1 + x
    return p

def deg2(x):
    q = 1 + x + ((x**2)/float(fact(2)))
    return q

def deg3(x):
    r = 1 + x + ((x**2)/float(fact(2))) + ((x**3)/float(fact(3)))
    return r

def deg4(x):
    s = 1 + x + ((x**2)/float(fact(2))) + ((x**3)/float(fact(3))) + ((x**4)/float(fact(4)))
    return s

x = np.arange(0,20,1)
line1, = plt.plot(x, deg1(x), color='g')
line2, = plt.plot(x, deg2(x), color='orange')
line3, = plt.plot(x, deg3(x), color='b')
line4, = plt.plot(x, deg4(x), color='r')
plt.xlabel('Values of x')
plt.ylabel('Value of e^x')
plt.title('Taylor series expansion of e^x')
plt.legend((line1, line2, line3, line4),('degree 1', 'degree 2', 'degree 3', 'degree 4'),loc='upper left')
plt.show()
