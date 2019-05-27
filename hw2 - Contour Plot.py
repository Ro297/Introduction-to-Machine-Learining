import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return ((y-x)**2+(1-x)**2)/2

x = np.linspace(-100, 100, 50)
y = np.linspace(-100, 100, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
#plt.contour(X, Y, Z, colors='black')
#plt.contour(X, Y, Z, 30, cmap='RdGy');
plt.contourf(X, Y, Z, 20, cmap='Blues')
plt.colorbar();
plt.show()
