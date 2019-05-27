import numpy as np
import matplotlib.pyplot as plt
import random

def f(x, y):
    return ((y-x)**2+(1-x)**2)/2

x = np.linspace(-100, 100, 50)
y = np.linspace(-100, 100, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contourf(X, Y, Z, 20, cmap='Blues')
plt.colorbar();

colors = ['black','gray','red']
for i in range(0,3):
	#Generating random values for starting coordinates of the vector
    x_val = x[random.randint(10,25)]
    y_val = y[random.randint(10,25)]

    #Calculates the endpoints of the vector
    gradient_vector = [(2*x_val)-y_val-1, y_val-x_val]
    print gradient_vector
    plt.quiver([x_val],[y_val],[gradient_vector[0] -x_val], [gradient_vector[1]-y_val], color = [colors[i%3]], scale_units = 'xy', scale=1)

plt.show()
