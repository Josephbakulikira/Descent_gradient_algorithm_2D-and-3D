import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm 
from sympy import symbols
from sympy import diff
import math

# 2D equation for Gradients descents Algorithms
def f(x):
	return  (x ** 4) - (4 * (x ** 2)) + 5

def df(x):
	return (4 * (x**3)) - (8 * x) 


x_array = np.linspace(-2, 2, 1000)

def gradient_descent(function, initial_input, learning_rate=0.02, precision= 0.0001):

	next_x = initial_input
	x_arr = [next_x]
	slope_array = [function(next_x)]

	for n in range(500):
		current_x = next_x
		gradient = function(current_x)
		next_x = current_x - learning_rate * gradient

		step_size = abs(next_x - current_x)
		x_arr.append(next_x)
		slope_array.append(function(next_x))

		if step_size < precision:
			break
	return next_x, x_arr, slope_array

minima, x_ar , deriv_ar = gradient_descent(df, 0.1)

# matplotlib graph
plt.figure(figsize=(15, 5))

#first graph 
plt.subplot(1,2,1)

plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)
plt.xlabel('X', fontsize=15)
plt.grid()
plt.ylabel('f(x)', fontsize=15)
plt.plot(x_array, f(x_array),color='green', linewidth=3, alpha=0.8)
plt.scatter(x_ar, f(np.array(x_ar)), color='red', s=100, alpha=0.6)

#first graph 
plt.subplot(1,2,2)

plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)
plt.xlabel('X', fontsize=15)
plt.grid()
plt.ylabel('g(x)', fontsize=15)
plt.plot(x_array, f(x_array),color='green', linewidth=3, alpha=0.8)
plt.scatter(x_ar, df(np.array(x_ar)), color='red', s=100, alpha=0.6)

plt.show()

# 3D graph 

def t(x, y):
    r = 3 ** (-x ** 2 - y **2)
    return 1/(r + 1)

x_4 = np.linspace(start=-2, stop=2, num=200)
y_4 = np.linspace(start=-2, stop=2, num=200)

x_4, y_4 =  np.meshgrid(x_4, y_4)
fig = plt.figure(figsize=(16, 12))
ax = fig.gca(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('t(x,y) - cost', fontsize=20)

ax.plot_surface(x_4, y_4, t(x_4, y_4), cmap=cm.hot ,alpha=0.6)
plt.show()