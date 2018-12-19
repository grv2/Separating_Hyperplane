# Separating_Hyperplane
Separating Hyperplane. 
creating random dataset and 3D hyperplane separating. 
coding Starts From Next line.
import numpy as np
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel='rbf', gamma=0.1)
svr_lin = SVR(kernel="linear")
svr_poly = SVR(kernel='poly', degree = 2)
y_rbf = svr_rbf.fit(x, y).predict(x)
y_lin = svr_lin.fit(x, y).predict(x)
y_poly = svr_poly.fit(x, y).predict(x)

lw = 2
ax.scatter(x, y, color='orange', label='data')
ax.plot(x, y_rbf, color='navy', lw=lw, label='Rbf model')
ax.plot(x, y_lin, color='c', lw=lw, label='Linear model')
ax.plot(x, y_poly, color='green', lw=lw, label='Polynomial model')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.legend()
plt.show()
