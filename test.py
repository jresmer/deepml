import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

indices = np.random.permutation(len(X) - 1)

print(X[indices])