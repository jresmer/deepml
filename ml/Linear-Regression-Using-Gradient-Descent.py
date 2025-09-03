import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:

    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(-1, 1)
    
    for i in range(iterations):
        theta = theta - alpha * (1/m) * X.T @ (X @ theta - y)

    return np.round(theta.flatten(), 4).tolist()