import numpy as np 
def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    B = A.T @ A
    if B[0, 0] == B[1, 1]:
        theta = 0.5 * (np.arctan2(2 * B[0, 1], B[0, 0] - B[1, 1]))
    else:
        theta = np.pi / 4

    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([[c, -s], [s, c]])
    D = R.T @ B @ R

    eigenvalues = np.linalg.eig(D)[0]
    sigma1 = np.sqrt(max(eigenvalues))
    sigma2 = np.sqrt(min(eigenvalues))
    s = np.diag([sigma1, sigma2])

    U = A @ R @ s

    return U, s, R.T