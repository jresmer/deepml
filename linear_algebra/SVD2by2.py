import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:

    ATA = A.T @ A

    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    sigma1 = np.sqrt(max(0, eigenvalues[0]))
    sigma2 = np.sqrt(max(0, eigenvalues[1]))

    s = np.diag([sigma1, sigma2])

    V = eigenvectors

    U = A @ V @ np.linalg.inv(s)

    return U, np.diag(s), V.T
