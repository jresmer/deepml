import numpy as np

def train(X: np.ndarray, y: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray):
    """
    Train a model on the training set and return the model.
    """

    if len(X.shape) == 1 or X.shape[1] == 1:
        return None

    X_train = X[train_indices]
    y_train = y[train_indices]

    return np.round(np.linalg.inv(X.T @ X) @ X.T @ y, 1).tolist()

def validate(model: list[float], X: np.ndarray, y: np.ndarray, test_indices: np.ndarray):

    X_test = X[test_indices]
    y_test = y[test_indices]

    loss = np.sum(np.power(X_test @ model - y_test, 2)) / len(y_test)

    return loss

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    if shuffle:
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))

    fold_size = len(X) // k

    folds = []
    for i in range(k):
        first_index = i * fold_size
        test_indices = indices[first_index:first_index + fold_size]
        train_indices = np.concatenate((indices[:first_index], indices[first_index + fold_size:]))
        folds.append((train_indices.tolist(), test_indices.tolist()))

        # model = train(X, y, train_indices, test_indices)
        # if model is None:
        #     continue
        # loss = validate(model, X, y, test_indices)
        # output = f"model:\n {model}\n, fold:\n {test_indices}\n, loss: {loss}"
        # print(output)

    return folds