import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	
    updated_weights = np.array(initial_weights)
    updated_bias = initial_bias

    features = np.array(features)
    labels = np.array(labels)

    mse_values = []
    n = labels.shape[0]

    for _ in range(epochs):

        z = features @ updated_weights + updated_bias
        sigma = 1 / (1 + np.exp(-z))

        loss = np.sum(np.power(sigma - labels, 2)) / n
        mse_values.append(loss)

        loss_gradient = 2/n * (sigma - labels)
        sigmoid_gradient = sigma * (1 - sigma)
        bias_gradient = loss_gradient @ sigmoid_gradient
        weight_gradient = loss_gradient * sigmoid_gradient @ features


        updated_weights -= learning_rate * weight_gradient
        updated_bias -= learning_rate * bias_gradient
    
    return np.round(updated_weights, 4), np.round(updated_bias, 4), mse_values
