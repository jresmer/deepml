import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Simulate the Central Limit Theorem (CLT).

    Args:
        num_samples: number of repeated samples to draw
        sample_size: size of each sample
        distribution: 'uniform' or 'exponential'

    Returns:
        Mean of the sample means (float)
    """

    match (distribution):
        case 'uniform':
            samples = [np.random.uniform(0, 1, sample_size) for _ in range(num_samples)]
        case 'exponential':
            samples = [np.random.exponential(1, sample_size) for _ in range(num_samples)]

    sample_means = [np.mean(sample) for sample in samples]

    return np.round(np.mean(sample_means), 4)
