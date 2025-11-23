import json
from typing import Tuple

import numpy as np


def generate_xor_dataset(n_samples: int = 1000, random_seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)

    # XOR
    base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    base_outputs = np.array([0, 1, 1, 0])

    samples_per_point = n_samples // 4
    x_list = []
    d_list = []

    for i in range(4):
        # dodaj do każdej próbki szum N(0, 0.1)
        noise = np.random.randn(samples_per_point, 2) * 0.1
        samples = base_inputs[i] + noise
        samples = np.clip(samples, 0, 1)

        x_list.append(samples)
        d_list.extend([base_outputs[i]] * samples_per_point)

    x = np.vstack(x_list).T
    d = np.array(d_list).reshape(1, -1)

    return x, d

def save_dataset(x: np.ndarray, d: np.ndarray, filepath: str) -> None:
    data = {
        'inputs': x.T.tolist(),
        'outputs': d.T.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    x = np.array(data['inputs']).T
    d = np.array(data['outputs']).T
    return x, d