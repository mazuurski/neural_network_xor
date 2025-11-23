import json
from pathlib import Path

from src.network import NetworkConfig, NeuralNetwork
from src.utils import generate_xor_dataset, save_dataset, load_dataset


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    config_path = project_root / 'config.json'
    results_path = project_root / 'results.txt'

    data_dir.mkdir(exist_ok=True)

    x_train, d_train = generate_xor_dataset(n_samples=210)
    x_test, d_test = generate_xor_dataset(n_samples=90, random_seed=321)

    save_dataset(x_train, d_train, str(data_dir / 'train_set.json'))
    save_dataset(x_test, d_test, str(data_dir / 'test_set.json'))

    if not config_path.exists():
        default_config = {
            'input_size': 2,
            'hidden_size': 2,
            'output_size': 1,
            'learning_rate': 0.01,
            'beta': 1.0,
            'epochs': 965,
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

    config = NetworkConfig.from_json(str(config_path))
    x_train, d_train = load_dataset(str(data_dir / 'train_set.json'))
    x_test, d_test = load_dataset(str(data_dir / 'test_set.json'))

    network = NeuralNetwork(config)
    network.train(x_train, d_train)
    network.save_report(x_test, d_test, str(results_path))


if __name__ == '__main__':
    main()