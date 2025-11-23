import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class NetworkConfig:
    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float
    beta: float
    epochs: int

    @staticmethod
    def from_json(filepath: str) -> 'NetworkConfig':
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return NetworkConfig(**config_dict)


class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config

        np.random.seed(123)

        # losowe wagi perceptronów w warstwie ukrytej
        self.w_hidden = np.random.randn(config.hidden_size, config.input_size) * 0.5
        self.w_hidden_bias = np.random.randn(config.hidden_size, 1) * 0.5

        # losowe wagi perceptronu w warstwie wyjściowej
        self.w_output = np.random.randn(config.output_size, config.hidden_size) * 0.5
        self.w_output_bias = np.random.randn(config.output_size, 1) * 0.5

        self.training_history: List[Dict[str, Any]] = []

    def _sigmoid(self, s: np.ndarray) -> np.ndarray:
        # funkcja aktywacji sigmoid
        return 1 / (1 + np.exp(-self.config.beta * s))

    def _sigmoid_derivative(self, s: np.ndarray) -> np.ndarray:
        # pochodna funkcji aktywacji sigmoid
        sig = self._sigmoid(s)
        return self.config.beta * sig * (1 - sig)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagacja w przód.

        Args:
            x: Wektor wejściowy

        Returns:
            s_hidden: Sumy ważone warstwy ukrytej
            v_hidden: Wyjścia warstwy ukrytej (po aktywacji)
            s_output: Sumy ważone warstwy wyjściowej
            y_output: Wyjście sieci (po aktywacji)
        """
        s_hidden = self.w_hidden @ x + self.w_hidden_bias
        v_hidden = self._sigmoid(s_hidden)

        s_output = self.w_output @ v_hidden + self.w_output_bias
        y_output = self._sigmoid(s_output)

        return s_hidden, v_hidden, s_output, y_output

    def backward(self, x: np.ndarray, d: np.ndarray, s_hidden: np.ndarray, v_hidden: np.ndarray, s_output: np.ndarray,
                 y_output: np.ndarray) -> float:
        """
        Propagacja wsteczna.

        Args:
            x: Wektor wejściowy
            d: Pożądane wyjście (label)
            s_hidden: Sumy ważone warstwy ukrytej
            v_hidden: Wyjścia warstwy ukrytej
            s_output: Sumy ważone warstwy wyjściowej
            y_output: Rzeczywiste wyjście sieci

        Returns:
            Wartość błędu dla tej próbki
        """
        # obliczenie błędu
        epsilon = d - y_output
        # MSE
        error = 0.5 * np.sum(epsilon ** 2)

        # obliczenie gradientu błędu po sumie dla warstwy wyjściowej
        de_ds_output = -epsilon * self._sigmoid_derivative(s_output)

        # obliczenie gradientów błędu po wagach dla warstwy wyjściowej
        # gradient dla wag
        de_dw_output = de_ds_output @ v_hidden.T
        # gradient dla biasu
        de_dw_output_bias = de_ds_output

        # obliczenie gradientu błędu po sumie dla warstwy wyjściowej
        # propagacja błędu do warstwy ukrytej
        de_ds_hidden = (self.w_output.T @ de_ds_output) * self._sigmoid_derivative(s_hidden)

        # obliczenie gradientów błędu po wagach dla warstwy ukrytej
        # gradient dla wag
        de_dw_hidden = de_ds_hidden @ x.T
        # gradient dla biasu
        de_dw_hidden_bias = de_ds_hidden

        # poruszamy się po funkcji celu w kierunku przeciwnym do gradientu (metoda gradientu prostego)
        # learning rate mówi jak duże kroki robimy
        # aktualizacja wag warstwy wyjściowej
        self.w_output -= self.config.learning_rate * de_dw_output
        self.w_output_bias -= self.config.learning_rate * de_dw_output_bias

        # aktualizacja wag warstwy ukrytej
        self.w_hidden -= self.config.learning_rate * de_dw_hidden
        self.w_hidden_bias -= self.config.learning_rate * de_dw_hidden_bias

        return error

    def train(self, x_train: np.ndarray, d_train: np.ndarray) -> None:
        n_samples = x_train.shape[1]

        for epoch in range(self.config.epochs):
            total_error = 0.0

            # trening na każdej próbce
            for i in range(n_samples):
                x = x_train[:, i:i + 1]
                d = d_train[:, i:i + 1]

                # propagacja w przód
                s_hidden, v_hidden, s_output, y_output = self.forward(x)

                # propagacja wsteczna i aktualizacja wag
                error = self.backward(x, d, s_hidden, v_hidden, s_output, y_output)
                total_error += error

            # średniego błędu dla epoki
            avg_error = total_error / n_samples
            self.training_history.append({
                'epoch': epoch + 1,
                'error': avg_error
            })

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, _, _, y_output = self.forward(x)
        return y_output.flatten()

    def get_weights_summary(self) -> Dict[str, Any]:
        return {
            'W_hidden': self.w_hidden,
            'w_hidden_bias': self.w_hidden_bias,
            'W_output': self.w_output,
            'w_output_bias': self.w_output_bias
        }

    def save_report(self, x_test: np.ndarray, d_test: np.ndarray, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("KONFIGURACJA SIECI:\n\n")
            f.write(f"Rozmiar wejścia: {self.config.input_size}\n")
            f.write(f"Perceptrony w warstwie ukrytej: {self.config.hidden_size}\n")
            f.write(f"Perceptrony w warstwie wyjściowej: {self.config.output_size}\n")
            f.write(f"Współczynnik uczenia (learning rate): {self.config.learning_rate}\n")
            f.write(f"Beta (parametr funkcji aktywacji sigmoid): {self.config.beta}\n")
            f.write(f"Liczba epok: {self.config.epochs}\n\n")

            f.write("STRUKTURA SIECI:\n\n")
            f.write("Warstwa 1 (Ukryta): 2 perceptrony\n")
            f.write("  - Perceptron 1: x(1), x(2) -> v(1)\n")
            f.write("  - Perceptron 2: x(1), x(2) -> v(2)\n")
            f.write("Warstwa 2 (Wyjściowa): 1 perceptron\n")
            f.write("  - Perceptron 3: v(1), v(2) -> y\n\n")

            f.write("HISTORIA TRENINGU:\n\n")
            f.write(f"{'Epoka':<10} {'Błąd':<20}\n")

            # pierwsze 10, ostatnie 10 i co setną epokę
            history = self.training_history
            epochs_to_show = set(range(min(10, len(history))))
            epochs_to_show.update(range(max(0, len(history) - 10), len(history)))
            epochs_to_show.update(range(0, len(history), 100))

            for idx in sorted(epochs_to_show):
                entry = history[idx]
                f.write(f"{entry['epoch']:<10} {entry['error']:<20.10f}\n")

            f.write("\n")

            f.write("KOŃCOWE WAGI:\n\n")
            weights = self.get_weights_summary()

            f.write("Wagi warstwy ukrytej:\n")
            f.write("  Perceptron 1: w1(1)={:.6f}, w1(2)={:.6f}, w1(0)={:.6f}\n".format(
                weights['W_hidden'][0, 0], weights['W_hidden'][0, 1],
                weights['w_hidden_bias'][0, 0]))
            f.write("  Perceptron 2: w2(1)={:.6f}, w2(2)={:.6f}, w2(0)={:.6f}\n\n".format(
                weights['W_hidden'][1, 0], weights['W_hidden'][1, 1],
                weights['w_hidden_bias'][1, 0]))

            f.write("Wagi warstwy wyjściowej:\n")
            f.write("  Perceptron 3: w3(1)={:.6f}, w3(2)={:.6f}, w3(0)={:.6f}\n\n".format(
                weights['W_output'][0, 0], weights['W_output'][0, 1],
                weights['w_output_bias'][0, 0]))

            f.write("WYNIKI NA ZBIORZE TESTOWYM:\n\n")
            predictions = self.predict(x_test)

            f.write(f"{'Wejście 1':<12} {'Wejście 2':<12} {'Oczekiwane':<12} {'Predykcja':<12} {'Zaokrąglone':<12} {'Poprawnie': <12}\n")
            f.write("-" * 80 + "\n")

            for i in range(x_test.shape[1]):
                x1, x2 = x_test[0, i], x_test[1, i]
                expected = d_test[0, i]
                predicted = predictions[i]
                rounded = round(predicted)
                correct = "TAK"
                if rounded != expected:
                    correct = "NIE"


                f.write(f"{x1:<12.4f} {x2:<12.4f} {expected:<12.0f} "
                        f"{predicted:<12.4f} {rounded:<12.0f} {correct:<12}\n")

            f.write("\n")

            # dokładność
            rounded_predictions = np.round(predictions)
            accuracy = np.mean(rounded_predictions == d_test[0, :]) * 100

            f.write("METRYKI:\n\n")
            f.write(f"Dokładność na zbiorze testowym (Accuracy): {accuracy:.2f}%\n")
            f.write(f"Średni błąd kwadratowy na zbiorze treningowym w ostatniej epoce (MSE): {self.training_history[-1]['error']:.10f}\n")
            f.write("\n")
