import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        for i in range(len(y_pred)):
            L += -1/len(y_pred) * (y_true[i] * np.log(y_pred[i]) + (1 - y_true[i])* np.log(1- y_pred[i]))
        return np.round(L, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        for i in range(len(y_true)):
            L += -1/len(y_true) * np.sum(y_true[i] * np.log(y_pred[i]))
        return float(np.round(L, 4))
