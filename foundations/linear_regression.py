import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        y = 0
        for i in range(len(weights)):
            y+= X[:,i] * weights[i]
        return np.round(y, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse = 0
        for i in range(len(model_prediction)):
            mse += 1/len(model_prediction) * (model_prediction[i] - ground_truth[i]) ** 2
        return float(np.round(mse[0], 5))

