import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)

        learning_rate = 0.01

        # learning_rate = 0.01

        # weights = initial_weights #[1,3]
        # for _ in range(num_iterations):
        #     pred = self.get_model_prediction(X, weights)
        #     weight_num = weights.shape[1]
        #     grads = np.zeros([1,3])
        #     for i in range(weight_num):
        #         grads[i] = self.get_derivative(pred, Y, len(X), X, i)
        #     weights = weights - learning_rate * grads
        # return np.round(weights, 5)
    
        weights = initial_weights #[1,3]
        for _ in range(num_iterations):
            pred = self.get_model_prediction(X, weights)
            weight_num = len(weights)
            grads = np.zeros([weight_num])
            for i in range(weight_num):
                grads[i] = self.get_derivative(pred, Y, len(X), X, i)
            weights = weights - learning_rate * grads
        return np.round(weights, 5)

def tmp_reshape(to_reshape):
        # torch.reshape() will be useful - check out the documentation
        m, n = to_reshape.shape[0], to_reshape.shape[1]
        out = torch.reshape(to_reshape, (m*n//2, 2))
        return out



if __name__ == "__main__":
    # X=np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    # Y=np.array([6.0, 3.0])
    # num_iterations=10
    # initial_weights=np.array([0.2, 0.1, 0.6])
    # s = Solution()
    # out = s.train_model(X, Y, num_iterations, initial_weights)
    # print(out)

    inp = torch.rand(4,3)
    out = tmp_reshape(inp)
    print(inp)
    inp2 = torch.rand(4,3)
    print(inp2)
    out2= torch.cat((inp, inp2), 0)
    print(out2)

    
